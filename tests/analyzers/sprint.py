from __future__ import annotations

from pathlib import Path
import math
import tempfile
from typing import Dict, List, Optional, Tuple

import pandas as pd

import soccer_ai.config as cfg
from soccer_ai.options import TouchOptions
from soccer_ai.pipelines import run_touch_detection

from .base import AnalysisResult, build_dummy_result


SNAPSHOT_MAX = 6
SNAPSHOT_WIDTH = 640
SNAPSHOT_JPEG_QUALITY = 90


def _resolve_weight(value: Optional[str], default: str) -> str:
    if not value or value == "Auto":
        return default
    return value


def _read_fps(video_path: str) -> float:
    try:
        import cv2  # type: ignore
    except Exception:
        return 30.0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 30.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else 30.0


def _save_snapshot(image_bgr, path: Path) -> tuple[int, int]:
    if image_bgr is None:
        return 0, 0
    try:
        import cv2  # type: ignore
    except Exception:
        return 0, 0

    image_to_save = image_bgr
    if SNAPSHOT_WIDTH and image_bgr.shape[1] > SNAPSHOT_WIDTH:
        scale = SNAPSHOT_WIDTH / image_bgr.shape[1]
        new_w = SNAPSHOT_WIDTH
        new_h = max(1, int(image_bgr.shape[0] * scale))
        image_to_save = cv2.resize(image_bgr, (new_w, new_h))
    cv2.imwrite(
        str(path),
        image_to_save,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(SNAPSHOT_JPEG_QUALITY)],
    )
    return image_to_save.shape[1], image_to_save.shape[0]


def _select_main_player(players: List[Dict]) -> Optional[Dict]:
    def area(player: Dict) -> float:
        bbox = player.get("bbox")
        if not bbox or len(bbox) != 4:
            return 0.0
        x1, y1, x2, y2 = bbox
        return max(0.0, (x2 - x1) * (y2 - y1))

    if not players:
        return None
    return max(players, key=area)


def _build_split_times(
    distance_series: List[Optional[float]],
    time_series: List[Optional[float]],
    split_distances: List[float],
) -> pd.DataFrame:
    rows = []
    for target in split_distances:
        split_time = None
        for dist, time_s in zip(distance_series, time_series):
            if dist is None or time_s is None:
                continue
            if dist >= target:
                split_time = time_s
                break
        rows.append({"distance_m": target, "time_s": split_time})
    return pd.DataFrame(rows)


def analyze(video_path: str, settings: dict) -> AnalysisResult:
    test_name = settings.get("test_name", "Sprint")
    expected_matrices = settings.get("expected_matrices", [])
    logs: List[str] = []

    try:
        import cv2  # type: ignore
    except Exception:
        fallback = build_dummy_result(test_name, expected_matrices, settings)
        fallback.status = "fallback"
        fallback.logs = ["OpenCV not available; returning placeholder result."] + fallback.logs
        return fallback

    detector_weights = _resolve_weight(settings.get("detector_weights"), cfg.DETECTOR_WEIGHTS)
    pose_weights = _resolve_weight(settings.get("pose_weights"), cfg.POSE_WEIGHTS)

    options = TouchOptions(
        detector_weights=detector_weights,
        pose_weights=pose_weights,
        display_stride=int(settings.get("display_stride", 1)),
        use_homography=bool(settings.get("use_homography", cfg.USE_HOMOGRAPHY)),
        ball_conf=float(settings.get("ball_conf", cfg.DET_CONF)),
        person_conf=float(settings.get("person_conf", cfg.DET_CONF)),
        det_imgsz=settings.get("det_imgsz"),
        ball_hold_frames=settings.get("ball_hold_frames"),
        ball_smoothing=settings.get("ball_smoothing"),
    )
    calibration_path = settings.get("calibration_path")
    if calibration_path:
        options.calibration_path = str(calibration_path)

    fps = _read_fps(video_path)
    logs.append(f"Loaded video: {Path(video_path).name}")
    logs.append(f"Using detector weights: {Path(detector_weights).name}")
    logs.append(f"Using pose weights: {Path(pose_weights).name}")
    if calibration_path:
        logs.append(f"Calibration loaded: {Path(calibration_path).name}")
    logs.append(f"Homography enabled: {options.use_homography}")

    live_callback = settings.get("live_callback")
    try:
        live_stride = int(settings.get("live_stride", 5))
    except (TypeError, ValueError):
        live_stride = 5
    live_stride = max(1, live_stride)
    try:
        live_tail_rows = int(settings.get("live_tail_rows", 8))
    except (TypeError, ValueError):
        live_tail_rows = 8
    live_tail_rows = max(1, live_tail_rows)
    total_frames_setting = settings.get("total_frames")

    runtime_store = settings.get("runtime_store")
    if isinstance(runtime_store, dict):
        frame_records = runtime_store.setdefault("frame_records", [])
        snapshots = runtime_store.setdefault("snapshots", [])
        shot_log = runtime_store.setdefault("shot_log", [])
    else:
        frame_records = None
        snapshots = None
        shot_log = None

    snapshot_dir = None
    best_speed_snapshot: Optional[float] = None

    time_series: List[Optional[float]] = []
    distance_series: List[Optional[float]] = []
    speed_series: List[Optional[float]] = []
    max_speed_series: List[Optional[float]] = []
    accel_series: List[Optional[float]] = []

    stride_rate_series: List[Optional[float]] = []
    step_rate_series: List[Optional[float]] = []
    contact_series: List[Dict[str, Optional[float] | str]] = []

    prev_time: Optional[float] = None
    prev_speed: Optional[float] = None
    distance_m = 0.0
    last_result = None
    peak_accel: Optional[float] = None
    peak_decel: Optional[float] = None

    contact_ratio = float(settings.get("sprint_contact_ratio", 0.9))
    contact_cooldown = int(settings.get("sprint_contact_cooldown_frames", 6))
    stride_window_s = float(settings.get("sprint_stride_window_s", 1.0))
    split_distances = settings.get("sprint_split_distances", [5, 10, 20, 30])
    try:
        split_distances = [float(x) for x in split_distances]
    except Exception:
        split_distances = [5, 10, 20, 30]

    left_contact_frame = -1000
    right_contact_frame = -1000
    contact_events: List[Tuple[float, str]] = []

    max_frames = settings.get("max_frames")
    try:
        max_frames_value = int(max_frames) if max_frames is not None else None
    except (TypeError, ValueError):
        max_frames_value = None
    if max_frames_value is not None and max_frames_value <= 0:
        max_frames_value = None

    try:
        for result in run_touch_detection(
            video_path,
            options=options,
            max_frames=max_frames_value,
        ):
            last_result = result
            time_s = result.total_time_sec
            if time_s is None:
                time_s = result.frame_idx / fps if fps > 0 else None

            avg_speed_mps = (
                result.avg_speed_kmh / 3.6 if result.avg_speed_kmh is not None else None
            )
            max_speed_mps = (
                result.max_speed_kmh / 3.6 if result.max_speed_kmh is not None else None
            )

            if result.total_distance_m is not None:
                distance_m = result.total_distance_m
                distance_val: Optional[float] = distance_m
            elif avg_speed_mps is not None and prev_time is not None and time_s is not None:
                dt = max(0.0, time_s - prev_time)
                distance_m += avg_speed_mps * dt
                distance_val = distance_m
            else:
                distance_val = None

            accel_mps2 = None
            if (
                prev_time is not None
                and time_s is not None
                and avg_speed_mps is not None
                and prev_speed is not None
            ):
                dt = max(1e-6, time_s - prev_time)
                accel_mps2 = (avg_speed_mps - prev_speed) / dt

            if accel_mps2 is not None:
                if peak_accel is None or accel_mps2 > peak_accel:
                    peak_accel = accel_mps2
                if accel_mps2 < 0:
                    decel_val = abs(accel_mps2)
                    if peak_decel is None or decel_val > peak_decel:
                        peak_decel = decel_val

            if avg_speed_mps is not None:
                prev_speed = avg_speed_mps
            prev_time = time_s

            time_series.append(time_s)
            distance_series.append(distance_val)
            speed_series.append(avg_speed_mps)
            max_speed_series.append(max_speed_mps)
            accel_series.append(accel_mps2)

            frame_meta = result.frame_meta or {}
            players_meta = frame_meta.get("players") or []
            main_player = _select_main_player(players_meta)
            contact_side = None

            if main_player:
                bbox = main_player.get("bbox")
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    height = max(1.0, float(y2) - float(y1))
                    left = main_player.get("left")
                    right = main_player.get("right")
                    if left is not None:
                        left_norm = (left[1] - y1) / height
                        if left_norm >= contact_ratio and result.frame_idx - left_contact_frame > contact_cooldown:
                            left_contact_frame = result.frame_idx
                            contact_side = "left"
                            if time_s is not None:
                                contact_events.append((time_s, "left"))
                    if right is not None:
                        right_norm = (right[1] - y1) / height
                        if right_norm >= contact_ratio and result.frame_idx - right_contact_frame > contact_cooldown:
                            right_contact_frame = result.frame_idx
                            contact_side = "right"
                            if time_s is not None:
                                contact_events.append((time_s, "right"))

            step_rate = None
            stride_rate = None
            if time_s is not None and stride_window_s > 0:
                cutoff = time_s - stride_window_s
                recent = [t for t, _side in contact_events if t >= cutoff]
                step_rate = len(recent) / stride_window_s if stride_window_s else None
                stride_rate = step_rate / 2.0 if step_rate is not None else None

            step_rate_series.append(step_rate)
            stride_rate_series.append(stride_rate)
            if contact_side:
                contact_series.append(
                    {"time_s": time_s, "side": contact_side, "step_rate_hz": step_rate}
                )

            if frame_records is not None:
                frame_records.append(
                    {
                        "frame_idx": result.frame_idx,
                        "meta": result.frame_meta,
                        "stats": {
                            "avg_speed_kmh": result.avg_speed_kmh,
                            "max_speed_kmh": result.max_speed_kmh,
                            "total_time_sec": time_s,
                            "total_distance_m": distance_val,
                            "peak_accel_mps2": peak_accel,
                            "peak_decel_mps2": peak_decel,
                            "stride_rate_hz": stride_rate,
                        },
                    }
                )

            if shot_log is not None and result.shot_events is not None:
                shot_log.clear()
                shot_log.extend(result.shot_events)

            if snapshots is not None:
                if (
                    avg_speed_mps is not None
                    and len(snapshots) < SNAPSHOT_MAX
                    and (best_speed_snapshot is None or avg_speed_mps > best_speed_snapshot + 0.5)
                ):
                    if snapshot_dir is None:
                        snapshot_dir = Path(tempfile.mkdtemp(prefix="sprint_snapshots_"))
                        runtime_store["snapshot_dir"] = str(snapshot_dir)
                    filename = f"speed_peak_{len(snapshots) + 1}_frame_{result.frame_idx}.jpg"
                    snapshot_path = snapshot_dir / filename
                    width, height = _save_snapshot(result.annotated, snapshot_path)
                    if width > 0 and snapshot_path.exists():
                        snapshots.append(
                            {
                                "id": len(snapshots) + 1,
                                "type": "speed_peak",
                                "frame_idx": result.frame_idx,
                                "time_sec": time_s,
                                "image_path": str(snapshot_path),
                                "width": width,
                                "height": height,
                            }
                        )
                        best_speed_snapshot = avg_speed_mps

            if callable(live_callback) and result.frame_idx % live_stride == 0:
                start = max(0, len(time_series) - live_tail_rows)
                speed_profile_tail = pd.DataFrame(
                    {
                        "time_s": time_series[start:],
                        "speed_mps": speed_series[start:],
                        "accel_mps2": accel_series[start:],
                        "distance_m": distance_series[start:],
                    }
                )
                stride_tail = pd.DataFrame(
                    {
                        "time_s": time_series[start:],
                        "step_rate_hz": step_rate_series[start:],
                        "stride_rate_hz": stride_rate_series[start:],
                    }
                )
                progress = None
                if total_frames_setting:
                    try:
                        total_frames_val = float(total_frames_setting)
                        if total_frames_val > 0:
                            progress = result.frame_idx / total_frames_val
                    except (TypeError, ValueError):
                        progress = None

                live_callback(
                    {
                        "frame_idx": result.frame_idx,
                        "frame_bgr": result.annotated,
                        "metrics": {
                            "total_time_s": time_s,
                            "total_distance_m": distance_val,
                            "avg_speed_mps": avg_speed_mps,
                            "max_speed_mps": max_speed_mps,
                            "peak_accel_mps2": peak_accel,
                            "peak_decel_mps2": peak_decel,
                            "stride_rate_hz": stride_rate,
                        },
                        "speed_profile_tail": speed_profile_tail,
                        "stride_rate_tail": stride_tail,
                        "progress": progress,
                    }
                )
    except Exception as exc:
        fallback = build_dummy_result(test_name, expected_matrices, settings)
        fallback.status = "fallback"
        fallback.logs = [f"Sprint analysis failed: {exc}"] + fallback.logs
        return fallback

    if last_result is None:
        fallback = build_dummy_result(test_name, expected_matrices, settings)
        fallback.status = "fallback"
        fallback.logs = ["No frames processed; returning placeholder result."] + fallback.logs
        return fallback

    total_time_s = last_result.total_time_sec
    if total_time_s is None and time_series:
        total_time_s = time_series[-1]

    total_distance_m = last_result.total_distance_m
    if total_distance_m is None and distance_series:
        total_distance_m = distance_series[-1]

    max_speed_mps = None
    max_candidates = [s for s in max_speed_series if s is not None]
    if max_candidates:
        max_speed_mps = max(max_candidates)

    avg_speed_mps = None
    if total_distance_m is not None and total_time_s:
        avg_speed_mps = total_distance_m / total_time_s

    time_to_90 = None
    time_to_95 = None
    if max_speed_mps and time_series:
        for t, s in zip(time_series, speed_series):
            if t is None or s is None:
                continue
            if time_to_90 is None and s >= 0.9 * max_speed_mps:
                time_to_90 = t
            if time_to_95 is None and s >= 0.95 * max_speed_mps:
                time_to_95 = t
        if time_to_90 is None:
            time_to_90 = total_time_s
        if time_to_95 is None:
            time_to_95 = total_time_s

    step_rates = [s for s in step_rate_series if s is not None]
    stride_rates = [s for s in stride_rate_series if s is not None]
    avg_step_rate = sum(step_rates) / len(step_rates) if step_rates else None
    avg_stride_rate = sum(stride_rates) / len(stride_rates) if stride_rates else None

    acceleration_phase = pd.DataFrame(
        {
            "time_s": time_series,
            "speed_mps": speed_series,
            "accel_mps2": accel_series,
            "pct_top_speed": [
                (s / max_speed_mps) if (s is not None and max_speed_mps) else None
                for s in speed_series
            ],
        }
    )

    top_speed_df = pd.DataFrame(
        [
            {
                "top_speed_mps": max_speed_mps,
                "time_to_90_pct_s": time_to_90,
                "time_to_95_pct_s": time_to_95,
                "avg_step_rate_hz": avg_step_rate,
                "avg_stride_rate_hz": avg_stride_rate,
            }
        ]
    )

    stride_freq_df = pd.DataFrame(
        {
            "time_s": time_series,
            "step_rate_hz": step_rate_series,
            "stride_rate_hz": stride_rate_series,
        }
    )

    speed_profile = pd.DataFrame(
        {
            "time_s": time_series,
            "speed_mps": speed_series,
            "accel_mps2": accel_series,
            "distance_m": distance_series,
        }
    )

    split_times = _build_split_times(distance_series, time_series, split_distances)

    metrics = {
        "total_time_s": total_time_s,
        "total_distance_m": total_distance_m,
        "avg_speed_mps": avg_speed_mps,
        "max_speed_mps": max_speed_mps,
        "peak_accel_mps2": peak_accel,
        "peak_decel_mps2": peak_decel,
        "time_to_90_pct_s": time_to_90,
        "time_to_95_pct_s": time_to_95,
        "avg_step_rate_hz": avg_step_rate,
        "avg_stride_rate_hz": avg_stride_rate,
    }

    logs.append(f"Processed frames: {len(time_series)}")

    return AnalysisResult(
        test_name=test_name,
        status="ok",
        metrics=metrics,
        matrices={
            "acceleration_phase": acceleration_phase,
            "top_speed": top_speed_df,
            "stride_frequency": stride_freq_df,
            "speed_profile": speed_profile,
            "split_times": split_times,
        },
        artifacts={},
        logs=logs,
    )
