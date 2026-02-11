from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Dict, List, Optional

import pandas as pd

import soccer_ai.config as cfg
from soccer_ai.options import TouchOptions
from soccer_ai.pipelines import run_touch_detection

from .base import AnalysisResult, build_dummy_result


SNAPSHOT_MAX = 8
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


def analyze(video_path: str, settings: dict) -> AnalysisResult:
    test_name = settings.get("test_name", "Dribbling")
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
    last_touch_snapshot_frame = -999

    time_series: List[Optional[float]] = []
    distance_series: List[Optional[float]] = []
    avg_speed_series: List[Optional[float]] = []
    max_speed_series: List[Optional[float]] = []
    accel_series: List[Optional[float]] = []
    touch_rate_series: List[Optional[float]] = []
    touch_count_series: List[int] = []

    touch_events: List[Dict[str, Optional[float] | str | int]] = []
    touch_window: List[float] = []
    touch_window_seconds = float(settings.get("touch_window_seconds", 2.0))

    prev_time: Optional[float] = None
    prev_speed: Optional[float] = None
    distance_m = 0.0
    last_result = None
    peak_accel_mps2: Optional[float] = None
    peak_decel_mps2: Optional[float] = None
    prev_left = 0
    prev_right = 0

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
                if peak_accel_mps2 is None or accel_mps2 > peak_accel_mps2:
                    peak_accel_mps2 = accel_mps2
                if accel_mps2 < 0:
                    decel_val = abs(accel_mps2)
                    if peak_decel_mps2 is None or decel_val > peak_decel_mps2:
                        peak_decel_mps2 = decel_val

            if result.peak_accel_mps2 is not None:
                if peak_accel_mps2 is None or result.peak_accel_mps2 > peak_accel_mps2:
                    peak_accel_mps2 = result.peak_accel_mps2
            if result.peak_decel_mps2 is not None:
                if peak_decel_mps2 is None or result.peak_decel_mps2 > peak_decel_mps2:
                    peak_decel_mps2 = result.peak_decel_mps2

            left = result.left_touches
            right = result.right_touches
            delta_left = max(0, left - prev_left)
            delta_right = max(0, right - prev_right)
            total_touches = left + right

            if time_s is not None:
                for _ in range(delta_left):
                    touch_events.append(
                        {
                            "frame_idx": result.frame_idx,
                            "time_s": time_s,
                            "foot": "left",
                            "total_touches": total_touches,
                            "left_touches": left,
                            "right_touches": right,
                        }
                    )
                    touch_window.append(time_s)
                for _ in range(delta_right):
                    touch_events.append(
                        {
                            "frame_idx": result.frame_idx,
                            "time_s": time_s,
                            "foot": "right",
                            "total_touches": total_touches,
                            "left_touches": left,
                            "right_touches": right,
                        }
                    )
                    touch_window.append(time_s)

                cutoff = time_s - touch_window_seconds
                touch_window = [t for t in touch_window if t >= cutoff]

            touch_rate = None
            if time_s is not None and touch_window_seconds > 0:
                touch_rate = len(touch_window) / touch_window_seconds

            time_series.append(time_s)
            distance_series.append(distance_val)
            avg_speed_series.append(avg_speed_mps)
            max_speed_series.append(max_speed_mps)
            accel_series.append(accel_mps2)
            touch_rate_series.append(touch_rate)
            touch_count_series.append(total_touches)

            if avg_speed_mps is not None:
                prev_speed = avg_speed_mps
            prev_time = time_s
            prev_left = left
            prev_right = right

            touches_per_min = None
            if time_s is not None and time_s > 0:
                touches_per_min = total_touches / time_s * 60.0

            if callable(live_callback) and result.frame_idx % live_stride == 0:
                start = max(0, len(time_series) - live_tail_rows)
                speed_profile_tail = pd.DataFrame(
                    {
                        "time_s": time_series[start:],
                        "avg_speed_mps": avg_speed_series[start:],
                        "max_speed_mps": max_speed_series[start:],
                        "accel_mps2": accel_series[start:],
                        "distance_m": distance_series[start:],
                        "touch_rate": touch_rate_series[start:],
                        "touch_count": touch_count_series[start:],
                    }
                )
                touch_rate_tail = pd.DataFrame(
                    {
                        "time_s": time_series[start:],
                        "touch_rate": touch_rate_series[start:],
                        "touch_count": touch_count_series[start:],
                    }
                )
                touch_log_tail = pd.DataFrame(touch_events[-live_tail_rows:])

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
                            "touch_count": total_touches,
                            "left_touches": left,
                            "right_touches": right,
                            "touch_rate": touch_rate,
                            "touches_per_min": touches_per_min,
                        },
                        "speed_profile_tail": speed_profile_tail,
                        "touch_rate_tail": touch_rate_tail,
                        "touch_log_tail": touch_log_tail,
                        "progress": progress,
                    }
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
                            "peak_accel_mps2": peak_accel_mps2,
                            "peak_decel_mps2": peak_decel_mps2,
                            "total_touches": total_touches,
                            "left_touches": left,
                            "right_touches": right,
                        },
                    }
                )

            if shot_log is not None and result.shot_events is not None:
                shot_log.clear()
                shot_log.extend(result.shot_events)

            if snapshots is not None:
                if delta_left + delta_right > 0 and result.frame_idx - last_touch_snapshot_frame > 8:
                    if len(snapshots) < SNAPSHOT_MAX:
                        if snapshot_dir is None:
                            snapshot_dir = Path(tempfile.mkdtemp(prefix="dribbling_snapshots_"))
                            runtime_store["snapshot_dir"] = str(snapshot_dir)
                        filename = f"touch_{len(snapshots) + 1}_frame_{result.frame_idx}.jpg"
                        snapshot_path = snapshot_dir / filename
                        width, height = _save_snapshot(result.annotated, snapshot_path)
                        if width > 0 and snapshot_path.exists():
                            snapshots.append(
                                {
                                    "id": len(snapshots) + 1,
                                    "type": "touch",
                                    "frame_idx": result.frame_idx,
                                    "time_sec": time_s,
                                    "image_path": str(snapshot_path),
                                    "width": width,
                                    "height": height,
                                }
                            )
                            last_touch_snapshot_frame = result.frame_idx
                elif (
                    avg_speed_mps is not None
                    and len(snapshots) < SNAPSHOT_MAX
                    and (best_speed_snapshot is None or avg_speed_mps > best_speed_snapshot + 0.5)
                ):
                    if snapshot_dir is None:
                        snapshot_dir = Path(tempfile.mkdtemp(prefix="dribbling_snapshots_"))
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
    except Exception as exc:
        fallback = build_dummy_result(test_name, expected_matrices, settings)
        fallback.status = "fallback"
        fallback.logs = [f"Dribbling analysis failed: {exc}"] + fallback.logs
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

    avg_speed_mps = (
        last_result.avg_speed_kmh / 3.6 if last_result.avg_speed_kmh is not None else None
    )
    if avg_speed_mps is None and total_distance_m is not None and total_time_s:
        avg_speed_mps = total_distance_m / total_time_s

    max_speed_mps = (
        last_result.max_speed_kmh / 3.6 if last_result.max_speed_kmh is not None else None
    )
    if max_speed_mps is None:
        max_candidates = [s for s in max_speed_series if s is not None]
        max_speed_mps = max(max_candidates) if max_candidates else None

    total_touches = last_result.left_touches + last_result.right_touches
    touches_per_min = None
    if total_time_s and total_time_s > 0:
        touches_per_min = total_touches / total_time_s * 60.0

    touches_per_meter = None
    if total_distance_m and total_distance_m > 0:
        touches_per_meter = total_touches / total_distance_m

    speed_profile = pd.DataFrame(
        {
            "time_s": time_series,
            "avg_speed_mps": avg_speed_series,
            "max_speed_mps": max_speed_series,
            "accel_mps2": accel_series,
            "distance_m": distance_series,
            "touch_rate": touch_rate_series,
            "touch_count": touch_count_series,
        }
    )

    touch_log = pd.DataFrame(touch_events)
    touch_rate_df = pd.DataFrame(
        {
            "time_s": time_series,
            "touch_rate": touch_rate_series,
            "touch_count": touch_count_series,
        }
    )

    metrics = {
        "total_time_s": total_time_s,
        "total_distance_m": total_distance_m,
        "avg_speed_mps": avg_speed_mps,
        "max_speed_mps": max_speed_mps,
        "peak_accel_mps2": peak_accel_mps2,
        "peak_decel_mps2": peak_decel_mps2,
        "touch_count": total_touches,
        "left_touches": last_result.left_touches,
        "right_touches": last_result.right_touches,
        "touches_per_min": touches_per_min,
        "touches_per_meter": touches_per_meter,
    }

    logs.append(f"Processed frames: {len(time_series)}")
    logs.append(f"Touch events: {len(touch_events)}")

    return AnalysisResult(
        test_name=test_name,
        status="ok",
        metrics=metrics,
        matrices={
            "speed_profile": speed_profile,
            "touch_log": touch_log,
            "touch_rate": touch_rate_df,
        },
        artifacts={},
        logs=logs,
    )
