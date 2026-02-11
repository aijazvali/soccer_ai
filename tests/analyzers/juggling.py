from __future__ import annotations

from pathlib import Path
from statistics import median
import math
import tempfile
from typing import Dict, List, Optional, Tuple

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


def _player_scale(players: List[Dict]) -> tuple[Optional[float], Optional[float]]:
    heights: List[float] = []
    ground_y: Optional[float] = None
    for person in players:
        bbox = person.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        _x1, y1, _x2, y2 = bbox
        height = float(y2) - float(y1)
        if height > 0:
            heights.append(height)
        if ground_y is None or y2 > ground_y:
            ground_y = float(y2)
    if not heights:
        return None, ground_y
    return median(heights), ground_y


def _calc_interval_std(values: List[float], window: int) -> Optional[float]:
    if window <= 1:
        return None
    sample = values[-window:]
    if len(sample) < 2:
        return None
    mean = sum(sample) / len(sample)
    variance = sum((v - mean) ** 2 for v in sample) / len(sample)
    return math.sqrt(variance)


def _calc_var(values: List[float], window: int) -> Optional[float]:
    if window <= 1:
        return None
    sample = values[-window:]
    if len(sample) < 2:
        return None
    mean = sum(sample) / len(sample)
    return sum((v - mean) ** 2 for v in sample) / len(sample)


def analyze(video_path: str, settings: dict) -> AnalysisResult:
    test_name = settings.get("test_name", "Juggling")
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

    gap_threshold = float(settings.get("juggling_gap_threshold_s", 1.0))
    missing_ball_frames = int(settings.get("juggling_missing_ball_frames", 10))
    ground_ratio = float(settings.get("juggling_ground_ratio", 0.92))
    ground_hold_frames = int(settings.get("juggling_ground_hold_frames", 3))
    use_player_height = bool(settings.get("juggling_use_player_height", True))
    min_height_ratio = float(settings.get("juggling_min_height_ratio", 0.1))
    stability_window = int(settings.get("juggling_stability_window", 10))
    touch_window_seconds = float(settings.get("juggling_touch_window_seconds", 2.0))

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
    max_streak_snapshot: Optional[int] = None

    time_series: List[Optional[float]] = []
    ball_height_series: List[Optional[float]] = []
    touch_rate_series: List[Optional[float]] = []
    touch_count_series: List[int] = []
    streak_series: List[int] = []

    touch_events: List[Dict[str, Optional[float] | str | int | bool]] = []
    stability_rows: List[Dict[str, Optional[float]]] = []
    ball_height_rows: List[Dict[str, Optional[float] | str]] = []
    touch_window: List[float] = []
    height_samples: List[float] = []

    prev_time: Optional[float] = None
    last_touch_time: Optional[float] = None
    prev_left = 0
    prev_right = 0

    current_streak = 0
    max_streak = 0
    drop_count = 0
    control_lost = False
    missing_ball_count = 0
    ground_contact_count = 0

    touch_intervals: List[float] = []
    last_result = None

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

            frame_meta = result.frame_meta or {}
            ball_meta = frame_meta.get("ball") or {}
            players_meta = frame_meta.get("players") or []

            frame_h = None
            if result.annotated is not None:
                frame_h = result.annotated.shape[0]

            ball_center = ball_meta.get("center")
            ball_y_norm = None
            height_norm = None
            height_ratio = None
            ground_y_norm = None
            height_proxy = None

            if ball_center is not None and frame_h:
                ball_y = float(ball_center[1])
                ball_y_norm = ball_y / frame_h
                height_norm = max(0.0, 1.0 - ball_y_norm)

                if use_player_height and players_meta:
                    player_height, ground_y = _player_scale(players_meta)
                    if player_height and player_height > 0:
                        height_px = max(0.0, ground_y - ball_y) if ground_y is not None else 0.0
                        height_ratio = height_px / player_height
                        if ground_y is not None:
                            ground_y_norm = ground_y / frame_h

            if use_player_height and height_ratio is not None:
                height_proxy = height_ratio
            elif height_norm is not None:
                height_proxy = height_norm

            if height_proxy is not None:
                height_samples.append(height_proxy)

            ball_height_rows.append(
                {
                    "time_s": time_s,
                    "ball_y_norm": ball_y_norm,
                    "height_norm": height_norm,
                    "height_ratio": height_ratio,
                    "ground_y_norm": ground_y_norm,
                    "height_proxy": height_proxy,
                    "source": "player_height" if height_ratio is not None else "image_y",
                }
            )

            if ball_center is None:
                missing_ball_count += 1
            else:
                missing_ball_count = 0

            ground_hit = False
            if ball_center is not None:
                if use_player_height and height_ratio is not None:
                    ground_hit = height_ratio <= min_height_ratio
                elif ball_y_norm is not None:
                    ground_hit = ball_y_norm >= ground_ratio

            if ground_hit:
                ground_contact_count += 1
            else:
                ground_contact_count = 0

            if missing_ball_count >= missing_ball_frames or ground_contact_count >= ground_hold_frames:
                control_lost = True

            left = result.left_touches
            right = result.right_touches
            delta_left = max(0, left - prev_left)
            delta_right = max(0, right - prev_right)

            event_left = prev_left
            event_right = prev_right

            def record_touch(foot: str, current_left: int, current_right: int) -> None:
                nonlocal current_streak, max_streak, drop_count, control_lost, missing_ball_count
                nonlocal ground_contact_count, last_touch_time

                interval = None
                if last_touch_time is not None and time_s is not None:
                    interval = time_s - last_touch_time

                streak_break = False
                if last_touch_time is not None:
                    if control_lost or (interval is not None and interval > gap_threshold):
                        streak_break = True

                if streak_break:
                    drop_count += 1
                    current_streak = 0

                current_streak += 1
                max_streak = max(max_streak, current_streak)

                if time_s is not None:
                    touch_window.append(time_s)
                    cutoff = time_s - touch_window_seconds
                    while touch_window and touch_window[0] < cutoff:
                        touch_window.pop(0)

                if interval is not None:
                    touch_intervals.append(interval)

                interval_std = _calc_interval_std(touch_intervals, stability_window)
                height_var = _calc_var(height_samples, stability_window)
                stability_score = None
                if interval_std is not None:
                    stability_score = 1.0 / (1.0 + interval_std)

                stability_rows.append(
                    {
                        "time_s": time_s,
                        "touch_interval_s": interval,
                        "interval_std": interval_std,
                        "height_var": height_var,
                        "stability_score": stability_score,
                    }
                )

                total_touches = current_left + current_right
                touch_events.append(
                    {
                        "frame_idx": result.frame_idx,
                        "time_s": time_s,
                        "foot": foot,
                        "total_touches": total_touches,
                        "left_touches": current_left,
                        "right_touches": current_right,
                        "streak_len": current_streak,
                        "touch_interval_s": interval,
                        "streak_break": streak_break,
                    }
                )

                last_touch_time = time_s
                control_lost = False
                missing_ball_count = 0
                ground_contact_count = 0

            for _ in range(delta_left):
                event_left += 1
                record_touch("left", event_left, event_right)
            for _ in range(delta_right):
                event_right += 1
                record_touch("right", event_left, event_right)

            total_touches = left + right
            touch_rate = None
            if time_s is not None and touch_window_seconds > 0:
                touch_rate = len(touch_window) / touch_window_seconds

            time_series.append(time_s)
            ball_height_series.append(height_proxy)
            touch_rate_series.append(touch_rate)
            touch_count_series.append(total_touches)
            streak_series.append(current_streak)

            prev_time = time_s
            prev_left = left
            prev_right = right

            touches_per_min = None
            if time_s is not None and time_s > 0:
                touches_per_min = total_touches / time_s * 60.0

            if callable(live_callback) and result.frame_idx % live_stride == 0:
                start = max(0, len(touch_events) - live_tail_rows)
                touch_tail = pd.DataFrame(touch_events[start:])
                stability_tail = pd.DataFrame(stability_rows[-live_tail_rows:])
                height_tail = pd.DataFrame(ball_height_rows[-live_tail_rows:])

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
                            "touch_count": total_touches,
                            "left_touches": left,
                            "right_touches": right,
                            "max_consecutive_touches": max_streak,
                            "current_streak": current_streak,
                            "touch_rate": touch_rate,
                            "touches_per_min": touches_per_min,
                            "ball_height": height_proxy,
                        },
                        "touch_count_tail": touch_tail,
                        "control_stability_tail": stability_tail,
                        "ball_height_tail": height_tail,
                        "progress": progress,
                    }
                )

            if frame_records is not None:
                frame_records.append(
                    {
                        "frame_idx": result.frame_idx,
                        "meta": result.frame_meta,
                        "stats": {
                            "total_time_sec": time_s,
                            "total_touches": total_touches,
                            "left_touches": left,
                            "right_touches": right,
                            "max_consecutive_touches": max_streak,
                        },
                    }
                )

            if shot_log is not None and result.shot_events is not None:
                shot_log.clear()
                shot_log.extend(result.shot_events)

            if snapshots is not None:
                if max_streak_snapshot is None or current_streak > max_streak_snapshot:
                    if len(snapshots) < SNAPSHOT_MAX and result.annotated is not None:
                        if snapshot_dir is None:
                            snapshot_dir = Path(tempfile.mkdtemp(prefix="juggling_snapshots_"))
                            runtime_store["snapshot_dir"] = str(snapshot_dir)
                        filename = f"streak_{current_streak}_frame_{result.frame_idx}.jpg"
                        snapshot_path = snapshot_dir / filename
                        width, height = _save_snapshot(result.annotated, snapshot_path)
                        if width > 0 and snapshot_path.exists():
                            snapshots.append(
                                {
                                    "id": len(snapshots) + 1,
                                    "type": "streak",
                                    "frame_idx": result.frame_idx,
                                    "time_sec": time_s,
                                    "image_path": str(snapshot_path),
                                    "width": width,
                                    "height": height,
                                    "value": current_streak,
                                }
                            )
                            max_streak_snapshot = current_streak
    except Exception as exc:
        fallback = build_dummy_result(test_name, expected_matrices, settings)
        fallback.status = "fallback"
        fallback.logs = [f"Juggling analysis failed: {exc}"] + fallback.logs
        return fallback

    if last_result is None:
        fallback = build_dummy_result(test_name, expected_matrices, settings)
        fallback.status = "fallback"
        fallback.logs = ["No frames processed; returning placeholder result."] + fallback.logs
        return fallback

    total_time_s = last_result.total_time_sec
    if total_time_s is None and time_series:
        total_time_s = time_series[-1]

    total_touches = last_result.left_touches + last_result.right_touches
    touches_per_min = None
    if total_time_s and total_time_s > 0:
        touches_per_min = total_touches / total_time_s * 60.0

    avg_touch_interval = None
    if touch_intervals:
        avg_touch_interval = sum(touch_intervals) / len(touch_intervals)

    stability_score = None
    interval_std_full = _calc_interval_std(touch_intervals, max(2, len(touch_intervals)))
    if interval_std_full is not None:
        stability_score = 1.0 / (1.0 + interval_std_full)

    max_height = None
    valid_heights = [h for h in ball_height_series if h is not None]
    if valid_heights:
        max_height = max(valid_heights)
        avg_height = sum(valid_heights) / len(valid_heights)
    else:
        avg_height = None

    touch_count_df = pd.DataFrame(touch_events)
    control_stability_df = pd.DataFrame(stability_rows)
    ball_height_df = pd.DataFrame(ball_height_rows)

    metrics = {
        "total_time_s": total_time_s,
        "touch_count": total_touches,
        "left_touches": last_result.left_touches,
        "right_touches": last_result.right_touches,
        "max_consecutive_touches": max_streak,
        "avg_touch_interval_s": avg_touch_interval,
        "touches_per_min": touches_per_min,
        "drop_count": drop_count,
        "stability_score": stability_score,
        "avg_ball_height": avg_height,
        "max_ball_height": max_height,
    }

    logs.append(f"Processed frames: {len(time_series)}")
    logs.append(f"Touch events: {len(touch_events)}")

    return AnalysisResult(
        test_name=test_name,
        status="ok",
        metrics=metrics,
        matrices={
            "touch_count": touch_count_df,
            "control_stability": control_stability_df,
            "ball_height": ball_height_df,
        },
        artifacts={},
        logs=logs,
    )
