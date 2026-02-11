from __future__ import annotations

from pathlib import Path
import math
from typing import Dict, List, Optional, Tuple

import pandas as pd

import soccer_ai.config as cfg
from soccer_ai.core import angle_diff_deg
from soccer_ai.options import TouchOptions
from soccer_ai.pipelines import run_touch_detection

from .base import AnalysisResult, build_dummy_result


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


def _as_point(value) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (float(value[0]), float(value[1]))
    return None


def _player_position(player: Optional[Dict]) -> tuple[Optional[Tuple[float, float]], bool]:
    if not player:
        return None, False

    left_field = _as_point(player.get("left_field"))
    right_field = _as_point(player.get("right_field"))
    bbox_field = _as_point(player.get("bbox_field"))

    if left_field or right_field or bbox_field:
        if left_field and right_field:
            return (
                ((left_field[0] + right_field[0]) / 2.0, (left_field[1] + right_field[1]) / 2.0),
                True,
            )
        if left_field:
            return left_field, True
        if right_field:
            return right_field, True
        return bbox_field, True

    left = _as_point(player.get("left"))
    right = _as_point(player.get("right"))
    if left or right:
        if left and right:
            return (
                ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0),
                False,
            )
        return left or right, False

    bbox = player.get("bbox")
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        return ((float(x1 + x2) / 2.0, float(y2)), False)
    return None, False


def _estimate_hr(
    speed_mps: Optional[float],
    ref_speed_mps: float,
    min_hr: float,
    max_hr: float,
) -> Optional[float]:
    if speed_mps is None:
        return None
    ref = max(ref_speed_mps, 0.1)
    effort = speed_mps / ref
    effort = max(0.0, min(1.2, effort))
    hr = min_hr + (max_hr - min_hr) * min(1.0, effort)
    return max(min_hr, min(max_hr, hr))


def _build_split_times(
    distances: List[Optional[float]],
    times: List[Optional[float]],
    total_distance_m: Optional[float],
) -> pd.DataFrame:
    split_rows: List[Dict[str, Optional[float] | str]] = []
    if total_distance_m is not None and total_distance_m > 0 and times:
        for frac in (0.25, 0.5, 0.75, 1.0):
            target = total_distance_m * frac
            time_at = None
            for distance, time_s in zip(distances, times):
                if distance is not None and time_s is not None and distance >= target:
                    time_at = time_s
                    break
            split_rows.append(
                {"segment": f"{int(frac * 100)}%", "distance_m": target, "time_s": time_at}
            )

        prev_time_val: Optional[float] = None
        for row in split_rows:
            current_time = row.get("time_s")
            if current_time is not None:
                if prev_time_val is None:
                    row["segment_time_s"] = current_time
                else:
                    row["segment_time_s"] = current_time - prev_time_val
                prev_time_val = current_time
            else:
                row["segment_time_s"] = None

    return pd.DataFrame(
        split_rows, columns=["segment", "distance_m", "time_s", "segment_time_s"]
    )


def analyze(video_path: str, settings: dict) -> AnalysisResult:
    test_name = settings.get("test_name", "Endurance")
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
        shot_log = runtime_store.setdefault("shot_log", [])
    else:
        frame_records = None
        shot_log = None

    time_series: List[Optional[float]] = []
    distance_series: List[Optional[float]] = []
    avg_speed_series: List[Optional[float]] = []
    max_speed_series: List[Optional[float]] = []
    accel_series: List[Optional[float]] = []

    heading_series: List[Optional[float]] = []
    turn_rate_series: List[Optional[float]] = []
    turn_angle_series: List[Optional[float]] = []
    turn_count_series: List[int] = []
    position_x_series: List[Optional[float]] = []
    position_y_series: List[Optional[float]] = []

    heart_rate_series: List[Optional[float]] = []
    rolling_speed_series: List[Optional[float]] = []
    fatigue_series: List[Optional[float]] = []

    turn_events: List[Dict[str, Optional[float] | str | int]] = []

    prev_time: Optional[float] = None
    prev_speed: Optional[float] = None
    distance_m = 0.0
    last_result = None
    peak_accel_mps2: Optional[float] = None
    peak_decel_mps2: Optional[float] = None

    prev_pos: Optional[Tuple[float, float]] = None
    prev_heading: Optional[float] = None
    prev_heading_time: Optional[float] = None
    last_turn_time: Optional[float] = None

    total_turns = 0
    total_turn_angle = 0.0
    max_turn_angle: Optional[float] = None
    max_turn_rate: Optional[float] = None

    min_turn_angle = float(settings.get("endurance_turn_min_angle_deg", 45.0))
    min_turn_move_m = float(settings.get("endurance_turn_min_move_m", 0.5))
    min_turn_move_px = float(settings.get("endurance_turn_min_move_px", 8.0))
    min_turn_gap_s = float(settings.get("endurance_turn_gap_s", 0.8))

    hr_min = float(settings.get("endurance_hr_min_bpm", 90.0))
    hr_max = float(settings.get("endurance_hr_max_bpm", 190.0))
    hr_ref_speed = float(settings.get("endurance_hr_speed_ref_mps", 7.0))

    fatigue_window_s = float(settings.get("endurance_fatigue_window_s", 10.0))
    baseline_sample_count = int(settings.get("endurance_fatigue_baseline_samples", 10))
    baseline_samples: List[float] = []
    baseline_speed: Optional[float] = None

    speed_window: List[Tuple[float, float]] = []
    window_sum = 0.0
    window_count = 0

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

            heading_deg_val: Optional[float] = None
            turn_rate_val: Optional[float] = None
            turn_angle_val: Optional[float] = None
            pos_x = None
            pos_y = None

            frame_meta = result.frame_meta or {}
            players = frame_meta.get("players") or []
            main_player = _select_main_player(players)
            pos, pos_is_field = _player_position(main_player)

            if pos is not None and time_s is not None:
                pos_x, pos_y = pos
                if prev_pos is None:
                    prev_pos = pos
                    prev_heading_time = time_s
                else:
                    dx = pos[0] - prev_pos[0]
                    dy = pos[1] - prev_pos[1]
                    move_dist = math.hypot(dx, dy)
                    min_move = min_turn_move_m if pos_is_field else min_turn_move_px
                    if move_dist >= min_move:
                        heading = math.atan2(dy, dx)
                        heading_deg_val = math.degrees(heading)
                        if prev_heading is not None and prev_heading_time is not None:
                            delta_deg = angle_diff_deg(heading, prev_heading)
                            dt = max(1e-6, time_s - prev_heading_time)
                            turn_rate_val = delta_deg / dt
                            turn_angle_val = delta_deg
                            if (
                                delta_deg >= min_turn_angle
                                and (last_turn_time is None or time_s - last_turn_time >= min_turn_gap_s)
                            ):
                                total_turns += 1
                                last_turn_time = time_s
                                total_turn_angle += delta_deg
                                if max_turn_angle is None or delta_deg > max_turn_angle:
                                    max_turn_angle = delta_deg
                                if turn_rate_val is not None:
                                    if max_turn_rate is None or turn_rate_val > max_turn_rate:
                                        max_turn_rate = turn_rate_val
                                turn_events.append(
                                    {
                                        "turn_id": total_turns,
                                        "time_s": time_s,
                                        "turn_angle_deg": delta_deg,
                                        "turn_rate_deg_s": turn_rate_val,
                                        "heading_deg": heading_deg_val,
                                        "pos_x": pos_x,
                                        "pos_y": pos_y,
                                        "pos_unit": "m" if pos_is_field else "px",
                                    }
                                )
                        prev_heading = heading
                        prev_heading_time = time_s
                        prev_pos = pos
                    else:
                        if prev_heading is not None:
                            heading_deg_val = math.degrees(prev_heading)

            hr_bpm = _estimate_hr(avg_speed_mps, hr_ref_speed, hr_min, hr_max)
            rolling_speed = None
            fatigue_score = None
            if time_s is not None and avg_speed_mps is not None:
                if baseline_speed is None:
                    baseline_samples.append(avg_speed_mps)
                    if len(baseline_samples) >= baseline_sample_count:
                        baseline_speed = sum(baseline_samples) / len(baseline_samples)

                speed_window.append((time_s, avg_speed_mps))
                window_sum += avg_speed_mps
                window_count += 1
                if fatigue_window_s > 0:
                    cutoff = time_s - fatigue_window_s
                    while speed_window and speed_window[0][0] < cutoff:
                        _t, _s = speed_window.pop(0)
                        window_sum -= _s
                        window_count -= 1
                if window_count > 0:
                    rolling_speed = window_sum / window_count
                if baseline_speed is not None and rolling_speed is not None and baseline_speed > 0:
                    fatigue_score = 1.0 - rolling_speed / baseline_speed
                    fatigue_score = max(0.0, min(1.0, fatigue_score))
            elif window_count > 0:
                rolling_speed = window_sum / window_count

            time_series.append(time_s)
            distance_series.append(distance_val)
            avg_speed_series.append(avg_speed_mps)
            max_speed_series.append(max_speed_mps)
            accel_series.append(accel_mps2)

            heading_series.append(heading_deg_val)
            turn_rate_series.append(turn_rate_val)
            turn_angle_series.append(turn_angle_val)
            turn_count_series.append(total_turns)
            position_x_series.append(pos_x)
            position_y_series.append(pos_y)

            heart_rate_series.append(hr_bpm)
            rolling_speed_series.append(rolling_speed)
            fatigue_series.append(fatigue_score)

            if avg_speed_mps is not None:
                prev_speed = avg_speed_mps
            prev_time = time_s

            if callable(live_callback) and result.frame_idx % live_stride == 0:
                start = max(0, len(time_series) - live_tail_rows)
                speed_profile_tail = pd.DataFrame(
                    {
                        "time_s": time_series[start:],
                        "avg_speed_mps": avg_speed_series[start:],
                        "max_speed_mps": max_speed_series[start:],
                        "accel_mps2": accel_series[start:],
                        "distance_m": distance_series[start:],
                    }
                )
                turn_tail = pd.DataFrame(
                    {
                        "time_s": time_series[start:],
                        "heading_deg": heading_series[start:],
                        "turn_rate_deg_s": turn_rate_series[start:],
                        "turns": turn_count_series[start:],
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

                turns_per_min_live = None
                if time_s is not None and time_s > 0:
                    turns_per_min_live = total_turns / time_s * 60.0

                live_callback(
                    {
                        "frame_idx": result.frame_idx,
                        "frame_bgr": result.annotated,
                        "metrics": {
                            "total_time_s": time_s,
                            "total_distance_m": distance_val,
                            "avg_speed_mps": avg_speed_mps,
                            "max_speed_mps": max_speed_mps,
                            "peak_accel_mps2": peak_accel_mps2,
                            "peak_decel_mps2": peak_decel_mps2,
                            "total_turns": total_turns,
                            "turns_per_min": turns_per_min_live,
                            "turn_rate_deg_s": turn_rate_val,
                            "heart_rate_bpm": hr_bpm,
                            "fatigue_index": fatigue_score,
                        },
                        "speed_profile_tail": speed_profile_tail,
                        "turn_profile_tail": turn_tail,
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
                            "total_turns": total_turns,
                        },
                    }
                )

            if shot_log is not None and result.shot_events is not None:
                shot_log.clear()
                shot_log.extend(result.shot_events)
    except Exception as exc:
        fallback = build_dummy_result(test_name, expected_matrices, settings)
        fallback.status = "fallback"
        fallback.logs = [f"Endurance analysis failed: {exc}"] + fallback.logs
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

    if baseline_speed is None:
        speed_candidates = [s for s in avg_speed_series if s is not None]
        if speed_candidates:
            baseline_speed = sum(speed_candidates[: baseline_sample_count or 1]) / len(
                speed_candidates[: baseline_sample_count or 1]
            )

    if max_turn_angle is None and turn_angle_series:
        turn_angles = [t for t in turn_angle_series if t is not None]
        max_turn_angle = max(turn_angles) if turn_angles else None

    avg_turn_angle = None
    if total_turns > 0:
        avg_turn_angle = total_turn_angle / total_turns

    turns_per_min = None
    if total_time_s and total_time_s > 0:
        turns_per_min = total_turns / total_time_s * 60.0

    avg_turn_rate = None
    turn_rate_candidates = [t for t in turn_rate_series if t is not None]
    if turn_rate_candidates:
        avg_turn_rate = sum(turn_rate_candidates) / len(turn_rate_candidates)

    avg_hr = None
    max_hr = None
    hr_candidates = [h for h in heart_rate_series if h is not None]
    if hr_candidates:
        avg_hr = sum(hr_candidates) / len(hr_candidates)
        max_hr = max(hr_candidates)

    fatigue_index = None
    for val in reversed(fatigue_series):
        if val is not None:
            fatigue_index = val
            break

    avg_pace_s_per_km = None
    avg_pace_min_per_km = None
    if avg_speed_mps and avg_speed_mps > 0:
        avg_pace_s_per_km = 1000.0 / avg_speed_mps
        avg_pace_min_per_km = avg_pace_s_per_km / 60.0

    pace_s_per_km_series: List[Optional[float]] = []
    pace_min_per_km_series: List[Optional[float]] = []
    for speed in avg_speed_series:
        if speed is not None and speed > 0:
            pace_s = 1000.0 / speed
            pace_s_per_km_series.append(pace_s)
            pace_min_per_km_series.append(pace_s / 60.0)
        else:
            pace_s_per_km_series.append(None)
            pace_min_per_km_series.append(None)

    speed_profile = pd.DataFrame(
        {
            "time_s": time_series,
            "avg_speed_mps": avg_speed_series,
            "max_speed_mps": max_speed_series,
            "accel_mps2": accel_series,
            "distance_m": distance_series,
        }
    )

    pace_profile = pd.DataFrame(
        {
            "time_s": time_series,
            "distance_m": distance_series,
            "speed_mps": avg_speed_series,
            "pace_s_per_km": pace_s_per_km_series,
            "pace_min_per_km": pace_min_per_km_series,
            "rolling_speed_mps": rolling_speed_series,
        }
    )

    heart_rate_df = pd.DataFrame(
        {
            "time_s": time_series,
            "heart_rate_bpm": heart_rate_series,
        }
    )

    fatigue_df = pd.DataFrame(
        {
            "time_s": time_series,
            "fatigue_score": fatigue_series,
            "rolling_speed_mps": rolling_speed_series,
            "baseline_speed_mps": [baseline_speed] * len(time_series),
        }
    )

    turn_profile = pd.DataFrame(
        {
            "time_s": time_series,
            "heading_deg": heading_series,
            "turn_rate_deg_s": turn_rate_series,
            "turn_angle_deg": turn_angle_series,
            "turns": turn_count_series,
            "pos_x": position_x_series,
            "pos_y": position_y_series,
        }
    )

    turn_events_df = pd.DataFrame(turn_events)

    split_times = _build_split_times(distance_series, time_series, total_distance_m)

    metrics = {
        "total_time_s": total_time_s,
        "total_distance_m": total_distance_m,
        "avg_speed_mps": avg_speed_mps,
        "max_speed_mps": max_speed_mps,
        "peak_accel_mps2": peak_accel_mps2,
        "peak_decel_mps2": peak_decel_mps2,
        "total_turns": total_turns,
        "turns_per_min": turns_per_min,
        "avg_turn_angle_deg": avg_turn_angle,
        "max_turn_angle_deg": max_turn_angle,
        "avg_turn_rate_deg_s": avg_turn_rate,
        "max_turn_rate_deg_s": max_turn_rate,
        "avg_heart_rate_bpm": avg_hr,
        "max_heart_rate_bpm": max_hr,
        "fatigue_index": fatigue_index,
        "avg_pace_s_per_km": avg_pace_s_per_km,
        "avg_pace_min_per_km": avg_pace_min_per_km,
    }

    logs.append(f"Processed frames: {len(time_series)}")
    if total_distance_m is None:
        logs.append("Distance unavailable; derived values may be empty.")
    if total_turns == 0:
        logs.append("No turn events detected with the current thresholds.")

    return AnalysisResult(
        test_name=test_name,
        status="ok",
        metrics=metrics,
        matrices={
            "speed_profile": speed_profile,
            "pace_profile": pace_profile,
            "heart_rate_estimate": heart_rate_df,
            "fatigue_index": fatigue_df,
            "turn_profile": turn_profile,
            "turn_events": turn_events_df,
            "split_times": split_times,
        },
        artifacts={},
        logs=logs,
    )
