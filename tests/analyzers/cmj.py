from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
from collections import deque

import numpy as np
import pandas as pd

import soccer_ai.config as cfg
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


def analyze(video_path: str, settings: dict) -> AnalysisResult:
    test_name = settings.get("test_name", "Counter Movement Jump (CMJ)")
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

    min_air_frames = int(settings.get("cmj_min_air_frames", cfg.JUMP_MIN_AIR_FRAMES))
    cooldown_frames = int(settings.get("cmj_cooldown_frames", cfg.JUMP_COOLDOWN_FRAMES))
    min_delta_px = float(settings.get("cmj_min_delta_px", cfg.JUMP_MIN_DELTA_PX))
    delta_ratio = float(settings.get("cmj_delta_ratio", cfg.JUMP_DELTA_RATIO))
    end_ratio = float(settings.get("cmj_end_ratio", cfg.JUMP_END_RATIO))
    up_px_per_frame = float(settings.get("cmj_up_px_per_frame", cfg.JUMP_UP_PX_PER_FRAME))
    up_streak_req = int(settings.get("cmj_up_streak", cfg.JUMP_UP_STREAK))
    landing_window_frames = int(settings.get("cmj_landing_window_frames", 8))
    landing_window_frames = max(1, landing_window_frames)

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

    time_series: List[Optional[float]] = []
    height_rows: List[Dict[str, Optional[float] | int | bool]] = []
    force_rows: List[Dict[str, Optional[float] | int]] = []
    landing_rows: List[Dict[str, Optional[float] | int]] = []
    jump_events: List[Dict[str, Optional[float] | int]] = []

    prev_time: Optional[float] = None
    prev_height_px: Optional[float] = None
    prev_height_m: Optional[float] = None
    prev_vel_px: Optional[float] = None
    prev_vel_m: Optional[float] = None
    prev_ankle_y: Optional[float] = None

    ground_ankle_history: deque = deque(maxlen=int(cfg.JUMP_GROUND_WINDOW))
    meters_per_px_history: deque = deque(maxlen=int(cfg.JUMP_SCALE_SMOOTHING))
    last_m_per_px: Optional[float] = None

    jump_active = False
    jump_id = 0
    jump_start_frame: Optional[int] = None
    jump_start_time: Optional[float] = None
    jump_peak_px = 0.0
    jump_peak_m: Optional[float] = None
    jump_air_streak = 0
    jump_up_streak = 0
    last_jump_end_frame = -1000

    total_jumps = 0
    highest_jump_px: Optional[float] = None
    highest_jump_m: Optional[float] = None
    last_jump_height_px: Optional[float] = None
    last_jump_height_m: Optional[float] = None
    last_flight_time_s: Optional[float] = None

    landing_collect: Optional[Dict[str, object]] = None
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
            players_meta = frame_meta.get("players") or []
            main_player = _select_main_player(players_meta)

            ankle_y = None
            ground_y = None
            bbox_height = None
            meters_per_px = None
            if main_player:
                bbox = main_player.get("bbox")
                if bbox and len(bbox) == 4:
                    _x1, y1, _x2, y2 = bbox
                    bbox_height = float(y2) - float(y1)
                    if bbox_height and bbox_height > 0:
                        meters_per_px = cfg.PLAYER_REF_HEIGHT_M / bbox_height
                left = main_player.get("left")
                right = main_player.get("right")
                if left is not None and right is not None:
                    ankle_y = float(max(left[1], right[1]))
                elif left is not None:
                    ankle_y = float(left[1])
                elif right is not None:
                    ankle_y = float(right[1])

            if meters_per_px is not None:
                meters_per_px_history.append(meters_per_px)
                if meters_per_px_history:
                    last_m_per_px = float(np.median(np.array(meters_per_px_history)))

            ankle_delta_px = None
            height_px = None
            height_m = None
            height_ratio = None
            base_threshold = None
            end_threshold = None
            noise_px = 0.0
            hip_ankle_px = (
                bbox_height * cfg.HIP_TO_ANKLE_RATIO
                if bbox_height is not None
                else None
            )

            if ankle_y is not None:
                if not ground_ankle_history:
                    ground_ankle_history.append(ankle_y)
                if ground_ankle_history:
                    ground_y = float(np.median(np.array(ground_ankle_history)))
                    if len(ground_ankle_history) >= 3:
                        arr = np.array(ground_ankle_history, dtype=np.float32)
                        med = float(np.median(arr))
                        noise_px = float(np.median(np.abs(arr - med)))
                    ankle_delta_px = max(0.0, ground_y - ankle_y)
                    base_threshold = max(
                        min_delta_px,
                        (hip_ankle_px if hip_ankle_px is not None else 0.0) * delta_ratio,
                        noise_px * cfg.JUMP_NOISE_SCALE + cfg.JUMP_NOISE_MARGIN_PX,
                    )
                    end_threshold = max(base_threshold * end_ratio, min_delta_px * 0.4)

            if ankle_delta_px is not None:
                height_px = ankle_delta_px
                if last_m_per_px is not None:
                    height_m = height_px * last_m_per_px
                if bbox_height and bbox_height > 0:
                    height_ratio = height_px / bbox_height

            vel_px_s = None
            vel_mps = None
            accel_px_s2 = None
            accel_mps2 = None
            if height_px is not None and prev_height_px is not None and prev_time is not None and time_s is not None:
                dt = time_s - prev_time
                if dt > 0:
                    vel_px_s = (height_px - prev_height_px) / dt
                    if prev_vel_px is not None:
                        accel_px_s2 = (vel_px_s - prev_vel_px) / dt
            if height_m is not None and prev_height_m is not None and prev_time is not None and time_s is not None:
                dt = time_s - prev_time
                if dt > 0:
                    vel_mps = (height_m - prev_height_m) / dt
                    if prev_vel_m is not None:
                        accel_mps2 = (vel_mps - prev_vel_m) / dt

            force_proxy = accel_mps2 if accel_mps2 is not None else accel_px_s2

            time_series.append(time_s)
            height_rows.append(
                {
                    "time_s": time_s,
                    "frame_idx": result.frame_idx,
                    "height_px": height_px,
                    "height_m": height_m,
                    "height_ratio": height_ratio,
                    "jump_active": jump_active,
                    "jump_id": jump_id if jump_active else None,
                }
            )
            force_rows.append(
                {
                    "time_s": time_s,
                    "frame_idx": result.frame_idx,
                    "height_px": height_px,
                    "height_m": height_m,
                    "vel_px_s": vel_px_s,
                    "vel_mps": vel_mps,
                    "accel_px_s2": accel_px_s2,
                    "accel_mps2": accel_mps2,
                    "force_proxy": force_proxy,
                }
            )

            if height_px is None or ankle_y is None:
                prev_time = time_s
                prev_height_px = height_px
                prev_height_m = height_m
                prev_vel_px = vel_px_s
                prev_vel_m = vel_mps
                prev_ankle_y = ankle_y
                jump_up_streak = 0
                jump_air_streak = 0
                continue

            up_vel = 0.0
            if prev_ankle_y is not None:
                up_vel = prev_ankle_y - ankle_y
            if up_vel >= up_px_per_frame:
                jump_up_streak += 1
            else:
                jump_up_streak = 0

            threshold = base_threshold if base_threshold is not None else min_delta_px
            end_threshold = end_threshold if end_threshold is not None else min_delta_px * 0.4

            trigger_delta = height_px >= threshold
            trigger_up = up_vel >= up_px_per_frame or jump_up_streak >= up_streak_req

            if not jump_active:
                if result.frame_idx - last_jump_end_frame >= cooldown_frames:
                    if trigger_delta:
                        jump_air_streak += 1
                    else:
                        jump_air_streak = 0
                    if trigger_delta and (jump_air_streak >= min_air_frames or trigger_up):
                        jump_active = True
                        jump_id += 1
                        jump_start_frame = result.frame_idx
                        jump_start_time = time_s
                        jump_peak_px = height_px
                        jump_peak_m = height_m
                        jump_air_streak = 0
            else:
                if height_px > jump_peak_px:
                    jump_peak_px = height_px
                if height_m is not None:
                    if jump_peak_m is None or height_m > jump_peak_m:
                        jump_peak_m = height_m

                if height_px <= end_threshold and jump_start_frame is not None:
                    if result.frame_idx - jump_start_frame >= min_air_frames:
                        flight_time = None
                        if time_s is not None and jump_start_time is not None:
                            flight_time = time_s - jump_start_time
                        jump_events.append(
                            {
                                "jump_id": jump_id,
                                "start_frame": jump_start_frame,
                                "end_frame": result.frame_idx,
                                "start_time_s": jump_start_time,
                                "end_time_s": time_s,
                                "flight_time_s": flight_time,
                                "peak_height_px": jump_peak_px,
                                "peak_height_m": jump_peak_m,
                            }
                        )
                        total_jumps += 1
                        last_jump_height_px = jump_peak_px
                        last_jump_height_m = jump_peak_m
                        last_flight_time_s = flight_time
                        if highest_jump_px is None or jump_peak_px > highest_jump_px:
                            highest_jump_px = jump_peak_px
                        if jump_peak_m is not None and (
                            highest_jump_m is None or jump_peak_m > highest_jump_m
                        ):
                            highest_jump_m = jump_peak_m

                        landing_collect = {
                            "jump_id": jump_id,
                            "start_frame": result.frame_idx,
                            "start_time_s": time_s,
                            "samples": [],
                        }

                        jump_active = False
                        last_jump_end_frame = result.frame_idx
                        jump_start_frame = None
                        jump_start_time = None
                        jump_peak_px = 0.0
                        jump_peak_m = None

            baseline_threshold = max(
                min_delta_px * 0.4,
                noise_px * cfg.JUMP_NOISE_SCALE * 0.5,
                (hip_ankle_px if hip_ankle_px is not None else 0.0)
                * delta_ratio
                * 0.4,
            )
            if not jump_active and ankle_delta_px is not None and ankle_delta_px <= baseline_threshold:
                ground_ankle_history.append(ankle_y)

            if landing_collect is not None and ankle_y is not None:
                samples = landing_collect["samples"]
                if isinstance(samples, list):
                    samples.append(ankle_y)
                    if len(samples) >= landing_window_frames:
                        std_px = float(np.std(samples)) if len(samples) > 1 else 0.0
                        std_m = std_px * last_m_per_px if last_m_per_px is not None else None
                        stability_score = 1.0 / (1.0 + std_px)
                        landing_rows.append(
                            {
                                "jump_id": landing_collect.get("jump_id"),
                                "frame_idx": result.frame_idx,
                                "time_s": time_s,
                                "landing_std_px": std_px,
                                "landing_std_m": std_m,
                                "stability_score": stability_score,
                            }
                        )
                        landing_collect = None

            display_total_jumps = (
                result.total_jumps if result.total_jumps is not None else total_jumps
            )
            display_highest_m = (
                result.highest_jump_m if result.highest_jump_m is not None else highest_jump_m
            )
            display_highest_px = (
                result.highest_jump_px if result.highest_jump_px is not None else highest_jump_px
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
                            "total_distance_m": result.total_distance_m,
                            "peak_accel_mps2": result.peak_accel_mps2,
                            "peak_decel_mps2": result.peak_decel_mps2,
                            "total_jumps": display_total_jumps,
                            "highest_jump_m": display_highest_m,
                            "highest_jump_px": display_highest_px,
                            "current_height_m": height_m,
                            "current_height_px": height_px,
                            "jump_active": jump_active,
                        },
                    }
                )

            if shot_log is not None and result.shot_events is not None:
                shot_log.clear()
                shot_log.extend(result.shot_events)

            if callable(live_callback) and result.frame_idx % live_stride == 0:
                start = max(0, len(height_rows) - live_tail_rows)
                height_tail = pd.DataFrame(height_rows[start:])
                force_tail = pd.DataFrame(force_rows[start:])
                landing_tail = pd.DataFrame(landing_rows[-live_tail_rows:])

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
                            "total_jumps": display_total_jumps,
                            "highest_jump_m": display_highest_m,
                            "highest_jump_px": display_highest_px,
                            "current_height_m": height_m,
                            "current_height_px": height_px,
                            "last_jump_height_m": last_jump_height_m,
                            "last_jump_height_px": last_jump_height_px,
                            "last_flight_time_s": last_flight_time_s,
                            "jump_active": jump_active,
                        },
                        "jump_height_tail": height_tail,
                        "force_time_tail": force_tail,
                        "landing_stability_tail": landing_tail,
                        "progress": progress,
                    }
                )

            prev_time = time_s
            prev_height_px = height_px
            prev_height_m = height_m
            prev_vel_px = vel_px_s
            prev_vel_m = vel_mps
            prev_ankle_y = ankle_y
    except Exception as exc:
        fallback = build_dummy_result(test_name, expected_matrices, settings)
        fallback.status = "fallback"
        fallback.logs = [f"CMJ analysis failed: {exc}"] + fallback.logs
        return fallback

    if last_result is None:
        fallback = build_dummy_result(test_name, expected_matrices, settings)
        fallback.status = "fallback"
        fallback.logs = ["No frames processed; returning placeholder result."] + fallback.logs
        return fallback

    total_time_s = last_result.total_time_sec
    if total_time_s is None and time_series:
        total_time_s = time_series[-1]

    pipeline_total_jumps = last_result.total_jumps
    pipeline_highest_m = last_result.highest_jump_m
    pipeline_highest_px = last_result.highest_jump_px

    total_jumps_display = (
        pipeline_total_jumps if pipeline_total_jumps is not None else total_jumps
    )
    highest_jump_m_display = (
        pipeline_highest_m if pipeline_highest_m is not None else highest_jump_m
    )
    highest_jump_px_display = (
        pipeline_highest_px if pipeline_highest_px is not None else highest_jump_px
    )

    avg_jump_height_px = None
    avg_jump_height_m = None
    avg_flight_time = None
    if jump_events:
        heights_px = [ev.get("peak_height_px") for ev in jump_events if ev.get("peak_height_px") is not None]
        heights_m = [ev.get("peak_height_m") for ev in jump_events if ev.get("peak_height_m") is not None]
        flights = [ev.get("flight_time_s") for ev in jump_events if ev.get("flight_time_s") is not None]
        if heights_px:
            avg_jump_height_px = float(sum(heights_px) / len(heights_px))
        if heights_m:
            avg_jump_height_m = float(sum(heights_m) / len(heights_m))
        if flights:
            avg_flight_time = float(sum(flights) / len(flights))

    avg_landing_stability = None
    best_landing_stability = None
    if landing_rows:
        scores = [row.get("stability_score") for row in landing_rows if row.get("stability_score") is not None]
        if scores:
            avg_landing_stability = float(sum(scores) / len(scores))
            best_landing_stability = float(max(scores))

    logs.append(f"Processed frames: {len(time_series)}")
    logs.append(f"Detected jumps: {total_jumps_display}")

    jump_height_df = pd.DataFrame(
        height_rows,
        columns=[
            "time_s",
            "frame_idx",
            "height_px",
            "height_m",
            "height_ratio",
            "jump_active",
            "jump_id",
        ],
    )
    force_time_df = pd.DataFrame(
        force_rows,
        columns=[
            "time_s",
            "frame_idx",
            "height_px",
            "height_m",
            "vel_px_s",
            "vel_mps",
            "accel_px_s2",
            "accel_mps2",
            "force_proxy",
        ],
    )
    landing_df = pd.DataFrame(
        landing_rows,
        columns=[
            "jump_id",
            "frame_idx",
            "time_s",
            "landing_std_px",
            "landing_std_m",
            "stability_score",
        ],
    )

    metrics = {
        "total_time_s": total_time_s,
        "total_jumps": total_jumps_display,
        "highest_jump_m": highest_jump_m_display,
        "highest_jump_px": highest_jump_px_display,
        "avg_jump_height_m": avg_jump_height_m,
        "avg_jump_height_px": avg_jump_height_px,
        "avg_flight_time_s": avg_flight_time,
        "avg_landing_stability": avg_landing_stability,
        "best_landing_stability": best_landing_stability,
    }

    return AnalysisResult(
        test_name=test_name,
        status="ok",
        metrics=metrics,
        matrices={
            "force_time": force_time_df,
            "jump_height": jump_height_df,
            "landing_stability": landing_df,
        },
        artifacts={},
        logs=logs,
    )
