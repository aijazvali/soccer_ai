from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    """Best-effort FPS read. Falls back to 30 if unavailable."""
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
    """Pick the largest bbox as the main player (simple but effective for single-athlete tests)."""

    def area(player: Dict) -> float:
        bbox = player.get("bbox")
        if not bbox or len(bbox) != 4:
            return 0.0
        x1, y1, x2, y2 = bbox
        return max(0.0, (x2 - x1) * (y2 - y1))

    if not players:
        return None
    return max(players, key=area)


def _ema(prev: Optional[float], x: float, alpha: float) -> float:
    return x if prev is None else (alpha * x + (1.0 - alpha) * prev)


def analyze(video_path: str, settings: dict) -> AnalysisResult:
    """
    Accurate Drop Jump (DJ) scoring from video:
    - Contact time: landing -> takeoff
    - Flight time: takeoff -> next landing
    - Jump height (m): h = g * t^2 / 8  (flight-time method)
    - RSI: jump_height / contact_time

    This avoids the common DJ bug: estimating "ground" from the initial frames (the athlete starts on a box).

    Assumptions about pose meta (same as your v1 file):
    - frame_meta["players"] is a list
    - each player has: bbox [x1,y1,x2,y2], and left/right ankle points in player["left"], player["right"] as (x,y)
    """
    test_name = settings.get("test_name", "Drop Jump")
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
    logs.append(f"FPS (best-effort): {fps:.2f}")

    # ------------------------ Tunables (settings -> defaults) ------------------------
    ema_alpha = float(settings.get("drop_jump_ema_alpha", 0.25))

    # Normalized velocities (1/s): v_norm = (dy/dt)/bbox_h
    drop_vel_norm_min = float(settings.get("drop_jump_drop_vel_norm_min", 0.55))         # falling phase
    takeoff_vel_norm_min = float(settings.get("drop_jump_takeoff_vel_norm_min", 0.45))  # upward phase (negative)
    descend_vel_norm_min = float(settings.get("drop_jump_descend_vel_norm_min", 0.45))  # downward in flight (positive)
    contact_vel_norm_max = float(settings.get("drop_jump_contact_vel_norm_max", 0.20))  # "still" band around 0

    drop_frames = int(settings.get("drop_jump_drop_frames", max(2, int(round(0.06 * fps)))))
    still_frames = int(settings.get("drop_jump_still_frames", max(2, int(round(0.04 * fps)))))
    up_frames = int(settings.get("drop_jump_up_frames", max(2, int(round(0.04 * fps)))))
    drop_delta_ratio = float(settings.get("drop_jump_drop_delta_ratio", 0.05))
    drop_delta_px_min = float(settings.get("drop_jump_drop_delta_px_min", 10.0))
    cooldown_frames = int(settings.get("drop_jump_cooldown_frames", cfg.JUMP_COOLDOWN_FRAMES))

    # Physics
    g = float(settings.get("drop_jump_gravity", 9.80665))

    # Plausibility filters (avoid false reps)
    min_contact_s = float(settings.get("drop_jump_min_contact_s", 0.05))
    max_contact_s = float(settings.get("drop_jump_max_contact_s", 1.20))
    min_flight_s = float(settings.get("drop_jump_min_flight_s", 0.08))
    max_flight_s = float(settings.get("drop_jump_max_flight_s", 1.50))

    # ------------------------ Runtime store ------------------------
    live_callback = settings.get("live_callback")
    live_stride = max(1, int(settings.get("live_stride", 5) or 5))
    live_tail_rows = max(1, int(settings.get("live_tail_rows", 8) or 8))
    total_frames_setting = settings.get("total_frames")

    runtime_store = settings.get("runtime_store")
    if isinstance(runtime_store, dict):
        frame_records = runtime_store.setdefault("frame_records", [])
        shot_log = runtime_store.setdefault("shot_log", [])
    else:
        frame_records = None
        shot_log = None

    # ------------------------ Outputs ------------------------
    ground_contact_rows: List[Dict[str, Optional[float] | int]] = []
    reactive_rows: List[Dict[str, Optional[float] | int]] = []
    landing_rows: List[Dict[str, Optional[float] | int]] = []

    # ------------------------ Signal state ------------------------
    prev_time: Optional[float] = None
    prev_y: Optional[float] = None
    y_s: Optional[float] = None
    prev_v: Optional[float] = None
    prev_bbox_h: Optional[float] = None
    bbox_h_s: Optional[float] = None
    v_norm_s: Optional[float] = None
    still_y_history: deque = deque(maxlen=max(6, still_frames * 2))

    # ------------------------ Phase machine ------------------------
    # SEEK_DROP -> SEEK_LANDING -> CONTACT -> FLIGHT -> SEEK_DROP (next rep)
    phase = "SEEK_DROP"
    last_event_frame = -10_000

    drop_streak = 0
    still_streak = 0
    up_streak = 0

    jump_id = 0
    total_jumps = 0

    # Events
    landing_frame: Optional[int] = None
    landing_time: Optional[float] = None
    landing_impact_v_px_s: Optional[float] = None
    landing_impact_v_mps: Optional[float] = None
    landing_peak_accel_px_s2: Optional[float] = None
    landing_peak_accel_mps2: Optional[float] = None
    landing_peak_v_px_s: Optional[float] = None
    landing_peak_v_mps: Optional[float] = None
    landing_descend_seen = False

    takeoff_frame: Optional[int] = None
    takeoff_time: Optional[float] = None
    takeoff_y: Optional[float] = None
    contact_time_s: Optional[float] = None

    descend_seen = False
    flight_apex_y: Optional[float] = None

    # Live metrics
    last_contact_time: Optional[float] = None
    last_jump_height_m: Optional[float] = None
    last_jump_height_px: Optional[float] = None
    last_rsi: Optional[float] = None
    current_height_px: Optional[float] = None
    current_height_m: Optional[float] = None

    best_jump_height_m: Optional[float] = None
    best_jump_height_px: Optional[float] = None
    best_rsi: Optional[float] = None
    best_contact_time: Optional[float] = None

    max_frames = settings.get("max_frames")
    try:
        max_frames_value = int(max_frames) if max_frames is not None else None
    except (TypeError, ValueError):
        max_frames_value = None
    if max_frames_value is not None and max_frames_value <= 0:
        max_frames_value = None

    last_result = None

    def emit_live(frame_idx: int, frame_bgr) -> None:
        if not callable(live_callback):
            return
        if frame_idx % live_stride != 0:
            return

        ground_tail = pd.DataFrame(ground_contact_rows[-live_tail_rows:])
        reactive_tail = pd.DataFrame(reactive_rows[-live_tail_rows:])
        landing_tail = pd.DataFrame(landing_rows[-live_tail_rows:])

        progress = None
        if total_frames_setting:
            try:
                total_frames_val = float(total_frames_setting)
                if total_frames_val > 0:
                    progress = frame_idx / total_frames_val
            except (TypeError, ValueError):
                progress = None

        live_callback(
            {
                "frame_idx": frame_idx,
                "frame_bgr": frame_bgr,
                "metrics": {
                    "total_jumps": total_jumps,
                    "phase": phase,
                    "last_contact_time_s": last_contact_time,
                    "last_jump_height_m": last_jump_height_m,
                    "last_jump_height_px": last_jump_height_px,
                    "last_rsi": last_rsi,
                    "current_height_m": current_height_m,
                    "current_height_px": current_height_px,
                },
                "ground_contact_tail": ground_tail,
                "reactive_strength_tail": reactive_tail,
                "landing_force_tail": landing_tail,
                "progress": progress,
            }
        )

    try:
        for result in run_touch_detection(video_path, options=options, max_frames=max_frames_value):
            last_result = result
            frame_idx = result.frame_idx

            time_s = result.total_time_sec
            if time_s is None:
                time_s = frame_idx / fps if fps > 0 else None

            frame_meta = result.frame_meta or {}
            players_meta = frame_meta.get("players") or []
            main_player = _select_main_player(players_meta)

            left_y = right_y = None
            bbox_h = None
            m_per_px = None
            if main_player:
                bbox = main_player.get("bbox")
                if bbox and len(bbox) == 4:
                    _x1, y1, _x2, y2 = bbox
                    bbox_h = float(y2) - float(y1)
                    if bbox_h and bbox_h > 1e-3:
                        # Same approach used elsewhere in your codebase.
                        m_per_px = float(cfg.PLAYER_REF_HEIGHT_M) / float(bbox_h)

                left = main_player.get("left")
                right = main_player.get("right")
                if left is not None:
                    left_y = float(left[1])
                if right is not None:
                    right_y = float(right[1])

            # Need at least one foot point
            if time_s is None or (left_y is None and right_y is None):
                prev_time = time_s
                prev_y = None
                prev_v = None
                if bbox_h is not None and bbox_h > 1e-3:
                    prev_bbox_h = bbox_h
                drop_streak = 0
                still_streak = 0
                up_streak = 0
                still_y_history.clear()
                emit_live(frame_idx, result.annotated)
                continue

            # Use the lower foot (max y in image coords) as contact proxy
            y = float(max(v for v in [left_y, right_y] if v is not None))

            # Smooth y to reduce jitter
            y_s = _ema(y_s, y, ema_alpha)

            # dt and derivatives
            dt = None
            if prev_time is not None:
                dt = time_s - prev_time
            if dt is None or dt <= 1e-6:
                dt = 1.0 / fps if fps > 0 else 1 / 30.0

            v_px_s = None
            a_px_s2 = None
            if prev_y is not None:
                v_px_s = (y_s - prev_y) / dt  # + down, - up
                if prev_v is not None:
                    a_px_s2 = (v_px_s - prev_v) / dt

            if bbox_h is not None and bbox_h > 1e-3:
                bbox_h_s = _ema(bbox_h_s, bbox_h, 0.2)

            # Normalized velocity (1/s)
            v_norm = None
            if v_px_s is not None:
                bh = bbox_h_s if (bbox_h_s is not None and bbox_h_s > 1e-3) else prev_bbox_h
                if bh is not None and bh > 1e-3:
                    v_norm = v_px_s / bh
                else:
                    v_norm = v_px_s / 1000.0

            v_norm_s = _ema(v_norm_s, v_norm, 0.35) if v_norm is not None else v_norm_s

            # Update streaks
            if v_norm_s is not None:
                drop_streak = drop_streak + 1 if (v_norm_s > drop_vel_norm_min) else 0
                still_streak = still_streak + 1 if (abs(v_norm_s) < contact_vel_norm_max) else 0
                up_streak = up_streak + 1 if (v_norm_s < -takeoff_vel_norm_min) else 0
                if abs(v_norm_s) < contact_vel_norm_max:
                    still_y_history.append(y_s)

            # Live height proxy while in flight
            current_height_px = None
            current_height_m = None
            if phase == "FLIGHT" and takeoff_y is not None:
                current_height_px = max(0.0, float(takeoff_y - y_s))
                if m_per_px is not None:
                    current_height_m = current_height_px * m_per_px

            # Phase machine
            if phase == "SEEK_DROP":
                baseline_y = float(np.median(still_y_history)) if still_y_history else None
                drop_delta_px = None
                if baseline_y is not None and bbox_h_s is not None:
                    drop_delta_px = y_s - baseline_y
                drop_delta_ok = False
                if drop_delta_px is not None and bbox_h_s is not None:
                    drop_delta_ok = drop_delta_px >= max(drop_delta_px_min, drop_delta_ratio * bbox_h_s)

                drop_detected = drop_streak >= drop_frames or drop_delta_ok
                if drop_detected and frame_idx - last_event_frame >= cooldown_frames:
                    phase = "SEEK_LANDING"
                    landing_descend_seen = False
                    flight_apex_y = None
                    takeoff_y = None
                    landing_peak_v_px_s = None
                    landing_peak_v_mps = None
                    landing_peak_accel_px_s2 = None
                    landing_peak_accel_mps2 = None
                    still_streak = 0
                    up_streak = 0

            elif phase == "SEEK_LANDING":
                if v_norm_s is not None and v_norm_s > descend_vel_norm_min:
                    landing_descend_seen = True
                if v_px_s is not None and v_px_s > 0:
                    landing_peak_v_px_s = (
                        v_px_s
                        if landing_peak_v_px_s is None
                        else max(landing_peak_v_px_s, v_px_s)
                    )
                    if m_per_px is not None:
                        v_mps = v_px_s * m_per_px
                        landing_peak_v_mps = (
                            v_mps
                            if landing_peak_v_mps is None
                            else max(landing_peak_v_mps, v_mps)
                        )
                if a_px_s2 is not None:
                    accel_val = abs(a_px_s2)
                    landing_peak_accel_px_s2 = (
                        accel_val
                        if landing_peak_accel_px_s2 is None
                        else max(landing_peak_accel_px_s2, accel_val)
                    )
                    if m_per_px is not None:
                        accel_m = accel_val * m_per_px
                        landing_peak_accel_mps2 = (
                            accel_m
                            if landing_peak_accel_mps2 is None
                            else max(landing_peak_accel_mps2, accel_m)
                        )

                sign_change_up = (
                    prev_v is not None
                    and v_px_s is not None
                    and prev_v > 0
                    and v_px_s < 0
                )
                landing_candidate = landing_descend_seen and (
                    still_streak >= still_frames
                    or (v_norm_s is not None and abs(v_norm_s) < contact_vel_norm_max)
                    or sign_change_up
                )

                if landing_candidate:
                    if still_streak >= still_frames:
                        landing_frame = frame_idx - still_frames + 1
                        landing_time = time_s - (still_frames - 1) * dt
                    elif sign_change_up and dt is not None:
                        landing_frame = frame_idx - 1
                        landing_time = time_s - dt
                    else:
                        landing_frame = frame_idx
                        landing_time = time_s

                    landing_impact_v_px_s = (
                        landing_peak_v_px_s if landing_peak_v_px_s is not None else v_px_s
                    )
                    if landing_impact_v_px_s is not None and m_per_px is not None:
                        landing_impact_v_mps = landing_impact_v_px_s * m_per_px
                    else:
                        landing_impact_v_mps = landing_peak_v_mps
                    landing_peak_accel_px_s2 = (
                        landing_peak_accel_px_s2
                        if landing_peak_accel_px_s2 is not None
                        else (abs(a_px_s2) if a_px_s2 is not None else None)
                    )
                    if landing_peak_accel_px_s2 is not None and m_per_px is not None:
                        landing_peak_accel_mps2 = landing_peak_accel_px_s2 * m_per_px
                    elif landing_peak_accel_mps2 is None and a_px_s2 is not None and m_per_px is not None:
                        landing_peak_accel_mps2 = abs(a_px_s2) * m_per_px

                    last_event_frame = landing_frame
                    phase = "CONTACT"
                    still_streak = 0
                    up_streak = 0
                    still_y_history.clear()

            elif phase == "CONTACT":
                if a_px_s2 is not None:
                    accel_val = abs(a_px_s2)
                    landing_peak_accel_px_s2 = (
                        accel_val
                        if landing_peak_accel_px_s2 is None
                        else max(landing_peak_accel_px_s2, accel_val)
                    )
                    if m_per_px is not None:
                        accel_m = accel_val * m_per_px
                        landing_peak_accel_mps2 = (
                            accel_m
                            if landing_peak_accel_mps2 is None
                            else max(landing_peak_accel_mps2, accel_m)
                        )
                sign_change_up = (
                    prev_v is not None
                    and v_px_s is not None
                    and prev_v > 0
                    and v_px_s < 0
                )
                takeoff_ready = up_streak >= up_frames or (
                    sign_change_up and v_norm_s is not None and abs(v_norm_s) >= takeoff_vel_norm_min * 0.5
                )
                if takeoff_ready:
                    takeoff_frame = frame_idx - up_frames + 1
                    takeoff_time = time_s - (up_frames - 1) * dt
                    takeoff_y = y_s

                    if landing_time is not None and takeoff_time is not None:
                        contact_time_s = max(0.0, takeoff_time - landing_time)
                    else:
                        contact_time_s = None

                    # Plausibility gate (avoid false positives)
                    if contact_time_s is None or not (min_contact_s <= contact_time_s <= max_contact_s):
                        phase = "SEEK_DROP"
                        drop_streak = still_streak = up_streak = 0
                        landing_frame = landing_time = None
                        contact_time_s = None
                        last_event_frame = frame_idx
                        continue

                    jump_id += 1
                    ground_contact_rows.append(
                        {
                            "jump_id": jump_id,
                            "contact_start_frame": landing_frame,
                            "contact_end_frame": takeoff_frame,
                            "contact_time_s": contact_time_s,
                            "impact_velocity_px_s": landing_impact_v_px_s,
                            "impact_velocity_mps": landing_impact_v_mps,
                            "peak_accel_px_s2": landing_peak_accel_px_s2,
                            "peak_accel_mps2": landing_peak_accel_mps2,
                        }
                    )
                    landing_rows.append(
                        {
                            "jump_id": jump_id,
                            "impact_velocity_px_s": landing_impact_v_px_s,
                            "impact_velocity_mps": landing_impact_v_mps,
                            "peak_accel_px_s2": landing_peak_accel_px_s2,
                            "peak_accel_mps2": landing_peak_accel_mps2,
                        }
                    )

                    last_event_frame = takeoff_frame
                    phase = "FLIGHT"
                    descend_seen = False
                    flight_apex_y = y_s
                    still_streak = 0
                    up_streak = 0
                    drop_streak = 0

            elif phase == "FLIGHT":
                if flight_apex_y is None or y_s < flight_apex_y:
                    flight_apex_y = y_s

                if v_norm_s is not None and v_norm_s > descend_vel_norm_min:
                    descend_seen = True

                landing_candidate = descend_seen and (
                    still_streak >= still_frames
                    or (v_norm_s is not None and abs(v_norm_s) < contact_vel_norm_max)
                )
                if landing_candidate:
                    if still_streak >= still_frames:
                        landing2_time = time_s - (still_frames - 1) * dt
                    else:
                        landing2_time = time_s

                    flight_time_s = None
                    if takeoff_time is not None:
                        flight_time_s = max(0.0, landing2_time - takeoff_time)

                    if flight_time_s is None or not (min_flight_s <= flight_time_s <= max_flight_s):
                        # Discard and reset
                        phase = "SEEK_DROP"
                        drop_streak = still_streak = up_streak = 0
                        landing_frame = landing_time = takeoff_frame = takeoff_time = None
                        contact_time_s = None
                        descend_seen = False
                        flight_apex_y = None
                        takeoff_y = None
                        last_event_frame = frame_idx
                        continue

                    jump_height_m = (g * (flight_time_s ** 2)) / 8.0

                    jump_height_px = None
                    if takeoff_y is not None and flight_apex_y is not None:
                        jump_height_px = max(0.0, float(takeoff_y - flight_apex_y))

                    rsi = None
                    if contact_time_s and contact_time_s > 1e-6:
                        rsi = jump_height_m / contact_time_s

                    reactive_rows.append(
                        {
                            "jump_id": jump_id,
                            "contact_time_s": contact_time_s,
                            "flight_time_s": flight_time_s,
                            "jump_height_px": jump_height_px,
                            "jump_height_m": jump_height_m,
                            "rsi": rsi,
                        }
                    )

                    total_jumps += 1
                    last_contact_time = contact_time_s
                    last_jump_height_m = jump_height_m
                    last_jump_height_px = jump_height_px
                    last_rsi = rsi

                    best_jump_height_m = jump_height_m if (best_jump_height_m is None or jump_height_m > best_jump_height_m) else best_jump_height_m
                    if jump_height_px is not None:
                        best_jump_height_px = jump_height_px if (best_jump_height_px is None or jump_height_px > best_jump_height_px) else best_jump_height_px
                    if rsi is not None:
                        best_rsi = rsi if (best_rsi is None or rsi > best_rsi) else best_rsi
                    best_contact_time = contact_time_s if (best_contact_time is None or contact_time_s < best_contact_time) else best_contact_time

                    # Reset for next rep
                    phase = "SEEK_DROP"
                    drop_streak = still_streak = up_streak = 0
                    landing_frame = landing_time = takeoff_frame = takeoff_time = None
                    contact_time_s = None
                    descend_seen = False
                    flight_apex_y = None
                    takeoff_y = None
                    last_event_frame = frame_idx
                    still_y_history.clear()

            # Store per-frame debug info
            if frame_records is not None:
                frame_records.append(
                    {
                        "frame_idx": frame_idx,
                        "meta": result.frame_meta,
                        "stats": {
                            "total_time_sec": time_s,
                            "phase": phase,
                            "y_ankle": y,
                            "y_ankle_s": y_s,
                            "v_px_s": v_px_s,
                            "v_norm": v_norm,
                            "a_px_s2": a_px_s2,
                            "meters_per_px": m_per_px,
                            "total_jumps": total_jumps,
                            "last_contact_time_s": last_contact_time,
                            "last_jump_height_m": last_jump_height_m,
                            "last_jump_height_px": last_jump_height_px,
                            "last_rsi": last_rsi,
                            "current_height_m": current_height_m,
                            "current_height_px": current_height_px,
                        },
                    }
                )

            if shot_log is not None and result.shot_events is not None:
                shot_log.clear()
                shot_log.extend(result.shot_events)

            emit_live(frame_idx, result.annotated)

            # advance
            prev_time = time_s
            prev_y = y_s
            prev_v = v_px_s
            if bbox_h is not None and bbox_h > 1e-3:
                prev_bbox_h = bbox_h_s if bbox_h_s is not None else bbox_h

    except Exception as exc:
        fallback = build_dummy_result(test_name, expected_matrices, settings)
        fallback.status = "fallback"
        fallback.logs = [f"Drop jump analysis failed: {exc}"] + fallback.logs
        return fallback

    if last_result is None:
        fallback = build_dummy_result(test_name, expected_matrices, settings)
        fallback.status = "fallback"
        fallback.logs = ["No frames processed; returning placeholder result."] + fallback.logs
        return fallback

    total_time_s = last_result.total_time_sec
    if total_time_s is None:
        total_time_s = last_result.frame_idx / fps if fps > 0 else None

    # Averages
    avg_contact_time = None
    avg_jump_height_px = None
    avg_jump_height_m = None
    avg_rsi = None
    if reactive_rows:
        contact_vals = [r["contact_time_s"] for r in reactive_rows if r.get("contact_time_s") is not None]
        if contact_vals:
            avg_contact_time = float(sum(contact_vals) / len(contact_vals))
        px_vals = [r["jump_height_px"] for r in reactive_rows if r.get("jump_height_px") is not None]
        if px_vals:
            avg_jump_height_px = float(sum(px_vals) / len(px_vals))
        m_vals = [r["jump_height_m"] for r in reactive_rows if r.get("jump_height_m") is not None]
        if m_vals:
            avg_jump_height_m = float(sum(m_vals) / len(m_vals))
        rsi_vals = [r["rsi"] for r in reactive_rows if r.get("rsi") is not None]
        if rsi_vals:
            avg_rsi = float(sum(rsi_vals) / len(rsi_vals))

    logs.append(f"Processed up to frame: {last_result.frame_idx}")
    logs.append(f"Detected drop jumps: {total_jumps}")

    metrics = {
        "total_time_s": total_time_s,
        "total_jumps": total_jumps,
        "last_contact_time_s": last_contact_time,
        "last_jump_height_m": last_jump_height_m,
        "last_jump_height_px": last_jump_height_px,
        "last_rsi": last_rsi,
        "best_jump_height_m": best_jump_height_m,
        "best_jump_height_px": best_jump_height_px,
        "best_contact_time_s": best_contact_time,
        "best_rsi": best_rsi,
        "avg_contact_time_s": avg_contact_time,
        "avg_jump_height_m": avg_jump_height_m,
        "avg_jump_height_px": avg_jump_height_px,
        "avg_rsi": avg_rsi,
    }

    ground_contact_df = pd.DataFrame(
        ground_contact_rows,
        columns=[
            "jump_id",
            "contact_start_frame",
            "contact_end_frame",
            "contact_time_s",
            "impact_velocity_px_s",
            "impact_velocity_mps",
            "peak_accel_px_s2",
            "peak_accel_mps2",
        ],
    )
    reactive_df = pd.DataFrame(
        reactive_rows,
        columns=[
            "jump_id",
            "contact_time_s",
            "flight_time_s",
            "jump_height_px",
            "jump_height_m",
            "rsi",
        ],
    )
    landing_df = pd.DataFrame(
        landing_rows,
        columns=[
            "jump_id",
            "impact_velocity_px_s",
            "impact_velocity_mps",
            "peak_accel_px_s2",
            "peak_accel_mps2",
        ],
    )

    return AnalysisResult(
        test_name=test_name,
        status="ok",
        metrics=metrics,
        matrices={
            "ground_contact": ground_contact_df,
            "reactive_strength": reactive_df,
            "landing_force": landing_df,
        },
        artifacts={},
        logs=logs,
    )
