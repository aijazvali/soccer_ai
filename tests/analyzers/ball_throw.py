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
    heights.sort()
    mid = len(heights) // 2
    median_height = heights[mid] if len(heights) % 2 else 0.5 * (heights[mid - 1] + heights[mid])
    return median_height, ground_y


def _angle_deg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.degrees(math.atan2(-(b[1] - a[1]), b[0] - a[0]))


def analyze(video_path: str, settings: dict) -> AnalysisResult:
    test_name = settings.get("test_name", "Ball Throw")
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

    release_speed_mps = float(settings.get("ball_throw_release_speed_mps", 3.0))
    release_speed_px = float(settings.get("ball_throw_release_speed_px_s", 120.0))
    release_window_frames = int(settings.get("ball_throw_release_window_frames", 8))
    missing_ball_frames = int(settings.get("ball_throw_missing_ball_frames", 8))
    use_player_height = bool(settings.get("ball_throw_use_player_height", True))
    min_height_ratio = float(settings.get("ball_throw_min_height_ratio", 0.1))
    ground_ratio = float(settings.get("ball_throw_ground_ratio", 0.92))
    ground_hold_frames = int(settings.get("ball_throw_ground_hold_frames", 3))

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
    release_snapshot_done = False
    max_height_snapshot_done = False

    trajectory_rows: List[Dict[str, Optional[float] | int | bool]] = []
    shoulder_rows: List[Dict[str, Optional[float]]] = []
    release_rows: List[Dict[str, Optional[float]]] = []

    missing_ball_count = 0
    last_ball_frame: Optional[int] = None
    prev_speed_for_release: Optional[float] = None
    release_frame_idx: Optional[int] = None
    release_point: Optional[Tuple[float, float]] = None
    release_time: Optional[float] = None
    release_speed_mps_val: Optional[float] = None
    release_speed_px_val: Optional[float] = None
    release_angle_deg: Optional[float] = None
    release_window_end: Optional[int] = None

    release_foot_px: Optional[Tuple[float, float]] = None
    release_meters_per_px: Optional[float] = None
    ground_contact_px: Optional[Tuple[float, float]] = None
    ground_contact_time: Optional[float] = None
    ground_contact_frames = 0
    throw_distance_px: Optional[float] = None
    throw_distance_m: Optional[float] = None

    max_height: Optional[float] = None
    hand_force_mps2: Optional[float] = None
    hand_force_px_s2: Optional[float] = None
    prev_wrist_point: Optional[Tuple[float, float]] = None
    prev_wrist_time: Optional[float] = None
    prev_wrist_speed_mps: Optional[float] = None
    prev_wrist_speed_px: Optional[float] = None
    prev_ball_point: Optional[Tuple[float, float]] = None
    prev_ball_time: Optional[float] = None
    ball_speed_px_hist: List[float] = []
    ball_speed_mps_hist: List[float] = []
    processed_frames = 0
    ball_detected_frames = 0
    ball_class_id: Optional[int] = None
    ball_class_name: Optional[str] = None

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
            processed_frames += 1
            last_result = result
            time_s = result.total_time_sec
            if time_s is None:
                time_s = result.frame_idx / fps if fps > 0 else None

            frame_meta = result.frame_meta or {}
            ball_meta = frame_meta.get("ball") or {}
            players_meta = frame_meta.get("players") or []
            if ball_class_id is None:
                ball_class_id = frame_meta.get("ball_class_id")
                ball_class_name = frame_meta.get("ball_class_name")

            frame_h = None
            frame_w = None
            if result.annotated is not None:
                frame_h, frame_w = result.annotated.shape[:2]

            ball_center = ball_meta.get("center")
            speed_mps = ball_meta.get("speed_mps")
            speed_px = ball_meta.get("speed_draw")
            if ball_center is not None:
                ball_detected_frames += 1

            main_player = None
            player_height_px = None
            meters_per_px = None
            left_foot = None
            right_foot = None
            release_foot_candidate = None
            wrist_point = None

            if players_meta:
                def _area(player: Dict) -> float:
                    bbox = player.get("bbox")
                    if not bbox or len(bbox) != 4:
                        return 0.0
                    x1, y1, x2, y2 = bbox
                    return max(0.0, (x2 - x1) * (y2 - y1))

                players_sorted = sorted(players_meta, key=_area, reverse=True)
                main_player = players_sorted[0] if players_sorted else None

            if main_player:
                bbox = main_player.get("bbox")
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    player_height_px = float(y2) - float(y1)
                    if player_height_px and player_height_px > 0:
                        meters_per_px = cfg.PLAYER_REF_HEIGHT_M / player_height_px
                    release_foot_candidate = ((x1 + x2) / 2.0, y2)

                left_foot = main_player.get("left")
                right_foot = main_player.get("right")
                if left_foot is not None and right_foot is not None:
                    release_foot_candidate = (
                        left_foot if left_foot[1] >= right_foot[1] else right_foot
                    )
                elif left_foot is not None:
                    release_foot_candidate = left_foot
                elif right_foot is not None:
                    release_foot_candidate = right_foot

                lw = main_player.get("left_wrist")
                rw = main_player.get("right_wrist")
                if lw is not None and rw is not None:
                    if ball_center is not None:
                        dl = math.hypot(lw[0] - ball_center[0], lw[1] - ball_center[1])
                        dr = math.hypot(rw[0] - ball_center[0], rw[1] - ball_center[1])
                        wrist_point = lw if dl <= dr else rw
                    else:
                        wrist_point = lw if lw[1] <= rw[1] else rw
                else:
                    wrist_point = lw or rw

            if ball_center is None:
                missing_ball_count += 1
            else:
                missing_ball_count = 0
                last_ball_frame = result.frame_idx

            height_ratio = None
            height_norm = None
            ball_y_norm = None
            if ball_center is not None and frame_h:
                ball_y = float(ball_center[1])
                ball_y_norm = ball_y / frame_h
                height_norm = max(0.0, 1.0 - ball_y_norm)
                if use_player_height:
                    ground_y = None
                    height_px = None
                    if main_player and main_player.get("bbox"):
                        _x1, _y1, _x2, y2 = main_player.get("bbox")
                        ground_y = float(y2)
                        if player_height_px and player_height_px > 0:
                            height_px = max(0.0, ground_y - ball_y)
                            height_ratio = height_px / player_height_px
                    if height_ratio is None and players_meta:
                        player_height, ground_y = _player_scale(players_meta)
                        if player_height and player_height > 0 and ground_y is not None:
                            height_px = max(0.0, ground_y - ball_y)
                            height_ratio = height_px / player_height
                    if height_ratio is not None and height_ratio < min_height_ratio:
                        height_ratio = 0.0

            height_proxy = height_ratio if use_player_height and height_ratio is not None else height_norm
            if height_proxy is not None:
                if max_height is None or height_proxy > max_height:
                    max_height = height_proxy

            speed_px_fallback = speed_px
            speed_mps_fallback = speed_mps
            if (
                ball_center is not None
                and prev_ball_point is not None
                and prev_ball_time is not None
                and time_s is not None
            ):
                dt_ball = time_s - prev_ball_time
                if dt_ball > 0:
                    dist_px = math.hypot(
                        ball_center[0] - prev_ball_point[0],
                        ball_center[1] - prev_ball_point[1],
                    )
                    calc_speed_px = dist_px / dt_ball
                    if speed_px_fallback is None:
                        speed_px_fallback = calc_speed_px
                    if speed_mps_fallback is None and meters_per_px is not None:
                        speed_mps_fallback = calc_speed_px * meters_per_px

            if ball_center is not None and time_s is not None:
                prev_ball_point = ball_center
                prev_ball_time = time_s

            if speed_px_fallback is not None:
                ball_speed_px_hist.append(speed_px_fallback)
                if len(ball_speed_px_hist) > 5:
                    ball_speed_px_hist.pop(0)
            if speed_mps_fallback is not None:
                ball_speed_mps_hist.append(speed_mps_fallback)
                if len(ball_speed_mps_hist) > 5:
                    ball_speed_mps_hist.pop(0)

            speed_px_smooth = None
            if ball_speed_px_hist:
                window = ball_speed_px_hist[-3:]
                speed_px_smooth = sorted(window)[len(window) // 2]
            speed_mps_smooth = None
            if ball_speed_mps_hist:
                window = ball_speed_mps_hist[-3:]
                speed_mps_smooth = sorted(window)[len(window) // 2]

            wrist_speed_px = None
            wrist_speed_mps = None
            wrist_accel_px = None
            wrist_accel_mps = None
            if (
                wrist_point is not None
                and prev_wrist_point is not None
                and prev_wrist_time is not None
                and time_s is not None
            ):
                dt = time_s - prev_wrist_time
                if dt > 0:
                    dist_px = math.hypot(
                        wrist_point[0] - prev_wrist_point[0],
                        wrist_point[1] - prev_wrist_point[1],
                    )
                    wrist_speed_px = dist_px / dt
                    if meters_per_px is not None:
                        wrist_speed_mps = wrist_speed_px * meters_per_px
                    if prev_wrist_speed_px is not None:
                        wrist_accel_px = (wrist_speed_px - prev_wrist_speed_px) / dt
                    if prev_wrist_speed_mps is not None and wrist_speed_mps is not None:
                        wrist_accel_mps = (wrist_speed_mps - prev_wrist_speed_mps) / dt

            if wrist_point is not None and time_s is not None:
                prev_wrist_point = wrist_point
                prev_wrist_time = time_s
                if wrist_speed_px is not None:
                    prev_wrist_speed_px = wrist_speed_px
                if wrist_speed_mps is not None:
                    prev_wrist_speed_mps = wrist_speed_mps

            in_release_window = False
            if release_window_end is not None and release_window_end >= 0:
                in_release_window = result.frame_idx <= release_window_end

            if in_release_window:
                if wrist_accel_mps is not None:
                    accel_val = abs(wrist_accel_mps)
                    if hand_force_mps2 is None or accel_val > hand_force_mps2:
                        hand_force_mps2 = accel_val
                if wrist_accel_px is not None:
                    accel_val = abs(wrist_accel_px)
                    if hand_force_px_s2 is None or accel_val > hand_force_px_s2:
                        hand_force_px_s2 = accel_val

            is_release = False
            speed_for_release = (
                speed_mps_smooth if speed_mps_smooth is not None else speed_px_smooth
            )
            threshold = (
                release_speed_mps if speed_mps_smooth is not None else release_speed_px
            )

            if (
                release_frame_idx is None
                and speed_for_release is not None
                and speed_for_release >= threshold
                and (prev_speed_for_release is None or prev_speed_for_release < threshold)
            ):
                release_frame_idx = result.frame_idx
                release_time = time_s
                release_point = ball_center
                release_speed_mps_val = speed_mps_smooth
                release_speed_px_val = speed_px_smooth
                release_window_end = result.frame_idx + max(1, release_window_frames)
                is_release = True
                if release_foot_candidate is not None:
                    release_foot_px = release_foot_candidate
                    release_meters_per_px = meters_per_px
                release_rows.append(
                    {
                        "time_s": time_s,
                        "frame_idx": result.frame_idx,
                        "release_speed_mps": speed_mps_fallback,
                        "release_speed_px_s": speed_px_fallback,
                        "release_angle_deg": None,
                        "height_proxy": height_proxy,
                        "hand_force_mps2": None,
                        "hand_force_px_s2": None,
                        "throw_distance_m": None,
                        "throw_distance_px": None,
                    }
                )

            if release_frame_idx is not None and release_window_end is not None:
                if speed_mps_smooth is not None:
                    if release_speed_mps_val is None or speed_mps_smooth > release_speed_mps_val:
                        release_speed_mps_val = speed_mps_smooth
                if speed_px_smooth is not None:
                    if release_speed_px_val is None or speed_px_smooth > release_speed_px_val:
                        release_speed_px_val = speed_px_smooth

            if (
                release_point is not None
                and release_angle_deg is None
                and ball_center is not None
                and result.frame_idx != release_frame_idx
            ):
                release_angle_deg = _angle_deg(release_point, ball_center)

            prev_speed_for_release = speed_for_release

            if release_window_end is not None and result.frame_idx > release_window_end:
                release_window_end = -1

            ground_hit = False
            if (
                release_frame_idx is not None
                and ball_center is not None
                and result.frame_idx > release_frame_idx
            ):
                if use_player_height and height_ratio is not None:
                    ground_hit = height_ratio <= min_height_ratio
                elif ball_y_norm is not None:
                    ground_hit = ball_y_norm >= ground_ratio

            ground_event = False
            if ground_hit:
                ground_contact_frames += 1
            else:
                ground_contact_frames = 0

            if (
                ground_contact_px is None
                and ground_contact_frames >= max(1, ground_hold_frames)
                and ball_center is not None
            ):
                ground_contact_px = ball_center
                ground_contact_time = time_s
                ground_event = True
                if release_foot_px is not None:
                    throw_distance_px = math.hypot(
                        ground_contact_px[0] - release_foot_px[0],
                        ground_contact_px[1] - release_foot_px[1],
                    )
                    if release_meters_per_px is not None:
                        throw_distance_m = throw_distance_px * release_meters_per_px

            x_norm = None
            y_norm = None
            if ball_center is not None and frame_w and frame_h:
                x_norm = ball_center[0] / frame_w
                y_norm = ball_center[1] / frame_h

            trajectory_rows.append(
                {
                    "time_s": time_s,
                    "frame_idx": result.frame_idx,
                    "ball_x": None if ball_center is None else float(ball_center[0]),
                    "ball_y": None if ball_center is None else float(ball_center[1]),
                    "x_norm": x_norm,
                    "y_norm": y_norm,
                    "height_proxy": height_proxy,
                    "speed_mps": speed_mps_fallback,
                    "speed_px_s": speed_px_fallback,
                    "is_release": is_release,
                    "is_ground_contact": ground_event,
                }
            )

            shoulder_line_angle = None
            left_arm_angle = None
            right_arm_angle = None
            dominant_arm_angle = None

            if players_meta:
                def _area(player: Dict) -> float:
                    bbox = player.get("bbox")
                    if not bbox or len(bbox) != 4:
                        return 0.0
                    x1, y1, x2, y2 = bbox
                    return max(0.0, (x2 - x1) * (y2 - y1))

                players_sorted = sorted(players_meta, key=_area, reverse=True)
                main_player = players_sorted[0] if players_sorted else None
                if main_player:
                    ls = main_player.get("left_shoulder")
                    rs = main_player.get("right_shoulder")
                    le = main_player.get("left_elbow")
                    re = main_player.get("right_elbow")
                    lw = main_player.get("left_wrist")
                    rw = main_player.get("right_wrist")

                    if ls is not None and rs is not None:
                        shoulder_line_angle = _angle_deg(ls, rs)
                    if ls is not None and le is not None:
                        left_arm_angle = _angle_deg(ls, le)
                    if rs is not None and re is not None:
                        right_arm_angle = _angle_deg(rs, re)

                    dominant_arm_angle = left_arm_angle
                    if lw is not None and rw is not None:
                        dominant_arm_angle = left_arm_angle if lw[1] <= rw[1] else right_arm_angle
                    elif rw is not None:
                        dominant_arm_angle = right_arm_angle
                    elif lw is None and rw is None:
                        dominant_arm_angle = left_arm_angle or right_arm_angle

            shoulder_rows.append(
                {
                    "time_s": time_s,
                    "left_arm_angle_deg": left_arm_angle,
                    "right_arm_angle_deg": right_arm_angle,
                    "shoulder_line_angle_deg": shoulder_line_angle,
                    "dominant_arm_angle_deg": dominant_arm_angle,
                }
            )

            if frame_records is not None:
                frame_records.append(
                    {
                        "frame_idx": result.frame_idx,
                        "meta": result.frame_meta,
                        "stats": {
                            "ball_speed_mps": speed_mps_fallback,
                            "ball_speed_px_s": speed_px_fallback,
                            "release_speed_mps": release_speed_mps_val,
                            "release_angle_deg": release_angle_deg,
                            "hand_force_mps2": hand_force_mps2,
                            "hand_force_px_s2": hand_force_px_s2,
                            "throw_distance_m": throw_distance_m,
                            "throw_distance_px": throw_distance_px,
                        },
                    }
                )

            if shot_log is not None and result.shot_events is not None:
                shot_log.clear()
                shot_log.extend(result.shot_events)

            if snapshots is not None and result.annotated is not None:
                if is_release and not release_snapshot_done and len(snapshots) < SNAPSHOT_MAX:
                    if snapshot_dir is None:
                        snapshot_dir = Path(tempfile.mkdtemp(prefix="ball_throw_snapshots_"))
                        runtime_store["snapshot_dir"] = str(snapshot_dir)
                    filename = f"release_frame_{result.frame_idx}.jpg"
                    snapshot_path = snapshot_dir / filename
                    width, height = _save_snapshot(result.annotated, snapshot_path)
                    if width > 0 and snapshot_path.exists():
                        snapshots.append(
                            {
                                "id": len(snapshots) + 1,
                                "type": "release",
                                "frame_idx": result.frame_idx,
                                "time_sec": time_s,
                                "image_path": str(snapshot_path),
                                "width": width,
                                "height": height,
                            }
                        )
                        release_snapshot_done = True

                if (
                    height_proxy is not None
                    and (max_height is not None and height_proxy >= max_height)
                    and not max_height_snapshot_done
                    and len(snapshots) < SNAPSHOT_MAX
                ):
                    if snapshot_dir is None:
                        snapshot_dir = Path(tempfile.mkdtemp(prefix="ball_throw_snapshots_"))
                        runtime_store["snapshot_dir"] = str(snapshot_dir)
                    filename = f"peak_height_frame_{result.frame_idx}.jpg"
                    snapshot_path = snapshot_dir / filename
                    width, height = _save_snapshot(result.annotated, snapshot_path)
                    if width > 0 and snapshot_path.exists():
                        snapshots.append(
                            {
                                "id": len(snapshots) + 1,
                                "type": "peak_height",
                                "frame_idx": result.frame_idx,
                                "time_sec": time_s,
                                "image_path": str(snapshot_path),
                                "width": width,
                                "height": height,
                            }
                        )
                        max_height_snapshot_done = True

            if callable(live_callback) and result.frame_idx % live_stride == 0:
                start = max(0, len(trajectory_rows) - live_tail_rows)
                trajectory_tail = pd.DataFrame(trajectory_rows[start:])
                shoulder_tail = pd.DataFrame(shoulder_rows[-live_tail_rows:])

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
                            "release_speed_mps": release_speed_mps_val,
                            "release_speed_px_s": release_speed_px_val,
                            "release_angle_deg": release_angle_deg,
                            "ball_height": height_proxy,
                            "hand_force_mps2": hand_force_mps2,
                            "hand_force_px_s2": hand_force_px_s2,
                            "throw_distance_m": throw_distance_m,
                            "throw_distance_px": throw_distance_px,
                            "dominant_arm_angle_deg": (
                                shoulder_rows[-1].get("dominant_arm_angle_deg")
                                if shoulder_rows
                                else None
                            ),
                        },
                        "trajectory_tail": trajectory_tail,
                        "shoulder_angle_tail": shoulder_tail,
                        "release_velocity_tail": pd.DataFrame(release_rows[-live_tail_rows:]),
                        "progress": progress,
                    }
                )

            if missing_ball_count >= missing_ball_frames:
                missing_ball_count = 0
    except Exception as exc:
        fallback = build_dummy_result(test_name, expected_matrices, settings)
        fallback.status = "fallback"
        fallback.logs = [f"Ball throw analysis failed: {exc}"] + fallback.logs
        return fallback

    if last_result is None:
        fallback = build_dummy_result(test_name, expected_matrices, settings)
        fallback.status = "fallback"
        fallback.logs = ["No frames processed; returning placeholder result."] + fallback.logs
        return fallback

    total_time_s = last_result.total_time_sec
    if total_time_s is None and trajectory_rows:
        total_time_s = trajectory_rows[-1].get("time_s")  # type: ignore[assignment]

    if release_rows:
        release_rows[-1]["release_angle_deg"] = release_angle_deg
        release_rows[-1]["hand_force_mps2"] = hand_force_mps2
        release_rows[-1]["hand_force_px_s2"] = hand_force_px_s2
        release_rows[-1]["throw_distance_m"] = throw_distance_m
        release_rows[-1]["throw_distance_px"] = throw_distance_px

    metrics = {
        "total_time_s": total_time_s,
        "release_speed_mps": release_speed_mps_val,
        "release_speed_px_s": release_speed_px_val,
        "release_angle_deg": release_angle_deg,
        "max_ball_height": max_height,
        "hand_force_mps2": hand_force_mps2,
        "hand_force_px_s2": hand_force_px_s2,
        "throw_distance_m": throw_distance_m,
        "throw_distance_px": throw_distance_px,
        "ball_detected_frames": ball_detected_frames,
        "processed_frames": processed_frames,
        "ball_detection_rate": (
            ball_detected_frames / processed_frames if processed_frames else None
        ),
        "ball_class_id": ball_class_id,
        "ball_class_name": ball_class_name,
    }

    logs.append(f"Processed frames: {len(trajectory_rows)}")
    logs.append(f"Release detected: {'yes' if release_frame_idx is not None else 'no'}")

    return AnalysisResult(
        test_name=test_name,
        status="ok",
        metrics=metrics,
        matrices={
            "release_velocity": pd.DataFrame(release_rows),
            "trajectory": pd.DataFrame(trajectory_rows),
            "shoulder_angle": pd.DataFrame(shoulder_rows),
        },
        artifacts={},
        logs=logs,
    )
