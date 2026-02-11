from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

from .formatting import format_distance_m, format_time
from .player_overlay import (
    collect_ball_trail,
    draw_ball_overlay,
    draw_ball_trail_overlay,
    draw_event_overlay,
    draw_player_overlays,
    draw_stats_overlay,
    format_accel,
    format_speed,
)


def _cv2():
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    return cv2


def _overlay_frame(
    frame_bgr,
    frame_idx: int,
    record: Dict,
    frame_lookup: Dict[int, Dict],
    settings: Dict,
) -> None:
    meta = record.get("meta", {}) or {}
    stats = record.get("stats", {}) or {}
    use_metric_display = settings.get("use_metric_display", True)
    use_homography = bool(meta.get("use_homography", settings.get("use_homography", False)))

    if settings.get("show_ball_trail"):
        trail_len = int(settings.get("trail_len", 12))
        max_gap = int(settings.get("trail_max_gap", 5))
        trail_points = collect_ball_trail(
            frame_lookup,
            frame_idx,
            max_len=max(2, trail_len),
            max_gap_frames=max(1, max_gap),
        )
        draw_ball_trail_overlay(frame_bgr, trail_points)

    if settings.get("show_players"):
        draw_player_overlays(
            frame_bgr,
            meta.get("players", []),
            show_ids=settings.get("show_ids", True),
            show_feet=settings.get("show_feet", True),
            show_speed=settings.get("show_player_speed", True),
            use_metric_display=use_metric_display,
        )

    if settings.get("show_ball"):
        draw_ball_overlay(
            frame_bgr,
            meta.get("ball"),
            show_vector=settings.get("show_ball_vector", True),
            show_speed=settings.get("show_ball_speed", True),
            use_homography=use_homography,
            vector_scale=float(settings.get("vector_scale", 10.0)),
        )

    if settings.get("show_annotations", True):
        draw_event_overlay(frame_bgr, meta.get("event_overlay"))

    overlay_lines = []
    avg_speed_text = format_speed(stats.get("avg_speed_kmh"), use_metric_display)
    max_speed_text = format_speed(stats.get("max_speed_kmh"), use_metric_display)
    overlay_lines.append(f"Speed (avg / max): {avg_speed_text} / {max_speed_text}")
    overlay_lines.append(
        f"Time / Distance: {format_time(stats.get('total_time_sec'))} / "
        f"{format_distance_m(stats.get('total_distance_m'), use_metric_display)}"
    )
    overlay_lines.append(
        f"Accel / Decel (peak): {format_accel(stats.get('peak_accel_mps2'))} / "
        f"{format_accel(stats.get('peak_decel_mps2'))}"
    )
    touches_val = stats.get("total_touches")
    if touches_val is not None:
        overlay_lines.append(f"Touches: {touches_val}")
    left_val = stats.get("left_touches")
    right_val = stats.get("right_touches")
    if left_val is not None or right_val is not None:
        left_disp = "--" if left_val is None else f"{left_val}"
        right_disp = "--" if right_val is None else f"{right_val}"
        overlay_lines.append(f"Left / Right: {left_disp} / {right_disp}")
    max_streak = stats.get("max_consecutive_touches")
    if max_streak is not None:
        overlay_lines.append(f"Max Streak: {max_streak}")
    draw_stats_overlay(frame_bgr, overlay_lines)


def export_annotated_video(
    video_path: str,
    frame_records: List[Dict],
    settings: Dict,
    output_path: Optional[str] = None,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> Optional[str]:
    cv2 = _cv2()
    if cv2 is None:
        raise RuntimeError("OpenCV is required to export annotated video.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    display_stride = int(settings.get("display_stride", 1))
    display_stride = max(1, display_stride)
    output_fps = fps / display_stride if display_stride > 1 else fps

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    if output_path is None:
        output_path = str(Path(video_path).with_suffix("")) + "_annotated.mp4"
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(output_file), fourcc, output_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Unable to initialize video writer.")

    frame_lookup = {
        int(rec.get("frame_idx")): rec
        for rec in frame_records
        if rec.get("frame_idx") is not None
    }

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if display_stride > 1 and frame_idx % display_stride != 0:
                continue

            record = frame_lookup.get(frame_idx)
            if record:
                _overlay_frame(frame, frame_idx, record, frame_lookup, settings)

            writer.write(frame)

            if progress_cb and total_frames:
                progress_cb(min(1.0, frame_idx / total_frames))
    finally:
        cap.release()
        writer.release()

    return str(output_file)
