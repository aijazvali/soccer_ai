from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _cv2():
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    return cv2


def format_speed(speed_kmh: Optional[float], use_metric: bool = True) -> str:
    if speed_kmh is None:
        return "--"
    if use_metric:
        speed_mps = speed_kmh / 3.6
        return f"{speed_mps:.1f} m/s"
    return f"{speed_kmh:.1f} km/h"


def format_accel(accel: Optional[float]) -> str:
    if accel is None:
        return "--"
    return f"{accel:.2f} m/s^2"


def draw_stats_overlay(
    frame_bgr,
    lines: List[str],
    header: Optional[str] = None,
    anchor: tuple[int, int] = (16, 16),
) -> None:
    cv2 = _cv2()
    if cv2 is None or frame_bgr is None:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    header_scale = 0.7
    thickness = 2
    header_thickness = 2
    padding = 10
    line_gap = 6

    items = []
    if header:
        items.append((header, header_scale, header_thickness))
    for line in lines:
        items.append((line, font_scale, thickness))

    max_width = 0
    total_height = padding
    metrics = []
    for text, scale, thick in items:
        (w, h), base = cv2.getTextSize(text, font, scale, thick)
        max_width = max(max_width, w)
        total_height += h + base + line_gap
        metrics.append((text, scale, thick, h, base))
    total_height += padding - line_gap
    box_width = max_width + padding * 2
    box_height = max(0, total_height)
    x, y = anchor
    y = max(0, y)
    x = max(0, x)

    cv2.rectangle(
        frame_bgr,
        (x, y),
        (x + box_width, y + box_height),
        (0, 0, 0),
        -1,
    )

    cursor_y = y + padding
    text_x = x + padding
    for text, scale, thick, height, base in metrics:
        cursor_y += height
        cv2.putText(
            frame_bgr,
            text,
            (text_x, cursor_y),
            font,
            scale,
            (255, 255, 255),
            thick,
            cv2.LINE_AA,
        )
        cursor_y += base + line_gap


def draw_event_overlay(frame_bgr, event: Optional[Dict[str, Any]]) -> None:
    cv2 = _cv2()
    if cv2 is None or frame_bgr is None or not event:
        return
    label = str(event.get("type") or "event").upper()
    power_val = event.get("power")
    if power_val is not None:
        label = f"{label} {power_val:.0f}"
    pos = event.get("pos")
    if not isinstance(pos, (list, tuple)) or len(pos) < 2:
        pos = (40.0, 80.0)
    x = int(pos[0])
    y = max(24, int(pos[1]) - 10)
    color_map = {
        "SHOT": (0, 0, 255),
        "PASS": (0, 255, 0),
        "DRIBBLE": (255, 215, 0),
    }
    cv2.putText(
        frame_bgr,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color_map.get(label.split()[0], (0, 255, 255)),
        3,
        cv2.LINE_AA,
    )


def draw_player_overlays(
    frame_bgr,
    players: List[Dict[str, Any]],
    show_ids: bool = True,
    show_feet: bool = True,
    show_speed: bool = True,
    use_metric_display: bool = True,
) -> None:
    cv2 = _cv2()
    if cv2 is None or frame_bgr is None or not players:
        return
    height = frame_bgr.shape[0]
    for player in players:
        bbox = player.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if show_ids:
            pid = player.get("id", "?")
            cv2.putText(
                frame_bgr,
                f"ID {pid}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        if show_feet:
            left = player.get("left")
            right = player.get("right")
            if left is not None:
                cv2.circle(frame_bgr, (int(left[0]), int(left[1])), 5, (255, 0, 0), -1)
            if right is not None:
                cv2.circle(frame_bgr, (int(right[0]), int(right[1])), 5, (0, 0, 255), -1)
        if show_speed:
            speed_kmh = player.get("speed_kmh")
            if speed_kmh is not None:
                speed_text = format_speed(speed_kmh, use_metric_display)
                speed_y = min(height - 8, y2 + 24)
                cv2.putText(
                    frame_bgr,
                    speed_text,
                    (x1, speed_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )


def draw_ball_overlay(
    frame_bgr,
    ball: Optional[Dict[str, Any]],
    show_vector: bool,
    show_speed: bool,
    use_homography: bool,
    vector_scale: float,
) -> None:
    cv2 = _cv2()
    if cv2 is None or frame_bgr is None or not ball:
        return
    center = ball.get("center")
    radius = ball.get("radius")
    if center is None or radius is None:
        return
    start_pt = (int(center[0]), int(center[1]))
    cv2.circle(frame_bgr, start_pt, int(radius), (0, 165, 255), 2)
    vel_draw = ball.get("vel_draw")
    if show_vector and vel_draw is not None:
        end_pt = (
            int(center[0] + vel_draw[0] * vector_scale),
            int(center[1] + vel_draw[1] * vector_scale),
        )
        cv2.arrowedLine(frame_bgr, start_pt, end_pt, (0, 255, 255), 2, tipLength=0.3)
        if show_speed:
            display_speed = ball.get("speed_mps") if use_homography else ball.get("speed_draw")
            speed_unit = "m/s" if use_homography else "px/s"
            if display_speed is not None:
                cv2.putText(
                    frame_bgr,
                    f"{display_speed:.1f}{speed_unit}",
                    (start_pt[0] + 6, start_pt[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )


def draw_ball_trail_overlay(
    frame_bgr,
    trail_points: List[Tuple[float, float]],
    color: Tuple[int, int, int] = (0, 165, 255),
    max_thickness: int = 6,
) -> None:
    cv2 = _cv2()
    if cv2 is None or frame_bgr is None or len(trail_points) < 2:
        return
    max_thickness = max(1, int(max_thickness))
    denom = max(1, len(trail_points) - 1)
    for idx in range(1, len(trail_points)):
        pt1 = trail_points[idx - 1]
        pt2 = trail_points[idx]
        t = idx / denom
        intensity = 0.2 + 0.8 * t
        thickness = max(1, int(round(1 + (max_thickness - 1) * t)))
        color_scaled = (
            int(color[0] * intensity),
            int(color[1] * intensity),
            int(color[2] * intensity),
        )
        cv2.line(
            frame_bgr,
            (int(pt1[0]), int(pt1[1])),
            (int(pt2[0]), int(pt2[1])),
            color_scaled,
            thickness,
            lineType=cv2.LINE_AA,
        )


def collect_ball_trail(
    frame_lookup: Dict[int, Dict[str, Any]],
    frame_idx: int,
    max_len: int,
    max_gap_frames: int,
) -> List[Tuple[float, float]]:
    if max_len <= 1:
        return []
    keys = [k for k in frame_lookup.keys() if k <= frame_idx]
    keys.sort(reverse=True)
    points: List[Tuple[float, float]] = []
    last_frame = None
    for k in keys:
        meta = frame_lookup[k].get("meta") if frame_lookup.get(k) else None
        ball = meta.get("ball") if meta else None
        center = ball.get("center") if ball else None
        if center is None:
            continue
        if last_frame is not None and last_frame - k > max_gap_frames:
            break
        points.append((float(center[0]), float(center[1])))
        last_frame = k
        if len(points) >= max_len:
            break
    return list(reversed(points))
