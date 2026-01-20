import argparse
from pathlib import Path

import cv2
import numpy as np

from soccer_ai import config as cfg
from soccer_ai.calibration import build_calibration, save_calibration


WINDOW_NAME = "Cone Calibration"
POINT_RADIUS = 8
HIT_RADIUS = 14
ORDER_LABELS = ("TL", "TR", "BR", "BL")


def _order_points(points):
    pts = np.array(points, dtype=np.float32)
    if pts.shape != (4, 2):
        return None
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return [tuple(map(float, pt)) for pt in ordered]


def _hit_test(points, x, y, radius=HIT_RADIUS):
    if not points:
        return None
    for idx, (px, py) in enumerate(points):
        if (px - x) ** 2 + (py - y) ** 2 <= radius ** 2:
            return idx
    return None


def _draw_points(frame, points, selected_idx=None):
    for idx, (x, y) in enumerate(points):
        color = (0, 128, 255) if idx == selected_idx else (0, 255, 255)
        cv2.circle(frame, (int(x), int(y)), POINT_RADIUS, color, -1)
        cv2.putText(
            frame,
            str(idx + 1),
            (int(x) + 8, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    if len(points) == 4:
        ordered = _order_points(points)
        if ordered is not None:
            poly = np.array(ordered, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [poly], True, (0, 255, 0), 2)
            for label, (x, y) in zip(ORDER_LABELS, ordered):
                cv2.putText(
                    frame,
                    label,
                    (int(x) + 10, int(y) + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )


def _draw_instructions(frame, width_m, height_m, count):
    lines = [
        f"Click 4 cones (rectangle {width_m:.1f}m x {height_m:.1f}m).",
        "Drag points to adjust. Right-click to delete.",
        "Keys: c=confirm, r=reset, u=undo, q=quit",
        f"Picked: {count}/4",
    ]
    y = 24
    for line in lines:
        cv2.putText(
            frame,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        y += 22


def main():
    parser = argparse.ArgumentParser(description="Calibrate a fixed-rectangle plane from cone clicks.")
    parser.add_argument("video", help="Path to the input video file.")
    parser.add_argument(
        "--width",
        type=float,
        default=cfg.CALIB_RECT_WIDTH_M,
        help="Rectangle width in meters (default from config).",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=cfg.CALIB_RECT_HEIGHT_M,
        help="Rectangle height in meters (default from config).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output calibration JSON path (default: <video>.calibration.json).",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise SystemExit("Unable to read the first frame.")

    points = []
    selected_idx = None
    dragging = False

    def on_mouse(event, x, y, _flags, _param):
        nonlocal selected_idx, dragging
        if event == cv2.EVENT_LBUTTONDOWN:
            hit_idx = _hit_test(points, x, y)
            if hit_idx is not None:
                selected_idx = hit_idx
                dragging = True
            elif len(points) < 4:
                points.append((float(x), float(y)))
                selected_idx = len(points) - 1
                dragging = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging and selected_idx is not None:
                points[selected_idx] = (float(x), float(y))
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
            selected_idx = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            hit_idx = _hit_test(points, x, y)
            if hit_idx is not None:
                points.pop(hit_idx)
                if selected_idx is not None:
                    if hit_idx == selected_idx:
                        selected_idx = None
                        dragging = False
                    elif hit_idx < selected_idx:
                        selected_idx -= 1

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    confirmed = False
    while True:
        canvas = frame.copy()
        _draw_points(canvas, points, selected_idx=selected_idx)
        _draw_instructions(canvas, args.width, args.height, len(points))
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("r"):
            points.clear()
            selected_idx = None
            dragging = False
        elif key == ord("u"):
            if points:
                removed_idx = len(points) - 1
                points.pop()
                if selected_idx is not None:
                    if removed_idx == selected_idx:
                        selected_idx = None
                        dragging = False
                    elif removed_idx < selected_idx:
                        selected_idx -= 1
        elif key == ord("q") or key == 27:
            break
        elif key == ord("c") and len(points) == 4:
            confirmed = True
            break

    cv2.destroyAllWindows()

    if not confirmed:
        raise SystemExit("Calibration canceled.")

    calibration = build_calibration(points, args.width, args.height, reorder=True)
    out_path = Path(args.out) if args.out else video_path.with_suffix(".calibration.json")
    save_calibration(str(out_path), calibration)
    print(f"Saved calibration to {out_path}")


if __name__ == "__main__":
    main()
