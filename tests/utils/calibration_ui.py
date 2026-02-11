from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from PIL import Image


def ensure_canvas_compat() -> None:
    try:
        import streamlit.elements.image as st_image
        if hasattr(st_image, "image_to_url"):
            return
        from streamlit.elements.lib import image_utils, layout_utils

        def _image_to_url_compat(image, width, clamp, channels, output_format, image_id):
            layout_config = layout_utils.LayoutConfig(width=width)
            return image_utils.image_to_url(
                image, layout_config, clamp, channels, output_format, image_id
            )

        st_image.image_to_url = _image_to_url_compat
    except Exception:
        return


def read_video_frame(path: str, frame_idx: int):
    try:
        import cv2  # type: ignore
    except Exception:
        return None

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    if frame_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    return frame


def prepare_canvas_frame(frame_bgr, max_width: int = 960) -> Tuple[Image.Image, float]:
    try:
        import cv2  # type: ignore
    except Exception:
        raise RuntimeError("OpenCV required for calibration preview.")

    height, width = frame_bgr.shape[:2]
    scale = min(1.0, max_width / max(1, width))
    new_w = int(width * scale)
    new_h = int(height * scale)
    resized = cv2.resize(frame_bgr, (new_w, new_h)) if scale < 1.0 else frame_bgr
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb), scale


def canvas_initial_drawing(points: Iterable[Tuple[float, float]], radius: int = 6) -> Dict[str, Any]:
    objects: List[Dict[str, Any]] = []
    for x, y in points:
        objects.append(
            {
                "type": "circle",
                "left": float(x - radius),
                "top": float(y - radius),
                "radius": float(radius),
                "fill": "rgba(255, 196, 0, 0.6)",
                "stroke": "rgba(255, 196, 0, 0.9)",
                "strokeWidth": 2,
            }
        )
    return {"version": "4.4.0", "objects": objects}


def extract_canvas_points(canvas_json: Dict[str, Any]) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    if not canvas_json:
        return points
    for obj in canvas_json.get("objects", []):
        x = obj.get("x")
        y = obj.get("y")
        if x is not None and y is not None:
            points.append((float(x), float(y)))
            continue
        left = float(obj.get("left", 0.0))
        top = float(obj.get("top", 0.0))
        radius = float(obj.get("radius", 0.0))
        scale_x = float(obj.get("scaleX", 1.0))
        scale_y = float(obj.get("scaleY", 1.0))
        points.append((left + radius * scale_x, top + radius * scale_y))
    return points
