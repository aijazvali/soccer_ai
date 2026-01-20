"""
Calibration helpers for mapping image points to a fixed ground-plane rectangle.

The homography is used to convert pixel coordinates into real-world meters for
distance and speed calculations.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


POINT_ORDER = ("top-left", "top-right", "bottom-right", "bottom-left")


@dataclass
class PlaneCalibration:
    homography: np.ndarray
    field_width_m: float
    field_height_m: float
    image_points: List[Tuple[float, float]]
    field_points: List[Tuple[float, float]]


def _order_points(points: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    """Return points ordered as top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("Expected 4 points of shape (x, y).")

    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    ordered = pts[order]

    sums = ordered.sum(axis=1)
    tl_idx = int(np.argmin(sums))
    ordered = np.roll(ordered, -tl_idx, axis=0)

    def _cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    if _cross(ordered[0], ordered[1], ordered[2]) < 0:
        ordered = np.array([ordered[0], ordered[3], ordered[2], ordered[1]], dtype=np.float32)

    return [tuple(map(float, pt)) for pt in ordered]


def _field_points(field_width_m: float, field_height_m: float) -> List[Tuple[float, float]]:
    return [
        (0.0, 0.0),
        (float(field_width_m), 0.0),
        (float(field_width_m), float(field_height_m)),
        (0.0, float(field_height_m)),
    ]


def build_calibration(
    image_points: Iterable[Sequence[float]],
    field_width_m: float,
    field_height_m: float,
    reorder: bool = True,
) -> PlaneCalibration:
    pts = list(image_points)
    if len(pts) != 4:
        raise ValueError("Expected exactly 4 image points.")
    ordered = _order_points(pts) if reorder else [tuple(map(float, p)) for p in pts]
    field_pts = _field_points(field_width_m, field_height_m)
    h_matrix, _ = cv2.findHomography(
        np.array(ordered, dtype=np.float32),
        np.array(field_pts, dtype=np.float32),
    )
    if h_matrix is None:
        raise RuntimeError("Unable to compute homography from the provided points.")
    return PlaneCalibration(
        homography=h_matrix,
        field_width_m=float(field_width_m),
        field_height_m=float(field_height_m),
        image_points=ordered,
        field_points=field_pts,
    )


def project_point(point: Tuple[float, float], homography: np.ndarray) -> Tuple[float, float]:
    arr = np.array([[point]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(arr, homography)
    return float(mapped[0][0][0]), float(mapped[0][0][1])


def project_points(
    points: Iterable[Tuple[float, float]], homography: np.ndarray
) -> List[Tuple[float, float]]:
    pts = np.array(list(points), dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, homography)
    return [(float(p[0][0]), float(p[0][1])) for p in mapped]


def save_calibration(path: str, calibration: PlaneCalibration) -> None:
    payload = {
        "field_width_m": calibration.field_width_m,
        "field_height_m": calibration.field_height_m,
        "image_points": calibration.image_points,
        "field_points": calibration.field_points,
        "homography": calibration.homography.tolist(),
        "point_order": list(POINT_ORDER),
        "source": "fixed-rectangle",
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=True))


def load_calibration(
    path: str,
    default_field_width_m: Optional[float] = None,
    default_field_height_m: Optional[float] = None,
) -> PlaneCalibration:
    payload = json.loads(Path(path).read_text())
    image_points = payload.get("image_points")
    if not image_points:
        raise ValueError("Calibration file is missing image_points.")
    field_width_m = payload.get("field_width_m", default_field_width_m)
    field_height_m = payload.get("field_height_m", default_field_height_m)
    if field_width_m is None or field_height_m is None:
        raise ValueError("Calibration file is missing field dimensions.")
    if "homography" in payload:
        h_matrix = np.array(payload["homography"], dtype=np.float32)
    else:
        h_matrix = None
    calibration = build_calibration(
        image_points=image_points,
        field_width_m=field_width_m,
        field_height_m=field_height_m,
        reorder=True,
    )
    if h_matrix is not None and h_matrix.shape == (3, 3):
        calibration.homography = h_matrix
    return calibration


__all__ = [
    "PlaneCalibration",
    "POINT_ORDER",
    "build_calibration",
    "load_calibration",
    "project_point",
    "project_points",
    "save_calibration",
]
