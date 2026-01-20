"""
Lightweight model loader helpers.

The loader is cached so repeated calls (e.g., from Streamlit reruns) reuse the
same YOLO instances.
"""

from functools import lru_cache
from typing import Tuple

from ultralytics import YOLO

import soccer_ai.config as cfg
from pathlib import Path


def _resolve_weights(path: str) -> str:
    """Return an absolute path, preferring files inside MODELS_DIR."""
    p = Path(path)
    if p.exists():
        return str(p.resolve())

    candidate = cfg.MODELS_DIR / p.name
    if candidate.exists():
        return str(candidate.resolve())

    # Fall back to the raw path (lets YOLO handle remote or custom URIs)
    return str(p)


@lru_cache(maxsize=4)
def load_models(
    detector_weights: str = cfg.DETECTOR_WEIGHTS,
    pose_weights: str = cfg.POSE_WEIGHTS,
) -> Tuple[YOLO, YOLO]:
    det_path = _resolve_weights(detector_weights)
    pose_path = _resolve_weights(pose_weights)

    try:
        det = YOLO(det_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load detector weights '{Path(detector_weights).name}': {exc}"
        ) from exc

    try:
        pose = YOLO(pose_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load pose weights '{Path(pose_weights).name}'. "
            "These weights may require a newer ultralytics build; try upgrading ultralytics or pick another file. "
            f"Underlying error: {exc}"
        ) from exc

    return det, pose


__all__ = ["load_models"]
