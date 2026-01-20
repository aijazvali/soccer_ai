"""
Shared utilities and configuration for the soccer touch-detection pipelines.

The package centralizes common constants, helpers, and model loaders used by the
Streamlit app and CLI scripts so individual entrypoints stay small and focused.
"""

from soccer_ai import config  # noqa: F401
from soccer_ai.core import (  # noqa: F401
    TrackState,
    clamp_box,
    smooth_point,
    pick_ball,
    smooth_ball,
    dist,
    angle_diff_deg,
)
from soccer_ai.options import TouchOptions  # noqa: F401
from soccer_ai.models import load_models  # noqa: F401

__all__ = [
    "config",
    "TouchOptions",
    "TrackState",
    "clamp_box",
    "smooth_point",
    "pick_ball",
    "smooth_ball",
    "dist",
    "angle_diff_deg",
    "load_models",
]
