"""
Core helpers and state containers for the touch-detection pipeline.

These utilities are intentionally framework-agnostic so they can be reused by
both the Streamlit UI and CLI scripts.
"""

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Optional

import numpy as np

from soccer_ai import config as cfg


@dataclass
class TrackState:
    active_foot: Optional[str] = None
    contact_streak: int = 0
    last_touch_frame: int = -1000
    left_touches: int = 0
    right_touches: int = 0
    passes: int = 0
    shots: int = 0
    shot_power_total: float = 0.0
    shot_power_count: int = 0
    last_contact_frame: int = -1000
    touch_locked: bool = False
    left_last_frame: int = -1000
    right_last_frame: int = -1000
    pending_contact_frame: int = -1000
    pending_foot: Optional[str] = None
    pending_ball_dist: float = 0.0
    left_hist: deque = field(
        default_factory=lambda: deque(maxlen=cfg.FOOT_SMOOTHING)
    )
    right_hist: deque = field(
        default_factory=lambda: deque(maxlen=cfg.FOOT_SMOOTHING)
    )
    prev_speed_point: Optional[tuple] = None
    prev_speed_field: Optional[tuple] = None
    last_speed_frame: int = -1000
    speed_history: deque = field(
        default_factory=lambda: deque(maxlen=cfg.PLAYER_SPEED_SMOOTHING)
    )
    speed_kmh: Optional[float] = None
    total_distance_m: float = 0.0
    last_speed_mps: Optional[float] = None
    peak_accel_mps2: float = 0.0
    peak_decel_mps2: float = 0.0
    ground_ankle_history: deque = field(
        default_factory=lambda: deque(maxlen=cfg.JUMP_GROUND_WINDOW)
    )
    meters_per_px_history: deque = field(
        default_factory=lambda: deque(maxlen=cfg.JUMP_SCALE_SMOOTHING)
    )
    last_m_per_px: Optional[float] = None
    jump_active: bool = False
    jump_air_streak: int = 0
    jump_start_frame: int = -1000
    jump_cooldown_frame: int = -1000
    jump_peak_delta_px: float = 0.0
    jump_count: int = 0
    max_jump_height_m: float = 0.0
    max_jump_height_px: float = 0.0
    prev_ankle_y: Optional[float] = None
    ankle_up_streak: int = 0


def clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int):
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def smooth_point(hist):
    if not hist:
        return None
    arr = np.array(hist, dtype=np.float32)
    x, y = np.median(arr, axis=0)
    return (float(x), float(y))


def pick_ball(candidates, last_center):
    if not candidates:
        return None
    if last_center is None:
        return max(candidates, key=lambda c: c[2])

    def score(cand):
        (cx, cy), _r, conf = cand
        dist = math.hypot(cx - last_center[0], cy - last_center[1])
        return dist - conf * 50.0

    return min(candidates, key=score)


def smooth_ball(hist):
    arr = np.array(hist, dtype=np.float32)
    cx, cy, r = np.median(arr, axis=0)
    return (float(cx), float(cy)), float(r)


def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def angle_diff_deg(a, b):
    diff = abs(a - b)
    if diff > math.pi:
        diff = 2 * math.pi - diff
    return math.degrees(diff)


__all__ = [
    "TrackState",
    "clamp_box",
    "smooth_point",
    "pick_ball",
    "smooth_ball",
    "dist",
    "angle_diff_deg",
]
