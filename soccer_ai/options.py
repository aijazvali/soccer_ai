"""
User-tunable options for the touch detection pipeline.
"""

from dataclasses import dataclass
from typing import Optional

import soccer_ai.config as cfg


@dataclass
class TouchOptions:
    detector_weights: str = cfg.DETECTOR_WEIGHTS
    pose_weights: str = cfg.POSE_WEIGHTS
    draw_ball_vector: bool = cfg.DRAW_BALL_VECTOR
    ball_vector_scale: float = cfg.BALL_VECTOR_SCALE
    show_ball_speed: bool = cfg.SHOW_BALL_SPEED
    show_ball_components: bool = cfg.SHOW_BALL_COMPONENTS
    show_player_speed: bool = cfg.SHOW_PLAYER_SPEED
    event_touch_enabled: bool = cfg.EVENT_TOUCH_ENABLED
    event_touch_dist_ratio: float = cfg.EVENT_TOUCH_DIST_RATIO
    display_stride: int = 1
    calibration_path: Optional[str] = None
    use_homography: bool = cfg.USE_HOMOGRAPHY
    # Extended ground plane visualization
    draw_extended_ground: bool = cfg.DRAW_EXTENDED_GROUND
    extended_ground_multiplier: float = cfg.EXTENDED_GROUND_MULTIPLIER
    draw_ground_grid: bool = cfg.DRAW_GROUND_GRID
    ground_grid_spacing_m: float = cfg.GROUND_GRID_SPACING_M
    grid_line_subdivisions: int = cfg.GRID_LINE_SUBDIVISIONS
    show_grid_distance_markers: bool = cfg.SHOW_GRID_DISTANCE_MARKERS
    # Accuracy improvements
    min_movement_m: float = cfg.MIN_MOVEMENT_M
    max_speed_gap_frames: int = cfg.MAX_SPEED_GAP_FRAMES
    max_human_speed_mps: float = cfg.MAX_HUMAN_SPEED_MPS
    max_ball_speed_mps: float = cfg.MAX_BALL_SPEED_MPS
    use_ema_smoothing: bool = cfg.USE_EMA_SMOOTHING
    ema_alpha: float = cfg.EMA_ALPHA
    # Unit system
    use_metric_display: bool = cfg.USE_METRIC_DISPLAY


__all__ = ["TouchOptions"]
