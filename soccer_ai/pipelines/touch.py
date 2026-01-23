"""
Touch detection pipeline extracted from the CLI prototype (final2.2.py).

The pipeline is exposed as a generator so both CLI scripts and Streamlit can
consume annotated frames and running stats while sharing the same core logic.
"""

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Generator, Optional, List, Dict, Tuple, Any

import cv2
import numpy as np

from soccer_ai.calibration import load_calibration, project_point
import soccer_ai.config as cfg
from soccer_ai.core import (
    TrackState,
    clamp_box,
    smooth_point,
    pick_ball,
    smooth_ball,
    dist,
    angle_diff_deg,
)
from soccer_ai.models import load_models
from soccer_ai.options import TouchOptions


@dataclass
class FrameResult:
    frame_idx: int
    annotated: np.ndarray
    left_touches: int
    right_touches: int
    avg_speed_kmh: Optional[float] = None
    max_speed_kmh: Optional[float] = None
    total_time_sec: Optional[float] = None
    total_distance_m: Optional[float] = None
    peak_accel_mps2: Optional[float] = None
    peak_decel_mps2: Optional[float] = None
    total_jumps: int = 0
    highest_jump_m: Optional[float] = None
    highest_jump_px: Optional[float] = None
    shot_count: int = 0
    shot_events: Optional[List[Dict]] = None
    pass_count: int = 0
    avg_shot_power: Optional[float] = None
    player_stats: Optional[Dict[str, Dict]] = None
    frame_meta: Optional[Dict[str, Any]] = None


@dataclass
class BallMotionSample:
    frame_idx: int
    center: Tuple[float, float]
    vx: float
    vy: float
    speed: float
    accel: float
    direction: Optional[float]


class BallMotionBuffer:
    """Rolling buffer for ball motion metrics."""

    def __init__(self, fps: float, maxlen: int):
        self.fps = fps if fps and fps > 0 else 30.0
        self.samples: deque[BallMotionSample] = deque(maxlen=maxlen)
        self.prev_center: Optional[Tuple[float, float]] = None
        self.prev_frame: Optional[int] = None
        self.prev_speed: Optional[float] = None

    def reset(self):
        self.samples.clear()
        self.prev_center = None
        self.prev_frame = None
        self.prev_speed = None

    def add(self, frame_idx: int, center: Tuple[float, float]) -> Optional[BallMotionSample]:
        if self.prev_center is None or self.prev_frame is None:
            sample = BallMotionSample(frame_idx, center, 0.0, 0.0, 0.0, 0.0, None)
            self.samples.append(sample)
            self.prev_center = center
            self.prev_frame = frame_idx
            self.prev_speed = 0.0
            return sample

        dt_frames = frame_idx - self.prev_frame
        if dt_frames <= 0:
            return None
        dt_seconds = dt_frames / self.fps
        vx_px_s = (center[0] - self.prev_center[0]) / dt_seconds
        vy_px_s = (center[1] - self.prev_center[1]) / dt_seconds
        speed = math.hypot(vx_px_s, vy_px_s)
        accel = 0.0
        if self.prev_speed is not None:
            accel = (speed - self.prev_speed) / dt_seconds if dt_seconds > 0 else 0.0
        direction = math.atan2(vy_px_s, vx_px_s) if speed > 0 else None

        sample = BallMotionSample(frame_idx, center, vx_px_s, vy_px_s, speed, accel, direction)
        self.samples.append(sample)
        self.prev_center = center
        self.prev_frame = frame_idx
        self.prev_speed = speed
        return sample

    def recent(self, start_frame: int) -> List[BallMotionSample]:
        return [s for s in self.samples if s.frame_idx >= start_frame]


@dataclass
class KickAnalysisResult:
    event_type: str
    kicker_id: str
    foot: Optional[str]
    frame_idx: int
    kick_frame: int
    avg_speed: float
    peak_accel: float
    dir_std_deg: float
    ground_fraction: float
    target_player_id: Optional[str]
    shot_power: Optional[float]
    display_center: Optional[Tuple[float, float]] = None


def _compute_direction_stats(samples: List[BallMotionSample]):
    angles = [s.direction for s in samples if s.direction is not None]
    if not angles:
        return None, float("inf")
    sin_mean = sum(math.sin(a) for a in angles) / len(angles)
    cos_mean = sum(math.cos(a) for a in angles) / len(angles)
    mean_angle = math.atan2(sin_mean, cos_mean)
    diffs = [angle_diff_deg(a, mean_angle) for a in angles]
    dir_std = float(np.std(diffs)) if diffs else 0.0
    return mean_angle, dir_std


def _compute_ground_fraction(
    samples: List[BallMotionSample], max_delta: float = cfg.GROUND_MAX_DELTA_PX
):
    if not samples:
        return 0.0
    eps = 1e-3
    stable = [
        s
        for s in samples
        if s.speed <= eps or abs(s.vy) / max(s.speed, eps) <= cfg.GROUND_MAX_VY_RATIO
    ]
    ys = [s.center[1] for s in samples]
    vertical_span = (max(ys) - min(ys)) if ys else 0.0
    if vertical_span > max_delta:
        return 0.0
    return len(stable) / len(samples)


def _find_receiver(
    ball_pos: Tuple[float, float],
    direction_angle: Optional[float],
    players: List[Dict],
    kicker_id,
    max_dist: float,
    max_angle: float,
):
    """Return (player_id, distance) for the nearest receiver in the kick direction."""
    if direction_angle is None:
        return None
    best = None
    for person in players:
        pid = person["id"]
        if pid == kicker_id:
            continue
        candidate = (
            person.get("left_field")
            or person.get("right_field")
            or person.get("left")
            or person.get("right")
        )
        if candidate is None:
            if "bbox_field" in person and person["bbox_field"] is not None:
                candidate = person["bbox_field"]
            else:
                x1, y1, x2, y2 = person["bbox"]
                candidate = ((x1 + x2) / 2.0, max(y1, y2))
        vec = (candidate[0] - ball_pos[0], candidate[1] - ball_pos[1])
        dist_px = math.hypot(vec[0], vec[1])
        if dist_px <= 0:
            continue
        if dist_px > max_dist:
            continue
        angle_to_player = math.atan2(vec[1], vec[0])
        angle_delta = angle_diff_deg(angle_to_player, direction_angle)
        if angle_delta <= max_angle:
            if best is None or dist_px < best[1]:
                best = (pid, dist_px)
    return best


def _normalize_shot_power(
    peak_accel: float, accel_min: float, accel_max: float
) -> float:
    span = max(accel_max - accel_min, 1e-3)
    score = (peak_accel - accel_min) / span
    return max(0.0, min(1.0, score)) * 100.0


def _draw_ball_trail(
    frame: np.ndarray,
    trail: deque,
    color: Tuple[int, int, int] = (0, 165, 255),
    max_thickness: int = 6,
) -> None:
    if len(trail) < 2:
        return
    max_thickness = max(1, int(max_thickness))
    denom = max(1, len(trail) - 1)
    for idx in range(1, len(trail)):
        pt1 = trail[idx - 1]
        pt2 = trail[idx]
        if pt1 is None or pt2 is None:
            continue
        t = idx / denom
        intensity = 0.2 + 0.8 * t
        thickness = max(1, int(round(1 + (max_thickness - 1) * t)))
        color_scaled = (
            int(color[0] * intensity),
            int(color[1] * intensity),
            int(color[2] * intensity),
        )
        cv2.line(
            frame,
            (int(pt1[0]), int(pt1[1])),
            (int(pt2[0]), int(pt2[1])),
            color_scaled,
            thickness,
            lineType=cv2.LINE_AA,
        )


def _to_point(pt: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if pt is None:
        return None
    return (float(pt[0]), float(pt[1]))


class KickAnalyzer:
    """State machine to classify post-kick outcomes."""

    def __init__(
        self,
        fps: float,
        use_homography: bool = False,
        meters_per_px: Optional[float] = None,
    ):
        self.fps = fps if fps and fps > 0 else 30.0
        self.use_homography = use_homography
        self.meters_per_px = meters_per_px
        self.kick_m_per_px: Optional[float] = None
        self.phase = "IDLE"
        self.cooldown_until = -1000
        self.kick_frame = -1000
        self.analysis_ready_frame = -1000
        self.analysis_deadline = -1000
        self.kicker_id: Optional[str] = None
        self.kicking_foot: Optional[str] = None
        self.kick_ball_center: Optional[Tuple[float, float]] = None
        self.samples: List[BallMotionSample] = []
        self.last_result: Optional[KickAnalysisResult] = None

    def tick(self, frame_idx: int):
        if self.phase in ("PASS", "SHOT", "DRIBBLE") and frame_idx >= self.cooldown_until:
            self.phase = "IDLE"

    def start_kick(
        self,
        frame_idx: int,
        kicker_id,
        foot: Optional[str],
        ball_center: Optional[Tuple[float, float]],
        meters_per_px: Optional[float] = None,
    ) -> bool:
        if frame_idx < self.cooldown_until and self.phase != "IDLE":
            return False
        if self.phase in ("FOOT_CONTACT", "POST_KICK_ANALYSIS"):
            return False
        self.phase = "FOOT_CONTACT"
        self.kick_frame = frame_idx
        self.analysis_ready_frame = frame_idx + cfg.POST_KICK_MIN_FRAMES
        self.analysis_deadline = frame_idx + cfg.POST_KICK_MAX_FRAMES
        self.kicker_id = str(kicker_id)
        self.kicking_foot = foot
        self.kick_ball_center = ball_center
        self.kick_m_per_px = meters_per_px
        self.samples.clear()
        return True

    def process_sample(
        self,
        sample: Optional[BallMotionSample],
        players: List[Dict],
        goal_angle: Optional[float],
    ) -> Optional[KickAnalysisResult]:
        if self.phase not in ("FOOT_CONTACT", "POST_KICK_ANALYSIS"):
            return None
        if sample is None or sample.frame_idx <= self.kick_frame:
            return None

        self.samples.append(sample)
        if self.phase == "FOOT_CONTACT":
            self.phase = "POST_KICK_ANALYSIS"

        if sample.frame_idx >= self.analysis_ready_frame:
            return self._finalize(sample.frame_idx, players, goal_angle)
        return None

    def finalize_if_due(
        self,
        frame_idx: int,
        players: List[Dict],
        goal_angle: Optional[float],
    ) -> Optional[KickAnalysisResult]:
        if self.phase in ("FOOT_CONTACT", "POST_KICK_ANALYSIS") and frame_idx >= self.analysis_deadline:
            return self._finalize(frame_idx, players, goal_angle)
        return None

    def _finalize(
        self,
        frame_idx: int,
        players: List[Dict],
        goal_angle: Optional[float],
    ) -> KickAnalysisResult:
        if not self.samples and self.kick_ball_center is not None:
            self.samples.append(
                BallMotionSample(
                    frame_idx, self.kick_ball_center, 0.0, 0.0, 0.0, 0.0, None
                )
            )

        result = self._classify(players, goal_angle, frame_idx)
        self.phase = result.event_type.upper()
        self.cooldown_until = frame_idx + cfg.POST_KICK_COOLDOWN_FRAMES
        self.last_result = result
        self._reset_tracking()
        return result

    def _classify(
        self,
        players: List[Dict],
        goal_angle: Optional[float],
        decision_frame: int,
    ) -> KickAnalysisResult:
        samples = self.samples[-cfg.POST_KICK_MAX_FRAMES :]
        # Aggregate basic kinematics over the post-kick window.
        speeds = [s.speed for s in samples if s.speed is not None]
        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        peak_speed = float(np.max(speeds)) if speeds else 0.0
        peak_accel = float(np.max([s.accel for s in samples])) if samples else 0.0
        accel_by_frame = {s.frame_idx: s.accel for s in samples}
        dir_mean, dir_std = _compute_direction_stats(samples)
        scale = self.kick_m_per_px or self.meters_per_px
        use_metric = self.use_homography and scale is not None
        ground_max_delta = (
            cfg.GROUND_MAX_DELTA_PX * scale if use_metric else cfg.GROUND_MAX_DELTA_PX
        )
        ground_fraction = _compute_ground_fraction(samples, max_delta=ground_max_delta)

        # Receiver detection: nearest player in the travel direction.
        receiver = None
        pass_target_max = (
            cfg.PASS_TARGET_MAX_DIST_PX * scale if use_metric else cfg.PASS_TARGET_MAX_DIST_PX
        )
        if samples:
            receiver = _find_receiver(
                samples[-1].center,
                dir_mean,
                players,
                self.kicker_id,
                pass_target_max,
                cfg.PASS_TARGET_MAX_ANGLE_DEG,
            )
        receiver_id = receiver[0] if receiver else None
        receiver_dist = receiver[1] if receiver else None

        goal_align_ok = True
        if goal_angle is not None and dir_mean is not None:
            goal_align_ok = angle_diff_deg(goal_angle, dir_mean) <= cfg.SHOT_GOAL_ALIGN_DEG

        # Require a strong impulse early in the window to qualify as a shot.
        peak_frame = max(accel_by_frame, key=accel_by_frame.get) if accel_by_frame else None
        accel_spike_early = (
            peak_frame is not None
            and self.kick_frame >= 0
            and (peak_frame - self.kick_frame) <= cfg.SHOT_ACCEL_WINDOW_FRAMES
        )

        # Simple deterministic rules for pass vs shot; fall back to dribble.
        shot_speed_min = cfg.SHOT_MIN_SPEED_MPS if use_metric else cfg.SHOT_SPEED_MIN_PX_S
        shot_accel_min = (
            cfg.SHOT_ACCEL_MIN_PX_S2 * scale if use_metric else cfg.SHOT_ACCEL_MIN_PX_S2
        )
        pass_min_speed = cfg.PASS_MIN_SPEED_MPS if use_metric else cfg.PASS_MIN_SPEED_PX_S
        pass_speed_max = (
            cfg.PASS_SPEED_MAX_PX_S * scale if use_metric else cfg.PASS_SPEED_MAX_PX_S
        )
        pass_accel_max = (
            cfg.PASS_ACCEL_MAX_PX_S2 * scale if use_metric else cfg.PASS_ACCEL_MAX_PX_S2
        )
        shot_receiver_exclude = (
            cfg.SHOT_RECEIVER_EXCLUDE_DIST_PX * scale
            if use_metric
            else cfg.SHOT_RECEIVER_EXCLUDE_DIST_PX
        )

        shot_candidate = (
            peak_speed >= shot_speed_min
            and peak_accel >= shot_accel_min
            and goal_align_ok
            and accel_spike_early
            and (receiver_dist is None or receiver_dist >= shot_receiver_exclude)
        )
        pass_candidate = (
            pass_min_speed <= avg_speed <= pass_speed_max
            and peak_accel <= pass_accel_max
            and dir_std <= cfg.PASS_DIR_VAR_MAX_DEG
            and ground_fraction >= cfg.PASS_GROUND_MIN_FRAC
            and receiver_id is not None
        )

        event_type = "dribble"
        shot_power = None
        if shot_candidate:
            event_type = "shot"
            power_min = cfg.SHOT_POWER_ACCEL_MIN * scale if use_metric else cfg.SHOT_POWER_ACCEL_MIN
            power_max = cfg.SHOT_POWER_ACCEL_MAX * scale if use_metric else cfg.SHOT_POWER_ACCEL_MAX
            shot_power = _normalize_shot_power(peak_accel, power_min, power_max)
        elif pass_candidate:
            event_type = "pass"

        display_center = samples[-1].center if samples else self.kick_ball_center

        return KickAnalysisResult(
            event_type=event_type,
            kicker_id=self.kicker_id or "unknown",
            foot=self.kicking_foot,
            frame_idx=decision_frame,
            kick_frame=self.kick_frame,
            avg_speed=avg_speed,
            peak_accel=peak_accel,
            dir_std_deg=dir_std if np.isfinite(dir_std) else float("inf"),
            ground_fraction=ground_fraction,
            target_player_id=receiver_id,
            shot_power=shot_power,
            display_center=display_center,
        )

    def _reset_tracking(self):
        self.kick_frame = -1000
        self.analysis_ready_frame = -1000
        self.analysis_deadline = -1000
        self.kicker_id = None
        self.kicking_foot = None
        self.kick_ball_center = None
        self.kick_m_per_px = None
        self.samples.clear()


def run_touch_detection(
    video_path: str,
    options: Optional[TouchOptions] = None,
    max_frames: Optional[int] = None,
) -> Generator[FrameResult, None, None]:
    opts = options or TouchOptions()

    det_model, pose_model = load_models(
        opts.detector_weights, opts.pose_weights
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps != fps:
        fps = 30.0

    calibration = None
    if opts.calibration_path:
        calibration = load_calibration(
            opts.calibration_path,
            default_field_width_m=cfg.CALIB_RECT_WIDTH_M,
            default_field_height_m=cfg.CALIB_RECT_HEIGHT_M,
        )
    global_m_per_px = None
    if calibration is not None and calibration.image_points:
        pts = np.array(calibration.image_points, dtype=np.float32)
        width_px = 0.5 * (dist(pts[0], pts[1]) + dist(pts[3], pts[2]))
        height_px = 0.5 * (dist(pts[0], pts[3]) + dist(pts[1], pts[2]))
        scales = []
        if width_px > 0:
            scales.append(calibration.field_width_m / width_px)
        if height_px > 0:
            scales.append(calibration.field_height_m / height_px)
        if scales:
            global_m_per_px = float(sum(scales) / len(scales))
    use_homography = bool(opts.use_homography and calibration is not None and global_m_per_px is not None)
    ground_poly = None
    extended_ground_poly = None
    ground_grid_lines = []
    grid_intersection_points = []  # Store (pixel_pos, field_x, field_y) for distance markers
    h_inv = None
    
    if calibration is not None:
        if calibration.image_points and len(calibration.image_points) == 4:
            ground_poly = np.array(calibration.image_points, dtype=np.float32)
        
        # Compute inverse homography for projecting field coords to image
        try:
            h_inv = np.linalg.inv(calibration.homography)
        except np.linalg.LinAlgError:
            h_inv = None
        
        if ground_poly is None and h_inv is not None:
            field_pts = np.array(
                [
                    [0.0, 0.0],
                    [calibration.field_width_m, 0.0],
                    [calibration.field_width_m, calibration.field_height_m],
                    [0.0, calibration.field_height_m],
                ],
                dtype=np.float32,
            ).reshape(-1, 1, 2)
            ground_poly = cv2.perspectiveTransform(field_pts, h_inv).reshape(-1, 2)
        
        # Compute extended ground plane (covers more area beyond calibration rect)
        if opts.draw_extended_ground and h_inv is not None:
            mult = opts.extended_ground_multiplier
            fw = calibration.field_width_m
            fh = calibration.field_height_m
            # Center of calibration rect in field coords
            cx, cy = fw / 2, fh / 2
            # Extended width and height
            ext_w = fw * mult
            ext_h = fh * mult
            # Extended corners centered on the calibration rect center
            ext_field_pts = np.array(
                [
                    [cx - ext_w / 2, cy - ext_h / 2],
                    [cx + ext_w / 2, cy - ext_h / 2],
                    [cx + ext_w / 2, cy + ext_h / 2],
                    [cx - ext_w / 2, cy + ext_h / 2],
                ],
                dtype=np.float32,
            ).reshape(-1, 1, 2)
            extended_ground_poly = cv2.perspectiveTransform(ext_field_pts, h_inv).reshape(-1, 2)
        
        # Compute grid lines for the extended ground plane
        # Use subdivided polylines for accurate perspective rendering
        if opts.draw_ground_grid and h_inv is not None:
            mult = opts.extended_ground_multiplier
            spacing = opts.ground_grid_spacing_m
            fw = calibration.field_width_m
            fh = calibration.field_height_m
            cx, cy = fw / 2, fh / 2
            ext_w = fw * mult
            ext_h = fh * mult
            
            # Compute grid boundaries in field coords (extended area)
            x_min = cx - ext_w / 2
            x_max = cx + ext_w / 2
            y_min = cy - ext_h / 2
            y_max = cy + ext_h / 2
            
            # Number of subdivisions per line for perspective accuracy
            num_subdivisions = opts.grid_line_subdivisions
            
            # Collect grid intersection points within the calibration rectangle
            # These are used for distance markers
            x_positions = []
            x = 0.0
            while x <= fw + 1e-6:
                x_positions.append(x)
                x += spacing
            
            y_positions = []
            y = 0.0
            while y <= fh + 1e-6:
                y_positions.append(y)
                y += spacing
            
            # Generate intersection points for distance markers
            for x_field in x_positions:
                for y_field in y_positions:
                    pt_field = np.array([[[x_field, y_field]]], dtype=np.float32)
                    pt_img = cv2.perspectiveTransform(pt_field, h_inv).reshape(2)
                    grid_intersection_points.append((pt_img, x_field, y_field))
            
            # Generate all X positions for vertical lines across the extended area
            # Start from x_min and increment by spacing
            x_grid_positions = []
            # First, find the first grid line position >= x_min that aligns to spacing
            first_x = spacing * math.ceil(x_min / spacing)
            x = first_x
            while x <= x_max + 1e-6:
                x_grid_positions.append(x)
                x += spacing
            
            # Generate all Y positions for horizontal lines across the extended area
            y_grid_positions = []
            first_y = spacing * math.ceil(y_min / spacing)
            y = first_y
            while y <= y_max + 1e-6:
                y_grid_positions.append(y)
                y += spacing
            
            # Vertical grid lines (constant X) - subdivided for perspective accuracy
            for x in x_grid_positions:
                y_values = np.linspace(y_min, y_max, num_subdivisions)
                line_field_pts = np.array([[[x, yv] for yv in y_values]], dtype=np.float32)
                line_img = cv2.perspectiveTransform(line_field_pts, h_inv).reshape(-1, 2)
                ground_grid_lines.append(line_img)
            
            # Horizontal grid lines (constant Y) - subdivided for perspective accuracy
            for y in y_grid_positions:
                x_values = np.linspace(x_min, x_max, num_subdivisions)
                line_field_pts = np.array([[[xv, y] for xv in x_values]], dtype=np.float32)
                line_img = cv2.perspectiveTransform(line_field_pts, h_inv).reshape(-1, 2)
                ground_grid_lines.append(line_img)
            
            # Add calibration rectangle boundary lines (0,0 to fw,fh) with thicker style
            # These should align exactly with the marked calibration points
            calib_boundary_lines = [
                # Top edge: y=0
                np.array([[[x_val, 0.0] for x_val in np.linspace(0, fw, num_subdivisions)]], dtype=np.float32),
                # Bottom edge: y=fh
                np.array([[[x_val, fh] for x_val in np.linspace(0, fw, num_subdivisions)]], dtype=np.float32),
                # Left edge: x=0
                np.array([[[0.0, y_val] for y_val in np.linspace(0, fh, num_subdivisions)]], dtype=np.float32),
                # Right edge: x=fw
                np.array([[[fw, y_val] for y_val in np.linspace(0, fh, num_subdivisions)]], dtype=np.float32),
            ]
            for line_field_pts in calib_boundary_lines:
                line_img = cv2.perspectiveTransform(line_field_pts, h_inv).reshape(-1, 2)
                ground_grid_lines.append(line_img)

    ground_overlay_data = None
    if (
        ground_poly is not None
        or extended_ground_poly is not None
        or ground_grid_lines
        or grid_intersection_points
    ):
        ground_overlay_data = {
            "ground_poly": ground_poly.tolist() if ground_poly is not None else None,
            "extended_ground_poly": (
                extended_ground_poly.tolist() if extended_ground_poly is not None else None
            ),
            "ground_grid_lines": (
                [line.tolist() for line in ground_grid_lines] if ground_grid_lines else []
            ),
            "grid_intersection_points": [
                {
                    "pos": (float(pt[0]), float(pt[1])),
                    "x": float(x_field),
                    "y": float(y_field),
                }
                for pt, x_field, y_field in grid_intersection_points
            ],
        }

    frames_required = max(1, int(round(cfg.CONTACT_SEC * fps)))
    cooldown_frames = max(1, int(round(cfg.COOLDOWN_SEC * fps)))
    soft_frames_required = max(1, int(round(cfg.SOFT_TOUCH_SEC * fps)))

    frame_idx = 0
    ground_overlay_sent = False

    left_touches = 0
    right_touches = 0
    pass_count = 0
    shot_count_total = 0
    shot_power_sum = 0.0
    shot_power_samples = 0
    speed_sum = 0.0
    speed_count = 0
    speed_max = 0.0
    total_distance_m = 0.0
    distance_samples = 0
    peak_accel_mps2 = 0.0
    peak_decel_mps2 = 0.0
    accel_samples = 0
    decel_samples = 0
    total_jumps = 0
    highest_jump_m = 0.0
    highest_jump_px = 0.0

    ball_history = deque(maxlen=cfg.BALL_SMOOTHING)
    last_ball_frame = -1000
    ball_trail_len = int(opts.ball_trail_length) if opts.ball_trail_length is not None else 0
    if ball_trail_len < 2:
        ball_trail_len = 0
    ball_trail = deque(maxlen=ball_trail_len) if ball_trail_len else deque()
    last_trail_frame = -1000

    ball_motion = deque(maxlen=cfg.BALL_VEL_SMOOTHING)
    ball_event_history = deque(maxlen=cfg.BALL_EVENT_WINDOW)
    prev_ball_center_px = None
    prev_ball_center_calc = None
    prev_ball_frame = None
    prev_ball_speed = None
    prev_ball_dir = None

    shot_events: List[Dict] = []
    player_totals: Dict[str, Dict] = {}
    last_overlay: Optional[Dict] = None
    overlay_expires = -1

    motion_buffer = BallMotionBuffer(fps, cfg.BALL_MOTION_BUFFER)
    kick_analyzer = KickAnalyzer(
        fps, use_homography=use_homography, meters_per_px=global_m_per_px
    )
    goal_angle = None
    if cfg.GOAL_VECTOR_X != 0.0 or cfg.GOAL_VECTOR_Y != 0.0:
        goal_angle = math.atan2(cfg.GOAL_VECTOR_Y, cfg.GOAL_VECTOR_X)

    def m_per_px_at(point: Optional[Tuple[float, float]]) -> Optional[float]:
        if not use_homography or point is None:
            return None
        try:
            base = project_point(point, calibration.homography)
            dx = project_point((point[0] + 1.0, point[1]), calibration.homography)
            dy = project_point((point[0], point[1] + 1.0), calibration.homography)
            mx = math.hypot(dx[0] - base[0], dx[1] - base[1])
            my = math.hypot(dy[0] - base[0], dy[1] - base[1])
            local = (mx + my) / 2.0
            if local > 0:
                return local
        except Exception:
            return global_m_per_px
        return global_m_per_px

    def px_to_m(value_px: float, ref_point: Optional[Tuple[float, float]]) -> Optional[float]:
        scale = m_per_px_at(ref_point) if ref_point is not None else global_m_per_px
        if scale is None:
            return None
        return value_px * scale

    def project_if_ready(point: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if not use_homography or point is None:
            return None
        return project_point(point, calibration.homography)

    def log_event(result: KickAnalysisResult, state: Optional[TrackState]):
        nonlocal pass_count, shot_count_total, shot_power_sum, shot_power_samples
        nonlocal last_overlay, overlay_expires

        timestamp = result.frame_idx / fps if fps > 0 else None
        if use_homography:
            avg_speed_mps = result.avg_speed
            peak_accel_mps2 = result.peak_accel
        else:
            meters_per_px = state.last_m_per_px if state is not None else None
            avg_speed_mps = (
                result.avg_speed * meters_per_px if meters_per_px is not None else None
            )
            peak_accel_mps2 = (
                result.peak_accel * meters_per_px if meters_per_px is not None else None
            )
        event_no = len(shot_events) + 1
        record = {
            "shot": event_no,
            "type": result.event_type,
            "frame_idx": result.frame_idx,
            "kick_frame": result.kick_frame,
            "time_sec": timestamp,
            "foot": "Left" if result.foot == "L" else "Right" if result.foot == "R" else result.foot,
            "track_id": str(result.kicker_id),
            "avg_speed_px_s": result.avg_speed,
            "peak_accel_px_s2": result.peak_accel,
            "dir_std_deg": result.dir_std_deg,
            "ground_fraction": result.ground_fraction,
            "target_player_id": result.target_player_id,
            "shot_power": result.shot_power,
            "avg_speed_mps": avg_speed_mps,
            "peak_accel_mps2": peak_accel_mps2,
        }
        shot_events.append(record)

        stats = player_totals.setdefault(
            str(result.kicker_id),
            {"passes": 0, "shots": 0, "shot_power_total": 0.0, "shot_power_count": 0},
        )

        if result.event_type == "pass":
            pass_count += 1
            stats["passes"] += 1
            if state is not None:
                state.passes += 1
        elif result.event_type == "shot":
            shot_count_total += 1
            stats["shots"] += 1
            if result.shot_power is not None:
                shot_power_sum += result.shot_power
                shot_power_samples += 1
                stats["shot_power_total"] += result.shot_power
                stats["shot_power_count"] += 1
            if state is not None:
                state.shots += 1
                if result.shot_power is not None:
                    state.shot_power_total += result.shot_power
                    state.shot_power_count += 1

        last_overlay = {
            "type": result.event_type,
            "power": result.shot_power,
            "pos": result.display_center,
            "frame": result.frame_idx,
        }
        overlay_expires = result.frame_idx + cfg.EVENT_OVERLAY_FRAMES

    def start_kick_analysis(
        track_id,
        foot_code: Optional[str],
        ball_pos_px: Optional[Tuple[float, float]],
        ball_pos_field: Optional[Tuple[float, float]],
    ):
        ref_ball = ball_pos_field if use_homography else ball_pos_px
        if ref_ball is None and ball_history:
            fallback = (ball_history[-1][0], ball_history[-1][1])
            ref_ball = project_if_ready(fallback) if use_homography else fallback
        kick_scale = m_per_px_at(ball_pos_px) if use_homography else None
        kick_analyzer.start_kick(
            frame_idx, track_id, foot_code, ref_ball, meters_per_px=kick_scale
        )

    def resolve_state_for(track_id):
        state = track_states.get(track_id)
        if state is None:
            try:
                state = track_states.get(int(track_id))
            except (ValueError, TypeError):
                state = None
        return state

    track_states = {}
    track_last_seen = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            if max_frames is not None and frame_idx > max_frames:
                break

            kick_analyzer.tick(frame_idx)

            annotated = frame.copy()
            h, w = frame.shape[:2]

            results = det_model.track(
                frame,
                persist=True,
                conf=cfg.DET_CONF,
                tracker="bytetrack.yaml",
                verbose=False,
            )

            ball_candidates = []
            people_dets = []

            if results and results[0].boxes is not None:
                for i, box in enumerate(results[0].boxes):
                    cls = int(box.cls[0])
                    conf = float(box.conf[0]) if box.conf is not None else 0.0
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if cls == cfg.BALL_CLASS_ID and conf >= cfg.DET_CONF:
                        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                        radius = 0.5 * max(cfg.BALL_RADIUS_MIN_PX, min(x2 - x1, y2 - y1))
                        ball_candidates.append((center, radius, conf))
                    elif cls == cfg.PERSON_CLASS_ID and conf >= cfg.DET_CONF:
                        track_id = None
                        if hasattr(box, "id") and box.id is not None:
                            track_id = int(box.id[0])
                        people_dets.append((track_id, x1, y1, x2, y2))

            ball_center = None
            ball_radius = None
            ball_center_raw = None
            ball_radius_raw = None
            ball_center_field = None
            ball_center_raw_field = None
            ball_radius_m = None
            ball_detected = False
            ball_speed = None
            ball_speed_draw = None
            ball_dir = None
            ball_vel = None
            ball_vel_draw = None
            motion_sample = None

            if ball_candidates:
                last_center = None
                if ball_history:
                    last_center = (ball_history[-1][0], ball_history[-1][1])
                choice = pick_ball(ball_candidates, last_center)
                ball_center_raw, ball_radius_raw, _conf = choice
                ball_history.append((ball_center_raw[0], ball_center_raw[1], ball_radius_raw))
                last_ball_frame = frame_idx
                ball_detected = True

            if ball_history and frame_idx - last_ball_frame <= cfg.BALL_HOLD_FRAMES:
                ball_center, ball_radius = smooth_ball(ball_history)
            else:
                ball_history.clear()

            if ball_trail_len:
                if ball_center is not None:
                    ball_trail.append((ball_center[0], ball_center[1]))
                    last_trail_frame = frame_idx
                elif frame_idx - last_trail_frame > opts.ball_trail_max_gap_frames:
                    ball_trail.clear()

            if use_homography and ball_center is not None:
                ball_center_field = project_if_ready(ball_center)
            if use_homography and ball_center_raw is not None:
                ball_center_raw_field = project_if_ready(ball_center_raw)
            if use_homography and ball_radius is not None and ball_center is not None:
                ball_radius_m = px_to_m(ball_radius, ball_center)

            if (
                motion_buffer.prev_frame is not None
                and frame_idx - motion_buffer.prev_frame > cfg.BALL_MOTION_MAX_GAP
            ):
                motion_buffer.reset()
            if ball_center is not None:
                motion_center = (
                    ball_center_field if use_homography and ball_center_field is not None else ball_center
                )
                motion_sample = motion_buffer.add(frame_idx, motion_center)

            ball_event = False
            if ball_detected:
                if prev_ball_frame is not None and frame_idx - prev_ball_frame > cfg.BALL_MOTION_MAX_GAP:
                    prev_ball_center_px = None
                    prev_ball_center_calc = None
                    prev_ball_frame = None
                    prev_ball_speed = None
                    prev_ball_dir = None
                    ball_motion.clear()

                ball_motion.append((ball_center_raw[0], ball_center_raw[1]))
                vel_center_px = smooth_point(ball_motion)
                if vel_center_px is not None:
                    vel_center_calc = (
                        project_if_ready(vel_center_px) if use_homography else vel_center_px
                    )
                    if vel_center_calc is None:
                        vel_center_calc = vel_center_px

                    if prev_ball_center_px is not None and prev_ball_frame is not None:
                        dt = frame_idx - prev_ball_frame
                        if dt > 0:
                            vx_px = (vel_center_px[0] - prev_ball_center_px[0]) / dt
                            vy_px = (vel_center_px[1] - prev_ball_center_px[1]) / dt
                            ball_speed_draw = math.hypot(vx_px, vy_px) * fps
                            ball_vel_draw = (vx_px, vy_px)

                            if prev_ball_center_calc is not None:
                                vx = (vel_center_calc[0] - prev_ball_center_calc[0]) / dt
                                vy = (vel_center_calc[1] - prev_ball_center_calc[1]) / dt
                                speed = math.hypot(vx, vy) * fps
                                ball_dir = math.atan2(vy, vx)
                                ball_speed = speed
                                ball_vel = (vx, vy)

                                radius_for_speed = (
                                    ball_radius_raw
                                    if ball_radius_raw is not None
                                    else (ball_radius if ball_radius is not None else 0.0)
                                )
                                if use_homography:
                                    local_scale = m_per_px_at(vel_center_px)
                                    speed_floor = (
                                        cfg.SPEED_MIN_PX_S * local_scale
                                        if local_scale is not None
                                        else None
                                    )
                                    radius_m = (
                                        radius_for_speed * local_scale
                                        if local_scale is not None
                                        else None
                                    )
                                    speed_min = (
                                        max(speed_floor, radius_m * fps * cfg.SPEED_MIN_RADIUS_RATIO)
                                        if speed_floor is not None and radius_m is not None
                                        else (speed_floor or radius_m or cfg.SPEED_MIN_PX_S)
                                    )
                                else:
                                    speed_min = max(
                                        cfg.SPEED_MIN_PX_S,
                                        radius_for_speed * fps * cfg.SPEED_MIN_RADIUS_RATIO,
                                    )

                                if prev_ball_speed is not None and prev_ball_dir is not None:
                                    dir_change = angle_diff_deg(ball_dir, prev_ball_dir)
                                    speed_gain = (speed - prev_ball_speed) / max(
                                        prev_ball_speed, speed_min
                                    )
                                    speed_drop = (prev_ball_speed - speed) / max(
                                        prev_ball_speed, speed_min
                                    )
                                    if (
                                        dir_change >= cfg.DIR_CHANGE_DEG
                                        or speed_gain >= cfg.SPEED_GAIN_RATIO
                                        or speed_drop >= cfg.SPEED_DROP_RATIO
                                    ):
                                        ball_event = True
                                    if prev_ball_speed < speed_min * 0.6 and speed >= speed_min:
                                        ball_event = True
                                    if prev_ball_speed >= speed_min and speed < speed_min * 0.6:
                                        ball_event = True

                                prev_ball_speed = speed
                                prev_ball_dir = ball_dir
                                ball_event_history.append((frame_idx, ball_event))

                    prev_ball_center_px = vel_center_px
                    prev_ball_center_calc = vel_center_calc
                    prev_ball_frame = frame_idx

            people = []
            for i, (track_id, x1, y1, x2, y2) in enumerate(people_dets):
                box = clamp_box(x1, y1, x2, y2, w, h)
                if box is None:
                    continue
                x1, y1, x2, y2 = box

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                pose = pose_model(crop, conf=cfg.POSE_CONF, verbose=False)
                if not pose or pose[0].keypoints is None:
                    continue

                kpts = pose[0].keypoints.xy.cpu().numpy()
                if kpts.size == 0:
                    continue

                confs = None
                if pose[0].keypoints.conf is not None:
                    confs = pose[0].keypoints.conf.cpu().numpy()

                if kpts.shape[0] > 1 and confs is not None:
                    idx = int(np.argmax(np.mean(confs, axis=1)))
                else:
                    idx = 0

                p = kpts[idx]
                p_conf = confs[idx] if confs is not None else None
                kpts_abs = p + np.array([x1, y1])

                left_foot = None
                right_foot = None
                if p_conf is None or p_conf[15] >= cfg.KPT_CONF:
                    left_foot = (kpts_abs[15][0], kpts_abs[15][1])
                if p_conf is None or p_conf[16] >= cfg.KPT_CONF:
                    right_foot = (kpts_abs[16][0], kpts_abs[16][1])

                foot_min_y = y1 + (y2 - y1) * cfg.FOOT_Y_MIN_RATIO
                if left_foot is not None:
                    if not (x1 <= left_foot[0] <= x2 and y1 <= left_foot[1] <= y2):
                        left_foot = None
                    elif left_foot[1] < foot_min_y:
                        left_foot = None
                if right_foot is not None:
                    if not (x1 <= right_foot[0] <= x2 and y1 <= right_foot[1] <= y2):
                        right_foot = None
                    elif right_foot[1] < foot_min_y:
                        right_foot = None

                if track_id is None:
                    track_id = f"tmp_{frame_idx}_{i}"

                state = track_states.setdefault(track_id, TrackState())
                if left_foot is not None:
                    state.left_hist.append(left_foot)
                    state.left_last_frame = frame_idx
                if right_foot is not None:
                    state.right_hist.append(right_foot)
                    state.right_last_frame = frame_idx

                if frame_idx - state.left_last_frame > cfg.FOOT_HOLD_FRAMES:
                    state.left_hist.clear()
                if frame_idx - state.right_last_frame > cfg.FOOT_HOLD_FRAMES:
                    state.right_hist.clear()

                left_foot = smooth_point(state.left_hist)
                right_foot = smooth_point(state.right_hist)
                left_field = project_if_ready(left_foot)
                right_field = project_if_ready(right_foot)

                speed_anchor = None
                ankle_for_scale = None
                if left_foot is not None and right_foot is not None:
                    speed_anchor = (
                        (left_foot[0] + right_foot[0]) / 2.0,
                        max(left_foot[1], right_foot[1]),
                    )
                    ankle_for_scale = left_foot if left_foot[1] >= right_foot[1] else right_foot
                elif left_foot is not None:
                    speed_anchor = left_foot
                    ankle_for_scale = left_foot
                elif right_foot is not None:
                    speed_anchor = right_foot
                    ankle_for_scale = right_foot

                hip_pt = None
                hip_ankle_px = None
                meters_per_pixel = None
                if p_conf is None or (p_conf[11] >= cfg.KPT_CONF and p_conf[12] >= cfg.KPT_CONF):
                    hip_pt = (
                        (kpts_abs[11][0] + kpts_abs[12][0]) / 2.0,
                        (kpts_abs[11][1] + kpts_abs[12][1]) / 2.0,
                    )
                if hip_pt is not None and ankle_for_scale is not None:
                    hip_ankle_px = abs(hip_pt[1] - ankle_for_scale[1])
                    if hip_ankle_px >= cfg.HIP_ANKLE_MIN_PX:
                        meters_per_pixel = (
                            cfg.PLAYER_REF_HEIGHT_M * cfg.HIP_TO_ANKLE_RATIO
                        ) / hip_ankle_px
                        state.meters_per_px_history.append(meters_per_pixel)
                        if state.meters_per_px_history:
                            state.last_m_per_px = float(np.median(state.meters_per_px_history))

                speed_kmh = None
                speed_mps_display = None
                if speed_anchor is not None:
                    speed_mps = None
                    speed_anchor_field = None
                    if use_homography and fps > 0:
                        speed_anchor_field = project_if_ready(speed_anchor)

                    # Gap detection: reset tracking if player lost for too long
                    gap_frames = frame_idx - state.last_speed_frame if state.last_speed_frame > 0 else 0
                    if gap_frames > opts.max_speed_gap_frames:
                        state.prev_speed_field = None
                        state.prev_speed_point = None
                        state.last_speed_mps = None
                        state.speed_history.clear()

                    if speed_anchor_field is not None and fps > 0:
                        if state.prev_speed_field is not None and state.last_speed_frame > 0:
                            dt_frames = frame_idx - state.last_speed_frame
                            if dt_frames > 0:
                                dist_m = dist(speed_anchor_field, state.prev_speed_field)
                                dt_seconds = dt_frames / fps
                                speed_mps = dist_m / dt_seconds
                                
                                # Outlier rejection: skip if speed is impossibly high
                                if speed_mps > opts.max_human_speed_mps:
                                    speed_mps = None
                                elif dist_m >= opts.min_movement_m:
                                    # Only count distance if above threshold (prevents jitter)
                                    total_distance_m += dist_m
                                    state.total_distance_m += dist_m
                                    distance_samples += 1
                                
                                if speed_mps is not None:
                                    if state.last_speed_mps is not None:
                                        accel_mps2 = (speed_mps - state.last_speed_mps) / dt_seconds
                                        if accel_mps2 >= 0:
                                            state.peak_accel_mps2 = max(
                                                state.peak_accel_mps2, accel_mps2
                                            )
                                            peak_accel_mps2 = max(peak_accel_mps2, accel_mps2)
                                            accel_samples += 1
                                        else:
                                            decel_mps2 = -accel_mps2
                                            state.peak_decel_mps2 = max(
                                                state.peak_decel_mps2, decel_mps2
                                            )
                                            peak_decel_mps2 = max(peak_decel_mps2, decel_mps2)
                                            decel_samples += 1
                                    state.last_speed_mps = speed_mps
                                    
                                    # Smoothing: EMA or median
                                    if opts.use_ema_smoothing:
                                        if state.speed_kmh is not None:
                                            speed_kmh = opts.ema_alpha * (speed_mps * 3.6) + (1 - opts.ema_alpha) * state.speed_kmh
                                        else:
                                            speed_kmh = speed_mps * 3.6
                                    else:
                                        state.speed_history.append(speed_mps * 3.6)
                                        if state.speed_history:
                                            speed_kmh = float(np.median(state.speed_history))
                                    state.speed_kmh = speed_kmh
                                    speed_mps_display = speed_mps
                        if speed_mps is None:
                            state.last_speed_mps = None
                        state.prev_speed_field = speed_anchor_field
                        state.prev_speed_point = speed_anchor
                        state.last_speed_frame = frame_idx
                    elif meters_per_pixel is not None and fps > 0:
                        if state.prev_speed_point is not None and state.last_speed_frame > 0:
                            dt_frames = frame_idx - state.last_speed_frame
                            if dt_frames > 0:
                                dist_px = dist(speed_anchor, state.prev_speed_point)
                                dist_m = dist_px * meters_per_pixel
                                dt_seconds = dt_frames / fps
                                speed_mps = dist_m / dt_seconds
                                
                                # Outlier rejection
                                if speed_mps > opts.max_human_speed_mps:
                                    speed_mps = None
                                elif dist_m >= opts.min_movement_m:
                                    total_distance_m += dist_m
                                    state.total_distance_m += dist_m
                                    distance_samples += 1
                                
                                if speed_mps is not None:
                                    if state.last_speed_mps is not None:
                                        accel_mps2 = (speed_mps - state.last_speed_mps) / dt_seconds
                                        if accel_mps2 >= 0:
                                            state.peak_accel_mps2 = max(
                                                state.peak_accel_mps2, accel_mps2
                                            )
                                            peak_accel_mps2 = max(peak_accel_mps2, accel_mps2)
                                            accel_samples += 1
                                        else:
                                            decel_mps2 = -accel_mps2
                                            state.peak_decel_mps2 = max(
                                                state.peak_decel_mps2, decel_mps2
                                            )
                                            peak_decel_mps2 = max(peak_decel_mps2, decel_mps2)
                                            decel_samples += 1
                                    state.last_speed_mps = speed_mps
                                    
                                    # Smoothing: EMA or median
                                    if opts.use_ema_smoothing:
                                        if state.speed_kmh is not None:
                                            speed_kmh = opts.ema_alpha * (speed_mps * 3.6) + (1 - opts.ema_alpha) * state.speed_kmh
                                        else:
                                            speed_kmh = speed_mps * 3.6
                                    else:
                                        state.speed_history.append(speed_mps * 3.6)
                                        if state.speed_history:
                                            speed_kmh = float(np.median(state.speed_history))
                                    state.speed_kmh = speed_kmh
                                    speed_mps_display = speed_mps
                        if speed_mps is None:
                            state.last_speed_mps = None
                        state.prev_speed_point = speed_anchor
                        state.prev_speed_field = None
                        state.last_speed_frame = frame_idx
                    else:
                        state.prev_speed_point = speed_anchor
                        state.prev_speed_field = None
                        state.last_speed_frame = frame_idx
                        state.last_speed_mps = None
                else:
                    state.prev_speed_point = None
                    state.prev_speed_field = None
                    state.speed_history.clear()
                    state.speed_kmh = None
                    state.last_speed_mps = None

                if speed_kmh is None:
                    speed_kmh = state.speed_kmh
                if speed_kmh is not None:
                    speed_sum += speed_kmh
                    speed_count += 1
                    speed_max = max(speed_max, speed_kmh)

                ankle_y = None
                if left_foot is not None and right_foot is not None:
                    ankle_y = max(left_foot[1], right_foot[1])
                elif left_foot is not None:
                    ankle_y = left_foot[1]
                elif right_foot is not None:
                    ankle_y = right_foot[1]

                ankle_delta_px = 0.0
                if ankle_y is not None:
                    if not state.ground_ankle_history:
                        state.ground_ankle_history.append(ankle_y)

                    ground_y = (
                        float(np.median(state.ground_ankle_history))
                        if state.ground_ankle_history
                        else None
                    )
                    noise_px = 0.0
                    if len(state.ground_ankle_history) >= 3:
                        arr = np.array(state.ground_ankle_history, dtype=np.float32)
                        med = float(np.median(arr))
                        noise_px = float(np.median(np.abs(arr - med)))
                    if ground_y is not None:
                        ankle_delta_px = max(0.0, ground_y - ankle_y)

                    base_threshold = max(
                        cfg.JUMP_MIN_DELTA_PX,
                        (hip_ankle_px if hip_ankle_px is not None else 0.0)
                        * cfg.JUMP_DELTA_RATIO,
                        noise_px * cfg.JUMP_NOISE_SCALE + cfg.JUMP_NOISE_MARGIN_PX,
                    )
                    end_threshold = max(
                        base_threshold * cfg.JUMP_END_RATIO, cfg.JUMP_MIN_DELTA_PX * 0.4
                    )

                    up_vel = 0.0
                    if state.prev_ankle_y is not None:
                        up_vel = state.prev_ankle_y - ankle_y
                    if up_vel >= cfg.JUMP_UP_PX_PER_FRAME:
                        state.ankle_up_streak += 1
                    else:
                        state.ankle_up_streak = 0

                    trigger_delta = ankle_delta_px >= base_threshold
                    trigger_up = (
                        up_vel >= cfg.JUMP_UP_PX_PER_FRAME
                        or state.ankle_up_streak >= cfg.JUMP_UP_STREAK
                    )

                    if not state.jump_active:
                        if trigger_delta:
                            state.jump_air_streak += 1
                        else:
                            state.jump_air_streak = 0

                        if (
                            frame_idx - state.jump_cooldown_frame >= cfg.JUMP_COOLDOWN_FRAMES
                            and trigger_delta
                            and (
                                state.jump_air_streak >= cfg.JUMP_MIN_AIR_FRAMES
                                or trigger_up
                            )
                        ):
                            state.jump_active = True
                            state.jump_start_frame = frame_idx
                            state.jump_peak_delta_px = ankle_delta_px
                            state.jump_air_streak = 0
                    else:
                        state.jump_peak_delta_px = max(state.jump_peak_delta_px, ankle_delta_px)
                        if (
                            ankle_delta_px <= end_threshold
                            and frame_idx - state.jump_start_frame >= cfg.JUMP_MIN_AIR_FRAMES
                        ):
                            jump_height_px = state.jump_peak_delta_px
                            if state.last_m_per_px is not None:
                                jump_height_m = jump_height_px * state.last_m_per_px
                                state.max_jump_height_m = max(state.max_jump_height_m, jump_height_m)
                                highest_jump_m = max(highest_jump_m, jump_height_m)
                            state.max_jump_height_px = max(state.max_jump_height_px, jump_height_px)
                            highest_jump_px = max(highest_jump_px, jump_height_px)
                            state.jump_count += 1
                            total_jumps += 1
                            state.jump_active = False
                            state.jump_cooldown_frame = frame_idx
                            state.jump_peak_delta_px = 0.0

                    baseline_threshold = max(
                        cfg.JUMP_MIN_DELTA_PX * 0.4,
                        noise_px * cfg.JUMP_NOISE_SCALE * 0.5,
                        (hip_ankle_px if hip_ankle_px is not None else 0.0)
                        * cfg.JUMP_DELTA_RATIO
                        * 0.4,
                    )
                    if not state.jump_active and ankle_delta_px <= baseline_threshold:
                        state.ground_ankle_history.append(ankle_y)

                    state.prev_ankle_y = ankle_y
                else:
                    state.prev_ankle_y = None
                    state.ankle_up_streak = 0
                    state.jump_air_streak = 0

                foot_radius = max(2, (y2 - y1) * cfg.FOOT_RADIUS_RATIO)
                foot_radius_m = None
                bbox_field = None
                if use_homography:
                    foot_ref = left_foot or right_foot or speed_anchor
                    if foot_ref is not None:
                        foot_radius_m = px_to_m(foot_radius, foot_ref)
                    if foot_radius_m is None and global_m_per_px is not None:
                        foot_radius_m = foot_radius * global_m_per_px
                    bbox_field = project_if_ready(((x1 + x2) / 2.0, max(y1, y2)))

                annotated[y1:y2, x1:x2] = pose[0].plot()
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"ID {track_id}",
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                if left_foot is not None:
                    cv2.circle(annotated, (int(left_foot[0]), int(left_foot[1])), 5, (255, 0, 0), -1)
                if right_foot is not None:
                    cv2.circle(annotated, (int(right_foot[0]), int(right_foot[1])), 5, (0, 0, 255), -1)

                if opts.show_player_speed and speed_kmh is not None:
                    speed_y = min(h - 8, y2 + 24)
                    cv2.putText(
                        annotated,
                        f"{speed_kmh:.1f} km/h",
                        (x1, speed_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                    )

                people.append(
                    {
                        "id": track_id,
                        "bbox": (x1, y1, x2, y2),
                        "bbox_field": bbox_field,
                        "left": left_foot,
                        "right": right_foot,
                        "left_field": left_field,
                        "right_field": right_field,
                        "foot_radius": foot_radius,
                        "foot_radius_m": foot_radius_m,
                        "speed_kmh": speed_kmh,
                    }
                )

                track_last_seen[track_id] = frame_idx

            ball_for_contact = (
                ball_center
                and ball_radius
                and frame_idx - last_ball_frame <= cfg.BALL_CONTACT_MAX_AGE
            )
            use_metric_contacts = (
                use_homography
                and ball_center_field is not None
                and ball_radius_m is not None
            )

            ball_event_recent = any(
                ev for f, ev in ball_event_history if frame_idx - f <= cfg.BALL_EVENT_WINDOW
            )

            contact_candidates = []
            event_candidates = []
            if ball_for_contact:
                for person in people:
                    if use_metric_contacts:
                        left = person.get("left_field")
                        right = person.get("right_field")
                        foot_radius = person.get("foot_radius_m")
                        ball_center_ref = ball_center_field
                        ball_radius_ref = ball_radius_m
                    else:
                        left = person["left"]
                        right = person["right"]
                        foot_radius = person["foot_radius"]
                        ball_center_ref = ball_center
                        ball_radius_ref = ball_radius
                    if left is None and right is None:
                        continue

                    d_left = dist(ball_center_ref, left) if left is not None else float("inf")
                    d_right = dist(ball_center_ref, right) if right is not None else float("inf")
                    if ball_radius_ref is None or foot_radius is None:
                        continue
                    threshold = ball_radius_ref + foot_radius
                    event_threshold = threshold * opts.event_touch_dist_ratio

                    candidate = None
                    distance = None
                    if d_left <= threshold and d_right > threshold:
                        candidate = "L"
                        distance = d_left
                    elif d_right <= threshold and d_left > threshold:
                        candidate = "R"
                        distance = d_right
                    elif d_left <= threshold and d_right <= threshold:
                        candidate = "L" if d_left < d_right else "R"
                        distance = min(d_left, d_right)

                    if candidate is not None:
                        contact_candidates.append((distance, person["id"], candidate))

                    candidate = None
                    distance = None
                    if d_left <= event_threshold and d_right > event_threshold:
                        candidate = "L"
                        distance = d_left
                    elif d_right <= event_threshold and d_left > event_threshold:
                        candidate = "R"
                        distance = d_right
                    elif d_left <= event_threshold and d_right <= event_threshold:
                        candidate = "L" if d_left < d_right else "R"
                        distance = min(d_left, d_right)

                    if candidate is not None:
                        event_candidates.append((distance, person["id"], candidate))

            active_contacts = {}
            if contact_candidates:
                contact_candidates.sort(key=lambda c: c[0])
                _dist, chosen_id, chosen_foot = contact_candidates[0]
                active_contacts[chosen_id] = (chosen_foot, _dist)

            if opts.event_touch_enabled and ball_event and event_candidates:
                event_candidates.sort(key=lambda c: c[0])
                _dist, chosen_id, chosen_foot = event_candidates[0]
                state = track_states.get(chosen_id)
                if state is not None and frame_idx - state.last_touch_frame > cooldown_frames:
                    if chosen_foot == "L":
                        state.left_touches += 1
                        left_touches += 1
                    else:
                        state.right_touches += 1
                        right_touches += 1
                    start_kick_analysis(
                        chosen_id,
                        chosen_foot,
                        ball_center if ball_center is not None else ball_center_raw,
                        ball_center_field
                        if ball_center_field is not None
                        else ball_center_raw_field,
                    )

                    state.last_touch_frame = frame_idx
                    state.last_contact_frame = frame_idx
                    state.contact_streak = 0
                    state.touch_locked = True
                    state.active_foot = None
                    state.pending_contact_frame = -1000
                    state.pending_foot = None
                    state.pending_ball_dist = 0.0

            for person in people:
                track_id = person["id"]
                state = track_states[track_id]
                candidate_info = active_contacts.get(track_id)
                candidate = None
                candidate_dist = None
                if candidate_info is not None:
                    candidate, candidate_dist = candidate_info

                if candidate is not None:
                    if state.active_foot is None:
                        state.active_foot = candidate
                    if state.active_foot != candidate:
                        candidate = state.active_foot

                    state.last_contact_frame = frame_idx
                    if not state.touch_locked:
                        state.contact_streak += 1
                else:
                    if frame_idx - state.last_contact_frame > cfg.CONTACT_GAP_ALLOW:
                        state.contact_streak = 0
                        state.touch_locked = False
                        if frame_idx - state.last_contact_frame > cfg.ACTIVE_FOOT_HOLD_FRAMES:
                            state.active_foot = None

                if (
                    candidate is not None
                    and state.contact_streak >= frames_required
                    and state.pending_contact_frame < 0
                ):
                    state.pending_contact_frame = frame_idx
                    state.pending_foot = state.active_foot
                    state.pending_ball_dist = candidate_dist if candidate_dist is not None else 0.0

                if state.pending_contact_frame >= 0:
                    if frame_idx - state.pending_contact_frame > cfg.IMPULSE_WINDOW:
                        state.pending_contact_frame = -1000
                        state.pending_foot = None
                        state.pending_ball_dist = 0.0
                    elif (
                        (ball_center_field and ball_radius_m)
                        if use_metric_contacts
                        else (ball_center and ball_radius)
                        and state.pending_foot in ("L", "R")
                        and frame_idx - state.last_touch_frame > cooldown_frames
                        and not state.touch_locked
                    ):
                        if use_metric_contacts:
                            foot_pt = (
                                person.get("left_field")
                                if state.pending_foot == "L"
                                else person.get("right_field")
                            )
                            ball_center_ref = ball_center_field
                            ball_radius_ref = ball_radius_m
                        else:
                            foot_pt = (
                                person["left"] if state.pending_foot == "L" else person["right"]
                            )
                            ball_center_ref = ball_center
                            ball_radius_ref = ball_radius
                        if foot_pt is not None:
                            current_dist = dist(ball_center_ref, foot_pt)
                            dist_gain = current_dist - state.pending_ball_dist
                            if use_metric_contacts:
                                local_scale = m_per_px_at(ball_center)
                                sep_floor = (
                                    cfg.SEPARATION_GAIN_PX * local_scale
                                    if local_scale is not None
                                    else None
                                )
                                sep_ratio = (
                                    ball_radius_ref * cfg.SEPARATION_GAIN_RATIO
                                    if ball_radius_ref is not None
                                    else None
                                )
                                separation_need = (
                                    max(sep_floor, sep_ratio)
                                    if sep_floor is not None and sep_ratio is not None
                                    else (sep_floor or sep_ratio or cfg.SEPARATION_GAIN_PX)
                                )
                            else:
                                separation_need = max(
                                    cfg.SEPARATION_GAIN_PX,
                                    ball_radius * cfg.SEPARATION_GAIN_RATIO,
                                )
                            separation_ok = dist_gain >= separation_need
                            if use_metric_contacts:
                                local_scale = m_per_px_at(ball_center)
                                speed_floor = (
                                    cfg.SPEED_MIN_PX_S * local_scale
                                    if local_scale is not None
                                    else None
                                )
                                speed_radius = (
                                    ball_radius_ref * fps * cfg.SPEED_MIN_RADIUS_RATIO
                                    if ball_radius_ref is not None
                                    else None
                                )
                                speed_min = (
                                    max(speed_floor, speed_radius)
                                    if speed_floor is not None and speed_radius is not None
                                    else (speed_floor or speed_radius or cfg.SPEED_MIN_PX_S)
                                )
                            else:
                                speed_min = max(
                                    cfg.SPEED_MIN_PX_S,
                                    ball_radius * fps * cfg.SPEED_MIN_RADIUS_RATIO,
                                )
                            rel = (
                                ball_center_ref[0] - foot_pt[0],
                                ball_center_ref[1] - foot_pt[1],
                            )
                            away_ok = False
                            if ball_vel is not None and ball_speed is not None and ball_speed >= speed_min:
                                away_ok = (ball_vel[0] * rel[0] + ball_vel[1] * rel[1]) > 0
                            impulse_signal = ball_event_recent
                            soft_touch_ok = False
                            if cfg.ALLOW_SOFT_TOUCH and state.contact_streak >= soft_frames_required:
                                close_enough = (
                                    current_dist
                                    <= (ball_radius_ref * cfg.SOFT_TOUCH_DIST_RATIO)
                                    if ball_radius_ref is not None
                                    else False
                                )
                                if close_enough:
                                    if ball_speed is None:
                                        soft_touch_ok = True
                                    else:
                                        soft_touch_ok = ball_speed <= speed_min * cfg.SOFT_TOUCH_SPEED_RATIO

                            signal_score = (
                                int(impulse_signal)
                                + int(separation_ok)
                                + int(away_ok)
                                + int(soft_touch_ok)
                            )
                            count_ok = signal_score >= cfg.REQUIRED_SIGNALS
                            if cfg.REQUIRE_BALL_IMPULSE and not (impulse_signal or soft_touch_ok):
                                count_ok = False
                            if cfg.REQUIRE_SEPARATION_GAIN and not (separation_ok or soft_touch_ok):
                                count_ok = False
                            if cfg.REQUIRE_AWAY_MOTION and not (away_ok or soft_touch_ok):
                                count_ok = False

                            if count_ok:
                                if state.pending_foot == "L":
                                    state.left_touches += 1
                                    left_touches += 1
                                else:
                                    state.right_touches += 1
                                    right_touches += 1
                                start_kick_analysis(
                                    track_id,
                                    state.pending_foot,
                                    ball_center if ball_center is not None else ball_center_raw,
                                    ball_center_field
                                    if ball_center_field is not None
                                    else ball_center_raw_field,
                                )

                                state.last_touch_frame = frame_idx
                                state.contact_streak = 0
                                state.touch_locked = True
                                state.active_foot = None
                                state.pending_contact_frame = -1000
                                state.pending_foot = None
                                state.pending_ball_dist = 0.0

            for track_id in list(track_last_seen.keys()):
                if frame_idx - track_last_seen[track_id] > cfg.TRACK_TTL:
                    track_last_seen.pop(track_id, None)
                    track_states.pop(track_id, None)

            analysis_result = None
            if motion_sample is not None:
                analysis_result = kick_analyzer.process_sample(
                    motion_sample, people, goal_angle
                )
            if analysis_result is None:
                analysis_result = kick_analyzer.finalize_if_due(
                    frame_idx, people, goal_angle
                )
            if analysis_result is not None:
                state_for_event = resolve_state_for(analysis_result.kicker_id)
                log_event(analysis_result, state_for_event)

            if (
                opts.draw_ball_trail
                and ball_trail_len
                and len(ball_trail) > 1
                and frame_idx - last_trail_frame <= opts.ball_trail_max_gap_frames
            ):
                _draw_ball_trail(annotated, ball_trail)

            if ball_center and ball_radius:
                cv2.circle(
                    annotated,
                    (int(ball_center[0]), int(ball_center[1])),
                    int(ball_radius),
                    (0, 165, 255),
                    2,
                )
                if opts.draw_ball_vector and ball_vel_draw is not None:
                    start_pt = (int(ball_center[0]), int(ball_center[1]))
                    end_pt = (
                        int(ball_center[0] + ball_vel_draw[0] * opts.ball_vector_scale),
                        int(ball_center[1] + ball_vel_draw[1] * opts.ball_vector_scale),
                    )
                    cv2.arrowedLine(
                        annotated,
                        start_pt,
                        end_pt,
                        (0, 255, 255),
                        2,
                        tipLength=0.3,
                    )
                    if opts.show_ball_speed:
                        display_speed = ball_speed if use_homography else ball_speed_draw
                        speed_unit = "m/s" if use_homography else "px/s"
                        if display_speed is not None:
                            cv2.putText(
                                annotated,
                                f"{display_speed:.1f}{speed_unit}",
                                (start_pt[0] + 6, start_pt[1] - 6),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 255),
                                2,
                            )
                    if opts.show_ball_components and ball_vel is not None:
                        vx = ball_vel[0] * fps
                        vy = ball_vel[1] * fps
                        cv2.putText(
                            annotated,
                            f"vx:{vx:+.1f} vy:{vy:+.1f} vz:+0.0",
                            (start_pt[0] + 6, start_pt[1] + 14),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            2,
                        )

            # Draw extended ground plane first (if enabled)
            if extended_ground_poly is not None:
                overlay = annotated.copy()
                ext_pts = np.round(extended_ground_poly).astype(int).reshape(-1, 1, 2)
                # Semi-transparent green fill for extended area
                cv2.fillPoly(overlay, [ext_pts], (50, 100, 50))
                cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0.0, annotated)
                # Dashed border for extended area (draw as thin line)
                cv2.polylines(annotated, [ext_pts], True, (100, 200, 100), 1)
            
            # Draw grid lines on the ground plane
            # Grid lines are now polylines with multiple points for perspective accuracy
            if ground_grid_lines:
                for line in ground_grid_lines:
                    # Convert to integer points for drawing
                    pts = np.round(line).astype(np.int32).reshape(-1, 1, 2)
                    # Draw as polyline to follow perspective accurately
                    cv2.polylines(annotated, [pts], isClosed=False, color=(80, 150, 80), thickness=1)
            
            # Draw calibration rectangle (on top of grid)
            if ground_poly is not None:
                overlay = annotated.copy()
                pts = np.round(ground_poly).astype(int).reshape(-1, 1, 2)
                cv2.fillPoly(overlay, [pts], (0, 120, 255))
                cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0.0, annotated)
                cv2.polylines(annotated, [pts], True, (0, 255, 255), 2)
                
                # Draw corner markers with field coordinate labels
                corner_labels = []
                if calibration is not None:
                    fw = calibration.field_width_m
                    fh = calibration.field_height_m
                    corner_labels = [
                        f"(0,0)",           # top-left
                        f"({fw:.0f},0)",    # top-right
                        f"({fw:.0f},{fh:.0f})",  # bottom-right
                        f"(0,{fh:.0f})",    # bottom-left
                    ]
                
                for i, pt in enumerate(pts):
                    pt_tuple = tuple(pt[0])
                    cv2.circle(annotated, pt_tuple, 6, (0, 255, 255), -1)
                    cv2.circle(annotated, pt_tuple, 8, (0, 0, 0), 2)
                    # Draw corner label
                    if i < len(corner_labels):
                        label_pos = (pt_tuple[0] + 10, pt_tuple[1] - 10)
                        cv2.putText(
                            annotated,
                            corner_labels[i],
                            label_pos,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            2,
                        )
            
            # Draw distance markers at grid intersections (within calibration rectangle)
            if opts.show_grid_distance_markers and grid_intersection_points:
                for pt_img, x_field, y_field in grid_intersection_points:
                    # Skip corner points (already drawn with labels)
                    if calibration is not None:
                        fw = calibration.field_width_m
                        fh = calibration.field_height_m
                        is_corner = (
                            (abs(x_field) < 0.01 and abs(y_field) < 0.01) or
                            (abs(x_field - fw) < 0.01 and abs(y_field) < 0.01) or
                            (abs(x_field - fw) < 0.01 and abs(y_field - fh) < 0.01) or
                            (abs(x_field) < 0.01 and abs(y_field - fh) < 0.01)
                        )
                        if is_corner:
                            continue
                    
                    pt_int = (int(round(pt_img[0])), int(round(pt_img[1])))
                    # Draw small circle at intersection
                    cv2.circle(annotated, pt_int, 3, (255, 200, 100), -1)
                    # Draw distance label
                    label = f"({x_field:.0f},{y_field:.0f})"
                    cv2.putText(
                        annotated,
                        label,
                        (pt_int[0] + 5, pt_int[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (255, 200, 100),
                        1,
                    )
            avg_power_display = shot_power_sum / shot_power_samples if shot_power_samples else None

            if last_overlay and frame_idx <= overlay_expires:
                label = last_overlay.get("type", "event")
                label_txt = label.upper()
                power_val = last_overlay.get("power")
                if power_val is not None:
                    label_txt = f"{label_txt} {power_val:.0f}"
                overlay_pos = last_overlay.get("pos")
                if (
                    overlay_pos is None
                    or not isinstance(overlay_pos, tuple)
                    or not all(isinstance(v, (int, float)) for v in overlay_pos)
                ):
                    overlay_pos = (40.0, 80.0)
                overlay_pt = (
                    int(overlay_pos[0]),
                    max(24, int(overlay_pos[1]) - 10),
                )
                color_map = {
                    "SHOT": (0, 0, 255),
                    "PASS": (0, 255, 0),
                    "DRIBBLE": (255, 215, 0),
                }
                cv2.putText(
                    annotated,
                    label_txt,
                    overlay_pt,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color_map.get(label_txt.split()[0], (0, 255, 255)),
                    3,
                )

            cv2.putText(
                annotated,
                f"L: {left_touches}   R: {right_touches}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3,
            )
            stats_text = f"Passes: {pass_count}   Shots: {shot_count_total}"
            if avg_power_display is not None:
                stats_text += f"   Power: {avg_power_display:.0f}"
            cv2.putText(
                annotated,
                stats_text,
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )

            if opts.display_stride > 0 and frame_idx % opts.display_stride == 0:
                avg_speed = speed_sum / speed_count if speed_count else None
                max_speed = speed_max if speed_count else None
                player_view = {}
                for pid, stats in player_totals.items():
                    avg_power = None
                    if stats["shot_power_count"] > 0:
                        avg_power = stats["shot_power_total"] / stats["shot_power_count"]
                    player_view[pid] = {
                        "passes": stats["passes"],
                        "shots": stats["shots"],
                        "avg_shot_power": avg_power,
                    }
                player_meta = []
                for person in people:
                    bbox = person.get("bbox")
                    if bbox is None or len(bbox) != 4:
                        continue
                    x1, y1, x2, y2 = bbox
                    player_meta.append(
                        {
                            "id": str(person.get("id")),
                            "bbox": (int(x1), int(y1), int(x2), int(y2)),
                            "left": _to_point(person.get("left")),
                            "right": _to_point(person.get("right")),
                            "speed_kmh": person.get("speed_kmh"),
                        }
                    )
                ball_meta = None
                if ball_center is not None and ball_radius is not None:
                    ball_meta = {
                        "center": _to_point(ball_center),
                        "radius": float(ball_radius),
                        "vel_draw": _to_point(ball_vel_draw),
                        "speed_draw": float(ball_speed_draw)
                        if ball_speed_draw is not None
                        else None,
                        "speed_mps": float(ball_speed) if ball_speed is not None else None,
                    }
                event_overlay = None
                if last_overlay and frame_idx <= overlay_expires:
                    overlay_pos = last_overlay.get("pos")
                    if isinstance(overlay_pos, (list, tuple)) and len(overlay_pos) >= 2:
                        overlay_pos = (float(overlay_pos[0]), float(overlay_pos[1]))
                    else:
                        overlay_pos = None
                    event_overlay = {
                        "type": last_overlay.get("type"),
                        "power": last_overlay.get("power"),
                        "pos": overlay_pos,
                        "frame": last_overlay.get("frame"),
                    }
                frame_meta: Dict[str, Any] = {
                    "players": player_meta,
                    "ball": ball_meta,
                    "event_overlay": event_overlay,
                    "use_homography": use_homography,
                }
                if not ground_overlay_sent and ground_overlay_data is not None:
                    frame_meta["ground_overlay"] = ground_overlay_data
                    ground_overlay_sent = True
                yield FrameResult(
                    frame_idx=frame_idx,
                    annotated=annotated,
                    left_touches=left_touches,
                    right_touches=right_touches,
                    avg_speed_kmh=avg_speed,
                    max_speed_kmh=max_speed,
                    total_time_sec=(frame_idx / fps if fps > 0 else None),
                    total_distance_m=(
                        total_distance_m if distance_samples > 0 else None
                    ),
                    peak_accel_mps2=(
                        peak_accel_mps2 if accel_samples > 0 else None
                    ),
                    peak_decel_mps2=(
                        peak_decel_mps2 if decel_samples > 0 else None
                    ),
                    total_jumps=total_jumps,
                    highest_jump_m=highest_jump_m if highest_jump_m > 0 else None,
                    highest_jump_px=highest_jump_px if highest_jump_px > 0 else None,
                    shot_count=shot_count_total,
                    pass_count=pass_count,
                    avg_shot_power=avg_power_display,
                    shot_events=shot_events.copy(),
                    player_stats=player_view,
                    frame_meta=frame_meta,
                )
    finally:
        cap.release()

    return {
        "left_touches": left_touches,
        "right_touches": right_touches,
        "total_jumps": total_jumps,
        "highest_jump_m": highest_jump_m if highest_jump_m > 0 else None,
        "highest_jump_px": highest_jump_px if highest_jump_px > 0 else None,
        "shot_count": shot_count_total,
        "pass_count": pass_count,
        "avg_shot_power": shot_power_sum / shot_power_samples if shot_power_samples else None,
        "shot_events": shot_events,
        "total_time_sec": (frame_idx / fps if fps > 0 else None),
        "total_distance_m": total_distance_m if distance_samples > 0 else None,
        "peak_accel_mps2": peak_accel_mps2 if accel_samples > 0 else None,
        "peak_decel_mps2": peak_decel_mps2 if decel_samples > 0 else None,
        "player_stats": {
            pid: {
                "passes": stats["passes"],
                "shots": stats["shots"],
                "avg_shot_power": (
                    stats["shot_power_total"] / stats["shot_power_count"]
                    if stats["shot_power_count"] > 0
                    else None
                ),
            }
            for pid, stats in player_totals.items()
        },
    }
