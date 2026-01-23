"""
Central configuration values shared across touch-detection entrypoints.

Importing these constants keeps Streamlit and CLI runners aligned while letting
us tweak defaults in one place.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Model weights
DETECTOR_WEIGHTS = str(MODELS_DIR / "yolo11m.pt")
POSE_WEIGHTS = str(MODELS_DIR / "yolo11n-pose.pt")

# Classes
BALL_CLASS_ID = 32
PERSON_CLASS_ID = 0

# Detection + pose thresholds
DET_CONF = 0.25
POSE_CONF = 0.4
KPT_CONF = 0.3

# Calibration (fixed rectangle in meters)
CALIB_RECT_WIDTH_M = 20.0
CALIB_RECT_HEIGHT_M = 10.0
USE_HOMOGRAPHY = True

# Extended ground plane visualization
DRAW_EXTENDED_GROUND = True  # Draw the extrapolated ground plane beyond calibration rect
EXTENDED_GROUND_MULTIPLIER = 3.0  # How many times larger than calibration rect to extend
DRAW_GROUND_GRID = True  # Draw grid lines on the ground plane
GROUND_GRID_SPACING_M = 5.0  # Spacing between grid lines in meters
GRID_LINE_SUBDIVISIONS = 20  # Number of subdivisions per grid line for perspective accuracy
SHOW_GRID_DISTANCE_MARKERS = True  # Show distance markers at grid intersections

# Touch timing (seconds; translated to frames at runtime)
CONTACT_SEC = 0.07
COOLDOWN_SEC = 0.25
SOFT_TOUCH_SEC = 0.12

# Rolling windows / smoothing
BALL_SMOOTHING = 5
FOOT_SMOOTHING = 3
BALL_HOLD_FRAMES = 2
BALL_CONTACT_MAX_AGE = 3
BALL_VEL_SMOOTHING = 3
BALL_MOTION_MAX_GAP = 2
BALL_EVENT_WINDOW = 6
BALL_MOTION_BUFFER = 25

# Motion heuristics
DIR_CHANGE_DEG = 20.0
SPEED_GAIN_RATIO = 0.2
SPEED_DROP_RATIO = 0.2
SPEED_MIN_PX_S = 8.0
SPEED_MIN_RADIUS_RATIO = 0.1
DRAW_BALL_VECTOR = True
BALL_VECTOR_SCALE = 12.0
SHOW_BALL_SPEED = False
SHOW_BALL_COMPONENTS = False
DRAW_BALL_TRAIL = False
BALL_TRAIL_LENGTH = 20
BALL_TRAIL_MAX_GAP_FRAMES = 6
EVENT_TOUCH_ENABLED = True
EVENT_TOUCH_DIST_RATIO = 1.2
REQUIRE_BALL_IMPULSE = False
IMPULSE_WINDOW = 8
SEPARATION_GAIN_PX = 6.0
SEPARATION_GAIN_RATIO = 0.15
REQUIRE_SEPARATION_GAIN = False
REQUIRE_AWAY_MOTION = False
REQUIRED_SIGNALS = 2
ALLOW_SOFT_TOUCH = True
SOFT_TOUCH_DIST_RATIO = 1.05
SOFT_TOUCH_SPEED_RATIO = 2.0
SHOW_PLAYER_SPEED = True
PLAYER_REF_HEIGHT_M = 1.70
HIP_TO_ANKLE_RATIO = 0.53
HIP_ANKLE_MIN_PX = 30.0
PLAYER_SPEED_SMOOTHING = 7  # Increased from 5 for better stability

# Accuracy improvements
MIN_MOVEMENT_M = 0.03  # Minimum movement to count as distance (prevents jitter accumulation)
MAX_SPEED_GAP_FRAMES = 5  # Reset speed tracking if player lost for this many frames
MAX_HUMAN_SPEED_MPS = 12.0  # Maximum realistic human sprint speed (~43 km/h, Usain Bolt peak)
MAX_BALL_SPEED_MPS = 50.0  # Maximum realistic ball speed (~180 km/h for hard shots)
USE_EMA_SMOOTHING = True  # Use Exponential Moving Average instead of median
EMA_ALPHA = 0.3  # EMA smoothing factor (0.1=very smooth, 0.5=more responsive)

# Unit system toggle
USE_METRIC_DISPLAY = True  # True = m/s and meters, False = km/h and km
# Passing / shooting heuristics
PASS_MIN_SPEED_PX_S = 90.0
SHOT_MIN_SPEED_PX_S = 140.0
PASS_MIN_SPEED_MPS = 5.0
SHOT_MIN_SPEED_MPS = 12.0
POSSESSION_RELEASE_RATIO = 1.35
PASS_SPEED_MAX_PX_S = 170.0
PASS_ACCEL_MAX_PX_S2 = 800.0
PASS_DIR_VAR_MAX_DEG = 18.0
PASS_GROUND_MIN_FRAC = 0.6
PASS_TARGET_MAX_DIST_PX = 220.0
PASS_TARGET_MAX_ANGLE_DEG = 35.0
SHOT_SPEED_MIN_PX_S = SHOT_MIN_SPEED_PX_S
SHOT_ACCEL_MIN_PX_S2 = 1200.0
SHOT_ACCEL_WINDOW_FRAMES = 3
SHOT_GOAL_ALIGN_DEG = 25.0
SHOT_RECEIVER_EXCLUDE_DIST_PX = 160.0
SHOT_POWER_ACCEL_MIN = 800.0
SHOT_POWER_ACCEL_MAX = 2800.0
POST_KICK_MIN_FRAMES = 5
POST_KICK_MAX_FRAMES = 10
POST_KICK_COOLDOWN_FRAMES = 12
GROUND_MAX_VY_RATIO = 0.35
GROUND_MAX_DELTA_PX = 22.0
EVENT_OVERLAY_FRAMES = 24
GOAL_VECTOR_X = 1.0
GOAL_VECTOR_Y = 0.0

# Jump detection
JUMP_GROUND_WINDOW = 45
JUMP_SCALE_SMOOTHING = 10
JUMP_MIN_DELTA_PX = 5.0
JUMP_DELTA_RATIO = 0.12
JUMP_MIN_AIR_FRAMES = 2
JUMP_END_RATIO = 0.55
JUMP_COOLDOWN_FRAMES = 6
JUMP_NOISE_SCALE = 1.6
JUMP_NOISE_MARGIN_PX = 2.0
JUMP_UP_PX_PER_FRAME = 1.0
JUMP_UP_STREAK = 2

# Tracking windows
TRACK_TTL = 30
FOOT_RADIUS_RATIO = 0.06
FOOT_HOLD_FRAMES = 3
CONTACT_GAP_ALLOW = 1
FOOT_Y_MIN_RATIO = 0.55
BALL_RADIUS_MIN_PX = 2
ACTIVE_FOOT_HOLD_FRAMES = 3

__all__ = [name for name in globals() if name.isupper()]
