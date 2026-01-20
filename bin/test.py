import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

from soccer_ai import config as cfg

# =============================
# CONFIG
# =============================

VIDEO_PATH = "input.mp4"
MODEL_PATH = str(cfg.MODELS_DIR / "yolo11n.pt")   # fine-tuned: classes = player, ball

BALL_CLASS_ID = 0
PLAYER_CLASS_ID = 1

FOOT_REGION_RATIO = 0.22          # bottom 22% of player mask
MIN_INTERSECTION_PIXELS = 12
CONTACT_FRAMES_REQUIRED = 3
COOLDOWN_FRAMES = 10
ACCEL_THRESHOLD_PERCENTILE = 85

# =============================
# INIT
# =============================

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

touch_buffer = defaultdict(int)
cooldown = defaultdict(int)

ball_velocity_hist = deque(maxlen=5)
last_ball_center = None

touch_count = defaultdict(int)

# =============================
# UTILS
# =============================

def compute_center(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return np.mean(xs), np.mean(ys)

def mask_intersection_area(mask1, mask2):
    return np.sum(np.logical_and(mask1, mask2))

def extract_foot_region(player_mask):
    h, w = player_mask.shape
    y_start = int(h * (1 - FOOT_REGION_RATIO))
    foot_mask = np.zeros_like(player_mask)
    foot_mask[y_start:h, :] = player_mask[y_start:h, :]
    return foot_mask

def ball_acceleration_ok():
    if len(ball_velocity_hist) < 3:
        return False
    accel = np.abs(ball_velocity_hist[-1] - ball_velocity_hist[-2])
    thresh = np.percentile(ball_velocity_hist, ACCEL_THRESHOLD_PERCENTILE)
    return accel > thresh

# =============================
# MAIN LOOP
# =============================

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.3,
        iou=0.5,
        verbose=False
    )[0]

    if results.masks is None:
        continue

    masks = results.masks.data.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    ids = results.boxes.id.cpu().numpy().astype(int)

    ball_mask = None
    ball_center = None

    # -------------------------
    # BALL PROCESSING
    # -------------------------

    for m, c in zip(masks, classes):
        if c == BALL_CLASS_ID:
            ball_mask = m.astype(bool)
            ball_center = compute_center(ball_mask)
            break

    if ball_center is not None and last_ball_center is not None:
        v = np.linalg.norm(np.array(ball_center) - np.array(last_ball_center))
        ball_velocity_hist.append(v)

    last_ball_center = ball_center

    # -------------------------
    # PLAYER PROCESSING
    # -------------------------

    for m, c, pid in zip(masks, classes, ids):
        if c != PLAYER_CLASS_ID:
            continue

        if cooldown[pid] > 0:
            cooldown[pid] -= 1
            continue

        player_mask = m.astype(bool)
        foot_mask = extract_foot_region(player_mask)

        if ball_mask is None:
            touch_buffer[pid] = 0
            continue

        intersection = mask_intersection_area(ball_mask, foot_mask)

        if intersection > MIN_INTERSECTION_PIXELS:
            touch_buffer[pid] += 1
        else:
            touch_buffer[pid] = 0

        # -------------------------
        # TOUCH CONFIRMATION
        # -------------------------

        if touch_buffer[pid] >= CONTACT_FRAMES_REQUIRED:
            if ball_acceleration_ok():
                touch_count[pid] += 1
                cooldown[pid] = COOLDOWN_FRAMES
                touch_buffer[pid] = 0
                print(f"TOUCH: Player {pid}, Total: {touch_count[pid]}")

        # -------------------------
        # VISUAL DEBUG
        # -------------------------

        overlay = frame.copy()
        overlay[ball_mask] = (0, 0, 255)
        overlay[foot_mask] = (0, 255, 0)

        cv2.putText(
            overlay,
            f"ID {pid} | touches {touch_count[pid]}",
            (20, 40 + pid * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    cv2.imshow("Ball Touch Debug", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
