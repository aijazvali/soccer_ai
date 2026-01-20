from ultralytics import YOLO
import cv2
import math
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

from soccer_ai import config as cfg

# ---------------- CONFIG ----------------
BALL_CLASS_ID = 32
PERSON_CLASS_ID = 0

DET_CONF = 0.25
POSE_CONF = 0.4
KPT_CONF = 0.3

CONTACT_SEC = 0.07
COOLDOWN_SEC = 0.25

BALL_SMOOTHING = 5
FOOT_SMOOTHING = 3
BALL_HOLD_FRAMES = 2
BALL_CONTACT_MAX_AGE = 1
TRACK_TTL = 30
FOOT_RADIUS_RATIO = 0.06
FOOT_HOLD_FRAMES = 3
FOOT_Y_MIN_RATIO = 0.55
BALL_RADIUS_MIN_PX = 2
ACTIVE_FOOT_HOLD_FRAMES = 3


@dataclass
class TrackState:
    active_foot: Optional[str] = None
    contact_streak: int = 0
    last_touch_frame: int = -1000
    left_touches: int = 0
    right_touches: int = 0
    last_contact_frame: int = -1000
    touch_locked: bool = False
    left_last_frame: int = -1000
    right_last_frame: int = -1000
    left_hist: deque = field(default_factory=lambda: deque(maxlen=FOOT_SMOOTHING))
    right_hist: deque = field(default_factory=lambda: deque(maxlen=FOOT_SMOOTHING))


def clamp_box(x1, y1, x2, y2, w, h):
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

    def score(c):
        (cx, cy), _r, conf = c
        dist = math.hypot(cx - last_center[0], cy - last_center[1])
        return dist - conf * 50.0

    return min(candidates, key=score)


def smooth_ball(hist):
    arr = np.array(hist, dtype=np.float32)
    cx, cy, r = np.median(arr, axis=0)
    return (float(cx), float(cy)), float(r)


def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# ---------------- MODELS ----------------
det_model = YOLO(str(cfg.MODELS_DIR / "yolo11m.pt"))
pose_model = YOLO(str(cfg.MODELS_DIR / "yolo11m-pose.pt"))

# ---------------- VIDEO ----------------
cap = cv2.VideoCapture("test_video.mp4")
if not cap.isOpened():
    raise RuntimeError("Unable to open video: test_video.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0 or fps != fps:
    fps = 30.0

FRAMES_REQUIRED = max(1, int(round(CONTACT_SEC * fps)))
COOLDOWN_FRAMES = max(1, int(round(COOLDOWN_SEC * fps)))

# ---------------- STATE ----------------
frame_idx = 0

left_touches = 0
right_touches = 0

ball_history = deque(maxlen=BALL_SMOOTHING)
last_ball_frame = -1000

track_states = {}
track_last_seen = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    annotated = frame.copy()
    h, w = frame.shape[:2]

    results = det_model.track(
        frame,
        persist=True,
        conf=DET_CONF,
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

            if cls == BALL_CLASS_ID and conf >= DET_CONF:
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                radius = 0.5 * max(BALL_RADIUS_MIN_PX, min(x2 - x1, y2 - y1))
                ball_candidates.append((center, radius, conf))
            elif cls == PERSON_CLASS_ID and conf >= DET_CONF:
                track_id = None
                if hasattr(box, "id") and box.id is not None:
                    track_id = int(box.id[0])
                people_dets.append((track_id, x1, y1, x2, y2))

    ball_center = None
    ball_radius = None

    if ball_candidates:
        last_center = None
        if ball_history:
            last_center = (ball_history[-1][0], ball_history[-1][1])
        choice = pick_ball(ball_candidates, last_center)
        ball_center, ball_radius, _conf = choice
        ball_history.append((ball_center[0], ball_center[1], ball_radius))
        last_ball_frame = frame_idx

    if ball_history and frame_idx - last_ball_frame <= BALL_HOLD_FRAMES:
        ball_center, ball_radius = smooth_ball(ball_history)
    else:
        ball_history.clear()

    people = []
    for i, (track_id, x1, y1, x2, y2) in enumerate(people_dets):
        box = clamp_box(x1, y1, x2, y2, w, h)
        if box is None:
            continue
        x1, y1, x2, y2 = box

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        pose = pose_model(crop, conf=POSE_CONF, verbose=False)
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

        left_foot = None
        right_foot = None
        if p_conf is None or p_conf[15] >= KPT_CONF:
            left_foot = (p[15][0] + x1, p[15][1] + y1)
        if p_conf is None or p_conf[16] >= KPT_CONF:
            right_foot = (p[16][0] + x1, p[16][1] + y1)

        foot_min_y = y1 + (y2 - y1) * FOOT_Y_MIN_RATIO
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

        if frame_idx - state.left_last_frame > FOOT_HOLD_FRAMES:
            state.left_hist.clear()
        if frame_idx - state.right_last_frame > FOOT_HOLD_FRAMES:
            state.right_hist.clear()

        left_foot = smooth_point(state.left_hist)
        right_foot = smooth_point(state.right_hist)

        foot_radius = max(2, (y2 - y1) * FOOT_RADIUS_RATIO)

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

        people.append(
            {
                "id": track_id,
                "bbox": (x1, y1, x2, y2),
                "left": left_foot,
                "right": right_foot,
                "foot_radius": foot_radius,
            }
        )

        track_last_seen[track_id] = frame_idx

    ball_for_contact = (
        ball_center
        and ball_radius
        and frame_idx - last_ball_frame <= BALL_CONTACT_MAX_AGE
    )

    contact_candidates = []
    if ball_for_contact:
        for person in people:
            left = person["left"]
            right = person["right"]
            if left is None and right is None:
                continue

            d_left = dist(ball_center, left) if left is not None else float("inf")
            d_right = dist(ball_center, right) if right is not None else float("inf")
            threshold = ball_radius + person["foot_radius"]

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

    active_contacts = {}
    if contact_candidates:
        contact_candidates.sort(key=lambda c: c[0])
        _dist, chosen_id, chosen_foot = contact_candidates[0]
        active_contacts[chosen_id] = chosen_foot

    for person in people:
        track_id = person["id"]
        state = track_states[track_id]
        candidate = active_contacts.get(track_id)

        if candidate is not None:
            if state.active_foot is None:
                state.active_foot = candidate
            if state.active_foot != candidate:
                candidate = state.active_foot

            state.last_contact_frame = frame_idx
            if not state.touch_locked:
                state.contact_streak += 1
        else:
            state.contact_streak = 0
            state.touch_locked = False
            if frame_idx - state.last_contact_frame > ACTIVE_FOOT_HOLD_FRAMES:
                state.active_foot = None

        if (
            state.contact_streak >= FRAMES_REQUIRED
            and frame_idx - state.last_touch_frame > COOLDOWN_FRAMES
            and state.active_foot is not None
            and not state.touch_locked
        ):
            if state.active_foot == "L":
                state.left_touches += 1
                left_touches += 1
            else:
                state.right_touches += 1
                right_touches += 1

            state.last_touch_frame = frame_idx
            state.contact_streak = 0
            state.touch_locked = True
            state.active_foot = None

    for track_id in list(track_last_seen.keys()):
        if frame_idx - track_last_seen[track_id] > TRACK_TTL:
            track_last_seen.pop(track_id, None)
            track_states.pop(track_id, None)

    if ball_center and ball_radius:
        cv2.circle(
            annotated,
            (int(ball_center[0]), int(ball_center[1])),
            int(ball_radius),
            (0, 165, 255),
            2,
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

    cv2.imshow("FOOT-LOCK TOUCH DETECTION", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\n======================")
print(f"LEFT TOUCHES:  {left_touches}")
print(f"RIGHT TOUCHES: {right_touches}")
print("======================")
