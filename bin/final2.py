import cv2
import math
import numpy as np
from collections import deque

from soccer_ai.config import *  # noqa: F401,F403
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

# ---------------- MODELS ----------------
det_model, pose_model = load_models()

# ---------------- VIDEO ----------------
cap = cv2.VideoCapture("test_video.mp4")
if not cap.isOpened():
    raise RuntimeError("Unable to open video: test_video.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0 or fps != fps:
    fps = 30.0

FRAMES_REQUIRED = max(1, int(round(CONTACT_SEC * fps)))
COOLDOWN_FRAMES = max(1, int(round(COOLDOWN_SEC * fps)))
SOFT_FRAMES_REQUIRED = max(1, int(round(SOFT_TOUCH_SEC * fps)))

# ---------------- STATE ----------------
frame_idx = 0

left_touches = 0
right_touches = 0

ball_history = deque(maxlen=BALL_SMOOTHING)
last_ball_frame = -1000

ball_motion = deque(maxlen=BALL_VEL_SMOOTHING)
ball_event_history = deque(maxlen=BALL_EVENT_WINDOW)
prev_ball_center = None
prev_ball_frame = None
prev_ball_speed = None
prev_ball_dir = None

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
    ball_center_raw = None
    ball_radius_raw = None
    ball_detected = False
    ball_speed = None
    ball_dir = None
    ball_vel = None

    if ball_candidates:
        last_center = None
        if ball_history:
            last_center = (ball_history[-1][0], ball_history[-1][1])
        choice = pick_ball(ball_candidates, last_center)
        ball_center_raw, ball_radius_raw, _conf = choice
        ball_history.append((ball_center_raw[0], ball_center_raw[1], ball_radius_raw))
        last_ball_frame = frame_idx
        ball_detected = True

    if ball_history and frame_idx - last_ball_frame <= BALL_HOLD_FRAMES:
        ball_center, ball_radius = smooth_ball(ball_history)
    else:
        ball_history.clear()

    ball_event = False
    if ball_detected:
        if prev_ball_frame is not None and frame_idx - prev_ball_frame > BALL_MOTION_MAX_GAP:
            prev_ball_center = None
            prev_ball_frame = None
            prev_ball_speed = None
            prev_ball_dir = None
            ball_motion.clear()

        ball_motion.append((ball_center_raw[0], ball_center_raw[1]))
        vel_center = smooth_point(ball_motion)
        if vel_center is not None:
            if prev_ball_center is not None and prev_ball_frame is not None:
                dt = frame_idx - prev_ball_frame
                if dt > 0:
                    vx = (vel_center[0] - prev_ball_center[0]) / dt
                    vy = (vel_center[1] - prev_ball_center[1]) / dt
                    speed = math.hypot(vx, vy) * fps
                    ball_dir = math.atan2(vy, vx)
                    ball_speed = speed
                    ball_vel = (vx, vy)
                    radius_for_speed = (
                        ball_radius_raw
                        if ball_radius_raw is not None
                        else (ball_radius if ball_radius is not None else 0.0)
                    )
                    speed_min = max(
                        SPEED_MIN_PX_S, radius_for_speed * fps * SPEED_MIN_RADIUS_RATIO
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
                            dir_change >= DIR_CHANGE_DEG
                            or speed_gain >= SPEED_GAIN_RATIO
                            or speed_drop >= SPEED_DROP_RATIO
                        ):
                            ball_event = True
                        if prev_ball_speed < speed_min * 0.6 and speed >= speed_min:
                            ball_event = True
                        if prev_ball_speed >= speed_min and speed < speed_min * 0.6:
                            ball_event = True

                    prev_ball_speed = speed
                    prev_ball_dir = ball_dir
                    ball_event_history.append((frame_idx, ball_event))

            prev_ball_center = vel_center
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

    ball_event_recent = any(
        ev for f, ev in ball_event_history if frame_idx - f <= BALL_EVENT_WINDOW
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
        active_contacts[chosen_id] = (chosen_foot, _dist)

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
            if frame_idx - state.last_contact_frame > CONTACT_GAP_ALLOW:
                state.contact_streak = 0
                state.touch_locked = False
                if frame_idx - state.last_contact_frame > ACTIVE_FOOT_HOLD_FRAMES:
                    state.active_foot = None

        if (
            candidate is not None
            and state.contact_streak >= FRAMES_REQUIRED
            and state.pending_contact_frame < 0
        ):
            state.pending_contact_frame = frame_idx
            state.pending_foot = state.active_foot
            state.pending_ball_dist = candidate_dist if candidate_dist is not None else 0.0

        if state.pending_contact_frame >= 0:
            if frame_idx - state.pending_contact_frame > IMPULSE_WINDOW:
                state.pending_contact_frame = -1000
                state.pending_foot = None
                state.pending_ball_dist = 0.0
            elif (
                ball_center
                and ball_radius
                and state.pending_foot in ("L", "R")
                and frame_idx - state.last_touch_frame > COOLDOWN_FRAMES
                and not state.touch_locked
            ):
                foot_pt = person["left"] if state.pending_foot == "L" else person["right"]
                if foot_pt is not None:
                    current_dist = dist(ball_center, foot_pt)
                    dist_gain = current_dist - state.pending_ball_dist
                    separation_need = max(
                        SEPARATION_GAIN_PX, ball_radius * SEPARATION_GAIN_RATIO
                    )
                    separation_ok = dist_gain >= separation_need
                    speed_min = max(
                        SPEED_MIN_PX_S, ball_radius * fps * SPEED_MIN_RADIUS_RATIO
                    )
                    rel = (ball_center[0] - foot_pt[0], ball_center[1] - foot_pt[1])
                    away_ok = False
                    if ball_vel is not None and ball_speed is not None and ball_speed >= speed_min:
                        away_ok = (ball_vel[0] * rel[0] + ball_vel[1] * rel[1]) > 0
                    impulse_signal = ball_event_recent
                    soft_touch_ok = False
                    if ALLOW_SOFT_TOUCH and state.contact_streak >= SOFT_FRAMES_REQUIRED:
                        close_enough = current_dist <= ball_radius * SOFT_TOUCH_DIST_RATIO
                        if close_enough:
                            if ball_speed is None:
                                soft_touch_ok = True
                            else:
                                soft_touch_ok = ball_speed <= speed_min * SOFT_TOUCH_SPEED_RATIO

                    signal_score = (
                        int(impulse_signal)
                        + int(separation_ok)
                        + int(away_ok)
                        + int(soft_touch_ok)
                    )
                    count_ok = signal_score >= REQUIRED_SIGNALS
                    if REQUIRE_BALL_IMPULSE and not (impulse_signal or soft_touch_ok):
                        count_ok = False
                    if REQUIRE_SEPARATION_GAIN and not (separation_ok or soft_touch_ok):
                        count_ok = False
                    if REQUIRE_AWAY_MOTION and not (away_ok or soft_touch_ok):
                        count_ok = False

                    if count_ok:
                        if state.pending_foot == "L":
                            state.left_touches += 1
                            left_touches += 1
                        else:
                            state.right_touches += 1
                            right_touches += 1

                        state.last_touch_frame = frame_idx
                        state.contact_streak = 0
                        state.touch_locked = True
                        state.active_foot = None
                        state.pending_contact_frame = -1000
                        state.pending_foot = None
                        state.pending_ball_dist = 0.0

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
