from ultralytics import YOLO
import cv2
import math
from collections import deque

from soccer_ai import config as cfg

# ---------------- MODELS ----------------
det_model = YOLO(str(cfg.MODELS_DIR / "yolo11m.pt"))
pose_model = YOLO(str(cfg.MODELS_DIR / "yolo11n-pose.pt"))

# ---------------- VIDEO ----------------
cap = cv2.VideoCapture("test_video.mp4")

# ---------------- TOUCH PARAMS ----------------
FRAMES_REQUIRED = 2
COOLDOWN_FRAMES = 10

# ---------------- STATE ----------------
frame_idx = 0
touch_count = 0
last_touch_frame = -100
collision_history = deque(maxlen=5)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    annotated = frame.copy()

    results = det_model.track(
        frame,
        persist=True,
        conf=0.25,
        tracker="bytetrack.yaml"
    )

    ball = None
    foot = None
    ball_radius = None
    foot_radius = None

    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ---------- BALL ----------
            if cls == 32:
                ball = ((x1 + x2) / 2, (y1 + y2) / 2)
                ball_radius = 0.5 * min(x2 - x1, y2 - y1)

                cv2.circle(
                    annotated,
                    (int(ball[0]), int(ball[1])),
                    int(ball_radius),
                    (0, 165, 255),
                    2
                )

            # ---------- PERSON ----------
            if cls != 0:
                continue

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            pose = pose_model(crop, conf=0.4)
            if pose[0].keypoints is None:
                continue

            kpts = pose[0].keypoints.xy.cpu().numpy()
            if len(kpts) == 0:
                continue

            p = kpts[0]

            # pose overlay
            annotated[y1:y2, x1:x2] = pose[0].plot()

            # ---------- FOOT CIRCLE ----------
            la, ra = p[15], p[16]
            foot_local = la if la[1] > ra[1] else ra
            foot = (foot_local[0] + x1, foot_local[1] + y1)

            # scale foot radius from body
            lh, rh = p[11], p[12]
            hip_y = (lh[1] + rh[1]) / 2
            hip_ankle_px = abs(hip_y - foot_local[1])

            if hip_ankle_px < 30:
                continue

            foot_radius = int(0.15 * hip_ankle_px)

            cv2.circle(
                annotated,
                (int(foot[0]), int(foot[1])),
                foot_radius,
                (255, 0, 0),
                2
            )

    # ---------- COLLISION TEST ----------
    if ball and foot and ball_radius and foot_radius:
        dist = math.dist(ball, foot)
        collision = dist <= (ball_radius + foot_radius)

        collision_history.append(collision)

        if (
            sum(collision_history) >= FRAMES_REQUIRED
            and frame_idx - last_touch_frame > COOLDOWN_FRAMES
        ):
            touch_count += 1
            last_touch_frame = frame_idx
            collision_history.clear()

    # ---------- HUD ----------
    cv2.putText(
        annotated,
        f"Touches: {touch_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    cv2.imshow("CIRCULAR TOUCH ACCURACY", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n======================")
print(f"TOTAL TOUCHES: {touch_count}")
print("======================")
