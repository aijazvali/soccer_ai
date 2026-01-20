from ultralytics import YOLO
import cv2
import math
from collections import deque

from soccer_ai import config as cfg

# ---------------- MODELS ----------------
det_model = YOLO(str(cfg.MODELS_DIR / "yolo11n.pt"))
pose_model = YOLO(str(cfg.MODELS_DIR / "yolo11n-pose.pt"))

# ---------------- VIDEO ----------------
cap = cv2.VideoCapture("test_video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

# ---------------- PARAMETERS ----------------
TOUCH_DIST_PX = 90
TOUCH_FRAMES_REQUIRED = 3
TOUCH_COOLDOWN = 12

# ---------------- STATE ----------------
frame_idx = 0
last_touch_frame = -100
touch_count = 0

prev_ankle = {}
speed_history = {}
close_frames = {}

max_speed_kmh = 0.0

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

    ball_center = None

    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ---------- BALL ----------
            if cls == 32:
                ball_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(annotated, "BALL", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            # ---------- PERSON ----------
            if cls != 0 or box.id is None:
                continue

            track_id = int(box.id[0])

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Crop for pose
            h, w, _ = frame.shape
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            crop = frame[y1c:y2c, x1c:x2c]

            if crop.size == 0:
                continue

            pose_results = pose_model(crop, conf=0.4)
            if pose_results[0].keypoints is None:
                continue

            kpts = pose_results[0].keypoints.xy.cpu().numpy()
            if len(kpts) == 0:
                continue

            p = kpts[0]

            # Pose overlay
            pose_img = pose_results[0].plot()
            annotated[y1c:y2c, x1c:x2c] = pose_img

            # ---------- FOOT POINT (contact foot) ----------
            la, ra = p[15], p[16]
            foot_local = la if la[1] > ra[1] else ra

            foot = (
                foot_local[0] + x1c,
                foot_local[1] + y1c
            )

            # ---------- SPEED (CORRECT) ----------
            lh, rh = p[11], p[12]
            hip = (
                (lh[0] + rh[0]) / 2,
                (lh[1] + rh[1]) / 2
            )

            hip_ankle_px = abs(hip[1] - foot_local[1])
            if hip_ankle_px > 30:
                meters_per_pixel = (0.53 * 1.7) / hip_ankle_px

                if track_id in prev_ankle:
                    dist_px = math.dist(prev_ankle[track_id], foot)
                    speed_mps = dist_px * meters_per_pixel * fps
                    speed_kmh = speed_mps * 3.6

                    max_speed_kmh = max(max_speed_kmh, speed_kmh)

                    cv2.putText(
                        annotated,
                        f"{speed_kmh:.1f} km/h",
                        (x1, y2 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )

                prev_ankle[track_id] = foot

            # ---------- TOUCH LOGIC (ROBUST) ----------
            if ball_center:
                dist = math.dist(ball_center, foot)

                cv2.line(
                    annotated,
                    (int(ball_center[0]), int(ball_center[1])),
                    (int(foot[0]), int(foot[1])),
                    (255, 0, 0),
                    2
                )

                close_frames.setdefault(track_id, deque(maxlen=5))
                close_frames[track_id].append(dist < TOUCH_DIST_PX)

                if (
                    sum(close_frames[track_id]) >= TOUCH_FRAMES_REQUIRED
                    and frame_idx - last_touch_frame > TOUCH_COOLDOWN
                ):
                    touch_count += 1
                    last_touch_frame = frame_idx

    # ---------- HUD ----------
    cv2.putText(annotated, f"Touches: {touch_count}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("ACCURACY DEBUG (FIXED)", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n======================")
print(f"TOTAL TOUCHES: {touch_count}")
print(f"MAX SPEED: {max_speed_kmh:.2f} km/h")
print("======================")
