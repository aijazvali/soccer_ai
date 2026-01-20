from ultralytics import YOLO
import cv2
import math

from soccer_ai import config as cfg

# Models
det_model = YOLO(str(cfg.MODELS_DIR / "yolo11n.pt"))
pose_model = YOLO(str(cfg.MODELS_DIR / "yolo11n-pose.pt"))

# Video
cap = cv2.VideoCapture("test_video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

# Touch logic parameters (RELAXED ON PURPOSE)
TOUCH_DIST_PX = 80          # relaxed threshold
TOUCH_COOLDOWN = 12         # frames between touches

# State
last_touch_frame = -100
touch_count = 0
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    results = det_model.track(
        frame,
        persist=True,
        conf=0.25,
        tracker="bytetrack.yaml"
    )

    ankle_center = None
    ball_center = None

    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ---------------- PERSON ----------------
            if cls == 0:
                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                pose_results = pose_model(crop, conf=0.4)
                if pose_results[0].keypoints is None:
                    continue

                kpts = pose_results[0].keypoints.xy.cpu().numpy()
                if len(kpts) == 0:
                    continue

                p = kpts[0]
                la, ra = p[15], p[16]

                ankle_center = (
                    (la[0] + ra[0]) / 2 + x1,
                    (la[1] + ra[1]) / 2 + y1
                )

            # ---------------- BALL ----------------
            # COCO sports ball = 32 (often misclassified, we accept that)
            elif cls == 32:
                ball_center = (
                    (x1 + x2) / 2,
                    (y1 + y2) / 2
                )

    # ---------------- TOUCH LOGIC ----------------
    if ankle_center and ball_center:
        dist = math.dist(ankle_center, ball_center)

        # DEBUG (very important)
        print(f"DEBUG frame={frame_idx} dist={dist:.1f}")

        if (
            dist < TOUCH_DIST_PX
            and frame_idx - last_touch_frame > TOUCH_COOLDOWN
        ):
            touch_count += 1
            last_touch_frame = frame_idx
            print(f"✅ TOUCH #{touch_count} at frame {frame_idx}")

    cv2.imshow("Touch Detection (Fixed)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n======================")
print(f"TOTAL TOUCHES: {touch_count}")
print("======================")
