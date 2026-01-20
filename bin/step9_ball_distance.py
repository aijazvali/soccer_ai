from ultralytics import YOLO
import cv2
import math

from soccer_ai import config as cfg

det_model = YOLO(str(cfg.MODELS_DIR / "yolo11n.pt"))
pose_model = YOLO(str(cfg.MODELS_DIR / "yolo11n-pose.pt"))

cap = cv2.VideoCapture("test_video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = det_model.track(
        frame,
        persist=True,
        conf=0.3,
        tracker="bytetrack.yaml"
    )

    ball_center = None
    ankle_center = None

    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Person
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

            # Sports ball (COCO = 32)
            elif cls == 32:
                ball_center = (
                    (x1 + x2) / 2,
                    (y1 + y2) / 2
                )

    if ankle_center and ball_center:
        dist = math.dist(ankle_center, ball_center)
        print(f"Ball–foot distance: {dist:.2f} px")

    cv2.imshow("Ball Distance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
