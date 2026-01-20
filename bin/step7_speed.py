from ultralytics import YOLO
import cv2
import math

from soccer_ai import config as cfg

det_model = YOLO(str(cfg.MODELS_DIR / "yolo11n.pt"))
pose_model = YOLO(str(cfg.MODELS_DIR / "yolo11n-pose.pt"))

cap = cv2.VideoCapture("test_video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

prev_ankle = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = det_model.track(
        frame,
        persist=True,
        conf=0.4,
        tracker="bytetrack.yaml"
    )

    if results[0].boxes.id is not None:
        for box in results[0].boxes:
            if int(box.cls[0]) != 0:
                continue

            track_id = int(box.id[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

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

            person_kpts = kpts[0]
            la, ra = person_kpts[15], person_kpts[16]

            ankle = (
                (la[0] + ra[0]) / 2,
                (la[1] + ra[1]) / 2
            )

            if track_id in prev_ankle:
                dist = math.dist(prev_ankle[track_id], ankle)
                speed = dist * fps
                print(f"ID {track_id} speed: {speed:.2f} px/s")

            prev_ankle[track_id] = ankle

    cv2.imshow("Speed Calc", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
