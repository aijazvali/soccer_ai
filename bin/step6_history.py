from ultralytics import YOLO
import cv2

from soccer_ai import config as cfg

det_model = YOLO(str(cfg.MODELS_DIR / "yolo11n.pt"))
pose_model = YOLO(str(cfg.MODELS_DIR / "yolo11n-pose.pt"))

cap = cv2.VideoCapture("test_video.mp4")

player_history = {}
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    results = det_model.track(
        frame,
        persist=True,
        conf=0.4,
        tracker="bytetrack.yaml"
    )

    if results[0].boxes.id is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            if cls != 0:
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

            left_ankle = person_kpts[15]
            right_ankle = person_kpts[16]

            ankle_center = (
                (left_ankle[0] + right_ankle[0]) / 2,
                (left_ankle[1] + right_ankle[1]) / 2
            )

            if track_id not in player_history:
                player_history[track_id] = {
                    "ankles": [],
                    "frames": []
                }

            player_history[track_id]["ankles"].append(ankle_center)
            player_history[track_id]["frames"].append(frame_idx)

            print(f"ID {track_id} | Frame {frame_idx} | Ankle {ankle_center}")

    cv2.imshow("History Builder", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
