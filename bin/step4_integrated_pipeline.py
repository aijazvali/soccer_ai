from ultralytics import YOLO
import cv2

from soccer_ai import config as cfg

det_model = YOLO(str(cfg.MODELS_DIR / "yolo11n.pt"))
pose_model = YOLO(str(cfg.MODELS_DIR / "yolo11n-pose.pt"))

cap = cv2.VideoCapture("test_video.mp4")

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

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

    annotated_frame = frame.copy()

    if results[0].boxes.id is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            if cls != 0:  # person only
                continue

            track_id = int(box.id[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Draw tracking ID
            cv2.putText(
                annotated_frame,
                f"ID {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Crop for pose
            h, w, _ = frame.shape
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            person_crop = frame[y1c:y2c, x1c:x2c]

            if person_crop.size == 0:
                continue

            pose_results = pose_model(person_crop, conf=0.4)
            pose_img = pose_results[0].plot()

            annotated_frame[y1c:y2c, x1c:x2c] = pose_img

    cv2.imshow("Detection + Tracking + Pose", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
