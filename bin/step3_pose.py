from ultralytics import YOLO
import cv2

from soccer_ai import config as cfg

# Load pose model
pose_model = YOLO(str(cfg.MODELS_DIR / "yolo11n-pose.pt"))

cap = cv2.VideoCapture("test_video.mp4")

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run pose inference
    results = pose_model(frame, conf=0.4)

    # Draw pose
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Pose Test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
