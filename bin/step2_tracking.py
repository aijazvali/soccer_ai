from ultralytics import YOLO
import cv2

from soccer_ai import config as cfg

# Load model
model = YOLO(str(cfg.MODELS_DIR / "yolo11n.pt"))

# Load video
cap = cv2.VideoCapture("test_video.mp4")

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection + tracking
    results = model.track(
        frame,
        persist=True,
        conf=0.4,
        tracker="bytetrack.yaml"
    )

    # Draw tracking results
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO + ByteTrack", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
