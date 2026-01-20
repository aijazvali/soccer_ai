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

    # Run detection
    results = model(frame, conf=0.4)

    # Draw detections
    annotated_frame = results[0].plot()

    # Show frame
    cv2.imshow("YOLO Video Test", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
