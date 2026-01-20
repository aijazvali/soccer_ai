from ultralytics import YOLO
import cv2

from soccer_ai import config as cfg

pose_model = YOLO(str(cfg.MODELS_DIR / "yolo11n-pose.pt"))
cap = cv2.VideoCapture("test_video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = pose_model(frame, conf=0.4)

    # Guard 1: keypoints exist
    if results[0].keypoints is not None:
        kpts = results[0].keypoints.xy.cpu().numpy()

        # Guard 2: at least one person detected
        if len(kpts) > 0:
            person_kpts = kpts[0]

            left_ankle = person_kpts[15]
            right_ankle = person_kpts[16]

            print("Left ankle:", left_ankle, "Right ankle:", right_ankle)
        else:
            print("No person keypoints in this frame")
    else:
        print("No detections")

    annotated = results[0].plot()
    cv2.imshow("Pose Keypoints", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
