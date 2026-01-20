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

            p = kpts[0]

            la, ra = p[15], p[16]
            lh, rh = p[11], p[12]

            ankle = ((la[0] + ra[0]) / 2, (la[1] + ra[1]) / 2)
            hip = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)

            # Estimate scale
            hip_ankle_px = abs(hip[1] - ankle[1])
            if hip_ankle_px < 30:
                continue

            meters_per_pixel = (0.53 * 1.7) / hip_ankle_px

            if track_id in prev_ankle:
                dist_px = math.dist(prev_ankle[track_id], ankle)
                speed_mps = dist_px * meters_per_pixel * fps
                speed_kmph = speed_mps * 3.6

                print(f"ID {track_id} speed: {speed_kmph:.2f} km/h")

            prev_ankle[track_id] = ankle

    cv2.imshow("Real Speed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
