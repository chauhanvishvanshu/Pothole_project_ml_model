from ultralytics import YOLO
import cv2

print("🚀 Loading YOLOv8 pothole segmentation model...")
model = YOLO("yolov8m_pothole_best.pt")

video_path = "input_video1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"❌ Could not open video: {video_path}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_pothole_detected.mp4", fourcc, 30.0,
                      (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Optional: improve brightness & contrast
    frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=30)

    # Run detection in segmentation mode
    results = model(frame, conf=0.15, task="segment")
    annotated_frame = results[0].plot()

    cv2.putText(annotated_frame, f"Potholes Detected: {len(results[0].boxes)}",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    out.write(annotated_frame)
    cv2.imshow("Pothole Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Detection complete. Saved as output_pothole_detected.mp4")
