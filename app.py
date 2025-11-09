# app.py — YOLOv12 Pothole Detection (v2 Final Optimized)
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, send_file
import cv2, os, time, threading, queue, traceback
import torch
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import csv, io
from werkzeug.utils import secure_filename

# ------------------ CONFIG ------------------
MODEL_PATH = "best.pt"
UPLOAD_DIR = "uploads"
CONF_THRESHOLD = 0.28
IOU_THRESHOLD = 0.45
FRAME_W, FRAME_H = 1280, 720
PIXELS_PER_METER = 100
ALLOWED_EXT = {"mp4", "mov", "avi", "mkv"}
# --------------------------------------------

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ------------------ GLOBALS ------------------
video_path = None
video_fps = 0.0
processed_frames = 0
total_frames = 0
processing_fps = 0.0
detection_logs = []
stop_flag = threading.Event()
frame_q, result_q = queue.Queue(maxsize=6), queue.Queue(maxsize=6)
fps_lock = threading.Lock()
# --------------------------------------------

# ------------------ GPU CONFIG ------------------
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Loading YOLOv12 model...")
model = YOLO(MODEL_PATH)
model.to(device)
print("✅ GPU detected. Using CUDA." if device == "cuda" else "⚠️ Using CPU mode")
print("✅ Model ready.\n")
# -------------------------------------------------

def allowed_file(fname):
    return "." in fname and fname.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def to_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.array(x)

def get_severity(area_m2):
    if area_m2 < 0.5: return "Minor", (0,255,0)
    elif area_m2 < 1.5: return "Moderate", (0,215,255)
    elif area_m2 < 3.0: return "Major", (0,140,255)
    else: return "Severe", (0,0,255)

def draw_boxes(frame, results):
    boxes = getattr(results, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return frame, 0, 0.0, []
    count, total_area, logs = 0, 0.0, []
    for box in boxes:
        xyxy = to_np(box.xyxy[0]).astype(int)
        x1, y1, x2, y2 = xyxy
        conf = float(to_np(box.conf)[0])
        if conf < CONF_THRESHOLD: continue
        w, h = x2 - x1, y2 - y1
        area_px = w * h
        if area_px < 800: continue
        area_m2 = round(area_px / (PIXELS_PER_METER**2), 2)
        severity, color = get_severity(area_m2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{area_m2} m² ({conf:.2f})"
        cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        count += 1; total_area += area_m2
        logs.append({
            "frame": processed_frames,
            "confidence": conf,
            "area_m2": area_m2,
            "severity": severity,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
    return frame, count, total_area, logs

# ------------------ THREADS ------------------
def infer_worker():
    global processed_frames, processing_fps
    stream = torch.cuda.Stream() if device == "cuda" else None
    while not stop_flag.is_set():
        try:
            frame_data = frame_q.get(timeout=1)
        except queue.Empty:
            continue
        fid, frame = frame_data
        t0 = time.time()
        try:
            with torch.cuda.stream(stream) if stream else torch.no_grad():
                results = model.predict(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, imgsz=960, verbose=False)[0]
            frame, dets, area, logs = draw_boxes(frame, results)
            with fps_lock:
                detection_logs.extend(logs)
                processed_frames += 1
                processing_fps = 1.0 / (time.time() - t0 + 1e-4)
            result_q.put((fid, frame))
        except Exception as e:
            print("⚠️ Inference error:", e)
    print("🛑 Inference thread stopped")

def fps_monitor():
    global processing_fps
    prev = processed_frames
    while True:
        time.sleep(1)
        now = processed_frames
        fps = now - prev
        prev = now
        with fps_lock:
            processing_fps = fps

def generate_frames():
    global video_fps, total_frames, processed_frames
    if not video_path:
        print("⚠️ No video selected.")
        return
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    print(f"🎥 Video loaded ({video_fps:.2f} FPS, {total_frames} frames)")
    processed_frames = 0
    stop_flag.clear()

    # Start inference thread if not active
    threading.Thread(target=infer_worker, daemon=True).start()

    fid = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        fid += 1
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        try:
            frame_q.put_nowait((fid, frame))
        except queue.Full:
            time.sleep(0.005)

        # Get processed frame
        try:
            _, out = result_q.get(timeout=0.5)
        except queue.Empty:
            continue

        ret, buf = cv2.imencode(".jpg", out)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

    cap.release()
    stop_flag.set()
    print("✅ Video completed")

# ------------------ ROUTES ------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_video():
    global video_path, processed_frames, detection_logs, stop_flag
    if "video" not in request.files:
        return redirect(url_for("index"))
    file = request.files["video"]
    if not file.filename or not allowed_file(file.filename):
        return "Invalid file", 400
    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    # ✅ Reset state cleanly before new run
    stop_flag.clear()
    while not frame_q.empty(): frame_q.get()
    while not result_q.empty(): result_q.get()
    detection_logs.clear()
    processed_frames = 0
    video_path = path
    print("📂 Uploaded:", path)
    return redirect(url_for("index"))

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/detection_count")
def detection_count_route():
    with fps_lock:
        total_det = len(detection_logs)
        total_area = round(sum(l["area_m2"] for l in detection_logs), 2) if detection_logs else 0.0
        avg_conf = round(sum(l["confidence"] for l in detection_logs)/total_det, 3) if total_det > 0 else 0.0
        return jsonify({"detections": total_det, "total_area": total_area, "avg_confidence": avg_conf})

@app.route("/progress")
def progress():
    with fps_lock:
        progress = (processed_frames / total_frames * 100) if total_frames > 0 else 0
        eta = (total_frames - processed_frames) / (processing_fps + 1e-4) if processing_fps > 0 else 0
        return jsonify({
            "processed_frames": processed_frames,
            "total_frames": total_frames,
            "progress_percent": round(progress, 2),
            "video_fps": round(video_fps, 2),
            "processing_fps": round(processing_fps, 2),
            "estimated_time_left": round(eta, 1)
        })

@app.route("/processing_status")
def processing_status():
    return jsonify({"processing": not stop_flag.is_set()})

@app.route("/export_csv")
def export_csv():
    if not detection_logs:
        return "No detections yet.", 404
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=detection_logs[0].keys())
    writer.writeheader(); writer.writerows(detection_logs)
    output.seek(0)
    filename = f"detections_{datetime.now().strftime('%H%M%S')}.csv"
    return send_file(io.BytesIO(output.getvalue().encode()), mimetype="text/csv", as_attachment=True, download_name=filename)

# ------------------ RUN APP ------------------
if __name__ == "__main__":
    threading.Thread(target=fps_monitor, daemon=True).start()
    app.run(debug=True, host="0.0.0.0", use_reloader=False, threaded=True)
