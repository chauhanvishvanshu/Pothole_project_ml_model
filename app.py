 # app.py — YOLOv12 Pothole Detection with FPS, ETA, and Live Stats
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, send_file
import cv2
import os
import time
import threading
import traceback
from ultralytics import YOLO
import numpy as np
import csv
import io
from datetime import datetime
from werkzeug.utils import secure_filename

# ---------- CONFIG ----------
MODEL_PATH = "best.pt"
UPLOAD_DIR = "uploads"
REPORT_DIR = "reports"
CONFIDENCE_THRESHOLD = 0.28
IOU_THRESHOLD = 0.45
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
INFERENCE_SIZE = 960
PIXELS_PER_METER = 100
ALLOWED_EXT = {"mp4", "mov", "avi", "mkv", "webm"}
# ----------------------------

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR

# Thread-safe state
state_lock = threading.Lock()
fps_lock = threading.Lock()

detection_count = 0
total_area_m2 = 0.0
detection_logs = []
current_video = None
video_path = None
processing = False
video_fps = 0.0
processing_fps = 0.0
frames_this_second = 0
total_frames = 0
processed_frames = 0

last_report = {
    "csv_path": None,
    "csv_url": None,
    "total_detections": 0,
    "total_area": 0.0,
    "average_confidence": 0.0,
    "video_name": None
}

# ---------------- MODEL LOAD ----------------
print("🚀 Loading YOLOv12 model...")
model = YOLO(MODEL_PATH)
try:
    model.model.float()
    model.fuse()
except Exception:
    pass
print("✅ Model loaded successfully!\n")
# --------------------------------------------

def ensure_dirs():
    for d in [UPLOAD_DIR, REPORT_DIR]:
        os.makedirs(d, exist_ok=True)


def allowed_file(filename):
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in ALLOWED_EXT


def to_numpy(x):
    try:
        if hasattr(x, "cpu"):
            return x.cpu().detach().numpy()
        if isinstance(x, np.ndarray):
            return x
        return np.array(x)
    except Exception:
        return np.array([])


def get_severity(area_m2):
    if area_m2 < 0.5:
        return "Minor", (0, 255, 0)
    elif area_m2 < 1.5:
        return "Moderate", (0, 215, 255)
    elif area_m2 < 3.0:
        return "Major", (0, 140, 255)
    else:
        return "Severe", (0, 0, 255)


def reset_detection_log(video_name):
    global detection_logs, detection_count, total_area_m2, current_video, processed_frames
    with state_lock:
        detection_logs = []
        detection_count = 0
        total_area_m2 = 0.0
        current_video = video_name
        processed_frames = 0
    print(f"\n🆕 New video session started: {video_name}\n")


def draw_boxes(frame, results, frame_number, video_time):
    boxes = getattr(results, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return frame, 0, 0.0, []

    count, total_area = 0, 0.0
    logs = []

    for box in boxes:
        try:
            xyxy = to_numpy(box.xyxy[0]).astype(int)
            x1, y1, x2, y2 = xyxy
            conf = float(to_numpy(box.conf)[0])
        except Exception:
            continue

        if conf < CONFIDENCE_THRESHOLD:
            continue

        w, h = x2 - x1, y2 - y1
        area_px = w * h
        if area_px < 1000 or area_px > FRAME_WIDTH * FRAME_HEIGHT * 0.35:
            continue

        ratio = w / (h + 1e-6)
        if ratio < 0.5 or ratio > 3.8:
            continue

        area_m2 = round(area_px / (PIXELS_PER_METER ** 2), 2)
        severity, color = get_severity(area_m2)
        total_area += area_m2
        count += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{area_m2} m² ({conf:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        logs.append({
            "frame_number": frame_number,
            "video_time_s": round(video_time, 2),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "confidence": round(conf, 3),
            "area_m2": area_m2,
            "severity": severity
        })

    return frame, count, total_area, logs


def infer_frame(fid, frame, video_time):
    """Run YOLO inference on one frame and count processed FPS."""
    global processed_frames
    try:
        start = time.time()
        results = model.predict(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=INFERENCE_SIZE,
            verbose=False
        )[0]
        frame, dets, area, logs = draw_boxes(frame, results, fid, video_time)
        end = time.time()

        with state_lock:
            detection_logs.extend(logs)
            global detection_count, total_area_m2
            detection_count += dets
            total_area_m2 += area
            processed_frames += 1

        with fps_lock:
            global frames_this_second
            frames_this_second += 1

        return fid, frame
    except Exception as e:
        print(f"⚠️ Inference error frame {fid}: {e}")
        processed_frames += 1
        return fid, frame


def generate_frames():
    global video_path, processing, total_frames, video_fps
    if not video_path:
        print("⚠️ No video selected.")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    reset_detection_log(video_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Cannot open video source: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    print(f"🎥 Video FPS: {video_fps:.2f}, Total Frames: {total_frames}")

    frame_number = 0
    with state_lock:
        processing = True

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame_number += 1
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            _, frame = infer_frame(frame_number, frame, video_time)

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            time.sleep(0.005)
    finally:
        cap.release()
        with state_lock:
            processing = False
        print(f"✅ Finished processing: {video_name}")


# ---------------- FPS Monitor Thread ----------------
def fps_monitor():
    global frames_this_second, processing_fps
    while True:
        time.sleep(1.0)
        with fps_lock:
            processing_fps = frames_this_second
            frames_this_second = 0


# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_video():
    global video_path
    if "video" not in request.files:
        return redirect(url_for("index"))

    file = request.files["video"]
    if file.filename == "":
        return redirect(url_for("index"))

    if file and allowed_file(file.filename):
        ensure_dirs()
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if os.path.exists(filepath):
            base, ext = os.path.splitext(filename)
            filename = f"{base}_{int(time.time())}{ext}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        video_path = filepath
        print(f"📂 Video uploaded: {filepath}")
        return redirect(url_for("index"))
    return "Invalid file type", 400


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/detection_count")
def detection_count_route():
    """Return live stats for detections."""
    with state_lock:
        logs_copy = list(detection_logs)

    if not logs_copy:
        return jsonify({
            "detections": 0,
            "total_area": 0.0,
            "avg_confidence": 0.0
        })

    total_det = len(logs_copy)
    total_area = round(sum(l.get("area_m2", 0.0) for l in logs_copy), 2)
    avg_conf = round(sum(l.get("confidence", 0.0) for l in logs_copy) / total_det, 3)

    return jsonify({
        "detections": total_det,
        "total_area": total_area,
        "avg_confidence": avg_conf
    })


@app.route("/progress")
def progress_status():
    with state_lock, fps_lock:
        progress = (processed_frames / total_frames * 100) if total_frames > 0 else 0
        remaining_frames = max(total_frames - processed_frames, 0)
        est_time = remaining_frames / processing_fps if processing_fps > 0 else 0
        speed_percent = (processing_fps / video_fps * 100) if video_fps > 0 else 0
        return jsonify({
            "processed_frames": processed_frames,
            "total_frames": total_frames,
            "progress_percent": round(progress, 2),
            "video_fps": round(video_fps, 2),
            "processing_fps": round(processing_fps, 2),
            "processing_speed_percent": round(speed_percent, 1),
            "estimated_time_left": round(est_time, 1)
        })


@app.route("/export_csv")
def export_csv():
    """Manual CSV export."""
    global detection_logs, last_report

    with state_lock:
        logs_copy = list(detection_logs)
        last_csv = last_report.get("csv_path")

    if logs_copy:
        video_name = current_video or "session"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{secure_filename(video_name)}_detections_{timestamp}.csv"
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=[
            "frame_number", "video_time_s", "timestamp",
            "confidence", "area_m2", "severity"
        ])
        writer.writeheader()
        writer.writerows(logs_copy)
        output.seek(0)
        return send_file(io.BytesIO(output.getvalue().encode()),
                         mimetype="text/csv",
                         as_attachment=True,
                         download_name=filename)
    elif last_csv and os.path.exists(last_csv):
        return send_file(last_csv, mimetype="text/csv", as_attachment=True,
                         download_name=os.path.basename(last_csv))
    return "No detections yet.", 404


# ----------------------------------------
if __name__ == "__main__":
    ensure_dirs()
    threading.Thread(target=fps_monitor, daemon=True).start()
    app.run(debug=True, host="0.0.0.0", threaded=True)
