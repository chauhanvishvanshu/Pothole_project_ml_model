# app.py — YOLOv12 Pothole Detection + Video Upload + Auto CSV per Video (Final)
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
detection_count = 0
total_area_m2 = 0.0
detection_logs = []
current_video = None
video_path = None
processing = False

last_report = {
    "csv_path": None,
    "csv_url": None,
    "total_detections": 0,
    "total_area": 0.0,
    "average_confidence": 0.0,
    "video_name": None
}

# ---------------- MODEL LOAD ----------------
print("🚀 Loading YOLOv12 model (with Upload Support)...")
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
    global detection_logs, detection_count, total_area_m2, current_video
    with state_lock:
        detection_logs = []
        detection_count = 0
        total_area_m2 = 0.0
        current_video = video_name
    print(f"\n🆕 New video session started: {video_name}\n")


def compute_aggregate_from_logs(logs):
    if not logs:
        return 0, 0.0, 0.0
    total_det = len(logs)
    total_area = sum(l.get("area_m2", 0.0) for l in logs)
    avg_conf = sum(l.get("confidence", 0.0) for l in logs) / total_det
    return total_det, round(total_area, 2), round(avg_conf, 3)


# ✅ FIXED — now uses Flask app context safely for url_for
def save_csv_to_disk(video_name):
    ensure_dirs()
    with state_lock:
        logs_copy = list(detection_logs)

    if not logs_copy:
        print("⚠️ No detections logged, skipping CSV save.")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = secure_filename(video_name)
    filename = f"{safe_name}_detections_{timestamp}.csv"
    filepath = os.path.join(REPORT_DIR, filename)

    try:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "frame_number", "video_time_s", "timestamp",
                "confidence", "area_m2", "severity"
            ])
            writer.writeheader()
            writer.writerows(logs_copy)

        total_det, total_area, avg_conf = compute_aggregate_from_logs(logs_copy)

        # create Flask context to safely build URL
        with app.app_context():
            csv_url = url_for("serve_report", filename=filename)

        with state_lock:
            last_report.update({
                "csv_path": filepath,
                "csv_url": csv_url,
                "total_detections": total_det,
                "total_area": total_area,
                "average_confidence": avg_conf,
                "video_name": safe_name
            })

        print(f"💾 CSV saved: {filepath}")
        print(f"🔗 Accessible at: {csv_url}")
        return filepath

    except Exception as e:
        print("❌ Failed to save CSV:", e)
        traceback.print_exc()
        return None


def update_stats(count, total_area, logs):
    global detection_count, total_area_m2, detection_logs
    with state_lock:
        detection_count = count
        total_area_m2 = total_area
        if logs:
            detection_logs.extend(logs)


def get_stats():
    with state_lock:
        return detection_count, round(total_area_m2, 2)


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


def generate_frames():
    global video_path, processing
    if not video_path:
        print("⚠️ No video selected.")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    reset_detection_log(video_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Cannot open video source: {video_path}")
        return

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

            try:
                results = model.predict(
                    source=frame,
                    conf=CONFIDENCE_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    imgsz=INFERENCE_SIZE,
                    verbose=False
                )[0]
                frame, detections, total_area, logs = draw_boxes(frame, results, frame_number, video_time)
                update_stats(detections, total_area, logs)
            except Exception as e:
                print("⚠️ Inference error:", e)
                traceback.print_exc()

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            time.sleep(0.005)
    finally:
        cap.release()
        save_csv_to_disk(video_name)
        with state_lock:
            processing = False
        print(f"✅ Finished processing: {video_name}")


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
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/detection_count")
def detection_count_route():
    count, total_area = get_stats()
    with state_lock:
        avg_conf = round(
            sum(d.get("confidence", 0.0) for d in detection_logs) / len(detection_logs), 3
        ) if detection_logs else 0.0
    return jsonify({"detections": count, "total_area": total_area, "avg_confidence": avg_conf})


@app.route("/processing_status")
def processing_status():
    with state_lock:
        return jsonify({"processing": processing})


@app.route("/last_report")
def last_report_route():
    with state_lock:
        if not last_report.get("csv_path"):
            return jsonify({}), 404
        return jsonify(last_report)


@app.route("/reports/<path:filename>")
def serve_report(filename):
    safe_path = os.path.normpath(os.path.join(REPORT_DIR, filename))
    if not safe_path.startswith(os.path.abspath(REPORT_DIR)):
        return "Forbidden", 403
    if not os.path.exists(safe_path):
        return "Not found", 404
    return send_file(safe_path, mimetype="text/csv", as_attachment=True, download_name=filename)


@app.route("/export_csv")
def export_csv():
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
    app.run(debug=True, host="0.0.0.0", threaded=True)
