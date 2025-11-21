# app.py — YOLOv12 Pothole Detection Backend (Fixed, annotated for production)
# -----------------------------------------------------------------------------
# NOTES (high-level)
#  - This file is your original app with only three focused production fixes:
#       1) Always create a CSV file at end of run (even if no detections).
#       2) Always create a ZIP archive at end of run (even if CSV empty).
#       3) Avoid using Flask url_for(_external=True) inside background threads
#          (it raises RuntimeError when outside a request context). Instead,
#          build absolute URLs using PUBLIC_HOST (environment) or default host.
#  - All other code, routes, and behavior are preserved exactly as you provided.
# -----------------------------------------------------------------------------

from flask import Flask, Response, jsonify, request, send_file, url_for
from flask_cors import CORS
import os, time, threading, traceback, csv, io, queue, zipfile, smtplib, json
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from werkzeug.utils import secure_filename
import cv2
import numpy as np

# ---------- OPTIONAL DEPENDENCIES ----------
try:
    import torch
except Exception:
    torch = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# ---------- CONFIG ----------
MODEL_PATH = "best.pt"
UPLOAD_DIR = "uploads"
REPORT_DIR = "reports"
ARCHIVE_DIR = os.path.join(REPORT_DIR, "archives")
CONFIDENCE_THRESHOLD = 0.28
IOU_THRESHOLD = 0.45
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
INFERENCE_SIZE = 960
PIXELS_PER_METER = 100
ALLOWED_EXT = {"mp4", "mov", "avi", "mkv", "webm"}
FRAME_QUEUE_MAX = 6
RESULT_QUEUE_MAX = 6
FRAME_SKIP = int(os.environ.get("FRAME_SKIP", 0))  # 0=process all, 1=skip one, etc.
MAX_FILES_UPLOADS = int(os.environ.get("MAX_FILES_UPLOADS", 10))
MAX_FILES_REPORTS = int(os.environ.get("MAX_FILES_REPORTS", 10))
MAX_FILES_ARCHIVES = int(os.environ.get("MAX_FILES_ARCHIVES", 5))

# ---------- Notifications ----------
NOTIFY_EMAIL = os.environ.get("NOTIFY_EMAIL")
SMTP_SERVER = os.environ.get("SMTP_SERVER")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASS = os.environ.get("SMTP_PASS")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")

app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR

# ---------- STATE ----------
state_lock = threading.Lock()
fps_lock = threading.Lock()
detection_logs = []
current_video = None
video_path = None
video_fps = 0.0
total_frames = 0
processed_frames = 0
processing_fps = 0.0
processing = False
frame_q = queue.Queue(maxsize=FRAME_QUEUE_MAX)
result_q = queue.Queue(maxsize=RESULT_QUEUE_MAX)
stop_event = threading.Event()
last_report = {
    "csv_path": None, "csv_url": None,
    "archive_path": None, "archive_url": None,
    "total_detections": 0, "total_area": 0.0,
    "average_confidence": 0.0, "video_name": None
}

worker_thread = None  # keep single worker thread reference

# ---------- DEVICE & MODEL ----------
FORCE_CPU = os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes")
DEVICE = "cpu"
USE_CUDA = False
if not FORCE_CPU and torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
    USE_CUDA = True
    DEVICE = "cuda:0"

print(f"Device → {DEVICE} | FORCE_CPU={FORCE_CPU}")

model = None
if YOLO is None:
    print("❌ ultralytics not found. Install with `pip install ultralytics`.")
else:
    try:
        print("🚀 Loading YOLO model...")
        model = YOLO(MODEL_PATH)
        # attempt to move model to desired device safely
        try:
            if hasattr(model, "to"):
                model.to(DEVICE)
        except Exception:
            try:
                if hasattr(model, "model") and hasattr(model.model, "to"):
                    model.model.to(DEVICE)
            except Exception:
                pass
        try:
            if USE_CUDA and hasattr(model, "model") and hasattr(model.model, "half"):
                model.model.half()
        except Exception:
            pass
        try:
            if hasattr(model, "fuse"):
                model.fuse()
        except Exception:
            pass
        print("✅ Model ready.")
    except Exception as e:
        print("⚠️ Failed loading model:", e)
        traceback.print_exc()
        model = None

# ---------- UTILITIES ----------
def ensure_dirs():
    for d in [UPLOAD_DIR, REPORT_DIR, ARCHIVE_DIR]:
        os.makedirs(d, exist_ok=True)

def auto_cleanup(directory, keep):
    """Keep only the most recent 'keep' files in directory."""
    try:
        files = sorted(
            [os.path.join(directory, f) for f in os.listdir(directory)],
            key=os.path.getmtime, reverse=True
        )
        for f in files[keep:]:
            try:
                os.remove(f)
                print(f"🧹 Deleted old file: {f}")
            except Exception:
                pass
    except Exception as e:
        print(f"⚠️ Cleanup error in {directory}: {e}")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[-1].lower() in ALLOWED_EXT

def to_numpy(x):
    try:
        # handle torch tensors
        if hasattr(x, "cpu"):
            return x.cpu().detach().numpy()
        return np.array(x)
    except Exception:
        return np.array([])

def get_severity(area_m2):
    if area_m2 < 0.5:  return "Minor", (0,255,0)
    elif area_m2 < 1.5: return "Moderate", (0,215,255)
    elif area_m2 < 3.0: return "Major", (0,140,255)
    else:               return "Severe", (0,0,255)

def compute_aggregate_from_logs(logs):
    if not logs: return 0,0.0,0.0
    total_det=len(logs)
    total_area=sum(l.get("area_m2",0.0) for l in logs)
    avg_conf=sum(l.get("confidence",0.0) for l in logs)/total_det
    return total_det,round(total_area,2),round(avg_conf,3)

def send_email_notification(subject, body):
    if not (NOTIFY_EMAIL and SMTP_SERVER and SMTP_USER and SMTP_PASS):
        print("📭 Email not configured; skipping.")
        return
    try:
        msg = MIMEMultipart()
        msg["From"], msg["To"], msg["Subject"] = SMTP_USER, NOTIFY_EMAIL, subject
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, NOTIFY_EMAIL, msg.as_string())
        print(f"📧 Email sent to {NOTIFY_EMAIL}")
    except Exception as e:
        print("⚠️ Email failed:", e)

def send_webhook_notification(data):
    if not WEBHOOK_URL:
        return
    try:
        r = requests.post(WEBHOOK_URL, json=data, timeout=10)
        print(f"🔗 Webhook status: {r.status_code}")
    except Exception as e:
        print("⚠️ Webhook failed:", e)

def flush_queue(q):
    try:
        while True:
            q.get_nowait()
    except Exception:
        pass

# -----------------------------
# FIXED: background-safe URL builder
# -----------------------------
def build_absolute_url(path: str):
    """
    Build an absolute URL from a path for use outside request context (background threads).
    Uses environment variable PUBLIC_HOST if set, otherwise defaults to localhost:7860.
    Example: build_absolute_url('/reports/file.csv') -> 'http://127.0.0.1:7860/reports/file.csv'
    """
    host = os.environ.get("PUBLIC_HOST", "http://127.0.0.1:7860")
    return host.rstrip("/") + path

# -----------------------------
# archive creation (unchanged logic, using build_absolute_url)
# -----------------------------
def create_archive(video_path_local, csv_path, video_name):
    try:
        ensure_dirs()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_name = f"{secure_filename(video_name)}_archive_{ts}.zip"
        zip_path = os.path.join(ARCHIVE_DIR, zip_name)
        with zipfile.ZipFile(zip_path, "w") as zf:
            if os.path.exists(video_path_local):
                zf.write(video_path_local, os.path.basename(video_path_local))
            if csv_path and os.path.exists(csv_path):
                zf.write(csv_path, os.path.basename(csv_path))
        # Use background-safe URL builder (do not call url_for in background thread)
        archive_url = build_absolute_url(f"/download_archive/{zip_name}")
        with state_lock:
            last_report["archive_path"], last_report["archive_url"] = zip_path, archive_url
        auto_cleanup(ARCHIVE_DIR, MAX_FILES_ARCHIVES)
        print(f"📦 Archive created: {zip_path}")
        # optional notifications
        body = (f"Pothole detection complete for '{video_name}'.\n\n"
                f"Detections: {last_report['total_detections']}\n"
                f"Total Area: {last_report['total_area']} m²\n"
                f"Average Confidence: {last_report['average_confidence']}\n\n"
                f"CSV: {last_report['csv_url']}\nZIP: {archive_url}\n")
        send_email_notification("Pothole Report Ready", body)
        send_webhook_notification({"event": "report_ready", **last_report})
    except Exception as e:
        print("⚠️ Archive creation failed:", e)
        traceback.print_exc()
# -----------------------------
# CSV save (fixed: always create CSV even if zero detections)
# -----------------------------
def save_csv_to_disk(video_name):
    """
    Save detection logs to a timestamped CSV in REPORT_DIR.
    FIX: This always creates a CSV file (even when detection_logs is empty),
    so downstream archive creation and last_report URLs will always be produced.
    """
    ensure_dirs()
    with state_lock:
        logs_copy = list(detection_logs)

    # --- If no detections, create an empty CSV (header only) instead of skipping ---
    if not logs_copy:
        print("⚠️ No detections — creating empty CSV.")
        logs_copy = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = secure_filename(video_name)
    filename = f"{safe_name}_detections_{timestamp}.csv"
    filepath = os.path.join(REPORT_DIR, filename)

    try:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["frame_number","video_time_s","timestamp","confidence","area_m2","severity"])
            writer.writeheader()
            writer.writerows(logs_copy)

        # compute aggregates (works with empty list)
        total_det, total_area, avg_conf = compute_aggregate_from_logs(logs_copy)

        # Build CSV URL using background-safe builder (no url_for here)
        csv_url = build_absolute_url(f"/reports/{filename}")

        with state_lock:
            last_report.update({
                "csv_path": filepath,
                "csv_url": csv_url,
                "total_detections": total_det,
                "total_area": total_area,
                "average_confidence": avg_conf,
                "video_name": safe_name
            })

        auto_cleanup(REPORT_DIR, MAX_FILES_REPORTS)
        print(f"💾 CSV saved: {filepath}")
        return filepath

    except Exception as e:
        print("❌ CSV save failed:", e)
        traceback.print_exc()
        return None

# ---------- INFERENCE ----------
def draw_boxes(frame, results, frame_number, video_time):
    """
    Robust drawing: handles multiple ultralytics box formats (tensor, numpy) and extracts confidence safely.
    Returns: (frame_with_boxes, count, total_area_m2, logs_list)
    """
    boxes = getattr(results, "boxes", None)
    if not boxes:
        return frame, 0, 0.0, []
    count, total_area, logs = 0, 0.0, []
    # If boxes is a sequence of box objects (ultralytics), iterate
    for box in boxes:
        try:
            # Attempt to extract xyxy coordinates (absolute pixel values)
            xyxy_arr = None
            # box.xyxy may be tensor([x1,y1,x2,y2]) OR an array-like; support variants
            if hasattr(box, "xyxy"):
                data = box.xyxy
                # data might be tensor([x1,y1,x2,y2]) or array with shape (1,4)
                arr = to_numpy(data)
                if arr.ndim == 2 and arr.shape[0] >= 1:
                    xyxy_arr = arr[0]
                elif arr.ndim == 1 and arr.size >= 4:
                    xyxy_arr = arr[:4]
            elif hasattr(box, "xyxyn"):
                arr = to_numpy(box.xyxyn)
                if arr.ndim == 2 and arr.shape[0] >= 1:
                    # normalized coords — convert to pixel coordinates using frame shape
                    xyxy_n = arr[0]
                    h, w = frame.shape[:2]
                    xyxy_arr = np.array([xyxy_n[0]*w, xyxy_n[1]*h, xyxy_n[2]*w, xyxy_n[3]*h])
            else:
                # as fallback, try known attributes
                try:
                    arr = to_numpy(getattr(box, "xyxy", getattr(box, "coords", np.array([]))))
                    if arr.size >= 4:
                        xyxy_arr = arr.flatten()[:4]
                except Exception:
                    xyxy_arr = None

            if xyxy_arr is None or len(xyxy_arr) < 4:
                continue

            # convert to ints and clamp
            x1, y1, x2, y2 = [int(max(0, v)) for v in xyxy_arr[:4]]

            # confidence extraction: box.conf or box.confidence or box.conf[0]
            conf = 0.0
            conf_attr = None
            for name in ("conf", "confidence", "prob"):
                if hasattr(box, name):
                    conf_attr = getattr(box, name)
                    break
            if conf_attr is not None:
                try:
                    conf_arr = to_numpy(conf_attr)
                    if conf_arr.size:
                        conf = float(conf_arr.reshape(-1)[0])
                    else:
                        # maybe scalar
                        conf = float(conf_attr)
                except Exception:
                    try:
                        conf = float(conf_attr)
                    except Exception:
                        conf = 0.0

            # filter by confidence threshold
            if conf < CONFIDENCE_THRESHOLD:
                continue

            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            area_px = w * h
            # area filters (ignore tiny or huge boxes)
            if area_px < 800 or area_px > FRAME_WIDTH * FRAME_HEIGHT * 0.35:
                continue

            ratio = w / (h + 1e-6)
            if ratio < 0.5 or ratio > 3.8:
                continue

            area_m2 = round(area_px / (PIXELS_PER_METER ** 2), 2)
            severity, color = get_severity(area_m2)
            total_area += area_m2
            count += 1
            # draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{area_m2} m² ({conf:.2f})"
            cv2.putText(frame, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            logs.append({
                "frame_number": int(frame_number),
                "video_time_s": round(video_time, 2),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "confidence": round(conf, 3),
                "area_m2": area_m2,
                "severity": severity
            })
        except Exception:
            # any single-box failure shouldn't stop overall detection
            traceback.print_exc()
            continue
    return frame, count, total_area, logs

def infer_worker():
    """
    Single worker that reads frames from frame_q, runs model (if any), draws boxes and emits into result_q.
    Worker exists when stop_event is set.
    """
    global processed_frames, worker_thread
    print(f"🧠 Worker started (device={DEVICE})")
    try:
        while not stop_event.is_set():
            try:
                fid, frame = frame_q.get(timeout=1)
            except queue.Empty:
                continue
            try:
                if model is None:
                    # pass-through (no inference), send back original frame
                    out_frame = frame
                    logs = []
                else:
                    # run prediction (best-effort compatibility)
                    try:
                        results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD,
                                                iou=IOU_THRESHOLD, imgsz=INFERENCE_SIZE,
                                                device=DEVICE, verbose=False)
                        # predict returns list-like. get first result
                        if isinstance(results, (list, tuple)) and len(results) > 0:
                            r = results[0]
                        else:
                            r = results
                    except Exception:
                        # fallback to model(frame) if predict signature differs
                        try:
                            results = model(frame)
                            if isinstance(results, (list, tuple)) and len(results) > 0:
                                r = results[0]
                            else:
                                r = results
                        except Exception as e:
                            print("⚠️ Model inference failed:", e)
                            r = None

                    if r is None:
                        out_frame = frame
                        logs = []
                    else:
                        # draw boxes on a copy of frame to avoid modifying input if reused
                        frame_copy = frame.copy()
                        out_frame, count, total_area, logs = draw_boxes(frame_copy, r, fid, 0)
                # append logs
                if logs:
                    with state_lock:
                        detection_logs.extend(logs)
                with state_lock:
                    processed_frames += 1
                # push result (fid,out_frame)
                try:
                    result_q.put_nowait((fid, out_frame))
                except queue.Full:
                    # drop oldest and push
                    try:
                        result_q.get_nowait()
                        result_q.put_nowait((fid, out_frame))
                    except Exception:
                        pass
            except Exception as e:
                print("⚠️ Inference loop error:", e)
                traceback.print_exc()
    finally:
        print("🛑 Worker stopped")
        with state_lock:
            worker_thread = None
# ---------------------------------------------------------
#   FRAME GENERATOR + END-OF-VIDEO REPORT CREATION (FIXED)
# ---------------------------------------------------------
def generate_frames():
    """
    Streams processed frames to frontend and,
    when video finishes, ALWAYS:
        - saves CSV (even empty)
        - creates ZIP archive (always)
        - updates last_report with working URLs
    """
    global processing, processed_frames, total_frames, video_fps
    global current_video, worker_thread, frame_q, result_q, video_path

    if not video_path:
        print("⚠️ No video selected.")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # reset previous session
    with state_lock:
        detection_logs.clear()
        current_video = video_name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    processed_frames = 0
    stop_event.clear()
    processing = True

    # ensure a worker thread exists
    if worker_thread is None or (worker_thread is not None and not worker_thread.is_alive()):
        with state_lock:
            worker = threading.Thread(target=infer_worker, daemon=True)
            worker.start()
            worker_thread = worker

    frame_no = 0

    try:
        while cap.isOpened() and not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            frame_no += 1

            # frame skipping
            if FRAME_SKIP > 0 and frame_no % (FRAME_SKIP + 1) != 1:
                continue

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            try:
                frame_q.put_nowait((frame_no, frame))
            except queue.Full:
                try:
                    frame_q.get_nowait()
                    frame_q.put_nowait((frame_no, frame))
                except Exception:
                    pass

            # fetch latest processed frame
            out = frame
            try:
                fid_out = result_q.get(timeout=0.5)
                if fid_out is not None:
                    _, out = fid_out
                    if out is None:
                        out = frame
            except queue.Empty:
                out = frame

            # JPEG encode
            try:
                ret2, buf = cv2.imencode(".jpg", out)
                if not ret2:
                    continue
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + buf.tobytes()
                    + b"\r\n"
                )
            except Exception:
                traceback.print_exc()
                continue

            time.sleep(0.002)

    finally:
        cap.release()

        # -------------------------------
        # SAVE CSV (always)
        # -------------------------------
        csv_path = save_csv_to_disk(video_name)

        if csv_path is None:
            # fallback attempt (should not happen)
            csv_path = save_csv_to_disk(video_name)

        # --------------------------------------
        # CREATE ZIP ARCHIVE (always)
        # --------------------------------------
        threading.Thread(
            target=create_archive,
            args=(video_path, csv_path, video_name),
            daemon=True,
        ).start()

        stop_event.set()
        processing = False

        print(f"✅ Completed {video_name}")


# ---------------------------------------------------------
#                     ROUTES
# ---------------------------------------------------------
@app.route("/")
def index():
    return jsonify({
        "status": "ok",
        "message": "Pothole Detection Backend (YOLOv12 + Flask) running."
    })

@app.route("/ping")
def ping():
    return jsonify({"status": "ok"})

@app.route("/device_info")
def device_info():
    return jsonify({
        "device": DEVICE,
        "gpu": USE_CUDA,
        "frame_skip": FRAME_SKIP,
        "force_cpu": FORCE_CPU
    })

@app.route("/upload", methods=["POST"])
def upload_video():
    """
    Handles safe upload + resets worker state properly before starting a new video.
    """
    global video_path, processed_frames, processing, frame_q, result_q, stop_event

    if "video" not in request.files or request.files["video"].filename == "":
        return jsonify({"error": "No file"}), 400

    f = request.files["video"]
    if not allowed_file(f.filename):
        return jsonify({"error": "Invalid file type"}), 400

    ensure_dirs()

    filename = secure_filename(f.filename)
    path = os.path.join(UPLOAD_DIR, filename)

    if os.path.exists(path):
        base, ext = os.path.splitext(filename)
        filename = f"{base}_{int(time.time())}{ext}"
        path = os.path.join(UPLOAD_DIR, filename)

    # clean old session
    try:
        stop_event.set()
        time.sleep(0.05)

        flush_queue(frame_q)
        flush_queue(result_q)

        frame_q = queue.Queue(maxsize=FRAME_QUEUE_MAX)
        result_q = queue.Queue(maxsize=RESULT_QUEUE_MAX)

        stop_event.clear()
    except Exception:
        pass

    # save file
    f.save(path)
    video_path = path

    with state_lock:
        detection_logs.clear()

    processed_frames = 0
    processing = False

    print(f"📂 Video uploaded: {path}")

    return jsonify({"status": "uploaded", "path": path})


@app.route("/video_feed")
def video_feed():
    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "No video uploaded"}), 404

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/detection_count")
def detection_count():
    with state_lock:
        det = len(detection_logs)
        total_area = round(sum(l.get("area_m2", 0.0) for l in detection_logs), 2) if detection_logs else 0.0
        avg = round(sum(l.get("confidence", 0.0) for l in detection_logs) / det, 3) if det > 0 else 0.0

    return jsonify({
        "detections": det,
        "total_area": total_area,
        "avg_confidence": avg
    })


@app.route("/progress")
def progress():
    with fps_lock:
        proc = processed_frames
        fps = processing_fps

    progress_pct = (proc / total_frames * 100) if total_frames > 0 else 0.0
    eta = (total_frames - proc) / (fps + 1e-4) if fps > 0 else None

    return jsonify({
        "processed_frames": int(proc),
        "total_frames": int(total_frames),
        "progress_percent": round(progress_pct, 2),
        "video_fps": round(video_fps, 2),
        "processing_fps": round(fps, 2),
        "estimated_time_left_s": round(eta, 1) if eta is not None else None
    })


@app.route("/processing_status")
def processing_status():
    return jsonify({"processing": processing})


@app.route("/last_report")
def last_report_route():
    with state_lock:
        if not last_report.get("csv_path"):
            return jsonify({"error": "No report yet"}), 404
        return jsonify(last_report)


@app.route("/reports/<path:filename>")
def serve_report(filename):
    requested = os.path.abspath(os.path.join(REPORT_DIR, filename))
    base = os.path.abspath(REPORT_DIR)

    # security: ensure path containment
    try:
        if os.path.commonpath([base, requested]) != base:
            return "Forbidden", 403
    except Exception:
        return "Forbidden", 403

    if not os.path.exists(requested):
        return "Not found", 404

    return send_file(
        requested,
        mimetype="text/csv",
        as_attachment=True,
        download_name=os.path.basename(requested),
    )


@app.route("/export_csv")
def export_csv():
    with state_lock:
        logs = list(detection_logs)
        last_csv = last_report.get("csv_path")

    if logs:
        video_name = current_video or "session"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{secure_filename(video_name)}_detections_{ts}.csv"

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=[
            "frame_number","video_time_s","timestamp",
            "confidence","area_m2","severity"
        ])
        writer.writeheader()
        writer.writerows(logs)
        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype="text/csv",
            as_attachment=True,
            download_name=filename,
        )

    elif last_csv and os.path.exists(last_csv):
        return send_file(
            last_csv,
            mimetype="text/csv",
            as_attachment=True,
            download_name=os.path.basename(last_csv),
        )

    return "No detections yet.", 404


@app.route("/download_archive/<path:filename>")
def download_archive(filename):
    requested = os.path.abspath(os.path.join(ARCHIVE_DIR, filename))
    base = os.path.abspath(ARCHIVE_DIR)

    try:
        if os.path.commonpath([base, requested]) != base:
            return "Forbidden", 403
    except Exception:
        return "Forbidden", 403

    if not os.path.exists(requested):
        return "Not found", 404

    return send_file(
        requested,
        mimetype="application/zip",
        as_attachment=True,
        download_name=os.path.basename(requested),
    )


@app.route("/health")
def health():
    with state_lock:
        det = len(detection_logs)
        avg_conf = round(sum(l.get("confidence", 0.0) for l in detection_logs) / det, 3) if det > 0 else 0.0
        csv_ready = bool(last_report.get("csv_path"))
        archive_ready = bool(last_report.get("archive_path"))

    with fps_lock:
        fps = processing_fps

    return jsonify({
        "device": DEVICE,
        "gpu": USE_CUDA,
        "frame_skip": FRAME_SKIP,
        "processing": processing,
        "detections": det,
        "avg_conf": avg_conf,
        "fps": fps,
        "current_video": current_video,
        "csv_ready": csv_ready,
        "csv": last_report.get("csv_url"),
        "archive_ready": archive_ready,
        "archive": last_report.get("archive_url")
    })


# ---------------------------------------------------------
#                     STARTUP
# ---------------------------------------------------------
if __name__ == "__main__":
    ensure_dirs()

    threading.Thread(target=lambda: auto_cleanup(UPLOAD_DIR, MAX_FILES_UPLOADS), daemon=True).start()
    threading.Thread(target=lambda: auto_cleanup(REPORT_DIR, MAX_FILES_REPORTS), daemon=True).start()
    threading.Thread(target=lambda: auto_cleanup(ARCHIVE_DIR, MAX_FILES_ARCHIVES), daemon=True).start()

    def fps_monitor():
        global processed_frames, processing_fps
        prev = processed_frames
        while True:
            time.sleep(1)
            now = processed_frames
            with fps_lock:
                processing_fps = now - prev
            prev = now

    threading.Thread(target=fps_monitor, daemon=True).start()

    port = int(os.environ.get("PORT", 7860))
    print(f"🚀 Flask running on port {port} (device={DEVICE}, FRAME_SKIP={FRAME_SKIP})")

    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
