#!/usr/bin/env python3
"""
app.py — Multi-session Pothole Detection Backend (Flask)
Updated with Ultralytics YOLO Tracking (ByteTrack/BoT-SORT) to prevent duplicate counting.
Includes Session Memory to track unique IDs.
"""
import os
import time
import uuid
import threading
import traceback
import zipfile
import csv
import io
from datetime import datetime
from functools import partial

from flask import Flask, jsonify, request, Response, send_file, abort, make_response
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Optional heavy deps
try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

try:
    import torch
except Exception:
    torch = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# ---------------------------
# Configuration (tweakable)
# ---------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
REPORT_DIR = os.environ.get("REPORT_DIR", "reports")
ARCHIVE_DIR = os.path.join(REPORT_DIR, "archives")
ALLOWED_EXT = {"mp4", "mov", "avi", "mkv", "webm"}
FRAME_WIDTH = int(os.environ.get("FRAME_WIDTH", 1280))
FRAME_HEIGHT = int(os.environ.get("FRAME_HEIGHT", 720))
INFERENCE_SIZE = int(os.environ.get("INFERENCE_SIZE", 960)) # 640 is standard for YOLO
PIXELS_PER_METER = float(os.environ.get("PIXELS_PER_METER", 100))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.28))
IOU_THRESHOLD = float(os.environ.get("IOU_THRESHOLD", 0.45))
FRAME_QUEUE_MAX = int(os.environ.get("FRAME_QUEUE_MAX", 6))
RESULT_QUEUE_MAX = int(os.environ.get("RESULT_QUEUE_MAX", 6))
FRAME_SKIP = int(os.environ.get("FRAME_SKIP", 0))  # 0 => process all frames
MAX_SESSION_COUNT = int(os.environ.get("MAX_SESSION_COUNT", 12))
MAX_FILES_UPLOADS = int(os.environ.get("MAX_FILES_UPLOADS", 20))
MAX_FILES_REPORTS = int(os.environ.get("MAX_FILES_REPORTS", 50))
MAX_FILES_ARCHIVES = int(os.environ.get("MAX_FILES_ARCHIVES", 20))
SESSION_TTL = int(os.environ.get("SESSION_TTL", 60 * 60 * 3))  # seconds, default 3 hours

# Notification (optional)
NOTIFY_EMAIL = os.environ.get("NOTIFY_EMAIL")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")

# Device selection
FORCE_CPU = os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes")
DEVICE = "cpu"
USE_CUDA = False
if not FORCE_CPU and torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
    USE_CUDA = True
    DEVICE = "cuda:0"

# Public host for building absolute links (for background threads)
PUBLIC_HOST = os.environ.get("PUBLIC_HOST", "").rstrip("/")

# Create app
app = Flask(__name__)
# Allow CORS broadly so Netlify or other origins can fetch
CORS(app, resources={r"/*": {"origins": "*"}})

# Ensure directories
for d in (UPLOAD_DIR, REPORT_DIR, ARCHIVE_DIR):
    os.makedirs(d, exist_ok=True)


# ---------------------------
# Utilities
# ---------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def now_ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def build_absolute_url(path: str):
    """
    Build background-safe absolute URL to a path under the server.
    If PUBLIC_HOST is provided, use it. Otherwise return relative path.
    """
    if not path:
        return path
    # ensure leading slash
    if not path.startswith("/"):
        path = "/" + path
    if PUBLIC_HOST:
        return PUBLIC_HOST.rstrip("/") + path
    return path  # frontend can prefix backend base if needed

def auto_cleanup(directory, keep):
    """Keep only the most recent 'keep' files in directory."""
    try:
        files = sorted(
            [os.path.join(directory, f) for f in os.listdir(directory)],
            key=os.path.getmtime,
            reverse=True,
        )
        for f in files[keep:]:
            try:
                os.remove(f)
            except Exception:
                pass
    except Exception:
        pass

def safe_commonpath(base, requested):
    """Return True if requested is within base directory."""
    try:
        return os.path.commonpath([base, requested]) == base
    except Exception:
        # fallback conservative: deny
        return False

# ---------------------------
# Lightweight YOLO wrapper
# ---------------------------
model = None
if YOLO is None:
    print("ultralytics not installed; running in pass-through mode.")
else:
    try:
        print("Loading YOLO model...", MODEL_PATH)
        model = YOLO(MODEL_PATH)
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
        print("Model loaded.")
    except Exception as e:
        print("Failed loading model:", e)
        traceback.print_exc()
        model = None

def to_numpy(x):
    try:
        if hasattr(x, "cpu"):
            return x.cpu().detach().numpy()
        return np.array(x)
    except Exception:
        try:
            return np.array(x)
        except Exception:
            return None

def extract_boxes_from_result(r):
    try:
        if hasattr(r, "boxes"):
            return r.boxes
    except Exception:
        pass
    try:
        if isinstance(r, (list, tuple)) and len(r) > 0 and hasattr(r[0], "boxes"):
            return r[0].boxes
    except Exception:
        pass
    try:
        if hasattr(r, "pred") and len(r.pred) > 0:
            return r.pred[0]
    except Exception:
        pass
    return []


# ---------------------------
# Session management
# ---------------------------
import queue
sessions = {}
sessions_lock = threading.Lock()

def create_session_entry(video_path, original_name):
    sid = uuid.uuid4().hex[:12]
    entry = {
        "session_id": sid,
        "video_path": video_path,
        "video_name": original_name,
        "frame_q": queue.Queue(maxsize=FRAME_QUEUE_MAX),
        "result_q": queue.Queue(maxsize=RESULT_QUEUE_MAX),
        "stop_event": threading.Event(),
        "worker": None,
        "detection_logs": [],
        
        # --- NEW: Memory for Tracking IDs ---
        "seen_ids": set(), 
        
        "csv_path": None,
        "csv_url": None,
        "archive_path": None,
        "archive_url": None,
        "total_frames": 0,
        "processed_frames": 0,
        "video_fps": 0.0,
        "processing_fps": 0.0,
        "created_at": time.time(),
        "last_activity": time.time(),
        "streaming": False,
        "archiving": False
    }
    with sessions_lock:
        # limit sessions
        if len(sessions) >= MAX_SESSION_COUNT:
            # remove oldest
            oldest = sorted(sessions.values(), key=lambda x: x["created_at"])[0]
            cleanup_session(oldest["session_id"])
        sessions[sid] = entry

    # start per-session FPS monitor thread
    def fps_monitor(sid_local):
        prev = 0
        while True:
            time.sleep(1)
            with sessions_lock:
                s = sessions.get(sid_local)
                if not s:
                    break
                now = s.get("processed_frames", 0)
                s["processing_fps"] = now - prev
                prev = now
                # update last_activity periodically if processing ongoing
                if s.get("processed_frames", 0) > 0:
                    s["last_activity"] = time.time()
                # stop monitor when no worker and not streaming
                w = s.get("worker")
                if (w is None or not w.is_alive()) and not s.get("streaming", False):
                    # allow final small grace period
                    # break after a couple of seconds idle
                    idle_since = time.time() - s.get("last_activity", s.get("created_at", time.time()))
                    if idle_since > 2:
                        break
        # monitor exits
    t = threading.Thread(target=fps_monitor, args=(sid,), daemon=True)
    t.start()

    return entry

def cleanup_session(sid):
    """Stop worker and clear queues and remove session record."""
    with sessions_lock:
        entry = sessions.get(sid)
        if not entry:
            return
        try:
            entry["stop_event"].set()
            # join worker if alive (allow up to 2s)
            w = entry.get("worker")
            if w and w.is_alive():
                w.join(timeout=2.0)
        except Exception:
            pass

        try:
            # flush queues
            while not entry["frame_q"].empty():
                entry["frame_q"].get_nowait()
            while not entry["result_q"].empty():
                entry["result_q"].get_nowait()
        except Exception:
            pass

        # remove record
        try:
            del sessions[sid]
        except Exception:
            pass

# ---------------------------
# Drawing + logging (UPDATED FOR TRACKING)
# ---------------------------
def get_severity(area_m2):
    if area_m2 < 0.5:
        return "Minor"
    if area_m2 < 1.5:
        return "Moderate"
    if area_m2 < 3.0:
        return "Major"
    return "Severe"

def draw_boxes_and_log(frame, boxes_obj, frame_number, video_time, seen_ids_set, conf_thresh=CONFIDENCE_THRESHOLD):
    """
    Draws boxes and IDs. 
    Only logs to the CSV list if the ID is NEW (not in seen_ids_set).
    Visual colors remain based on severity for good UX.
    """
    logs = []
    h, w = frame.shape[:2]
    
    if boxes_obj is None:
        return frame, logs

    # Iterate over boxes
    try:
        # Convert to list if iterable
        iterable = boxes_obj if isinstance(boxes_obj, (list, tuple)) else list(boxes_obj)
    except:
        return frame, logs

    for box in iterable:
        try:
            # 1. Get Tracking ID
            track_id = None
            if hasattr(box, 'id') and box.id is not None:
                track_id = int(box.id.item())

            # 2. Get Confidence
            conf = 0.0
            if hasattr(box, 'conf'):
                 conf = float(box.conf[0].cpu().numpy())
            if conf < conf_thresh:
                continue

            # 3. Get Coordinates
            xyxy = None
            if hasattr(box, "xyxy"):
                arr = to_numpy(box.xyxy)
                if arr is not None and len(arr) > 0:
                     xyxy = arr[0] if arr.ndim == 2 else arr
            
            if xyxy is None or len(xyxy) < 4:
                continue

            x1, y1, x2, y2 = [int(v) for v in xyxy[:4]]
            
            # 4. Calculate Area
            wbox = x2 - x1
            hbox = y2 - y1
            area_px = wbox * hbox
            
            # Filter noise
            if area_px < 800 or area_px > (w * h * 0.35):
                continue
            
            # Filter aspect ratio
            ratio = wbox / (hbox + 1e-6)
            if ratio < 0.5 or ratio > 3.8:
                continue

            area_m2 = round(area_px / (PIXELS_PER_METER ** 2), 2)
            severity = get_severity(area_m2)

            # 5. Color Logic (Based on Severity for UX)
            color = (0, 140, 255) # Orange default
            if severity == "Minor": color = (0, 255, 0)       # Green
            elif severity == "Moderate": color = (0, 215, 255) # Cyan/Yellow
            elif severity == "Major": color = (0, 140, 255)    # Orange
            elif severity == "Severe": color = (0, 0, 255)     # Red

            # 6. Logging Logic (Only if NEW ID)
            if track_id is not None:
                if track_id not in seen_ids_set:
                    # NEW DETECTION
                    seen_ids_set.add(track_id)
                    logs.append({
                        "frame_number": int(frame_number),
                        "video_time_s": round(float(video_time or 0.0), 2),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "confidence": round(conf, 3),
                        "area_m2": area_m2,
                        "severity": severity,
                        "track_id": track_id
                    })
                # If ID is in set, we skip logging, but we still draw below
            else:
                # Fallback if tracking fails (no ID), optional log
                pass

            # 7. Draw Visuals (Always draw for UX)
            if cv2 is not None:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Label with ID
                label = f"ID:{track_id} | {area_m2}m2" if track_id is not None else f"{area_m2}m2"
                
                # Text Background
                (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w_text, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        except Exception:
            # traceback.print_exc()
            continue

    return frame, logs

# ---------------------------
# Worker (per-session)
# ---------------------------
def session_infer_worker(sid):
    with sessions_lock:
        entry = sessions.get(sid)
    if not entry:
        return

    print(f"[{sid}] Worker started (device={DEVICE})")
    stop_ev = entry["stop_event"]
    fq = entry["frame_q"]
    rq = entry["result_q"]
    
    # Get reference to this session's memory
    current_seen_ids = entry["seen_ids"]

    try:
        while not stop_ev.is_set():
            try:
                fid, frame, video_time = fq.get(timeout=0.5)
            except Exception:
                continue

            logs = []
            out_frame = frame
            try:
                if model is None:
                    out_frame = frame
                else:
                    r = None
                    try:
                        # --- UPDATED: Use model.track() with persist=True ---
                        r = model.track(source=frame, 
                                      persist=True,  # Critical for tracking IDs
                                      conf=CONFIDENCE_THRESHOLD,
                                      iou=IOU_THRESHOLD, 
                                      imgsz=INFERENCE_SIZE,
                                      device=DEVICE, 
                                      verbose=False)
                    except Exception:
                        try:
                            # Fallback to simple predict if track fails
                            r = model(frame)
                        except Exception:
                            r = None

                    if r is None:
                        out_frame = frame
                    else:
                        boxes = extract_boxes_from_result(r)
                        frame_copy = frame.copy()
                        # Pass seen_ids to drawing function
                        out_frame, logs = draw_boxes_and_log(frame_copy, boxes, fid, video_time, current_seen_ids)
            except Exception:
                traceback.print_exc()
                out_frame = frame

            if logs:
                with sessions_lock:
                    s = sessions.get(sid)
                    if s:
                        s["detection_logs"].extend(logs)
                        s["last_activity"] = time.time()

            with sessions_lock:
                s = sessions.get(sid)
                if s:
                    s["processed_frames"] += 1
                    s["last_activity"] = time.time()

            try:
                rq.put_nowait((fid, out_frame))
            except Exception:
                try:
                    rq.get_nowait()
                    rq.put_nowait((fid, out_frame))
                except Exception:
                    pass

    finally:
        print(f"[{sid}] Worker stopped")

# ---------------------------
# CSV & ZIP saving (per-session)
# ---------------------------
def save_csv_for_session(sid):
    with sessions_lock:
        s = sessions.get(sid)
        if not s:
            return None
        logs = list(s["detection_logs"])
        video_name = secure_filename(s.get("video_name") or f"session_{sid}")

    timestamp = now_ts()
    filename = f"{video_name}_detections_{timestamp}.csv"
    filepath = os.path.join(REPORT_DIR, filename)

    try:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            # Added 'track_id' to CSV columns
            writer = csv.DictWriter(f, fieldnames=["frame_number", "video_time_s", "timestamp", "confidence", "area_m2", "severity", "track_id"])
            writer.writeheader()
            if logs:
                writer.writerows(logs)

        total_det = len(logs)
        total_area = round(sum(l.get("area_m2", 0.0) for l in logs), 2) if logs else 0.0
        avg_conf = round(sum(l.get("confidence", 0.0) for l in logs) / total_det, 3) if total_det > 0 else 0.0
        csv_url = build_absolute_url(f"/reports/{os.path.basename(filepath)}")

        with sessions_lock:
            s = sessions.get(sid)
            if s:
                s["csv_path"] = filepath
                s["csv_url"] = csv_url
                s["total_detections"] = total_det
                s["total_area"] = total_area
                s["average_confidence"] = avg_conf
                s["last_activity"] = time.time()

        auto_cleanup(REPORT_DIR, MAX_FILES_REPORTS)
        print(f"[{sid}] CSV saved: {filepath}")
        return filepath
    except Exception:
        traceback.print_exc()
        return None

def create_archive_for_session(sid):
    # mark archiving state
    with sessions_lock:
        s = sessions.get(sid)
        if not s:
            return None
        s["archiving"] = True

    try:
        # brief wait for CSV to be available
        time.sleep(0.5)

        with sessions_lock:
            s = sessions.get(sid)
            if not s:
                return None
            video_path = s.get("video_path")
            video_name = secure_filename(s.get("video_name") or f"session_{sid}")
            csv_path = s.get("csv_path")

        ts = now_ts()
        zip_name = f"{video_name}_archive_{ts}.zip"
        zip_path = os.path.join(ARCHIVE_DIR, zip_name)

        with zipfile.ZipFile(zip_path, "w") as zf:
            if video_path and os.path.exists(video_path):
                zf.write(video_path, os.path.basename(video_path))
            if csv_path and os.path.exists(csv_path):
                zf.write(csv_path, os.path.basename(csv_path))

        archive_url = build_absolute_url(f"/download_archive/{os.path.basename(zip_path)}")
        with sessions_lock:
            s = sessions.get(sid)
            if s:
                s["archive_path"] = zip_path
                s["archive_url"] = archive_url
                s["last_activity"] = time.time()

        auto_cleanup(ARCHIVE_DIR, MAX_FILES_ARCHIVES)
        print(f"[{sid}] Archive created: {zip_path}")
        return zip_path
    except Exception:
        traceback.print_exc()
        return None
    finally:
        with sessions_lock:
            s = sessions.get(sid)
            if s:
                s["archiving"] = False

# ---------------------------
# Video frame generator (per session)
# ---------------------------
def mjpeg_stream_for_session(sid):
    with sessions_lock:
        s = sessions.get(sid)
        if not s:
            yield b''
            return

    video_path = s["video_path"]
    if not os.path.exists(video_path):
        yield b''
        return

    cap = None
    try:
        if cv2 is None:
            print("OpenCV not available; cannot stream video.")
            return
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        with sessions_lock:
            s = sessions.get(sid)
            if s:
                s["total_frames"] = total_frames
                s["video_fps"] = cap.get(cv2.CAP_PROP_FPS) or 0.0
                s["streaming"] = True
                s["last_activity"] = time.time()

        # ensure worker exists
        with sessions_lock:
            s = sessions.get(sid)
            if not s:
                return
            if s.get("worker") is None or not s.get("worker").is_alive():
                w = threading.Thread(target=session_infer_worker, args=(sid,), daemon=True)
                s["worker"] = w
                w.start()

        frame_no = 0
        last_sent_frame = None
        start_t = time.time()
        while cap.isOpened() and not s["stop_event"].is_set():
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1

            if FRAME_SKIP > 0 and (frame_no % (FRAME_SKIP + 1)) != 1:
                continue

            try:
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            except Exception:
                pass

            video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # enqueue frame
            try:
                s["frame_q"].put_nowait((frame_no, frame, video_time))
            except Exception:
                try:
                    s["frame_q"].get_nowait()
                    s["frame_q"].put_nowait((frame_no, frame, video_time))
                except Exception:
                    pass

            out = frame
            try:
                fid_out = s["result_q"].get(timeout=0.05)
                if fid_out:
                    _, out = fid_out
                    if out is None:
                        out = frame
            except Exception:
                if last_sent_frame is not None:
                    out = last_sent_frame
                else:
                    out = frame

            try:
                ret2, buf = cv2.imencode(".jpg", out)
                if not ret2:
                    continue
                mjpeg_frame = (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + buf.tobytes()
                    + b"\r\n"
                )
                last_sent_frame = out
                yield mjpeg_frame
            except Exception:
                traceback.print_exc()
                continue

            # update activity
            with sessions_lock:
                s = sessions.get(sid)
                if s:
                    s["last_activity"] = time.time()

            time.sleep(0.002)

    finally:
        try:
            if cap:
                cap.release()
        except Exception:
            pass

        # finalized: video ended or stop_event set
        # always create CSV and ZIP (CSV first, then archive in background)
        try:
            save_csv_for_session(sid)
            # run archive creation in background thread
            threading.Thread(target=create_archive_for_session, args=(sid,), daemon=True).start()
        except Exception:
            traceback.print_exc()

        with sessions_lock:
            s = sessions.get(sid)
            if s:
                s["streaming"] = False
                s["last_activity"] = time.time()

        print(f"[{sid}] Streaming finished.")

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def index():
    return jsonify({"status": "ok", "message": "Pothole Detection Backend (Multi-session) running."})

@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No file 'video' in request"}), 400
    f = request.files["video"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "Invalid file type"}), 400

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filename = secure_filename(f.filename)
    path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(path):
        base, ext = os.path.splitext(filename)
        filename = f"{base}_{int(time.time())}{ext}"
        path = os.path.join(UPLOAD_DIR, filename)

    f.save(path)

    entry = create_session_entry(path, f.filename)
    sid = entry["session_id"]

    print(f"[{sid}] Uploaded {path}")
    # schedule cleanup of old uploads
    threading.Thread(target=lambda: auto_cleanup(UPLOAD_DIR, MAX_FILES_UPLOADS), daemon=True).start()

    return jsonify({"status": "uploaded", "session_id": sid, "video_name": f.filename})

@app.route("/video_feed/<sid>")
def video_feed(sid):
    with sessions_lock:
        if sid not in sessions:
            return jsonify({"error": "Invalid session id"}), 404

    # Return MJPEG stream with permissive CORS header so remote frontends can render it in <img>
    response = Response(mjpeg_stream_for_session(sid), mimetype="multipart/x-mixed-replace; boundary=frame")
    # Add CORS header explicitly for image streaming
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

@app.route("/progress/<sid>")
def progress(sid):
    with sessions_lock:
        s = sessions.get(sid)
        if not s:
            return jsonify({"error": "Invalid session id"}), 404
        proc = s.get("processed_frames", 0)
        total = s.get("total_frames", 0)
        fps = s.get("processing_fps", 0.0)
        video_fps = s.get("video_fps", 0.0)

    progress_pct = (proc / total * 100.0) if total > 0 else 0.0
    eta = None
    if fps and fps > 0:
        eta = (total - proc) / (fps + 1e-6)

    return jsonify({
        "processed_frames": int(proc),
        "total_frames": int(total),
        "progress_percent": round(progress_pct, 2),
        "video_fps": round(video_fps, 2),
        "processing_fps": round(fps, 2),
        "estimated_time_left_s": round(eta, 1) if eta is not None else None
    })

@app.route("/detection_count/<sid>")
def detection_count(sid):
    with sessions_lock:
        s = sessions.get(sid)
        if not s:
            return jsonify({"error": "Invalid session id"}), 404
        logs = list(s.get("detection_logs", []))

    det = len(logs)
    total_area = round(sum(l.get("area_m2", 0.0) for l in logs), 2) if logs else 0.0
    avg = round(sum(l.get("confidence", 0.0) for l in logs) / det, 3) if det > 0 else 0.0

    return jsonify({"detections": det, "total_area": total_area, "avg_confidence": avg})

@app.route("/processing_status/<sid>")
def processing_status(sid):
    with sessions_lock:
        s = sessions.get(sid)
        if not s:
            return jsonify({"error": "Invalid session id"}), 404
        # consider processing active if worker alive OR streaming is True OR archiving is True
        processing = bool((s.get("worker") and s.get("worker").is_alive()) or s.get("streaming") or s.get("archiving"))
    return jsonify({"processing": processing})

@app.route("/last_report/<sid>")
def last_report(sid):
    with sessions_lock:
        s = sessions.get(sid)
        if not s:
            return jsonify({"error": "Invalid session id"}), 404
        if not s.get("csv_path"):
            return jsonify({"error": "No report yet"}), 404
        return jsonify({
            "csv_path": s.get("csv_path"),
            "csv_url": s.get("csv_url"),
            "archive_path": s.get("archive_path"),
            "archive_url": s.get("archive_url"),
            "total_detections": s.get("total_detections", 0),
            "total_area": s.get("total_area", 0.0),
            "average_confidence": s.get("average_confidence", 0.0),
            "video_name": s.get("video_name")
        })

@app.route("/reports/<path:filename>")
def serve_report(filename):
    requested = os.path.abspath(os.path.join(REPORT_DIR, filename))
    base = os.path.abspath(REPORT_DIR)
    if not safe_commonpath(base, requested):
        return "Forbidden", 403
    if not os.path.exists(requested):
        return "Not found", 404
    return send_file(requested, mimetype="text/csv", as_attachment=True, download_name=os.path.basename(requested))

@app.route("/export_csv/<sid>")
def export_csv(sid):
    with sessions_lock:
        s = sessions.get(sid)
        if not s:
            return jsonify({"error": "Invalid session id"}), 404
        logs = list(s.get("detection_logs", []))
        csv_path = s.get("csv_path")

    if logs:
        output = io.StringIO()
        # Updated fieldnames to include track_id
        writer = csv.DictWriter(output, fieldnames=["frame_number","video_time_s","timestamp","confidence","area_m2","severity", "track_id"])
        writer.writeheader()
        writer.writerows(logs)
        output.seek(0)
        fname = f"{secure_filename(s.get('video_name') or sid)}_detections_{now_ts()}.csv"
        return send_file(io.BytesIO(output.getvalue().encode()), mimetype="text/csv", as_attachment=True, download_name=fname)
    elif csv_path and os.path.exists(csv_path):
        return send_file(csv_path, mimetype="text/csv", as_attachment=True, download_name=os.path.basename(csv_path))
    return "No detections yet.", 404

@app.route("/download_archive/<path:filename>")
def download_archive(filename):
    requested = os.path.abspath(os.path.join(ARCHIVE_DIR, filename))
    base = os.path.abspath(ARCHIVE_DIR)
    if not safe_commonpath(base, requested):
        return "Forbidden", 403
    if not os.path.exists(requested):
        return "Not found", 404
    return send_file(requested, mimetype="application/zip", as_attachment=True, download_name=os.path.basename(requested))

@app.route("/health")
def health():
    with sessions_lock:
        active = len(sessions)
    return jsonify({"device": DEVICE, "gpu": USE_CUDA, "active_sessions": active})

# ---------------------------
# Background housekeeping: prune idle sessions
# ---------------------------
def session_ttl_pruner():
    while True:
        time.sleep(60)
        now = time.time()
        with sessions_lock:
            to_remove = []
            for sid, s in list(sessions.items()):
                idle = now - s.get("last_activity", s.get("created_at", now))
                if idle > SESSION_TTL:
                    to_remove.append(sid)
            for sid in to_remove:
                print(f"[{sid}] TTL exceeded, cleaning up session.")
                cleanup_session(sid)

# ---------------------------
# Startup helpers
# ---------------------------
if __name__ == "__main__":
    # housekeeping threads
    threading.Thread(target=lambda: auto_cleanup(UPLOAD_DIR, MAX_FILES_UPLOADS), daemon=True).start()
    threading.Thread(target=lambda: auto_cleanup(REPORT_DIR, MAX_FILES_REPORTS), daemon=True).start()
    threading.Thread(target=lambda: auto_cleanup(ARCHIVE_DIR, MAX_FILES_ARCHIVES), daemon=True).start()
    threading.Thread(target=session_ttl_pruner, daemon=True).start()

    port = int(os.environ.get("PORT", 7860))
    host = "0.0.0.0"
    print(f"Starting Flask on {host}:{port} (device={DEVICE})")
    app.run(host=host, port=port, debug=False, threaded=True)