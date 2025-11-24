#!/usr/bin/env python3
"""
app.py — Multi-session Pothole Detection Backend (Flask)
INTEGRATED: Working Snapshot Logic + Robust GPS (GPX) Support + Fixed UI Data
"""
import os
import time
import uuid
import threading
import traceback
import zipfile
import csv
import io
import math
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
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
SNAPSHOT_DIR = os.path.join(REPORT_DIR, "snapshots")

ALLOWED_EXT = {"mp4", "mov", "avi", "mkv", "webm"}
FRAME_WIDTH = int(os.environ.get("FRAME_WIDTH", 1280))
FRAME_HEIGHT = int(os.environ.get("FRAME_HEIGHT", 720))
INFERENCE_SIZE = int(os.environ.get("INFERENCE_SIZE", 960)) # 640 is standard for YOLO
PIXELS_PER_METER = float(os.environ.get("PIXELS_PER_METER", 100))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.28))
IOU_THRESHOLD = float(os.environ.get("IOU_THRESHOLD", 0.45))
FRAME_QUEUE_MAX = int(os.environ.get("FRAME_QUEUE_MAX", 6))
RESULT_QUEUE_MAX = int(os.environ.get("RESULT_QUEUE_MAX", 6))
FRAME_SKIP = int(os.environ.get("FRAME_SKIP", 0))
MAX_SESSION_COUNT = int(os.environ.get("MAX_SESSION_COUNT", 12))
MAX_FILES_UPLOADS = int(os.environ.get("MAX_FILES_UPLOADS", 20))
MAX_FILES_REPORTS = int(os.environ.get("MAX_FILES_REPORTS", 50))
MAX_FILES_ARCHIVES = int(os.environ.get("MAX_FILES_ARCHIVES", 20))
SESSION_TTL = int(os.environ.get("SESSION_TTL", 60 * 60 * 3))

# Device selection
FORCE_CPU = os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes")
DEVICE = "cpu"
USE_CUDA = False
if not FORCE_CPU and torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
    USE_CUDA = True
    DEVICE = "cuda:0"

# Public host for building absolute links
PUBLIC_HOST = os.environ.get("PUBLIC_HOST", "").rstrip("/")

# Create app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Ensure directories
for d in (UPLOAD_DIR, REPORT_DIR, ARCHIVE_DIR, SNAPSHOT_DIR):
    os.makedirs(d, exist_ok=True)


# ---------------------------
# Utilities
# ---------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def now_ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def build_absolute_url(path: str):
    if not path: return path
    if not path.startswith("/"): path = "/" + path
    if PUBLIC_HOST: return PUBLIC_HOST.rstrip("/") + path
    return path

def auto_cleanup(directory, keep):
    try:
        files = sorted(
            [os.path.join(directory, f) for f in os.listdir(directory)],
            key=os.path.getmtime,
            reverse=True,
        )
        for f in files[keep:]:
            try: os.remove(f)
            except Exception: pass
    except Exception: pass

def safe_commonpath(base, requested):
    try: return os.path.commonpath([base, requested]) == base
    except Exception: return False

def to_numpy(x):
    try:
        if hasattr(x, "cpu"): return x.cpu().detach().numpy()
        return np.array(x)
    except Exception:
        try: return np.array(x)
        except Exception: return None

def extract_boxes_from_result(r):
    try:
        if hasattr(r, "boxes"): return r.boxes
    except Exception: pass
    try:
        if isinstance(r, (list, tuple)) and len(r) > 0 and hasattr(r[0], "boxes"): return r[0].boxes
    except Exception: pass
    try:
        if hasattr(r, "pred") and len(r.pred) > 0: return r.pred[0]
    except Exception: pass
    return []

# ---------------------------
# GPX / GPS Handler (Robust)
# ---------------------------
class GPXHandler:
    def __init__(self, gpx_path):
        self.points = [] 
        self.valid = False
        try:
            if gpx_path: self._parse(gpx_path)
        except Exception as e:
            print(f"GPX Init Error: {e}")

    def _parse(self, gpx_path):
        if not os.path.exists(gpx_path): return
        try:
            tree = ET.parse(gpx_path)
            root = tree.getroot()
            # Support with and without namespace
            trkpts = root.findall(".//{http://www.topografix.com/GPX/1/1}trkpt")
            if not trkpts: trkpts = root.findall(".//trkpt")

            parsed_points = []
            for pt in trkpts:
                try:
                    lat = float(pt.get("lat"))
                    lon = float(pt.get("lon"))
                    time_elem = pt.find("{http://www.topografix.com/GPX/1/1}time")
                    if time_elem is None: time_elem = pt.find("time")
                    
                    if time_elem is not None and time_elem.text:
                        t_str = time_elem.text
                        dt = None
                        # Basic ISO8601 parsing
                        try:
                            clean_ts = t_str.replace("Z", "").split(".")[0]
                            dt = datetime.strptime(clean_ts, "%Y-%m-%dT%H:%M:%S")
                        except: pass
                        if dt: parsed_points.append((dt, lat, lon))
                except: continue

            parsed_points.sort(key=lambda x: x[0])
            if parsed_points:
                start_time = parsed_points[0][0]
                for dt, lat, lon in parsed_points:
                    rel_sec = (dt - start_time).total_seconds()
                    self.points.append((rel_sec, lat, lon))
                self.valid = True
                print(f"GPX Loaded: {len(self.points)} points.")
        except Exception as e:
            print(f"GPX Parse Error: {e}")

    def get_location_at(self, video_time_s):
        if not self.valid or not self.points: return None, None
        # Boundary checks
        if video_time_s <= self.points[0][0]: return self.points[0][1], self.points[0][2]
        if video_time_s >= self.points[-1][0]: return self.points[-1][1], self.points[-1][2]
        # Linear search
        for i in range(len(self.points) - 1):
            t1, lat1, lon1 = self.points[i]
            t2, lat2, lon2 = self.points[i+1]
            if t1 <= video_time_s <= t2:
                ratio = (video_time_s - t1) / (t2 - t1) if (t2 - t1) > 0 else 0
                lat = lat1 + (lat2 - lat1) * ratio
                lon = lon1 + (lon2 - lon1) * ratio
                return round(lat, 6), round(lon, 6)
        return None, None

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
            if hasattr(model, "to"): model.to(DEVICE)
        except: pass
        print("Model loaded.")
    except Exception as e:
        print("Failed loading model:", e)
        model = None

# ---------------------------
# Session management
# ---------------------------
import queue
sessions = {}
sessions_lock = threading.Lock()

def create_session_entry(video_path, original_name):
    sid = uuid.uuid4().hex[:12]
    
    session_snapshot_dir = os.path.join(SNAPSHOT_DIR, sid)
    os.makedirs(session_snapshot_dir, exist_ok=True)

    # Auto-detect GPX
    gpx_path = None
    try:
        base = os.path.splitext(video_path)[0]
        pot_gpx = base + ".gpx"
        if os.path.exists(pot_gpx):
            gpx_path = pot_gpx
            print(f"[{sid}] GPX Found: {gpx_path}")
    except: pass

    entry = {
        "session_id": sid,
        "video_path": video_path,
        "gpx_path": gpx_path, # Store path
        "video_name": original_name,
        "frame_q": queue.Queue(maxsize=FRAME_QUEUE_MAX),
        "result_q": queue.Queue(maxsize=RESULT_QUEUE_MAX),
        "stop_event": threading.Event(),
        "worker": None,
        "detection_logs": [],
        "seen_ids": set(),
        "snapshot_dir": session_snapshot_dir,
        "csv_path": None, "csv_url": None,
        "archive_path": None, "archive_url": None,
        "total_frames": 0, "processed_frames": 0,
        "video_fps": 0.0, "processing_fps": 0.0,
        "created_at": time.time(), "last_activity": time.time(),
        "streaming": False, "archiving": False
    }
    with sessions_lock:
        if len(sessions) >= MAX_SESSION_COUNT:
            oldest = sorted(sessions.values(), key=lambda x: x["created_at"])[0]
            cleanup_session(oldest["session_id"])
        sessions[sid] = entry

    def fps_monitor(sid_local):
        prev = 0
        while True:
            time.sleep(1)
            with sessions_lock:
                s = sessions.get(sid_local)
                if not s: break
                now = s.get("processed_frames", 0)
                s["processing_fps"] = now - prev
                prev = now
                if s.get("processed_frames", 0) > 0:
                    s["last_activity"] = time.time()
                w = s.get("worker")
                if (w is None or not w.is_alive()) and not s.get("streaming", False):
                    idle = time.time() - s.get("last_activity", s.get("created_at", time.time()))
                    if idle > 2: break
    t = threading.Thread(target=fps_monitor, args=(sid,), daemon=True)
    t.start()
    return entry

def cleanup_session(sid):
    with sessions_lock:
        entry = sessions.get(sid)
        if not entry: return
        try:
            entry["stop_event"].set()
            w = entry.get("worker")
            if w and w.is_alive(): w.join(timeout=2.0)
        except: pass
        try:
            while not entry["frame_q"].empty(): entry["frame_q"].get_nowait()
            while not entry["result_q"].empty(): entry["result_q"].get_nowait()
        except: pass
        try: del sessions[sid]
        except: pass

# ---------------------------
# Drawing + logging
# ---------------------------
def get_severity(area_m2):
    if area_m2 < 0.5: return "Minor"
    if area_m2 < 1.5: return "Moderate"
    if area_m2 < 3.0: return "Major"
    return "Severe"

def draw_boxes_and_log(frame, boxes_obj, frame_number, video_time, seen_ids_set, snapshot_dir, gps_coords=(None, None), conf_thresh=CONFIDENCE_THRESHOLD):
    logs = []
    h, w = frame.shape[:2]
    
    orig_frame_clean = None
    if cv2 is not None:
        orig_frame_clean = frame.copy()

    if boxes_obj is None: return frame, logs

    try:
        iterable = boxes_obj if isinstance(boxes_obj, (list, tuple)) else list(boxes_obj)
    except: return frame, logs

    for box in iterable:
        try:
            track_id = None
            if hasattr(box, 'id') and box.id is not None:
                track_id = int(box.id.item())

            conf = 0.0
            if hasattr(box, 'conf'): conf = float(box.conf[0].cpu().numpy())
            if conf < conf_thresh: continue

            xyxy = None
            if hasattr(box, "xyxy"):
                arr = to_numpy(box.xyxy)
                if arr is not None and len(arr) > 0: xyxy = arr[0] if arr.ndim == 2 else arr
            
            if xyxy is None or len(xyxy) < 4: continue
            x1, y1, x2, y2 = [int(v) for v in xyxy[:4]]
            
            wbox = x2 - x1
            hbox = y2 - y1
            area_px = wbox * hbox
            if area_px < 800 or area_px > (w * h * 0.35): continue
            
            ratio = wbox / (hbox + 1e-6)
            if ratio < 0.5 or ratio > 3.8: continue

            area_m2 = round(area_px / (PIXELS_PER_METER ** 2), 2)
            severity = get_severity(area_m2)

            color = (0, 140, 255)
            if severity == "Minor": color = (0, 255, 0)
            elif severity == "Moderate": color = (0, 215, 255)
            elif severity == "Major": color = (0, 140, 255)
            elif severity == "Severe": color = (0, 0, 255)

            if track_id is not None:
                if track_id not in seen_ids_set:
                    seen_ids_set.add(track_id)
                    
                    # Snapshot
                    snapshot_filename = f"frame_{frame_number}_id_{track_id}.jpg"
                    snapshot_path = os.path.join(snapshot_dir, snapshot_filename)
                    try:
                        if cv2 is not None and orig_frame_clean is not None:
                            snap_img = orig_frame_clean.copy()
                            cv2.rectangle(snap_img, (x1, y1), (x2, y2), color, 2)
                            label_snap = f"ID:{track_id} | {area_m2}m2"
                            (w_t, h_t), _ = cv2.getTextSize(label_snap, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(snap_img, (x1, y1 - 20), (x1 + w_t, y1), color, -1)
                            cv2.putText(snap_img, label_snap, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.imwrite(snapshot_path, snap_img)
                    except Exception as e:
                        print(f"Snap Error: {e}")
                        snapshot_filename = "error_saving.jpg"

                    lat, lon = gps_coords

                    logs.append({
                        "frame_number": int(frame_number),
                        "video_time_s": round(float(video_time or 0.0), 2),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "confidence": round(conf, 3),
                        "area_m2": area_m2,
                        "severity": severity,
                        "track_id": track_id,
                        "snapshot_file": snapshot_filename,
                        "latitude": lat if lat else "",
                        "longitude": lon if lon else ""
                    })

            if cv2 is not None:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{track_id} | {area_m2}m2"
                (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w_text, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        except: continue
    return frame, logs

# ---------------------------
# Worker
# ---------------------------
def session_infer_worker(sid):
    with sessions_lock:
        entry = sessions.get(sid)
    if not entry: return

    print(f"[{sid}] Worker started")
    stop_ev = entry["stop_event"]
    fq = entry["frame_q"]
    rq = entry["result_q"]
    current_seen_ids = entry["seen_ids"]
    current_snapshot_dir = entry["snapshot_dir"]
    gpx_path = entry.get("gpx_path")

    # Init GPS
    gps_handler = None
    try: gps_handler = GPXHandler(gpx_path)
    except: pass

    try:
        while not stop_ev.is_set():
            try: fid, frame, video_time = fq.get(timeout=0.5)
            except: continue

            logs = []
            out_frame = frame
            try:
                if model is None: out_frame = frame
                else:
                    r = None
                    try:
                        r = model.track(source=frame, persist=True, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, imgsz=INFERENCE_SIZE, device=DEVICE, verbose=False)
                    except:
                        try: r = model(frame)
                        except: r = None
                    
                    if r is None: out_frame = frame
                    else:
                        boxes = extract_boxes_from_result(r)
                        frame_copy = frame.copy()
                        
                        # Get GPS
                        lat, lon = (None, None)
                        if gps_handler:
                            try: lat, lon = gps_handler.get_location_at(video_time)
                            except: pass

                        out_frame, logs = draw_boxes_and_log(
                            frame_copy, boxes, fid, video_time, 
                            current_seen_ids, current_snapshot_dir, 
                            gps_coords=(lat, lon)
                        )
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

            try: rq.put_nowait((fid, out_frame))
            except:
                try: rq.get_nowait(); rq.put_nowait((fid, out_frame))
                except: pass
    finally:
        print(f"[{sid}] Worker stopped")

# ---------------------------
# CSV & ZIP
# ---------------------------
def save_csv_for_session(sid):
    with sessions_lock:
        s = sessions.get(sid)
        if not s: return None
        logs = list(s["detection_logs"])
        video_name = secure_filename(s.get("video_name") or f"session_{sid}")

    timestamp = now_ts()
    filename = f"{video_name}_detections_{timestamp}.csv"
    filepath = os.path.join(REPORT_DIR, filename)

    try:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            # Updated fields with GPS
            fields = ["frame_number", "video_time_s", "timestamp", "confidence", "area_m2", "severity", "track_id", "snapshot_file", "latitude", "longitude"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            if logs: writer.writerows(logs)

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
        return filepath
    except:
        traceback.print_exc()
        return None

def create_archive_for_session(sid):
    with sessions_lock:
        s = sessions.get(sid)
        if not s: return None
        s["archiving"] = True
    try:
        time.sleep(0.5)
        with sessions_lock:
            s = sessions.get(sid)
            if not s: return None
            video_path = s.get("video_path")
            video_name = secure_filename(s.get("video_name") or f"session_{sid}")
            csv_path = s.get("csv_path")
            snap_dir = s.get("snapshot_dir")
            gpx_path = s.get("gpx_path")

        ts = now_ts()
        zip_name = f"{video_name}_archive_{ts}.zip"
        zip_path = os.path.join(ARCHIVE_DIR, zip_name)

        with zipfile.ZipFile(zip_path, "w") as zf:
            if video_path and os.path.exists(video_path):
                zf.write(video_path, os.path.basename(video_path))
            if csv_path and os.path.exists(csv_path):
                zf.write(csv_path, os.path.basename(csv_path))
            if gpx_path and os.path.exists(gpx_path):
                zf.write(gpx_path, os.path.basename(gpx_path))
            
            if snap_dir and os.path.exists(snap_dir):
                for root, dirs, files in os.walk(snap_dir):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            zf.write(os.path.join(root, file), os.path.join("snapshots", file))

        archive_url = build_absolute_url(f"/download_archive/{os.path.basename(zip_path)}")
        with sessions_lock:
            s = sessions.get(sid)
            if s:
                s["archive_path"] = zip_path
                s["archive_url"] = archive_url
                s["last_activity"] = time.time()
        auto_cleanup(ARCHIVE_DIR, MAX_FILES_ARCHIVES)
        return zip_path
    except: return None
    finally:
        with sessions_lock:
            s = sessions.get(sid)
            if s: s["archiving"] = False

# ---------------------------
# Video stream
# ---------------------------
def mjpeg_stream_for_session(sid):
    with sessions_lock:
        s = sessions.get(sid)
        if not s: yield b''; return
    video_path = s["video_path"]
    if not os.path.exists(video_path): yield b''; return

    cap = None
    try:
        if cv2 is None: return
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps_val = cap.get(cv2.CAP_PROP_FPS) or 30.0
        with sessions_lock:
            s = sessions.get(sid)
            if s:
                s["total_frames"] = total_frames
                s["video_fps"] = fps_val
                s["streaming"] = True
                s["last_activity"] = time.time()

        with sessions_lock:
            s = sessions.get(sid)
            if not s: return
            if s.get("worker") is None or not s.get("worker").is_alive():
                w = threading.Thread(target=session_infer_worker, args=(sid,), daemon=True)
                s["worker"] = w
                w.start()

        frame_no = 0
        last_sent_frame = None
        while cap.isOpened() and not s["stop_event"].is_set():
            ret, frame = cap.read()
            if not ret: break
            frame_no += 1
            if FRAME_SKIP > 0 and (frame_no % (FRAME_SKIP + 1)) != 1: continue
            
            try: frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            except: pass
            video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
            try: s["frame_q"].put_nowait((frame_no, frame, video_time))
            except:
                try: s["frame_q"].get_nowait(); s["frame_q"].put_nowait((frame_no, frame, video_time))
                except: pass

            out = frame
            try:
                fid_out = s["result_q"].get(timeout=0.05)
                if fid_out: _, out = fid_out
                if out is None: out = frame
            except:
                if last_sent_frame is not None: out = last_sent_frame
                else: out = frame

            try:
                ret2, buf = cv2.imencode(".jpg", out)
                if not ret2: continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
                last_sent_frame = out
            except: pass
            with sessions_lock:
                s = sessions.get(sid)
                if s: s["last_activity"] = time.time()
            time.sleep(0.002)
    finally:
        try: cap.release()
        except: pass
        try:
            save_csv_for_session(sid)
            threading.Thread(target=create_archive_for_session, args=(sid,), daemon=True).start()
        except: pass
        with sessions_lock:
            s = sessions.get(sid)
            if s: 
                s["streaming"] = False
                s["last_activity"] = time.time()

# ---------------------------
# Routes (RESTORED FULL FIELDS)
# ---------------------------
@app.route("/")
def index(): return jsonify({"status": "ok"})

@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files: return jsonify({"error": "No file"}), 400
    f = request.files["video"]
    if f.filename == "": return jsonify({"error": "Empty"}), 400
    if not allowed_file(f.filename): return jsonify({"error": "Invalid"}), 400
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
    threading.Thread(target=lambda: auto_cleanup(UPLOAD_DIR, MAX_FILES_UPLOADS), daemon=True).start()
    return jsonify({"status": "uploaded", "session_id": sid, "video_name": f.filename})

@app.route("/video_feed/<sid>")
def video_feed(sid):
    return Response(mjpeg_stream_for_session(sid), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/progress/<sid>")
def progress(sid):
    with sessions_lock:
        s = sessions.get(sid)
        if not s: return jsonify({"error": "Invalid"}), 404
        proc, total, fps = s.get("processed_frames", 0), s.get("total_frames", 0), s.get("processing_fps", 0)
        # RESTORED: video_fps for frontend UI
        v_fps = s.get("video_fps", 0.0)
    pct = (proc / total * 100.0) if total > 0 else 0.0
    eta = (total - proc) / (fps + 1e-6) if fps > 0 else None
    return jsonify({
        "processed_frames": proc, 
        "total_frames": total, 
        "progress_percent": round(pct, 2), 
        "video_fps": round(v_fps, 2), # Restored field
        "processing_fps": round(fps, 2), 
        "estimated_time_left_s": round(eta, 1) if eta else None
    })

@app.route("/detection_count/<sid>")
def detection_count(sid):
    with sessions_lock:
        s = sessions.get(sid)
        if not s: return jsonify({"error": "Invalid"}), 404
        logs = list(s.get("detection_logs", []))
    det = len(logs)
    total_area = round(sum(l.get("area_m2", 0) for l in logs), 2)
    # RESTORED: avg_confidence for frontend UI
    avg = round(sum(l.get("confidence", 0.0) for l in logs) / det, 3) if det > 0 else 0.0
    return jsonify({"detections": det, "total_area": total_area, "avg_confidence": avg})

@app.route("/processing_status/<sid>")
def processing_status(sid):
    with sessions_lock:
        s = sessions.get(sid)
        if not s: return jsonify({"error": "Invalid"}), 404
        # Fixed variable name collision logic
        proc = bool((s.get("worker") and s.get("worker").is_alive()) or s.get("streaming") or s.get("archiving"))
    return jsonify({"processing": proc})

@app.route("/last_report/<sid>")
def last_report(sid):
    with sessions_lock:
        s = sessions.get(sid)
        if not s or not s.get("csv_path"): return jsonify({"error": "No report"}), 404
        return jsonify({
            "csv_path": s.get("csv_path"), "csv_url": s.get("csv_url"),
            "archive_path": s.get("archive_path"), "archive_url": s.get("archive_url"),
            "total_detections": s.get("total_detections", 0),
            "total_area": s.get("total_area", 0.0),
            "average_confidence": s.get("average_confidence", 0.0),
            "video_name": s.get("video_name")
        })

@app.route("/reports/<path:filename>")
def serve_report(filename):
    req = os.path.abspath(os.path.join(REPORT_DIR, filename))
    if not safe_commonpath(os.path.abspath(REPORT_DIR), req) or not os.path.exists(req): return "Not found", 404
    return send_file(req, mimetype="text/csv", as_attachment=True, download_name=os.path.basename(req))

@app.route("/download_archive/<path:filename>")
def download_archive(filename):
    req = os.path.abspath(os.path.join(ARCHIVE_DIR, filename))
    if not safe_commonpath(os.path.abspath(ARCHIVE_DIR), req) or not os.path.exists(req): return "Not found", 404
    return send_file(req, mimetype="application/zip", as_attachment=True, download_name=os.path.basename(req))

@app.route("/export_csv/<sid>")
def export_csv(sid):
    with sessions_lock:
        s = sessions.get(sid)
        if not s: return jsonify({"error": "Invalid"}), 404
        logs, csv_path = list(s.get("detection_logs", [])), s.get("csv_path")
    if logs:
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=["frame_number","video_time_s","timestamp","confidence","area_m2","severity", "track_id", "snapshot_file", "latitude", "longitude"])
        writer.writeheader()
        writer.writerows(logs)
        out.seek(0)
        return send_file(io.BytesIO(out.getvalue().encode()), mimetype="text/csv", as_attachment=True, download_name=f"{sid}.csv")
    elif csv_path and os.path.exists(csv_path): return send_file(csv_path, mimetype="text/csv", as_attachment=True)
    return "No detections", 404

@app.route("/health")
def health(): return jsonify({"device": DEVICE, "gpu": USE_CUDA})

def session_ttl_pruner():
    while True:
        time.sleep(60)
        with sessions_lock:
            to_del = [sid for sid, s in sessions.items() if time.time() - s.get("last_activity", 0) > SESSION_TTL]
            for sid in to_del: cleanup_session(sid)

if __name__ == "__main__":
    for f in [lambda: auto_cleanup(UPLOAD_DIR, MAX_FILES_UPLOADS), lambda: auto_cleanup(REPORT_DIR, MAX_FILES_REPORTS), lambda: auto_cleanup(ARCHIVE_DIR, MAX_FILES_ARCHIVES), lambda: auto_cleanup(SNAPSHOT_DIR, MAX_FILES_REPORTS)]:
        threading.Thread(target=f, daemon=True).start()
    threading.Thread(target=session_ttl_pruner, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), debug=False, threaded=True)