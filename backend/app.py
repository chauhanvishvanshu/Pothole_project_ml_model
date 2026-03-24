#!/usr/bin/env python3
"""
app.py — Multi-session Pothole Detection Backend (Flask)
COMPLETE SUITE: Tracking + Snapshots + GPS + PDF Reports + KML Map Export.
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
import base64
import json
import shutil
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

# --- ReportLab for PDF Generation ---
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    print("ReportLab not installed. PDF generation disabled. (pip install reportlab)")
    REPORTLAB_AVAILABLE = False
# -----------------------------------------

# ---------------------------
# Configuration
# ---------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_config_path(path_value, default_value):
    raw_value = path_value or default_value
    return raw_value if os.path.isabs(raw_value) else os.path.join(APP_DIR, raw_value)

MODEL_PATH = resolve_config_path(os.environ.get("MODEL_PATH"), "best.pt")
UPLOAD_DIR = resolve_config_path(os.environ.get("UPLOAD_DIR"), "uploads")
REPORT_DIR = resolve_config_path(os.environ.get("REPORT_DIR"), "reports")
ARCHIVE_DIR = os.path.join(REPORT_DIR, "archives")
SNAPSHOT_DIR = os.path.join(REPORT_DIR, "snapshots")
PHOTO_DIR = os.path.join(REPORT_DIR, "photos")
LIVE_DIR = os.path.join(REPORT_DIR, "live")
MANIFEST_DIR = os.path.join(REPORT_DIR, "manifests")

ALLOWED_EXT = {"mp4", "mov", "avi", "mkv", "webm"}
ALLOWED_IMAGE_EXT = {"jpg", "jpeg", "png", "webp"}
FRAME_WIDTH = int(os.environ.get("FRAME_WIDTH", 1280))
FRAME_HEIGHT = int(os.environ.get("FRAME_HEIGHT", 720))
VIDEO_FRAME_WIDTH = int(os.environ.get("VIDEO_FRAME_WIDTH", FRAME_WIDTH))
VIDEO_FRAME_HEIGHT = int(os.environ.get("VIDEO_FRAME_HEIGHT", FRAME_HEIGHT))
INFERENCE_SIZE = int(os.environ.get("INFERENCE_SIZE", 960))
LIVE_FRAME_WIDTH = int(os.environ.get("LIVE_FRAME_WIDTH", 960))
LIVE_FRAME_HEIGHT = int(os.environ.get("LIVE_FRAME_HEIGHT", 540))
LIVE_INFERENCE_SIZE = int(os.environ.get("LIVE_INFERENCE_SIZE", 640))
LIVE_SAVE_SNAPSHOTS = os.environ.get("LIVE_SAVE_SNAPSHOTS", "").lower() in ("1", "true", "yes")
LIVE_USE_TRACKING = os.environ.get("LIVE_USE_TRACKING", "true").lower() in ("1", "true", "yes")
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

DEVICE = "cpu"
USE_CUDA = False
FORCE_CPU = os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes")
if not FORCE_CPU and torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
    USE_CUDA = True
    DEVICE = "cuda:0"
INFERENCE_HALF = USE_CUDA and os.environ.get("INFERENCE_HALF", "true").lower() in ("1", "true", "yes")
if USE_CUDA and torch is not None:
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

TRACKER_NAME = os.environ.get("TRACKER_NAME", "bytetrack").strip().lower()
TRACKER_CONFIG = None
if TRACKER_NAME and TRACKER_NAME not in ("default", "none"):
    tracker_dir = os.path.join(os.path.dirname(__file__), ".venv", "Lib", "site-packages", "ultralytics", "cfg", "trackers")
    tracker_candidate = os.path.join(tracker_dir, f"{TRACKER_NAME}.yaml")
    if os.path.exists(tracker_candidate):
        TRACKER_CONFIG = tracker_candidate

PUBLIC_HOST = os.environ.get("PUBLIC_HOST", "").rstrip("/")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
WARMED_INFERENCE_KEYS = set()

for d in (UPLOAD_DIR, REPORT_DIR, ARCHIVE_DIR, SNAPSHOT_DIR, PHOTO_DIR, LIVE_DIR, MANIFEST_DIR):
    os.makedirs(d, exist_ok=True)


# ---------------------------
# Utilities
# ---------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def allowed_image_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXT

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

def managed_storage_roots():
    roots = []
    for base in (UPLOAD_DIR, REPORT_DIR):
        for candidate in (os.path.abspath(base), os.path.abspath(os.path.join(APP_DIR, base))):
            if candidate not in roots:
                roots.append(candidate)
    return roots

def resolve_managed_path(path):
    if not path:
        return None
    candidates = []
    if os.path.isabs(path):
        candidates.append(os.path.abspath(path))
    else:
        candidates.append(os.path.abspath(path))
        candidates.append(os.path.abspath(os.path.join(APP_DIR, path)))
    roots = managed_storage_roots()
    for candidate in candidates:
        if any(safe_commonpath(root, candidate) for root in roots):
            return candidate
    return None

def delete_path_if_managed(path):
    resolved = resolve_managed_path(path)
    if not resolved or not os.path.exists(resolved):
        return 0
    try:
        if os.path.isdir(resolved):
            shutil.rmtree(resolved)
        else:
            os.remove(resolved)
        return 1
    except FileNotFoundError:
        return 0
    except Exception:
        traceback.print_exc()
        return 0

def purge_directory_contents(directory):
    resolved = resolve_managed_path(directory)
    if not resolved or not os.path.exists(resolved):
        return 0
    deleted = 0
    try:
        for name in os.listdir(resolved):
            deleted += delete_path_if_managed(os.path.join(resolved, name))
    except Exception:
        traceback.print_exc()
    return deleted

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

SEVERITY_LEVELS = ("Minor", "Moderate", "Major", "Severe")
SEVERITY_WEIGHTS = {"Minor": 1.0, "Moderate": 2.8, "Major": 6.0, "Severe": 10.0}

def severity_rank(severity):
    try:
        return SEVERITY_LEVELS.index(severity)
    except Exception:
        return -1

def default_severity_counts():
    return {severity: 0 for severity in SEVERITY_LEVELS}

def safe_float(value, default=None):
    try:
        if value in ("", None):
            raise ValueError("empty")
        return float(value)
    except Exception:
        return default

def format_mmss(seconds):
    total = safe_float(seconds, 0.0) or 0.0
    mins = int(total // 60)
    secs = int(total % 60)
    return f"{mins:02d}:{secs:02d}"

def estimate_hazard_count(logs):
    track_ids = set()
    for log in logs or []:
        track_id = log.get("track_id")
        if track_id in (None, "", "None"):
            continue
        track_ids.add(str(track_id))
    return len(track_ids) if track_ids else len(logs or [])

def build_hotspots(logs, limit=3):
    if not logs:
        return []

    gps_clusters = {}
    for log in logs:
        lat = safe_float(log.get("latitude"))
        lon = safe_float(log.get("longitude"))
        if lat is None or lon is None:
            continue
        key = (round(lat, 4), round(lon, 4))
        cluster = gps_clusters.setdefault(key, {
            "kind": "gps",
            "label": f"{key[0]:.4f}, {key[1]:.4f}",
            "count": 0,
            "total_area": 0.0,
            "top_severity": "Minor",
        })
        cluster["count"] += 1
        cluster["total_area"] += safe_float(log.get("area_m2"), 0.0) or 0.0
        sev = log.get("severity", "Minor")
        if severity_rank(sev) > severity_rank(cluster["top_severity"]):
            cluster["top_severity"] = sev

    if gps_clusters:
        values = sorted(
            gps_clusters.values(),
            key=lambda item: (item["count"], item["total_area"], severity_rank(item["top_severity"])),
            reverse=True,
        )
        return [{
            "kind": item["kind"],
            "label": item["label"],
            "count": item["count"],
            "total_area": round(item["total_area"], 2),
            "top_severity": item["top_severity"],
        } for item in values[:limit]]

    times = [safe_float(log.get("video_time_s"), 0.0) or 0.0 for log in logs]
    if not times or max(times) <= 0.01:
        counts = default_severity_counts()
        total_area = 0.0
        for log in logs:
            sev = log.get("severity", "Minor")
            if sev in counts:
                counts[sev] += 1
            total_area += safe_float(log.get("area_m2"), 0.0) or 0.0
        top_severity = next((severity for severity in reversed(SEVERITY_LEVELS) if counts[severity] > 0), "Minor")
        return [{
            "kind": "frame",
            "label": "Captured frame",
            "count": len(logs),
            "total_area": round(total_area, 2),
            "top_severity": top_severity,
        }]

    window = 15 if max(times) >= 90 else 10
    time_clusters = {}
    for log in logs:
        time_s = safe_float(log.get("video_time_s"), 0.0) or 0.0
        bucket = int(time_s // window) * window
        label = f"{format_mmss(bucket)}-{format_mmss(bucket + window)}"
        cluster = time_clusters.setdefault(bucket, {
            "kind": "time",
            "label": label,
            "count": 0,
            "total_area": 0.0,
            "top_severity": "Minor",
        })
        cluster["count"] += 1
        cluster["total_area"] += safe_float(log.get("area_m2"), 0.0) or 0.0
        sev = log.get("severity", "Minor")
        if severity_rank(sev) > severity_rank(cluster["top_severity"]):
            cluster["top_severity"] = sev

    values = sorted(
        time_clusters.values(),
        key=lambda item: (item["count"], item["total_area"], severity_rank(item["top_severity"])),
        reverse=True,
    )
    return [{
        "kind": item["kind"],
        "label": item["label"],
        "count": item["count"],
        "total_area": round(item["total_area"], 2),
        "top_severity": item["top_severity"],
    } for item in values[:limit]]

def build_highlights(logs, limit=3):
    ranked_logs = sorted(
        logs or [],
        key=lambda log: (
            severity_rank(log.get("severity", "Minor")),
            safe_float(log.get("area_m2"), 0.0) or 0.0,
            safe_float(log.get("confidence"), 0.0) or 0.0,
        ),
        reverse=True,
    )
    highlights = []
    seen = set()
    for log in ranked_logs:
        time_s = safe_float(log.get("video_time_s"), 0.0) or 0.0
        track_id = log.get("track_id")
        snapshot_file = log.get("snapshot_file") or ""
        dedupe_key = (str(track_id), round(time_s, 1), snapshot_file)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        severity = log.get("severity", "Minor")
        lat = safe_float(log.get("latitude"))
        lon = safe_float(log.get("longitude"))
        if lat is not None and lon is not None:
            location_label = f"{lat:.5f}, {lon:.5f}"
        elif time_s > 0.01:
            location_label = format_mmss(time_s)
        else:
            location_label = "Still frame"

        highlights.append({
            "title": f"{severity} defect",
            "severity": severity,
            "area_m2": round(safe_float(log.get("area_m2"), 0.0) or 0.0, 2),
            "confidence": round(safe_float(log.get("confidence"), 0.0) or 0.0, 3),
            "time_label": format_mmss(time_s) if time_s > 0.01 else "Still frame",
            "location_label": location_label,
        })
        if len(highlights) >= limit:
            break
    return highlights

def summarize_logs(logs):
    detections = len(logs)
    total_area = round(sum(l.get("area_m2", 0.0) for l in logs), 2) if logs else 0.0
    avg_conf = round(sum(l.get("confidence", 0.0) for l in logs) / detections, 3) if detections > 0 else 0.0
    gps_detection_count = 0
    severity_counts = default_severity_counts()
    for log in logs or []:
        severity = log.get("severity", "Minor")
        if severity in severity_counts:
            severity_counts[severity] += 1
        if safe_float(log.get("latitude")) is not None and safe_float(log.get("longitude")) is not None:
            gps_detection_count += 1

    unique_hazards = estimate_hazard_count(logs)
    weighted_severity = sum(severity_counts[severity] * SEVERITY_WEIGHTS[severity] for severity in SEVERITY_LEVELS)
    severity_penalty = min(55.0, weighted_severity * 1.8)
    hazard_penalty = min(28.0, unique_hazards * 3.2)
    area_penalty = min(22.0, total_area * 2.0)
    road_health_score = int(max(8, min(100, round(100 - severity_penalty - hazard_penalty - area_penalty)))) if detections > 0 else 100

    if detections == 0:
        maintenance_priority = "Low"
        health_band = "Excellent"
        insight_headline = "Road surface looks clear."
        recommended_action = "No immediate maintenance signal. Keep monitoring with routine captures."
    else:
        severe_count = severity_counts["Severe"]
        major_count = severity_counts["Major"]
        moderate_count = severity_counts["Moderate"]
        if road_health_score <= 35 or severe_count >= 2 or total_area >= 10 or unique_hazards >= 12:
            maintenance_priority = "Critical"
            health_band = "Critical"
            insight_headline = "Urgent repair candidate detected."
            recommended_action = "Escalate this route for immediate inspection and repair scheduling."
        elif road_health_score <= 55 or severe_count >= 1 or major_count >= 3 or total_area >= 5 or unique_hazards >= 6:
            maintenance_priority = "High"
            health_band = "Damaged"
            insight_headline = "High-priority damage cluster detected."
            recommended_action = "Plan a targeted maintenance pass soon and review highlighted hotspots."
        elif road_health_score <= 75 or major_count >= 1 or moderate_count >= 3 or total_area >= 2 or unique_hazards >= 3:
            maintenance_priority = "Moderate"
            health_band = "Watchlist"
            insight_headline = "Surface degradation is emerging."
            recommended_action = "Track this segment and prepare preventive maintenance before it worsens."
        else:
            maintenance_priority = "Low"
            health_band = "Stable"
            insight_headline = "Only limited surface damage detected."
            recommended_action = "Keep this segment in routine monitoring with no urgent intervention."

    top_severity = next((severity for severity in reversed(SEVERITY_LEVELS) if severity_counts[severity] > 0), "None")
    summary = {
        "detections": detections,
        "total_area": total_area,
        "avg_confidence": avg_conf,
        "severity_counts": dict(severity_counts),
        "unique_hazards": unique_hazards,
        "top_severity": top_severity,
        "road_health_score": road_health_score,
        "health_band": health_band,
        "maintenance_priority": maintenance_priority,
        "insight_headline": insight_headline,
        "recommended_action": recommended_action,
        "gps_detection_count": gps_detection_count,
        "has_map": gps_detection_count > 0,
    }
    hotspots = build_hotspots(logs, limit=3)
    highlights = build_highlights(logs, limit=3)
    summary["hotspot_count"] = len(hotspots)
    summary["hotspots"] = hotspots
    summary["highlights"] = highlights
    return summary

def parse_csv_logs(csv_path):
    resolved = resolve_managed_path(csv_path)
    if not resolved or not os.path.exists(resolved):
        return []
    logs = []
    try:
        with open(resolved, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                logs.append({
                    "frame_number": int(safe_float(row.get("frame_number"), 0) or 0),
                    "video_time_s": round(safe_float(row.get("video_time_s"), 0.0) or 0.0, 2),
                    "timestamp": row.get("timestamp") or "",
                    "confidence": round(safe_float(row.get("confidence"), 0.0) or 0.0, 3),
                    "area_m2": round(safe_float(row.get("area_m2"), 0.0) or 0.0, 2),
                    "severity": row.get("severity") or "Minor",
                    "track_id": row.get("track_id") or "",
                    "snapshot_file": row.get("snapshot_file") or "",
                    "latitude": row.get("latitude") or "",
                    "longitude": row.get("longitude") or "",
                })
    except Exception:
        traceback.print_exc()
    return logs

def load_session_data(sid):
    with sessions_lock:
        entry = sessions.get(sid)
        if entry:
            return {
                "session": entry,
                "payload": session_report_payload(entry),
                "logs": list(entry.get("detection_logs", [])),
            }

    manifest = load_manifest_record(sid)
    if not manifest:
        return None
    logs = parse_csv_logs(manifest.get("csv_path"))
    return {
        "session": None,
        "payload": manifest,
        "logs": logs,
    }

def build_map_payload(sid, payload, logs):
    points = []
    seen_coords = set()
    for log in sorted(logs or [], key=lambda item: safe_float(item.get("video_time_s"), 0.0) or 0.0):
        lat = safe_float(log.get("latitude"))
        lon = safe_float(log.get("longitude"))
        if lat is None or lon is None:
            continue
        severity = log.get("severity", "Minor")
        point = {
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "severity": severity,
            "area_m2": round(safe_float(log.get("area_m2"), 0.0) or 0.0, 2),
            "confidence": round(safe_float(log.get("confidence"), 0.0) or 0.0, 3),
            "time_s": round(safe_float(log.get("video_time_s"), 0.0) or 0.0, 2),
            "time_label": format_mmss(safe_float(log.get("video_time_s"), 0.0) or 0.0),
            "track_id": str(log.get("track_id") or ""),
        }
        points.append(point)
        coord_key = (point["lat"], point["lon"])
        seen_coords.add(coord_key)

    if not points:
        return None

    route_points = []
    route_seen = set()
    for point in points:
        key = (point["lat"], point["lon"])
        if key in route_seen:
            continue
        route_seen.add(key)
        route_points.append([point["lat"], point["lon"]])

    latitudes = [point["lat"] for point in points]
    longitudes = [point["lon"] for point in points]
    bounds = {
        "south": min(latitudes),
        "west": min(longitudes),
        "north": max(latitudes),
        "east": max(longitudes),
    }

    summary = summarize_logs(logs)
    hotspot_markers = []
    for hotspot in summary.get("hotspots", []):
        if hotspot.get("kind") != "gps":
            continue
        try:
            lat_str, lon_str = str(hotspot.get("label", "")).split(",", 1)
            hotspot_markers.append({
                "lat": round(float(lat_str.strip()), 6),
                "lon": round(float(lon_str.strip()), 6),
                "label": hotspot.get("label"),
                "count": hotspot.get("count", 0),
                "total_area": hotspot.get("total_area", 0.0),
                "top_severity": hotspot.get("top_severity", "Minor"),
            })
        except Exception:
            continue

    return {
        "session_id": sid,
        "video_name": payload.get("video_name") or sid,
        "source_type": payload.get("source_type", "video"),
        "has_map": True,
        "gps_detection_count": summary.get("gps_detection_count", 0),
        "road_health_score": summary.get("road_health_score", payload.get("road_health_score", 100)),
        "maintenance_priority": summary.get("maintenance_priority", payload.get("maintenance_priority", "Low")),
        "health_band": summary.get("health_band", payload.get("health_band", "Excellent")),
        "insight_headline": summary.get("insight_headline", payload.get("insight_headline", "")),
        "recommended_action": summary.get("recommended_action", payload.get("recommended_action", "")),
        "bounds": bounds,
        "center": {"lat": round(sum(latitudes) / len(latitudes), 6), "lon": round(sum(longitudes) / len(longitudes), 6)},
        "route": route_points,
        "detections": points,
        "hotspots": hotspot_markers,
    }

def read_image_from_upload(file_storage):
    if cv2 is None or np is None or file_storage is None:
        return None
    try:
        raw = file_storage.read()
        if not raw:
            return None
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

def resize_frame(frame, width=FRAME_WIDTH, height=FRAME_HEIGHT):
    if cv2 is None or frame is None:
        return frame
    try:
        return cv2.resize(frame, (int(width), int(height)))
    except Exception:
        return frame

def encode_frame_to_jpeg(frame, quality=82):
    if cv2 is None or frame is None:
        return None
    try:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return buf.tobytes() if ok else None
    except Exception:
        return None

def encode_jpeg_bytes_to_data_url(jpeg_bytes):
    if not jpeg_bytes:
        return None
    try:
        return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii")
    except Exception:
        return None

def encode_frame_to_data_url(frame, quality=82):
    return encode_jpeg_bytes_to_data_url(encode_frame_to_jpeg(frame, quality=quality))

def save_frame_image(frame, directory, stem, jpeg_bytes=None):
    if cv2 is None or (frame is None and jpeg_bytes is None):
        return None, None
    try:
        safe_stem = secure_filename(stem) or "frame"
        filename = f"{safe_stem}_{now_ts()}.jpg"
        filepath = os.path.join(directory, filename)
        if jpeg_bytes is None:
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                return None, None
            jpeg_bytes = buf.tobytes()
        with open(filepath, "wb") as f:
            f.write(jpeg_bytes)
        return filepath, build_absolute_url(f"/reports/{os.path.relpath(filepath, REPORT_DIR).replace(os.sep, '/')}")
    except Exception:
        return None, None

# ---------------------------
# GPS Handlers
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
        if video_time_s <= self.points[0][0]: return self.points[0][1], self.points[0][2]
        if video_time_s >= self.points[-1][0]: return self.points[-1][1], self.points[-1][2]
        for i in range(len(self.points) - 1):
            t1, lat1, lon1 = self.points[i]
            t2, lat2, lon2 = self.points[i+1]
            if t1 <= video_time_s <= t2:
                ratio = (video_time_s - t1) / (t2 - t1) if (t2 - t1) > 0 else 0
                lat = lat1 + (lat2 - lat1) * ratio
                lon = lon1 + (lon2 - lon1) * ratio
                return round(lat, 6), round(lon, 6)
        return None, None

class LinearGPSHandler:
    def __init__(self, start_coords, end_coords, total_duration_s):
        self.valid = False
        self.start_lat = 0.0
        self.start_lon = 0.0
        self.end_lat = 0.0
        self.end_lon = 0.0
        self.duration = float(total_duration_s) if total_duration_s else 0.0

        try:
            if start_coords and end_coords and self.duration > 0:
                s_lat, s_lon = map(float, start_coords.split(','))
                e_lat, e_lon = map(float, end_coords.split(','))
                self.start_lat, self.start_lon = s_lat, s_lon
                self.end_lat, self.end_lon = e_lat, e_lon
                self.valid = True
                print(f"Linear GPS Initialized.")
        except Exception as e:
            print(f"Linear GPS Init Failed: {e}")

    def get_location_at(self, video_time_s):
        if not self.valid: return None, None
        t = max(0.0, min(video_time_s, self.duration))
        progress = t / self.duration
        curr_lat = self.start_lat + (self.end_lat - self.start_lat) * progress
        curr_lon = self.start_lon + (self.end_lon - self.start_lon) * progress
        return round(curr_lat, 6), round(curr_lon, 6)

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
        print(f"Runtime device: {DEVICE} | tracker: {TRACKER_NAME or 'default'} | imgsz: {INFERENCE_SIZE} | fp16: {INFERENCE_HALF} | video: {VIDEO_FRAME_WIDTH}x{VIDEO_FRAME_HEIGHT}")
    except Exception as e:
        print("Failed loading model:", e)
        model = None

# ---------------------------
# Session management
# ---------------------------
import queue
sessions = {}
sessions_lock = threading.Lock()

def create_session_entry(video_path, original_name, start_coords=None, end_coords=None, source_type="video"):
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
        "source_type": source_type,
        "video_path": video_path,
        "gpx_path": gpx_path, 
        "start_coords": start_coords,
        "end_coords": end_coords,
        "video_name": original_name,
        "frame_q": queue.Queue(maxsize=FRAME_QUEUE_MAX),
        "result_q": queue.Queue(maxsize=RESULT_QUEUE_MAX),
        "stop_event": threading.Event(),
        "producer": None,
        "worker": None,
        "detection_logs": [],
        "seen_ids": set(),
        "snapshot_dir": session_snapshot_dir,
        
        "csv_path": None, "csv_url": None,
        "pdf_path": None, "pdf_url": None,
        "kml_path": None, "kml_url": None, # NEW: KML Fields
        "archive_path": None, "archive_url": None,
        "annotated_path": None, "annotated_url": None,
        "last_live_frame_bytes": None,
        "last_video_frame_bytes": None,
        
        "total_frames": 0, "processed_frames": 0,
        "video_fps": 0.0, "processing_fps": 0.0,
        "duration_s": 0.0,
        "severity_counts": default_severity_counts(),
        "gps_detection_count": 0,
        "has_map": False,
        "unique_hazards": 0,
        "top_severity": "None",
        "road_health_score": 100,
        "health_band": "Excellent",
        "maintenance_priority": "Low",
        "insight_headline": "Road surface looks clear.",
        "recommended_action": "No immediate maintenance signal. Keep monitoring with routine captures.",
        "hotspot_count": 0,
        "hotspots": [],
        "highlights": [],
        "live_active": source_type == "live",
        "live_config": {
            "frame_width": LIVE_FRAME_WIDTH,
            "frame_height": LIVE_FRAME_HEIGHT,
            "inference_size": LIVE_INFERENCE_SIZE,
            "save_snapshots": LIVE_SAVE_SNAPSHOTS,
            "use_tracking": LIVE_USE_TRACKING,
            "jpeg_quality": 78,
        },
        "created_at": time.time(), "last_activity": time.time(),
        "streaming": False, "archiving": False,
        "processing_active": False,
        "processing_complete": False,
    }
    sid_to_cleanup = None
    with sessions_lock:
        if len(sessions) >= MAX_SESSION_COUNT and sessions:
            oldest = sorted(sessions.values(), key=lambda x: x["created_at"])[0]
            sid_to_cleanup = oldest["session_id"]
    if sid_to_cleanup:
        cleanup_session(sid_to_cleanup)
    with sessions_lock:
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
                p = s.get("producer")
                session_busy = (
                    s.get("processing_active", False)
                    or bool(p and p.is_alive())
                    or s.get("streaming", False)
                    or s.get("live_active", False)
                    or s.get("archiving", False)
                )
                if (w is None or not w.is_alive()) and not session_busy:
                    idle = time.time() - s.get("last_activity", s.get("created_at", time.time()))
                    if idle > 2: break
    t = threading.Thread(target=fps_monitor, args=(sid,), daemon=True)
    t.start()
    return entry

def apply_summary_to_session(session_obj, summary):
    if not session_obj or not summary:
        return
    session_obj["total_detections"] = summary.get("detections", 0)
    session_obj["total_area"] = summary.get("total_area", 0.0)
    session_obj["average_confidence"] = summary.get("avg_confidence", 0.0)
    session_obj["severity_counts"] = summary.get("severity_counts", default_severity_counts())
    session_obj["unique_hazards"] = summary.get("unique_hazards", 0)
    session_obj["top_severity"] = summary.get("top_severity", "None")
    session_obj["road_health_score"] = summary.get("road_health_score", 100)
    session_obj["health_band"] = summary.get("health_band", "Excellent")
    session_obj["maintenance_priority"] = summary.get("maintenance_priority", "Low")
    session_obj["insight_headline"] = summary.get("insight_headline", "")
    session_obj["recommended_action"] = summary.get("recommended_action", "")
    session_obj["gps_detection_count"] = summary.get("gps_detection_count", 0)
    session_obj["has_map"] = summary.get("has_map", False)
    session_obj["hotspot_count"] = summary.get("hotspot_count", 0)
    session_obj["hotspots"] = summary.get("hotspots", [])
    session_obj["highlights"] = summary.get("highlights", [])
    session_obj["last_activity"] = time.time()

def cleanup_session(sid):
    with sessions_lock:
        entry = sessions.pop(sid, None)
    if not entry:
        return None
    try:
        entry["stop_event"].set()
        p = entry.get("producer")
        if p and p.is_alive():
            p.join(timeout=2.0)
        w = entry.get("worker")
        if w and w.is_alive():
            w.join(timeout=2.0)
    except Exception:
        pass
    try:
        while not entry["frame_q"].empty():
            entry["frame_q"].get_nowait()
        while not entry["result_q"].empty():
            entry["result_q"].get_nowait()
    except Exception:
        pass
    try:
        entry.get("detection_logs", []).clear()
        entry.get("seen_ids", set()).clear()
        entry["last_live_frame_bytes"] = None
        entry["last_video_frame_bytes"] = None
        entry["live_active"] = False
        entry["streaming"] = False
        entry["archiving"] = False
        entry["processing_active"] = False
        entry["processing_complete"] = False
    except Exception:
        pass
    return entry

# ---------------------------
# Drawing + logging
# ---------------------------
def get_severity(area_m2):
    if area_m2 < 0.5: return "Minor"
    if area_m2 < 1.5: return "Moderate"
    if area_m2 < 3.0: return "Major"
    return "Severe"

def draw_boxes_and_log(
    frame,
    boxes_obj,
    frame_number,
    video_time,
    seen_ids_set,
    snapshot_dir,
    gps_coords=(None, None),
    conf_thresh=CONFIDENCE_THRESHOLD,
    log_every_detection=False,
    save_snapshots=True,
):
    logs = []
    h, w = frame.shape[:2]
    
    orig_frame_clean = None
    if cv2 is not None:
        orig_frame_clean = frame.copy()

    if boxes_obj is None: return frame, logs

    try:
        iterable = boxes_obj if isinstance(boxes_obj, (list, tuple)) else list(boxes_obj)
    except: return frame, logs

    for idx, box in enumerate(iterable, start=1):
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

            log_id = track_id if track_id is not None else idx
            should_log = False
            if track_id is not None:
                if log_every_detection or seen_ids_set is None or track_id not in seen_ids_set:
                    should_log = True
                    if seen_ids_set is not None and not log_every_detection:
                        seen_ids_set.add(track_id)
            elif log_every_detection:
                should_log = True

            if should_log:
                snapshot_filename = ""
                if save_snapshots and snapshot_dir:
                    snapshot_filename = f"frame_{frame_number}_id_{log_id}.jpg"
                    snapshot_path = os.path.join(snapshot_dir, snapshot_filename)
                    try:
                        if cv2 is not None and orig_frame_clean is not None:
                            snap_img = orig_frame_clean.copy()
                            cv2.rectangle(snap_img, (x1, y1), (x2, y2), color, 2)
                            label_snap = f"ID:{log_id} | {area_m2}m2"
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
                    "track_id": log_id,
                    "snapshot_file": snapshot_filename,
                    "latitude": lat if lat is not None else "",
                    "longitude": lon if lon is not None else ""
                })

            if cv2 is not None:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{log_id} | {area_m2}m2"
                (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w_text, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        except: continue
    return frame, logs

def run_inference_on_frame(frame, use_tracking=True, inference_size=INFERENCE_SIZE):
    if model is None:
        return None
    try:
        if use_tracking:
            track_kwargs = {
                "source": frame,
                "persist": True,
                "conf": CONFIDENCE_THRESHOLD,
                "iou": IOU_THRESHOLD,
                "imgsz": inference_size,
                "device": DEVICE,
                "half": INFERENCE_HALF,
                "verbose": False,
            }
            if TRACKER_CONFIG:
                track_kwargs["tracker"] = TRACKER_CONFIG
            return model.track(**track_kwargs)
        return model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, imgsz=inference_size, device=DEVICE, half=INFERENCE_HALF, verbose=False)
    except Exception:
        traceback.print_exc()
        return None

def analyze_frame(
    frame,
    frame_number=1,
    video_time=0.0,
    seen_ids_set=None,
    snapshot_dir=None,
    gps_coords=(None, None),
    use_tracking=True,
    log_every_detection=False,
    save_snapshots=True,
    target_size=(FRAME_WIDTH, FRAME_HEIGHT),
    inference_size=INFERENCE_SIZE,
):
    target_width, target_height = target_size
    if frame is not None:
        current_h, current_w = frame.shape[:2]
        if current_w != int(target_width) or current_h != int(target_height):
            frame = resize_frame(frame, width=target_width, height=target_height)
    out_frame = frame
    logs = []
    if frame is None:
        return out_frame, logs
    result = run_inference_on_frame(frame, use_tracking=use_tracking, inference_size=inference_size)
    if result is None:
        return out_frame, logs
    boxes = extract_boxes_from_result(result)
    try:
        out_frame, logs = draw_boxes_and_log(
            frame.copy(),
            boxes,
            frame_number,
            video_time,
            seen_ids_set,
            snapshot_dir,
            gps_coords=gps_coords,
            log_every_detection=log_every_detection,
            save_snapshots=save_snapshots,
        )
    except Exception:
        traceback.print_exc()
        out_frame = frame
        logs = []
    return out_frame, logs

def update_session_summary_fields(sid):
    with sessions_lock:
        s = sessions.get(sid)
        if not s:
            return None
        summary = summarize_logs(list(s.get("detection_logs", [])))
        apply_summary_to_session(s, summary)
        return summary

def ensure_session_worker(sid):
    with sessions_lock:
        s = sessions.get(sid)
        if not s:
            return None
        w = s.get("worker")
        if w and w.is_alive():
            return w
        w = threading.Thread(target=session_infer_worker, args=(sid,), daemon=True)
        s["worker"] = w
        w.start()
        return w

def drain_latest_processed_frame(result_queue):
    latest_frame = None
    try:
        while True:
            item = result_queue.get_nowait()
            if item:
                _, latest_frame = item
    except queue.Empty:
        pass
    except Exception:
        pass
    return latest_frame

def update_video_preview_frame(sid, frame=None, jpeg_bytes=None):
    if cv2 is None:
        return None
    if jpeg_bytes is None and frame is not None:
        jpeg_bytes = encode_frame_to_jpeg(frame, quality=82)
    if not jpeg_bytes:
        return None
    with sessions_lock:
        s = sessions.get(sid)
        if not s:
            return None
        s["last_video_frame_bytes"] = jpeg_bytes
        s["last_activity"] = time.time()
    return jpeg_bytes

def persist_video_preview_frame(sid, jpeg_bytes):
    if not jpeg_bytes:
        return None, None
    with sessions_lock:
        s = sessions.get(sid)
        if not s:
            return None, None
        video_name = secure_filename(s.get("video_name") or f"session_{sid}")
    annotated_path, annotated_url = save_frame_image(None, REPORT_DIR, f"{video_name}_annotated", jpeg_bytes=jpeg_bytes)
    with sessions_lock:
        s = sessions.get(sid)
        if s:
            s["annotated_path"] = annotated_path
            s["annotated_url"] = annotated_url
    return annotated_path, annotated_url

def session_video_producer(sid):
    with sessions_lock:
        entry = sessions.get(sid)
    if not entry:
        return

    video_path = entry.get("video_path")
    stop_ev = entry["stop_event"]
    frame_queue = entry["frame_q"]
    result_queue = entry["result_q"]
    cap = None
    last_preview_bytes = None
    last_rendered_frame = None

    try:
        if cv2 is None or not video_path or not os.path.exists(video_path):
            return
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps_val = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration_s = total_frames / fps_val if fps_val > 0 else 0.0

        with sessions_lock:
            s = sessions.get(sid)
            if not s:
                return
            s["total_frames"] = total_frames
            s["video_fps"] = fps_val
            s["duration_s"] = duration_s
            s["processing_active"] = True
            s["processing_complete"] = False
            s["last_activity"] = time.time()

        ensure_session_worker(sid)

        frame_no = 0
        while cap.isOpened() and not stop_ev.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1
            if FRAME_SKIP > 0 and (frame_no % (FRAME_SKIP + 1)) != 1:
                continue

            try:
                frame = cv2.resize(frame, (VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT))
            except Exception:
                pass
            video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            queued = False
            while not queued and not stop_ev.is_set():
                try:
                    frame_queue.put((frame_no, frame, video_time), timeout=0.05)
                    queued = True
                except queue.Full:
                    latest_frame = drain_latest_processed_frame(result_queue)
                    if latest_frame is not None:
                        last_rendered_frame = latest_frame
                        last_preview_bytes = update_video_preview_frame(sid, frame=latest_frame)
                    else:
                        fallback_frame = last_rendered_frame if last_rendered_frame is not None else frame
                        last_preview_bytes = update_video_preview_frame(sid, frame=fallback_frame)
                except Exception:
                    break
            if not queued:
                break

            latest_frame = drain_latest_processed_frame(result_queue)
            if latest_frame is not None:
                last_rendered_frame = latest_frame
                last_preview_bytes = update_video_preview_frame(sid, frame=latest_frame)
            else:
                fallback_frame = last_rendered_frame if last_rendered_frame is not None else frame
                last_preview_bytes = update_video_preview_frame(sid, frame=fallback_frame)
            time.sleep(0.002)

        with sessions_lock:
            s = sessions.get(sid)
            if s:
                s["processing_active"] = False
                s["last_activity"] = time.time()

        flush_deadline = time.time() + max(10.0, duration_s + 5.0)
        while time.time() < flush_deadline and not stop_ev.is_set():
            latest_frame = drain_latest_processed_frame(result_queue)
            if latest_frame is not None:
                last_rendered_frame = latest_frame
                last_preview_bytes = update_video_preview_frame(sid, frame=latest_frame)

            pending_frames = 0
            pending_results = 0
            worker_alive = False
            with sessions_lock:
                s = sessions.get(sid)
                if s:
                    pending_frames = s["frame_q"].qsize()
                    pending_results = s["result_q"].qsize()
                    worker = s.get("worker")
                    worker_alive = bool(worker and worker.is_alive())
            if not worker_alive and pending_frames == 0 and pending_results == 0:
                break
            time.sleep(0.05)

        if last_preview_bytes:
            persist_video_preview_frame(sid, last_preview_bytes)
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        with sessions_lock:
            s = sessions.get(sid)
            if s:
                s["processing_active"] = False
                s["processing_complete"] = True
                s["last_activity"] = time.time()
                s["producer"] = None
        try:
            save_csv_for_session(sid)
            threading.Thread(target=create_archive_for_session, args=(sid,), daemon=True).start()
        except Exception:
            traceback.print_exc()

def start_video_processing(sid):
    with sessions_lock:
        s = sessions.get(sid)
        if not s or s.get("source_type") != "video":
            return False
        producer = s.get("producer")
        if producer and producer.is_alive():
            return True
        if s.get("processing_complete", False):
            return True
        s["processing_active"] = True
        s["processing_complete"] = False
        producer = threading.Thread(target=session_video_producer, args=(sid,), daemon=True)
        s["producer"] = producer
        producer.start()
        return True

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
    start_coords = entry.get("start_coords")
    end_coords = entry.get("end_coords")
    duration = entry.get("duration_s", 0.0)

    gps_provider = None
    try: 
        if gpx_path:
            gps_provider = GPXHandler(gpx_path)
        elif start_coords and end_coords:
            gps_provider = LinearGPSHandler(start_coords, end_coords, duration)
    except: pass

    try:
        while not stop_ev.is_set():
            try:
                fid, frame, video_time = fq.get(timeout=0.5)
            except queue.Empty:
                with sessions_lock:
                    s = sessions.get(sid)
                    if not s:
                        break
                    if s.get("source_type") == "video" and not s.get("processing_active", False):
                        break
                continue
            except Exception:
                continue

            logs = []
            out_frame = frame
            try:
                lat, lon = (None, None)
                if gps_provider:
                    try: lat, lon = gps_provider.get_location_at(video_time)
                    except: pass
                out_frame, logs = analyze_frame(
                    frame,
                    frame_number=fid,
                    video_time=video_time,
                    seen_ids_set=current_seen_ids,
                    snapshot_dir=current_snapshot_dir,
                    gps_coords=(lat, lon),
                    use_tracking=True,
                    log_every_detection=False,
                    save_snapshots=True,
                    target_size=(VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT),
                )
            except Exception:
                traceback.print_exc()
                out_frame = frame

            if logs:
                with sessions_lock:
                    s = sessions.get(sid)
                    if s:
                        s["detection_logs"].extend(logs)
                        summary = summarize_logs(s["detection_logs"])
                        apply_summary_to_session(s, summary)

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
# Generate Reports (PDF + KML)
# ---------------------------
def generate_pdf_report(sid):
    if not REPORTLAB_AVAILABLE: return None
    with sessions_lock:
        s = sessions.get(sid)
        if not s: return None
        logs = list(s["detection_logs"])
        video_name = s.get("video_name") or f"Session {sid}"
        snap_dir = s.get("snapshot_dir")
        summary = summarize_logs(logs)

    timestamp = now_ts()
    filename = f"{secure_filename(video_name)}_report_{timestamp}.pdf"
    pdf_path = os.path.join(REPORT_DIR, filename)

    try:
        doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=18)
        styles = getSampleStyleSheet()
        story = []
        story.append(Paragraph("Road Watch - Pothole Detection Report", styles['Title']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"<b>Video:</b> {video_name}", styles['Normal']))
        story.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
        story.append(Paragraph(f"<b>Total Detections:</b> {len(logs)}", styles['Normal']))
        story.append(Paragraph(f"<b>Road Health Score:</b> {summary.get('road_health_score', 100)} ({summary.get('health_band', 'Excellent')})", styles['Normal']))
        story.append(Paragraph(f"<b>Maintenance Priority:</b> {summary.get('maintenance_priority', 'Low')}", styles['Normal']))
        story.append(Paragraph(f"<b>Top Severity:</b> {summary.get('top_severity', 'None')} &nbsp; <b>Unique Hazards:</b> {summary.get('unique_hazards', 0)}", styles['Normal']))
        story.append(Paragraph(f"<b>Recommendation:</b> {summary.get('recommended_action', 'No immediate maintenance signal.')}", styles['Normal']))
        story.append(Spacer(1, 12))

        if logs:
            counts = summary.get("severity_counts", default_severity_counts())
            data = [["Severity", "Count"]] + [[k, str(v)] for k,v in counts.items()]
            t = Table(data, colWidths=[200, 100])
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.grey),('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),('GRID',(0,0),(-1,-1),1,colors.black)]))
            story.append(t)
            story.append(Spacer(1, 20))

            hotspots = summary.get("hotspots", [])
            if hotspots:
                story.append(Paragraph("<b>Top Hotspots</b>", styles['Heading3']))
                hotspot_data = [["Hotspot", "Count", "Area (m2)", "Top Severity"]]
                for item in hotspots:
                    hotspot_data.append([
                        str(item.get("label", "Hotspot")),
                        str(item.get("count", 0)),
                        str(item.get("total_area", 0.0)),
                        str(item.get("top_severity", "Minor")),
                    ])
                ht = Table(hotspot_data, colWidths=[180, 60, 80, 100])
                ht.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.lightgrey),('GRID',(0,0),(-1,-1),1,colors.black)]))
                story.append(ht)
                story.append(Spacer(1, 16))

        for log in logs:
            tid = log.get("track_id","N/A")
            sev = log.get("severity","N/A")
            area = log.get("area_m2",0)
            ts = log.get("video_time_s",0)
            lat = log.get("latitude")
            lon = log.get("longitude")
            snap = log.get("snapshot_file")
            
            txt = f"<b>ID:</b> {tid} &nbsp; <b>Time:</b> {ts}s &nbsp; <b>Sev:</b> {sev} &nbsp; <b>Area:</b> {area}m2<br/>"
            if safe_float(lat) is not None and safe_float(lon) is not None:
                txt += f"<b>GPS:</b> {lat}, {lon}<br/>"
            
            p = Paragraph(txt, styles['Normal'])
            img_flow = None
            if snap and snap_dir:
                fp = os.path.join(snap_dir, snap)
                if os.path.exists(fp):
                    try:
                        img = RLImage(fp)
                        img.drawHeight = 2.5*inch * (img.imageHeight/img.imageWidth)
                        img.drawWidth = 2.5*inch
                        img_flow = img
                    except: pass
            
            if img_flow:
                story.append(Table([[p], [img_flow]], colWidths=[450]))
            else:
                story.append(p)
            story.append(Spacer(1, 10))
            story.append(Paragraph("_"*60, styles['Normal']))
            story.append(Spacer(1, 10))

        doc.build(story)
        with sessions_lock:
            s = sessions.get(sid)
            if s:
                s["pdf_path"] = pdf_path
                s["pdf_url"] = build_absolute_url(f"/reports/{filename}")
        return pdf_path
    except: return None

# --- NEW: Generate KML for Google Earth ---
def generate_kml_report(sid):
    with sessions_lock:
        s = sessions.get(sid)
        if not s: return None
        logs = list(s["detection_logs"])
        video_name = s.get("video_name") or f"Session {sid}"

    # Filter logs that have valid GPS
    gps_logs = [l for l in logs if safe_float(l.get("latitude")) is not None and safe_float(l.get("longitude")) is not None]
    if not gps_logs:
        return None # No GPS data to map

    timestamp = now_ts()
    filename = f"{secure_filename(video_name)}_map_{timestamp}.kml"
    kml_path = os.path.join(REPORT_DIR, filename)

    try:
        # Basic KML Structure
        kml_content = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<kml xmlns="http://www.opengis.net/kml/2.2">',
            '<Document>',
            f'<name>Potholes: {video_name}</name>',
            '<Style id="minorStyle"><IconStyle><scale>1.0</scale><Icon><href>http://maps.google.com/mapfiles/kml/paddle/grn-circle.png</href></Icon></IconStyle></Style>',
            '<Style id="moderateStyle"><IconStyle><scale>1.05</scale><Icon><href>http://maps.google.com/mapfiles/kml/paddle/ylw-circle.png</href></Icon></IconStyle></Style>',
            '<Style id="majorStyle"><IconStyle><scale>1.1</scale><Icon><href>http://maps.google.com/mapfiles/kml/paddle/orange-circle.png</href></Icon></IconStyle></Style>',
            '<Style id="severeStyle"><IconStyle><scale>1.15</scale><Icon><href>http://maps.google.com/mapfiles/kml/paddle/red-circle.png</href></Icon></IconStyle></Style>'
        ]

        route_coords = []
        seen_route_coords = set()
        for log in sorted(gps_logs, key=lambda item: safe_float(item.get("video_time_s"), 0.0) or 0.0):
            lat = safe_float(log["latitude"])
            lon = safe_float(log["longitude"])
            if lat is None or lon is None:
                continue
            tid = log["track_id"]
            sev = log["severity"]
            area = log["area_m2"]
            style_id = {
                "Minor": "minorStyle",
                "Moderate": "moderateStyle",
                "Major": "majorStyle",
                "Severe": "severeStyle",
            }.get(sev, "minorStyle")
            
            kml_content.append('<Placemark>')
            kml_content.append(f'<name>Pothole #{tid}</name>')
            kml_content.append(f'<description>Severity: {sev}\nArea: {area} m2</description>')
            kml_content.append(f'<styleUrl>#{style_id}</styleUrl>')
            kml_content.append('<Point>')
            kml_content.append(f'<coordinates>{lon},{lat},0</coordinates>') # KML uses lon,lat
            kml_content.append('</Point>')
            kml_content.append('</Placemark>')
            route_key = (round(lat, 6), round(lon, 6))
            if route_key not in seen_route_coords:
                seen_route_coords.add(route_key)
                route_coords.append(f"{lon},{lat},0")

        if len(route_coords) > 1:
            kml_content.append('<Placemark>')
            kml_content.append('<name>Survey Route</name>')
            kml_content.append('<Style><LineStyle><color>ff165dff</color><width>4</width></LineStyle></Style>')
            kml_content.append('<LineString><tessellate>1</tessellate><coordinates>')
            kml_content.append(" ".join(route_coords))
            kml_content.append('</coordinates></LineString>')
            kml_content.append('</Placemark>')

        kml_content.append('</Document>')
        kml_content.append('</kml>')

        with open(kml_path, "w", encoding="utf-8") as f:
            f.write("\n".join(kml_content))

        with sessions_lock:
            s = sessions.get(sid)
            if s:
                s["kml_path"] = kml_path
                s["kml_url"] = build_absolute_url(f"/reports/{filename}")
        
        print(f"[{sid}] KML Map Generated: {kml_path}")
        return kml_path
    except Exception as e:
        print(f"KML Gen Error: {e}")
        return None
# ------------------------------------------

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
            fields = ["frame_number", "video_time_s", "timestamp", "confidence", "area_m2", "severity", "track_id", "snapshot_file", "latitude", "longitude"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            if logs: writer.writerows(logs)

        csv_url = build_absolute_url(f"/reports/{os.path.basename(filepath)}")

        with sessions_lock:
            s = sessions.get(sid)
            if s:
                s["csv_path"] = filepath
                s["csv_url"] = csv_url
                summary = summarize_logs(logs)
                apply_summary_to_session(s, summary)
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
        
        # Generate Reports
        pdf_path = generate_pdf_report(sid)
        kml_path = generate_kml_report(sid) # Generate KML

        with sessions_lock:
            s = sessions.get(sid)
            if not s: return None
            video_path = s.get("video_path")
            video_name = secure_filename(s.get("video_name") or f"session_{sid}")
            csv_path = s.get("csv_path")
            snap_dir = s.get("snapshot_dir")
            gpx_path = s.get("gpx_path")
            pdf_path_stored = s.get("pdf_path") 
            kml_path_stored = s.get("kml_path")
            annotated_path = s.get("annotated_path")

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
            if pdf_path_stored and os.path.exists(pdf_path_stored):
                zf.write(pdf_path_stored, os.path.basename(pdf_path_stored))
            if kml_path_stored and os.path.exists(kml_path_stored):
                zf.write(kml_path_stored, os.path.basename(kml_path_stored))
            if annotated_path and os.path.exists(annotated_path):
                zf.write(annotated_path, os.path.basename(annotated_path))
            
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
        save_session_manifest(sid)
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
        if not s:
            yield b''
            return
        video_path = s["video_path"]
        s["streaming"] = True
        s["last_activity"] = time.time()
    if not video_path or not os.path.exists(video_path):
        with sessions_lock:
            s = sessions.get(sid)
            if s:
                s["streaming"] = False
        yield b''
        return
    start_video_processing(sid)

    last_sent_bytes = None
    try:
        while True:
            processing = False
            complete = False
            current_bytes = None
            with sessions_lock:
                s = sessions.get(sid)
                if not s:
                    break
                producer = s.get("producer")
                worker = s.get("worker")
                processing = bool(
                    s.get("processing_active", False)
                    or bool(producer and producer.is_alive())
                    or bool(worker and worker.is_alive())
                )
                complete = bool(s.get("processing_complete", False))
                current_bytes = s.get("last_video_frame_bytes")

            if current_bytes and current_bytes != last_sent_bytes:
                last_sent_bytes = current_bytes
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + current_bytes + b"\r\n")
                continue

            if complete and not processing:
                if current_bytes and current_bytes == last_sent_bytes:
                    break
                if not current_bytes:
                    break

            time.sleep(0.05)
    finally:
        with sessions_lock:
            s = sessions.get(sid)
            if s:
                s["streaming"] = False
                s["last_activity"] = time.time()

def session_report_payload(s):
    if not s:
        return {"error": "Invalid"}
    return {
        "session_id": s.get("session_id"),
        "source_type": s.get("source_type", "video"),
        "csv_path": s.get("csv_path"),
        "csv_url": s.get("csv_url"),
        "pdf_path": s.get("pdf_path"),
        "pdf_url": s.get("pdf_url"),
        "kml_path": s.get("kml_path"),
        "kml_url": s.get("kml_url"),
        "archive_path": s.get("archive_path"),
        "archive_url": s.get("archive_url"),
        "annotated_path": s.get("annotated_path"),
        "annotated_url": s.get("annotated_url"),
        "total_detections": s.get("total_detections", 0),
        "total_area": s.get("total_area", 0.0),
        "average_confidence": s.get("average_confidence", 0.0),
        "severity_counts": s.get("severity_counts", default_severity_counts()),
        "unique_hazards": s.get("unique_hazards", 0),
        "top_severity": s.get("top_severity", "None"),
        "road_health_score": s.get("road_health_score", 100),
        "health_band": s.get("health_band", "Excellent"),
        "maintenance_priority": s.get("maintenance_priority", "Low"),
        "insight_headline": s.get("insight_headline", ""),
        "recommended_action": s.get("recommended_action", ""),
        "gps_detection_count": s.get("gps_detection_count", 0),
        "has_map": s.get("has_map", False),
        "hotspot_count": s.get("hotspot_count", 0),
        "hotspots": s.get("hotspots", []),
        "highlights": s.get("highlights", []),
        "video_name": s.get("video_name"),
    }

def clamp_int(value, default, minimum, maximum):
    try:
        return max(minimum, min(maximum, int(value)))
    except Exception:
        return default

def save_session_manifest(sid):
    with sessions_lock:
        s = sessions.get(sid)
        if not s:
            return None
        payload = session_report_payload(s)
        payload["created_at"] = s.get("created_at")
        payload["last_activity"] = s.get("last_activity")
        payload["generated_at"] = time.time()
        payload["video_fps"] = s.get("video_fps", 0.0)
        payload["processing_fps"] = s.get("processing_fps", 0.0)
        payload["_storage"] = {
            "video_path": s.get("video_path"),
            "gpx_path": s.get("gpx_path"),
            "snapshot_dir": s.get("snapshot_dir"),
        }

    try:
        manifest_name = f"{secure_filename(payload.get('video_name') or sid)}_{sid}.json"
        manifest_path = os.path.join(MANIFEST_DIR, manifest_name)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return manifest_path
    except Exception:
        traceback.print_exc()
        return None

def load_recent_manifests(limit=30):
    manifests = []
    try:
        files = sorted(
            [os.path.join(MANIFEST_DIR, name) for name in os.listdir(MANIFEST_DIR) if name.lower().endswith(".json")],
            key=os.path.getmtime,
            reverse=True,
        )
        for path in files[:limit]:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    item = json.load(f)
                item.pop("_storage", None)
                item["manifest_file"] = os.path.basename(path)
                item["manifest_mtime"] = os.path.getmtime(path)
                manifests.append(item)
            except Exception:
                continue
    except Exception:
        pass
    return manifests

def iter_manifest_paths():
    try:
        return sorted(
            [os.path.join(MANIFEST_DIR, name) for name in os.listdir(MANIFEST_DIR) if name.lower().endswith(".json")],
            key=os.path.getmtime,
            reverse=True,
        )
    except Exception:
        return []

def load_manifest_record(sid):
    for path in iter_manifest_paths():
        try:
            with open(path, "r", encoding="utf-8") as f:
                item = json.load(f)
            if item.get("session_id") == sid:
                item["_manifest_path"] = path
                return item
        except Exception:
            continue
    return None

def collect_session_artifact_paths(sid, entry=None, manifest_payload=None):
    paths = []
    sources = []
    if entry:
        sources.append(entry)
    if manifest_payload:
        storage = manifest_payload.get("_storage") or {}
        merged = dict(storage)
        for key in ("video_path", "gpx_path", "snapshot_dir", "csv_path", "pdf_path", "kml_path", "archive_path", "annotated_path"):
            if manifest_payload.get(key) and key not in merged:
                merged[key] = manifest_payload.get(key)
        sources.append(merged)

    for source in sources:
        for key in ("video_path", "gpx_path", "csv_path", "pdf_path", "kml_path", "archive_path", "annotated_path", "snapshot_dir"):
            value = source.get(key)
            if value:
                paths.append(value)

    if sid:
        paths.append(os.path.join(SNAPSHOT_DIR, sid))

    deduped = []
    seen = set()
    for path in paths:
        resolved = resolve_managed_path(path)
        if resolved and resolved not in seen:
            seen.add(resolved)
            deduped.append(resolved)
    return deduped

def delete_session_records(sid):
    manifest_payload = load_manifest_record(sid)
    entry = cleanup_session(sid)
    if not entry and not manifest_payload:
        return {"found": False, "session_id": sid, "deleted_files": 0}

    deleted_files = 0
    for path in collect_session_artifact_paths(sid, entry=entry, manifest_payload=manifest_payload):
        deleted_files += delete_path_if_managed(path)

    manifest_path = (manifest_payload or {}).get("_manifest_path")
    if manifest_path:
        deleted_files += delete_path_if_managed(manifest_path)

    return {
        "found": True,
        "session_id": sid,
        "deleted_files": deleted_files,
        "had_manifest": bool(manifest_payload),
        "was_active": bool(entry),
    }

def delete_all_session_records():
    manifest_ids = set()
    for path in iter_manifest_paths():
        try:
            with open(path, "r", encoding="utf-8") as f:
                item = json.load(f)
            sid = item.get("session_id")
            if sid:
                manifest_ids.add(sid)
        except Exception:
            continue

    with sessions_lock:
        active_sids = list(sessions.keys())

    unique_session_ids = set(active_sids) | manifest_ids
    for sid in active_sids:
        cleanup_session(sid)

    deleted_files = 0
    deleted_files += purge_directory_contents(UPLOAD_DIR)
    deleted_files += purge_directory_contents(REPORT_DIR)

    for d in (UPLOAD_DIR, REPORT_DIR, ARCHIVE_DIR, SNAPSHOT_DIR, PHOTO_DIR, LIVE_DIR, MANIFEST_DIR):
        os.makedirs(d, exist_ok=True)

    return {
        "deleted_sessions": len(unique_session_ids),
        "deleted_files": deleted_files,
    }

def warmup_model(frame_width, frame_height, inference_size, use_tracking):
    if model is None or np is None:
        return False
    key = (int(frame_width), int(frame_height), int(inference_size), bool(use_tracking))
    if key in WARMED_INFERENCE_KEYS:
        return True
    try:
        dummy = np.zeros((int(frame_height), int(frame_width), 3), dtype=np.uint8)
        run_inference_on_frame(dummy, use_tracking=bool(use_tracking), inference_size=int(inference_size))
        WARMED_INFERENCE_KEYS.add(key)
        return True
    except Exception:
        traceback.print_exc()
        return False

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
    
    start_c = request.form.get("start_coords")
    end_c = request.form.get("end_coords")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filename = secure_filename(f.filename)
    path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(path):
        base, ext = os.path.splitext(filename)
        filename = f"{base}_{int(time.time())}{ext}"
        path = os.path.join(UPLOAD_DIR, filename)
    f.save(path)
    
    entry = create_session_entry(path, f.filename, start_c, end_c, source_type="video")
    sid = entry["session_id"]
    start_video_processing(sid)
    threading.Thread(target=lambda: auto_cleanup(UPLOAD_DIR, MAX_FILES_UPLOADS), daemon=True).start()
    return jsonify({"status": "uploaded", "session_id": sid, "video_name": f.filename})

@app.route("/upload_photo", methods=["POST"])
def upload_photo():
    if "photo" not in request.files:
        return jsonify({"error": "No photo"}), 400
    f = request.files["photo"]
    if f.filename == "":
        return jsonify({"error": "Empty"}), 400
    if not allowed_image_file(f.filename):
        return jsonify({"error": "Invalid"}), 400
    raw = f.read()
    if not raw:
        return jsonify({"error": "Unreadable"}), 400
    if cv2 is None or np is None:
        return jsonify({"error": "OpenCV unavailable"}), 500
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Decode failed"}), 400

    safe_name = secure_filename(f.filename) or "photo.jpg"
    base, ext = os.path.splitext(safe_name)
    source_filename = f"{base}_{now_ts()}{ext or '.jpg'}"
    source_path = os.path.join(PHOTO_DIR, source_filename)
    with open(source_path, "wb") as out:
        out.write(raw)

    entry = create_session_entry(source_path, safe_name, source_type="photo")
    sid = entry["session_id"]

    out_frame, logs = analyze_frame(
        frame,
        frame_number=1,
        video_time=0.0,
        seen_ids_set=entry["seen_ids"],
        snapshot_dir=entry["snapshot_dir"],
        use_tracking=False,
        log_every_detection=True,
        save_snapshots=True,
    )
    annotated_path, annotated_url = save_frame_image(out_frame, PHOTO_DIR, f"{base}_annotated")
    with sessions_lock:
        s = sessions.get(sid)
        if not s:
            return jsonify({"error": "Session init failed"}), 500
        s["detection_logs"].extend(logs)
        s["processed_frames"] = 1
        s["total_frames"] = 1
        s["annotated_path"] = annotated_path
        s["annotated_url"] = annotated_url
    update_session_summary_fields(sid)
    save_csv_for_session(sid)
    create_archive_for_session(sid)
    with sessions_lock:
        s = sessions.get(sid)
        payload = session_report_payload(s)
    payload["status"] = "analyzed"
    payload["session_id"] = sid
    payload["image_data"] = encode_frame_to_data_url(out_frame)
    payload["frame_detections"] = len(logs)
    return jsonify(payload)

@app.route("/live/start", methods=["POST"])
def live_start():
    data = request.get_json(silent=True) or {}
    source_name = secure_filename(data.get("source_name") or "live_dashcam") or "live_dashcam"
    start_c = data.get("start_coords")
    end_c = data.get("end_coords")
    entry = create_session_entry("", source_name, start_c, end_c, source_type="live")
    sid = entry["session_id"]
    live_config = {
        "frame_width": clamp_int(data.get("frame_width"), LIVE_FRAME_WIDTH, 320, 1920),
        "frame_height": clamp_int(data.get("frame_height"), LIVE_FRAME_HEIGHT, 180, 1080),
        "inference_size": clamp_int(data.get("inference_size"), LIVE_INFERENCE_SIZE, 320, 1280),
        "jpeg_quality": clamp_int(data.get("jpeg_quality"), 78, 45, 95),
        "save_snapshots": bool(data.get("save_snapshots", LIVE_SAVE_SNAPSHOTS)),
        "use_tracking": bool(data.get("use_tracking", LIVE_USE_TRACKING)),
    }
    with sessions_lock:
        s = sessions.get(sid)
        if s:
            s["live_config"] = live_config
    warmed = warmup_model(
        live_config["frame_width"],
        live_config["frame_height"],
        live_config["inference_size"],
        live_config["use_tracking"],
    )
    return jsonify({"status": "started", "session_id": sid, "source_type": "live", "live_config": live_config, "warmed": warmed})

@app.route("/live/frame/<sid>", methods=["POST"])
def live_frame(sid):
    with sessions_lock:
        s = sessions.get(sid)
        if not s or s.get("source_type") != "live":
            return jsonify({"error": "Invalid"}), 404
        if not s.get("live_active", False):
            return jsonify({"error": "Stopped"}), 409
        live_config = dict(s.get("live_config") or {})

    frame_upload = request.files.get("frame")
    frame = read_image_from_upload(frame_upload)
    if frame is None:
        return jsonify({"error": "No frame"}), 400

    now = time.time()
    with sessions_lock:
        s = sessions.get(sid)
        if not s:
            return jsonify({"error": "Invalid"}), 404
        prev_frame_at = s.get("last_frame_received_at")
        if prev_frame_at:
            current_fps = 1.0 / max(now - prev_frame_at, 1e-6)
            old_fps = s.get("video_fps", 0.0)
            s["video_fps"] = round(current_fps if old_fps <= 0 else (old_fps * 0.65 + current_fps * 0.35), 2)
        s["last_frame_received_at"] = now
        frame_number = s.get("processed_frames", 0) + 1
        video_time = max(0.0, now - s.get("created_at", now))
        snapshot_dir = s.get("snapshot_dir")
        seen_ids = s.get("seen_ids")

    started = time.perf_counter()
    out_frame, logs = analyze_frame(
        frame,
        frame_number=frame_number,
        video_time=video_time,
        seen_ids_set=seen_ids,
        snapshot_dir=snapshot_dir,
        use_tracking=bool(live_config.get("use_tracking", LIVE_USE_TRACKING)),
        log_every_detection=False,
        save_snapshots=bool(live_config.get("save_snapshots", LIVE_SAVE_SNAPSHOTS)),
        target_size=(
            live_config.get("frame_width", LIVE_FRAME_WIDTH),
            live_config.get("frame_height", LIVE_FRAME_HEIGHT),
        ),
        inference_size=live_config.get("inference_size", LIVE_INFERENCE_SIZE),
    )
    processing_ms = round((time.perf_counter() - started) * 1000.0, 1)
    jpeg_quality = live_config.get("jpeg_quality", 78)
    annotated_bytes = encode_frame_to_jpeg(out_frame, quality=jpeg_quality)

    with sessions_lock:
        s = sessions.get(sid)
        if not s:
            return jsonify({"error": "Invalid"}), 404
        if logs:
            s["detection_logs"].extend(logs)
            summary = summarize_logs(list(s.get("detection_logs", [])))
            apply_summary_to_session(s, summary)
        s["processed_frames"] = frame_number
        s["duration_s"] = video_time
        s["last_activity"] = time.time()
        s["last_live_frame_bytes"] = annotated_bytes
        response = {
            "status": "processed",
            "session_id": sid,
            "source_type": "live",
            "frame_number": frame_number,
            "new_detections": len(logs),
            "total_detections": s["total_detections"],
            "total_area": s["total_area"],
            "avg_confidence": s["average_confidence"],
            "severity_counts": s.get("severity_counts", default_severity_counts()),
            "unique_hazards": s.get("unique_hazards", 0),
            "top_severity": s.get("top_severity", "None"),
            "road_health_score": s.get("road_health_score", 100),
            "health_band": s.get("health_band", "Excellent"),
            "maintenance_priority": s.get("maintenance_priority", "Low"),
            "insight_headline": s.get("insight_headline", ""),
            "recommended_action": s.get("recommended_action", ""),
            "hotspot_count": s.get("hotspot_count", 0),
            "hotspots": s.get("hotspots", []),
            "highlights": s.get("highlights", []),
            "video_fps": round(s.get("video_fps", 0.0), 2),
            "processing_fps": round(s.get("processing_fps", 0.0), 2),
            "processing_ms": processing_ms,
            "image_data": encode_jpeg_bytes_to_data_url(annotated_bytes),
        }
    return jsonify(response)

@app.route("/live/stop/<sid>", methods=["POST"])
def live_stop(sid):
    with sessions_lock:
        s = sessions.get(sid)
        if not s or s.get("source_type") != "live":
            return jsonify({"error": "Invalid"}), 404
        s["live_active"] = False
        s["last_activity"] = time.time()
        final_frame_bytes = s.get("last_live_frame_bytes")
        video_name = secure_filename(s.get("video_name") or f"session_{sid}")
    if final_frame_bytes:
        annotated_path, annotated_url = save_frame_image(None, LIVE_DIR, f"{video_name}_live", jpeg_bytes=final_frame_bytes)
        with sessions_lock:
            s = sessions.get(sid)
            if s:
                s["annotated_path"] = annotated_path
                s["annotated_url"] = annotated_url
    save_csv_for_session(sid)
    create_archive_for_session(sid)
    with sessions_lock:
        s = sessions.get(sid)
        payload = session_report_payload(s)
    payload["status"] = "stopped"
    payload["session_id"] = sid
    return jsonify(payload)

@app.route("/video_feed/<sid>")
def video_feed(sid):
    with sessions_lock:
        s = sessions.get(sid)
        if not s or s.get("source_type") != "video":
            return jsonify({"error": "Invalid"}), 404
    start_video_processing(sid)
    return Response(mjpeg_stream_for_session(sid), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/progress/<sid>")
def progress(sid):
    with sessions_lock:
        s = sessions.get(sid)
        if not s: return jsonify({"error": "Invalid"}), 404
        proc, total, fps = s.get("processed_frames", 0), s.get("total_frames", 0), s.get("processing_fps", 0)
        v_fps = s.get("video_fps", 0.0)
    pct = (proc / total * 100.0) if total > 0 else 0.0
    eta = (total - proc) / (fps + 1e-6) if fps > 0 else None
    return jsonify({
        "processed_frames": proc, 
        "total_frames": total, 
        "progress_percent": round(pct, 2), 
        "video_fps": round(v_fps, 2),
        "processing_fps": round(fps, 2), 
        "estimated_time_left_s": round(eta, 1) if eta else None
    })

@app.route("/detection_count/<sid>")
def detection_count(sid):
    with sessions_lock:
        s = sessions.get(sid)
        if not s: return jsonify({"error": "Invalid"}), 404
        logs = list(s.get("detection_logs", []))
    return jsonify(summarize_logs(logs))

@app.route("/processing_status/<sid>")
def processing_status(sid):
    with sessions_lock:
        s = sessions.get(sid)
        if not s: return jsonify({"error": "Invalid"}), 404
        worker = s.get("worker")
        producer = s.get("producer")
        proc = bool(
            s.get("processing_active")
            or bool(worker and worker.is_alive())
            or bool(producer and producer.is_alive())
            or s.get("streaming")
            or s.get("archiving")
            or s.get("live_active")
        )
        complete = bool(s.get("processing_complete"))
        queued = bool(s.get("source_type") == "video" and not complete and not proc)
    return jsonify({"processing": proc, "complete": complete, "queued": queued})

@app.route("/last_report/<sid>")
def last_report(sid):
    with sessions_lock:
        s = sessions.get(sid)
        if not s or not s.get("csv_path"): return jsonify({"error": "No report"}), 404
        return jsonify(session_report_payload(s))

@app.route("/recent_reports")
def recent_reports():
    try:
        limit = clamp_int(request.args.get("limit"), 20, 1, 100)
    except Exception:
        limit = 20
    return jsonify({"items": load_recent_manifests(limit=limit)})

@app.route("/session_map/<sid>")
def session_map(sid):
    loaded = load_session_data(sid)
    if not loaded:
        return jsonify({"error": "Invalid session"}), 404
    payload = build_map_payload(sid, loaded.get("payload") or {}, loaded.get("logs") or [])
    if not payload:
        return jsonify({"error": "No GPS data for this session"}), 404
    return jsonify(payload)

@app.route("/sessions/<sid>", methods=["DELETE"])
def delete_session_route(sid):
    result = delete_session_records(sid)
    if not result.get("found"):
        return jsonify({"error": "Invalid session"}), 404
    return jsonify({"status": "deleted", **result})

@app.route("/sessions", methods=["DELETE"])
def delete_all_sessions_route():
    result = delete_all_session_records()
    return jsonify({"status": "deleted", **result})

@app.route("/export_pdf/<sid>")
def export_pdf(sid):
    with sessions_lock:
        s = sessions.get(sid)
        if not s: return jsonify({"error": "Invalid"}), 404
        pdf_path = s.get("pdf_path")
    if not pdf_path or not os.path.exists(pdf_path):
        pdf_path = generate_pdf_report(sid)
    if pdf_path and os.path.exists(pdf_path):
        return send_file(pdf_path, mimetype="application/pdf", as_attachment=True, download_name=os.path.basename(pdf_path))
    return "PDF generation failed or ReportLab not installed.", 500

# --- NEW: Route to trigger/download KML ---
@app.route("/export_kml/<sid>")
def export_kml(sid):
    with sessions_lock:
        s = sessions.get(sid)
        if not s: return jsonify({"error": "Invalid"}), 404
        kml_path = s.get("kml_path")
    
    if not kml_path or not os.path.exists(kml_path):
        kml_path = generate_kml_report(sid)
    
    if kml_path and os.path.exists(kml_path):
        return send_file(kml_path, mimetype="application/vnd.google-earth.kml+xml", as_attachment=True, download_name=os.path.basename(kml_path))
    
    return "KML generation failed (No GPS data?).", 500
# ------------------------------------------

@app.route("/reports/<path:filename>")
def serve_report(filename):
    req = os.path.abspath(os.path.join(REPORT_DIR, filename))
    if not safe_commonpath(os.path.abspath(REPORT_DIR), req) or not os.path.exists(req): return "Not found", 404
    mime = "text/csv"
    if filename.endswith(".pdf"): mime = "application/pdf"
    if filename.endswith(".kml"): mime = "application/vnd.google-earth.kml+xml"
    if filename.endswith(".jpg") or filename.endswith(".jpeg"): mime = "image/jpeg"
    if filename.endswith(".png"): mime = "image/png"
    if filename.endswith(".webp"): mime = "image/webp"
    return send_file(req, mimetype=mime, as_attachment=True, download_name=os.path.basename(req))

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
def health():
    with sessions_lock:
        active_sessions = len(sessions)
    return jsonify({"device": DEVICE, "gpu": USE_CUDA, "active_sessions": active_sessions, "tracker": TRACKER_NAME})

def session_ttl_pruner():
    while True:
        time.sleep(60)
        with sessions_lock:
            to_del = [sid for sid, s in sessions.items() if time.time() - s.get("last_activity", 0) > SESSION_TTL]
        for sid in to_del:
            cleanup_session(sid)

if __name__ == "__main__":
    for f in [lambda: auto_cleanup(UPLOAD_DIR, MAX_FILES_UPLOADS), lambda: auto_cleanup(REPORT_DIR, MAX_FILES_REPORTS), lambda: auto_cleanup(ARCHIVE_DIR, MAX_FILES_ARCHIVES), lambda: auto_cleanup(SNAPSHOT_DIR, MAX_FILES_REPORTS)]:
        threading.Thread(target=f, daemon=True).start()
    threading.Thread(target=session_ttl_pruner, daemon=True).start()
    app.run(host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", 7860)), debug=False, threaded=True)
