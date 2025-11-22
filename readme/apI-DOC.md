
---

# 🚀 **Pothole Detection API — Full Documentation**

Multi-Session YOLO Video Analysis Backend
Supports Local, Netlify, HuggingFace Spaces, and external clients.

---

# 📌 **Base URL**

Your backend may run on any of the following:

| Environment            | Base URL                                 |
| ---------------------- | ---------------------------------------- |
| **Local Development**  | `http://127.0.0.1:7860`                  |
| **HuggingFace Spaces** | `https://<your-space>.hf.space`          |
| **Netlify Frontend**   | Use the configured backend URL in `.env` |

All endpoints below assume `{BASE_URL}` is one of the above.

---

---

# 🔑 **Authentication**

❌ No authentication required (public service).
(Optional: You can add API keys later.)

---

---

# 🎞 **Overview**

This backend performs **real-time YOLO detection on video files**.
Each upload creates a **unique processing session** with:

* MJPEG live stream
* Frame-by-frame inference
* Live stats (FPS, detections, area, confidence)
* Automatic CSV report generation
* Automatic ZIP archive creation
* Full session life-cycle management

---

# 📚 **Endpoints Summary**

| Category   | Endpoint                   | Method | Description                         |
| ---------- | -------------------------- | ------ | ----------------------------------- |
| System     | `/health`                  | GET    | Check server/device status          |
| Upload     | `/upload`                  | POST   | Upload a video → returns session_id |
| Streaming  | `/video_feed/<sid>`        | GET    | MJPEG processed video stream        |
| Progress   | `/progress/<sid>`          | GET    | Frame progress + ETA                |
| Statistics | `/detection_count/<sid>`   | GET    | YOLO detection totals               |
| Status     | `/processing_status/<sid>` | GET    | Processing active/not               |
| Reports    | `/last_report/<sid>`       | GET    | Get report CSV + ZIP URLs           |
| CSV        | `/export_csv/<sid>`        | GET    | Download CSV                        |
| ZIP        | `/download_archive/<file>` | GET    | Download ZIP archive                |
| Reports FS | `/reports/<file>`          | GET    | Serve CSV securely                  |

---

---

# 🚦 **1. Health Check**

### `GET /health`

Returns current device (CPU/GPU), CUDA availability, and active sessions.

#### Example Response

```json
{
  "device": "cpu",
  "gpu": false,
  "active_sessions": 0
}
```

---

---

# 📤 **2. Upload & Start Processing**

### `POST /upload`

Uploads a video and creates a new session.

#### Form-Data Fields

| Field   | Type | Required | Description                                     |
| ------- | ---- | -------- | ----------------------------------------------- |
| `video` | file | ✔        | Video file (`mp4`, `mov`, `avi`, `mkv`, `webm`) |

#### Example Response

```json
{
  "status": "uploaded",
  "session_id": "f3a9d12c8e40",
  "video_name": "road.mp4"
}
```

---

---

# 🎥 **3. Live Processed Video Stream**

### `GET /video_feed/<session_id>`

Returns **real-time MJPEG stream** of YOLO-processed frames.

Use in frontend:

```html
<img src="https://backend/video_feed/<session_id>" />
```

#### Response Type

```
Content-Type: multipart/x-mixed-replace; boundary=frame
```

#### Special Streaming Headers Added

```
Access-Control-Allow-Origin: *
Connection: keep-alive
Cache-Control: no-cache, no-store
X-Accel-Buffering: no
```

---

---

# 📊 **4. Processing Progress**

### `GET /progress/<session_id>`

Returns frame counters, FPS, and estimated time remaining.

#### Example Response

```json
{
  "processed_frames": 142,
  "total_frames": 900,
  "progress_percent": 15.78,
  "video_fps": 30.0,
  "processing_fps": 12.0,
  "estimated_time_left_s": 63.4
}
```

---

---

# 🔍 **5. Real-Time Detection Statistics**

### `GET /detection_count/<session_id>`

Returns YOLO detection stats cumulative so far.

#### Example Response

```json
{
  "detections": 12,
  "total_area": 4.68,
  "avg_confidence": 0.79
}
```

---

---

# 🔄 **6. Processing Status**

### `GET /processing_status/<session_id>`

Returns processing state.

Backend counts as **processing = true** if:

* Worker thread alive
* MJPEG stream running
* Archive (ZIP) creation active

#### Example Response

```json
{
  "processing": true
}
```

---

---

# 📄 **7. Session Reports**

### `GET /last_report/<session_id>`

Returns URLs to CSV & ZIP report files
(after the video has finished processing).

#### Example Response

```json
{
  "csv_path": "reports/road_detections_20250120.csv",
  "csv_url": "https://backend/reports/road_detections_20250120.csv",
  "archive_path": "reports/archives/road_archive_20250120.zip",
  "archive_url": "https://backend/download_archive/road_archive_20250120.zip",
  "total_detections": 37,
  "total_area": 8.22,
  "average_confidence": 0.81,
  "video_name": "road.mp4"
}
```

---

---

# 📥 **8. Download CSV Directly**

### `GET /export_csv/<session_id>`

Returns CSV either:

* dynamically generated from live logs
* or the final saved CSV

File is downloaded as:

```
<video_name>_detections_<timestamp>.csv
```

---

---

# 📦 **9. Download ZIP Archive**

### `GET /download_archive/<filename>`

ZIP archive contains:

* original uploaded video
* final CSV detection report

#### Example Response

(File download)

---

---

# 📁 **10. Serve a Report File (Secure)**

### `GET /reports/<filename>`

Serves CSV from the reports directory.
Path traversal is fully blocked.

Responds with:

```
text/csv
```

---

---

# 🗑 **Background Cleanup Behavior (Automatic)**

| Feature          | Behavior                                 |
| ---------------- | ---------------------------------------- |
| Session TTL      | Auto-delete after 3 hours (default)      |
| Upload cleanup   | Keeps last **20** uploaded videos        |
| CSV cleanup      | Keeps last **50** reports                |
| ZIP cleanup      | Keeps last **20** archives               |
| Session overflow | If more than 12 sessions → delete oldest |

No endpoints needed — backend manages itself.

---

---

# 🔒 **Security**

* Full CORS support
* MJPEG streaming works across origins (Netlify → HF)
* Safe path handling prevents directory traversal
* Unique session IDs per upload
* Optional public host override via `PUBLIC_HOST`

---

---

# 🛠 **Error Response Format**

All errors follow:

```json
{
  "error": "Message here"
}
```

Examples:

```
400 — No file 'video'
404 — Invalid session id
404 — No report yet
403 — Forbidden (path violation)
```

---

---

# 🧪 **Testing Guide**

## Upload video via curl

```bash
curl -X POST -F "video=@test.mp4" http://127.0.0.1:7860/upload
```

## Read MJPEG live stream

Open in browser:

```
http://127.0.0.1:7860/video_feed/<session_id>
```

---

---

