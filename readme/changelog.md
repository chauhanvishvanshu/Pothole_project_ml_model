

---

# 🧾 **PyResearch — Pothole Detection System**

### **Changelog**

---

## 🧩 **v3 — Production-Ready YOLOv12 System**

**Release Date:** *Dec 2025*
**Core Model:** YOLOv12 (best.pt)

### ✨ **Major Enhancements**

* 🧱 **Stabilized Real-Time Engine** — Complete backend rewrite using a production-safe worker thread + non-blocking frame queues.
* 💾 **Guaranteed CSV Generation** — CSV file is always produced, even if *zero* detections are found.
* 📦 **ZIP Report Packaging** — Automatically creates a ZIP containing the processed video + CSV.
* 🌐 **Background-Safe URL Handling** — Eliminates Flask context errors using a thread-safe absolute URL builder.
* 🧠 **YOLOv12 Robustness Update** — Improved compatibility for different YOLO output formats; safer extraction of boxes & confidence.
* 📈 **Real-Time Performance Metrics** — Accurate `processing FPS`, `video FPS`, and `ETA` reported live.
* 🎥 **Smoother Streaming** — Lower latency, frame skip support, and stable MJPEG pipeline during long video processing.
* 🧮 **Extended Log Details** — Logs now include frame timestamps, severity, area (m²), and processed-time metadata.
* 📨 **Notification Support** — Optional email and webhook alerts when reports finish generating.
* 🧹 **Auto-Cleanup System** — Automatically trims old uploads, reports, and archives.
* ⚙️ **Backend–Frontend Split** — Clear folder separation for deployment:

  * `/backend` → Flask + YOLO
  * `/frontend` → Dashboard UI
* ⚡ **One-Click Automation Scripts Added**

  * `setup.bat` — Automatically sets up environment + installs dependencies
  * `run.bat` — Launches backend + frontend + browser in one click

---

### ⚡ **Performance Gains**

| Metric            | v2                     | v3                               |
| ----------------- | ---------------------- | -------------------------------- |
| Report Generation | CSV only (conditional) | CSV + ZIP (always generated)     |
| Backend Stability | High                   | Very High (production-safe)      |
| Stream Smoothness | Smooth                 | Ultra-smooth (no queue blocking) |
| Error Handling    | Limited                | Robust recovery & isolation      |
| Deployment        | Manual                 | Automated (setup.bat + run.bat)  |

---

## 🧩 **v2 — Hybrid Real-Time GPU Pipeline**

**Release Date:** *Nov 09, 2025*
**Core Model:** YOLOv12 (best.pt)

### ✨ Major Enhancements

* 🚀 Hybrid GPU streaming pipeline (smooth playback).
* ⚙️ CUDA acceleration with CPU fallback.
* 📊 Live dashboard with FPS, Speed%, ETA.
* 🔁 Multi-video processing without restart.
* 💾 Improved logging + CSV export.
* 🎯 Up to 40% faster detection with full accuracy.

---

### ⚡ Performance Gains

| Metric               | v1       | v2                    |
| -------------------- | -------- | --------------------- |
| Processing FPS       | ~4–8     | ~16–25                |
| Video Smoothness     | Moderate | Seamless (no stutter) |
| GPU Utilization      | Partial  | Fully optimized       |
| Multi-Video Handling | ❌        | ✅ Supported           |

---

## 🧠 **v1 — Base Real-Time Detection System**

**Release Date:** *Oct 2025*
**Core Model:** YOLOv12 (best.pt)

### 🧩 Initial Features

* 🎥 Real-time pothole detection using YOLOv12.
* 🧮 Live metrics: count, area, confidence.
* 💻 OpenCV frame-by-frame inference.
* 🧱 Basic Flask dashboard.
* 📂 Simple folder structure.

### 🧩 Known Limitations

* ❗ Low FPS on CPU.
* ⚠️ Frame stutter during streaming.
* 🔁 Single-run only.
* ❌ No ETA / Processing FPS indicators.

---

## 🧭 **Version Summary**

| Version | Focus                                     | Performance    | Accuracy | Stability      |
| ------- | ----------------------------------------- | -------------- | -------- | -------------- |
| **v1**  | Baseline YOLOv12 System                   | 🟡 Medium      | 🟢 High  | 🟡 Medium      |
| **v2**  | Hybrid GPU + Smooth Real-Time Pipeline    | 🟢 High        | 🟢 High  | 🟢 High        |
| **v3**  | Production Build + Automation + Reporting | 🟢🟢 Very High | 🟢 High  | 🟢🟢 Very High |

---

## 🏁 **Current Status**

> ✅ **v3 is the recommended production version** with stable backend, guaranteed reporting, smooth streaming, and one-click automation.
> 💡 Ideal setup: *Python 3.11 + CUDA 12.x + RTX GPU (optional)*.

---

