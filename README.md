
---
# 🕳️ PyResearch – Intelligent Pothole Detection System (YOLOv12 + Flask)

> 🚀 Real-Time Computer Vision Dashboard for Automated Pothole Detection and Analysis  
> Built with **YOLOv12**, **OpenCV**, and **Flask** — complete with per-video CSV reporting.

---

## 📘 Overview

**PyResearch** is an AI-powered pothole detection web system built using **YOLOv12** (Ultralytics) and **Flask**.  
It allows users to upload road inspection videos, automatically detects potholes frame-by-frame, calculates the area (m²) and confidence, and generates structured CSV reports per video.

The system is ideal for:
- Road maintenance data collection  
- Infrastructure safety analysis  
- Smart city research projects  
- Computer vision and ML-based road condition monitoring

---

## 🧠 System Architecture

```

```
              ┌────────────────────────────┐
              │     User Uploads Video      │
              └─────────────┬───────────────┘
                            │
                            ▼
              ┌────────────────────────────┐
              │     Flask Backend (app.py)  │
              │  - Manages uploads          │
              │  - Runs YOLOv12 inference   │
              │  - Logs detections          │
              └─────────────┬───────────────┘
                            │
                            ▼
            ┌────────────────────────────┐
            │ YOLOv12 Model (Ultralytics) │
            │ - Frame-wise detection      │
            │ - Area & confidence calc.   │
            └─────────────┬───────────────┘
                            │
                            ▼
     ┌──────────────────────────────────────────────────┐
     │              Flask Web Dashboard (HTML)           │
     │  - Live frame stream                              │
     │  - Dynamic stats update                           │
     │  - Auto CSV download after completion             │
     └──────────────────────────────────────────────────┘
```

```

---

## 📂 Folder Structure

```

Pothole-Computer-Vision-Project/
│
├── app.py                  # Flask backend + YOLO logic
├── best.pt                 # Trained YOLOv12 model weights
├── requirements.txt        # All Python dependencies
│
├── templates/
│   └── index.html          # Frontend dashboard (Flask template)
│
├── uploads/                # Uploaded videos (auto-created)
└── reports/                # CSV results per processed video

````

---

## ⚙️ Installation & Setup

### 1️⃣ Create and Activate Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate      # (Windows)
# OR
source .venv/bin/activate   # (Linux/Mac)
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

> Your `requirements.txt` (example):
>
> ```
> Flask==3.1.2
> torch==2.9.0+cpu
> torchvision==0.24.0+cpu
> opencv-python==4.11.0.86
> numpy==1.26.4
> pandas==2.3.3
> ultralytics @ git+https://github.com/sunsmarterjie/yolov12.git
> supervision==0.26.1
> tqdm==4.67.1
> ```

### 3️⃣ Place YOLO Model File

Copy your trained YOLOv12 model file into the project root:

```
best.pt
```

### 4️⃣ Run Flask Application

```bash
python app.py
```

### 5️⃣ Access Dashboard

Open in your browser:

```
http://127.0.0.1:5000/
```

---

## 🧩 How It Works

1. Upload a road inspection video via the dashboard.
2. YOLOv12 detects potholes frame-by-frame using a confidence threshold (default: 0.28).
3. Each detection is processed for:

   * Bounding box area (converted to m²)
   * Confidence level
   * Severity category (Minor / Moderate / Major / Severe)
4. The live dashboard displays:

   * Total detections
   * Total detected area
   * Average confidence
5. When processing finishes:

   * A summary appears.
   * A downloadable CSV report is auto-generated.

---

## 📊 CSV Report Format

Each processed video generates a unique CSV file:

**File name format:**

```
videoName_detections_YYYYMMDD_HHMMSS.csv
```

### Columns

| Column         | Description                             |
| -------------- | --------------------------------------- |
| `frame_number` | Index of video frame                    |
| `video_time_s` | Timestamp (seconds)                     |
| `timestamp`    | Real date-time during processing        |
| `confidence`   | YOLOv12 confidence (0–1)                |
| `area_m2`      | Calculated pothole area (square meters) |
| `severity`     | Severity category based on area         |

---

## 🧮 Severity Classification

| Area (m²)   | Severity | Color     |
| ----------- | -------- | --------- |
| `< 0.5`     | Minor    | 🟩 Green  |
| `0.5 – 1.5` | Moderate | 🟨 Yellow |
| `1.5 – 3.0` | Major    | 🟧 Orange |
| `> 3.0`     | Severe   | 🟥 Red    |

---

## 🖥️ Frontend (Dashboard UI)

### ✳️ Live Statistics

* YOLO model name (YOLOv12)
* Status (`Processing` / `Idle`)
* Detected pothole count
* Total detected area
* Average confidence
* CSV export button

### ✅ Post-Processing Summary

Displays final:

* Total potholes detected
* Total area (m²)
* Average confidence
* Direct download link to CSV

---

## 🧠 Model Details

| Parameter                | Value                        |
| ------------------------ | ---------------------------- |
| **Model**                | YOLOv12                      |
| **Framework**            | Ultralytics                  |
| **Inference Size**       | 960x960                      |
| **Confidence Threshold** | 0.28                         |
| **IoU Threshold**        | 0.45                         |
| **Input Source**         | Uploaded videos              |
| **Output**               | Frame annotations + CSV file |

> Training was done on a custom **Roboflow pothole dataset**, exported in YOLO format and fine-tuned with Ultralytics YOLOv12.

---

## 📈 Example Summary Output

| Metric          | Example |
| --------------- | ------- |
| Total potholes  | 23      |
| Total area (m²) | 8.46    |
| Avg confidence  | 0.87    |

Auto-generated CSV:
`reports/road1_detections_20251106_174530.csv`

---

## 🧰 Tech Stack

| Component           | Technology              |
| ------------------- | ----------------------- |
| **Frontend**        | HTML5, CSS3, JavaScript |
| **Backend**         | Flask (Python)          |
| **AI Model**        | YOLOv12 (Ultralytics)   |
| **Computer Vision** | OpenCV                  |
| **Data Handling**   | NumPy, Pandas           |
| **File Format**     | CSV Reports             |

---

## 🧾 Research Implementation Note

You can describe this system in your **project documentation** as:

> “The system integrates YOLOv12 for pothole detection and Flask for interactive web deployment.
> Uploaded road videos are processed frame-by-frame, and detected regions are quantified by area and severity.
> The processed data is exported to structured CSV format, supporting detailed post-analysis for urban road maintenance.”

---

## 🧪 Example Results Folder

```
reports/
├── highway_demo_detections_20251106_183012.csv
├── city_street_detections_20251106_191840.csv
└── rural_road_detections_20251106_194300.csv
```

---

## 🧩 Troubleshooting

| Issue                 | Solution                                |
| --------------------- | --------------------------------------- |
| **App not starting**  | Activate venv and install dependencies  |
| **Model not loading** | Ensure `best.pt` path is correct        |
| **Video not visible** | Check browser (Chrome/Edge recommended) |
| **CSV empty**         | Try lowering confidence threshold       |
| **Slow performance**  | Resize video or use GPU                 |

---

## 👨‍💻 Author & Credits

**Developed by:** *VISHVANSHU CHAUHAN*
**Under Guidance of:** *MR. HIMANSHU NANDANWAR* 
**Institution:** *[GL BAJAJ ]*
**Academic Year:** *2025*
**Project Title:** *Pothole Detection and Severity Analysis Using YOLOv12 and Flask*

**Core Tools:** Flask • YOLOv12 • OpenCV • Pandas • NumPy

---

## 🧾 License

This project is released for **academic and research purposes only**.
You are free to reuse, modify, and distribute this code with attribution to
**PyResearch (2025)**.

---

## 💬 Acknowledgment

Special thanks to:

* **Ultralytics Team** – for YOLOv12 framework
* **Roboflow** – for dataset preparation tools
* **Your Institution / Guide** – for academic support

---

## Documented By -- Vishvanshu Chauhan
## Contact -- chauhanvishvanshu@gmail.com