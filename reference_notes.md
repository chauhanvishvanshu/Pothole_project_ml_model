# 🧠 Project Resources & References  
> Personal developer documentation for my pothole detection project (YOLOv12 + Flask)  
> Created & maintained by **Vishvanshu Chauhan**  

---

### 📅 Last Updated: *November 2025*  
### 🧩 Project: *Pothole Detection and Severity Analysis using YOLOv12 and Flask*  

---

## 📘 About This Document
This file contains all **external resources, datasets, training links, dependencies, and setup notes**  
used while developing the project.  

It’s meant for **personal reference only** — so that I can revisit any part of the project in the future  
without searching through multiple sources or code files.

---

## 📺 **Tutorials & Learning Materials**
- **YouTube Tutorial:** [YOLOv12 Object Detection Explained](https://www.youtube.com/watch?v=X5CWP_YEiP4)  
  *Helped understand YOLOv12 architecture, configuration, and training workflow.*

---

## 📊 **Datasets**
- **Pothole Detection Dataset (Roboflow Universe):**  
  [https://universe.roboflow.com/potholes-r7qcn/pothole-jujbl/dataset/1](https://universe.roboflow.com/potholes-r7qcn/pothole-jujbl/dataset/1)  
  *Used for model training — contains annotated road images (YOLO format) with pothole bounding boxes.*

---

## 💻 **Training & Implementation Resources**
- **Google Colab Notebook (YOLOv12 Training):**  
  [https://colab.research.google.com/github/pyresearch/notebooks/blob/main/notebook/train_yolov12_object_detection.ipynb](https://colab.research.google.com/github/pyresearch/notebooks/blob/main/notebook/train_yolov12_object_detection.ipynb)  
  *Base structure for YOLOv12 model training and fine-tuning pipeline.*

- **Reference Project (Pothole Computer Vision):**  
  [https://github.com/pyresearch/Pothole-Computer-Vision-Project](https://github.com/pyresearch/Pothole-Computer-Vision-Project)  
  *Used for understanding Flask integration, inference logic, and video stream processing.*

- **YOLOv12 Implementation Repository:**  
  [https://github.com/sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12)  
  *Official YOLOv12 repo used to install `ultralytics` directly via Git link in `requirements.txt`.*

---

## 🚀 **My Project Repository**
- **Pothole Detection ML Model (by Vishvanshu Chauhan):**  
  [https://github.com/chauhanvishvanshu/Pothole_project_ml_model](https://github.com/chauhanvishvanshu/Pothole_project_ml_model)  
  *Final implementation combining YOLOv12, Flask, and OpenCV for real-time pothole detection and severity analysis.*

---

## ⚙️ **Environment Setup (Quick Reference)**

### 🔧 Create Virtual Environment
```bash
python -m venv .venv
# Activate environment:
.venv\Scripts\activate      # (Windows)
source .venv/bin/activate   # (Linux/Mac)
📦 Install Dependencies
bash
Copy code
pip install -r requirements.txt
🧠 Run the Application
bash
Copy code
python app.py
🌐 Access Web Dashboard
cpp
Copy code
http://127.0.0.1:5000/
📋 Dependencies Summary (requirements.txt)
txt
Copy code
# Core AI & Computer Vision
torch>=2.5.0
torchvision>=0.24.0
ultralytics @ git+https://github.com/sunsmarterjie/yolov12.git
opencv-python==4.11.0.86
numpy==1.26.4
pandas==2.3.3
supervision==0.26.1

# Flask Web Framework
Flask==3.1.2
Werkzeug==3.1.3
itsdangerous==2.2.0
click==8.3.0
Jinja2==3.1.6

# Optional / Supporting Dependencies
roboflow==1.2.11
requests==2.32.5
tqdm==4.67.1
PyYAML==6.0.1
python-dotenv==1.2.1
huggingface_hub
🧩 Notes for Future Reference
Model Used: YOLOv12 (ultralytics package)

Training Source: Roboflow pothole dataset

Frameworks: PyTorch + Flask

Core Libraries: OpenCV, Pandas, NumPy, Supervision

Output: Frame-wise pothole detections, confidence scores, and CSV-based reports

Purpose: To automate pothole identification, severity categorization, and area quantification from video data

🗂️ File/Folder Recap
bash
Copy code
Pothole_project_ml_model/
│
├── app.py               # Flask backend with YOLO inference
├── best.pt              # YOLOv12 trained model weights
├── requirements.txt     # Python dependencies
│
├── templates/
│   └── index.html       # Frontend (Flask template)
│
├── uploads/             # Uploaded videos
└── reports/             # CSV results for each processed video

📚 Quick Reminders
If you retrain the model → update best.pt and note date/version here.

If new dependencies are added → update both requirements.txt and this file.

Use git pull before modifying to avoid version mismatches.

You can use this document as your own developer changelog for future updates.

<p align="center"> <sub>🧠 Personal Dev Notes by <b>Vishvanshu Chauhan</b> • Updated November 2025</sub><br> <sup>YOLOv12 + Flask | Computer Vision | Road Safety AI</sup> </p> ```
