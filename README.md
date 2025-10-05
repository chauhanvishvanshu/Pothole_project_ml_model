# 🕳️ Pothole Detection using Pretrained YOLOv8 Model (Windows)

This project detects **potholes from a video** using a pretrained **YOLOv8 model**.
It processes video input, highlights detected potholes, and displays whether a pothole is present or not.

---

## 🚀 Features

* Detect potholes in any road or dashcam video
* Uses pretrained YOLOv8 model (no training required)
* Shows real-time detection results on screen
* Saves output video with detection overlays

---

## 🧠 Model Used

We are using a **pretrained YOLOv8 model** trained on pothole datasets.

**Model Download:**
[Pothole YOLOv8 Pretrained Model (.pt)](https://github.com/arpy8/Pothole_Detection_YOLOv8/releases/download/v1.0/pothole_best.pt)

---

## 🗂️ Folder Structure

```
pothole-detection/
│
├── input_video.mp4               # Your test video
├── output_pothole_detected.mp4   # Output after detection
├── detect_pothole.py             # Main Python script
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation (Windows)

### 1️⃣ Clone or Download the Repository

```bash
git clone https://github.com/yourusername/pothole-detection.git
cd pothole-detection
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🎥 Run the Model

Make sure your input video is named **input_video.mp4** and placed in the same folder.

```bash
python detect_pothole.py
```

### Output:

* Annotated video: **output_pothole_detected.mp4**
* Real-time window showing detection:

  * 🟥 *POTHOLE DETECTED!* if pothole found
  * 🟩 *NO POTHOLE* if not found

Press **'Q'** to quit the window.

---

## 🧰 Example Test Video

Download a sample pothole video to test:

* [Mixkit Free Pothole Road Video](https://mixkit.co/free-stock-video/potholes-in-a-rural-road-25208/)
* [Mendeley Pothole Dataset](https://data.mendeley.com/datasets/tp95cdvgm8/1)

Save the file as `input_video.mp4` inside your project folder.

---

## 📦 requirements.txt

```
ultralytics==8.2.31
opencv-python==4.10.0.84
numpy==1.26.4
```

---

## 📧 Future Enhancements

* Store detections in a CSV file (timestamp + frame)
* Add GPS-based location logging
* Send email alerts to municipal authorities with pothole info

---

## 👨‍💻 Author

**Your Name**
B.Tech CSE (AI) Student
Project: Pothole Detection System (Windows Version)
