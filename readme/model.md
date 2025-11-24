
---

# 🕳️ **YOLOv12S Pothole Detection Model – Complete Documentation**

This README provides a **single highly-detailed guide** combining all features and capabilities of your YOLOv12S model (`best.pt`).
It includes **detection**, **tracking**, **unique counting**, **deployment**, **benchmarking**, **model info**, **JSON outputs**, and **supported input types**.

---

# ⭐ **1. Overview**

Your model is a **YOLOv12S object detection model** trained on **1 class only**:

```python
{0: "Pothole"}
```

### ✔ Supports:

* Image detection
* Video detection
* Webcam / live stream detection
* ByteTrack object tracking
* Unique pothole counting
* JSON output
* Batch/folder processing
* Inference on NumPy arrays (OpenCV frames)
* Model export (ONNX, TorchScript, TensorRT)
* Model benchmarking & speed test
* Model architecture info
* Training & validation

### ❌ Does NOT support:

* Segmentation / polygon masks
* Multiple road defect classes
* Crack detection
* Pothole depth/size estimation
* Damage severity classification

---

# 🟦 **2. Input Types Supported by the Model**

Your YOLO model accepts:

| Input Type           | Example              | Notes                        |
| -------------------- | -------------------- | ---------------------------- |
| Image file           | `"image.jpg"`        | Automatic detection          |
| Video file           | `"video.mp4"`        | Frame-by-frame               |
| Webcam               | `source=0`           | Real-time                    |
| IP Camera            | `"rtsp://..."`       | Streaming                    |
| HTTP Video Stream    | `"http://..."`       | Streaming                    |
| NumPy array (OpenCV) | `model(frame)`       | Perfect for custom pipelines |
| Folder of images     | `"./images/"`        | Batch detection              |
| List of inputs       | `["1.jpg", "2.jpg"]` | Multi-input batch            |

---

# 🟩 **3. Detection Features**

## 📌 **3.1 Image Detection**

```python
model("image.jpg", save=True)
```

Output includes:

* Bounding boxes
* Confidence
* Class label

---

## 📌 **3.2 Video Detection**

```python
model.predict("video.mp4", save=True)
```

Performs frame-wise pothole detection and saves annotated output.

---

## 📌 **3.3 Real-Time Webcam Detection**

```python
model(source=0, show=True)
```

Supports:

* Default laptop camera
* USB camera
* IP cameras
* RTSP streams

---

## 📌 **3.4 Batch / Folder Detection**

```python
model("images_folder/", save=True)
```

OR batch list:

```python
model(["a.jpg", "b.jpg", "c.jpg"], save=True)
```

---

## 📌 **3.5 Programmatic Detection (Using OpenCV Frames)**

Essential for MJPEG, RTSP pipelines, custom video readers:

```python
import cv2

frame = cv2.imread("img.jpg")
results = model(frame)
```

---

# 🟨 **4. Object Tracking & Unique Counting**

The **most important capability** of your model.

## 📌 **4.1 Object Tracking (ByteTrack)**

```python
model.track(
    source="video.mp4",
    tracker="bytetrack.yaml",
    save=True
)
```

Tracking provides:

* Persistent pothole ID across frames
* Stable tracking even with occlusion
* Motion-aware unique pothole detection

---

## 📌 **4.2 Unique Pothole Counting**

ID is available through:

```python
track_id = int(box.id[0])
```

Count uniquely:

```python
unique_ids = set()

for result in results:
    for box in result.boxes:
        if box.id is not None:
            unique_ids.add(int(box.id[0]))
```

---

# 🟧 **5. Model Information & Performance**

## 📌 **5.1 Model Architecture & Parameters**

```python
model.info()
```

Returns:

* Layer-by-layer breakdown
* Total parameters
* GFLOPs

---

## 📌 **5.2 Benchmarking (Latency & Throughput)**

```python
model.benchmark()
```

Notes:

* Windows may show multiprocessing warnings
* Detection still works perfectly

---

## 📌 **5.3 Optimizing Performance**

Compile model (PyTorch 2.x):

```python
model.compile()
```

Move model to GPU:

```python
model.to("cuda")
```

Enable half precision:

```python
model.half()
```

---

# 🟥 **6. Export the Model for Deployment**

Supports:

* ONNX
* TorchScript
* TensorRT (engine)
* CoreML
* OpenVINO

### Examples:

```python
model.export(format="onnx")
model.export(format="torchscript")
model.export(format="engine")  # TensorRT
```

---

# 🟪 **7. JSON Output (For API & Analytics)**

Save detection output as standardized JSON:

```python
model("img.jpg", save_json=True)
```

---

# 🟫 **8. Training & Evaluation**

## 📌 **8.1 Evaluate (mAP, Precision, Recall)**

```python
model.val()
```

## 📌 **8.2 Retrain or Fine-tune on New Data**

```python
model.train(data="data.yaml", epochs=100)
```

---

# 🟩 **9. Full Feature Summary**

| Feature           | Command                                      |
| ----------------- | -------------------------------------------- |
| Image Detection   | `model("image.jpg")`                         |
| Video Detection   | `model.predict("video.mp4")`                 |
| Live Webcam       | `model(source=0)`                            |
| Object Tracking   | `model.track(..., tracker="bytetrack.yaml")` |
| Unique Counting   | `box.id`                                     |
| Benchmarking      | `model.benchmark()`                          |
| Model Info        | `model.info()`                               |
| Export            | `model.export("onnx")`                       |
| JSON Results      | `save_json=True`                             |
| Numpy Frame Input | `model(frame)`                               |
| Folder Detection  | `model("folder/")`                           |
| Validation        | `model.val()`                                |
| Training          | `model.train()`                              |

---

# 🏁 **10. Summary**

Your **YOLOv12S pothole detection model** is fully production-ready and supports:

* Real-time pothole detection
* Video analytics
* IP camera monitoring
* Automated road inspection
* Unique pothole counting with ByteTrack
* Deployment to mobile, CUDA, or TensorRT

This documentation now contains **every function** your model can perform **with clean examples** and **correct usage**.

---

