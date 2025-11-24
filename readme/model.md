# Pothole Detection Model – README

This README explains **exactly** what your YOLO model (best.pt) can do and **how to use every function** clearly and practically.

---

# ⭐ Overview

Your model is a **YOLOV12S detection model** trained only on **1 class: Pothole**.
It supports:

* Image detection
* Video detection
* Object tracking (ByteTrack)
* Real-time webcam detection
* Unique pothole counting
* Model info & benchmarking
* Exporting model to ONNX/TorchScript/etc.

---

# 📌 Model Class

```
{0: "Pothole"}
```

✔ Only **Pothole** detection
❌ No other classes (cars, cracks, road wear etc.)

---

# 🟢 1. Image Detection

Detect potholes in images.

```python
model("image.jpg", save=True)
```

Output: bounding box, confidence, label.

---

# 🟢 2. Video Detection

Frame-by-frame pothole detection.

```python
model.predict("video.mp4", save=True)
```

---

# 🟢 3. Object Tracking (ByteTrack)

Track the **same pothole across frames** using unique IDs.

```python
model.track(
    source="video.mp4",
    tracker="bytetrack.yaml",
    save=True
)
```

Tracking enables:

* Unique pothole identification
* Real-world pothole counting

---

# 🟢 4. Unique Pothole Counting

Use tracking IDs from `box.id`.

```python
track_id = int(box.id[0])
```

Store IDs in a `set()` to get unique count.

---

# 🟢 5. Real-time Webcam Detection

Detect potholes live using your camera.

```python
model(source=0, show=True)
```

---

# 🟢 6. Model Information (Architecture + Params)

Know model layers, parameters, GFLOPs.

```python
model.info()
```

---

# 🟢 7. Benchmarking (Speed Test)

Measures model speed and latency.

```python
model.benchmark()
```

⚠ Windows may show multiprocessing errors – detection is NOT affected.

---

# 🟢 8. Export Model (Mobile / Deployment)

Export to ONNX, TorchScript, TensorRT, CoreML.

```python
model.export(format="onnx")
model.export(format="torchscript")
model.export(format="engine")
```

---

# 🟢 9. JSON Output (API Use)

Useful for backend or analytics.

```python
model("image.jpg", save_json=True)
```

---

# 🟢 10. Folder Batch Detection

Detect potholes from a folder of images.

```python
model("images_folder/", save=True)
```

---

# ❌ What the Model Cannot Do

Your pothole model **CANNOT**:

* Segment potholes (no masks)
* Detect multiple road damages
* Detect cracks
* Give severity (small/medium/deep)
* Classify road quality
* Measure depth/size

These require extra training or additional models.

---

# 🎯 Summary of Functions

| Feature          | Function                     |
| ---------------- | ---------------------------- |
| Image Detection  | `model("img.jpg")`           |
| Video Detection  | `model.predict("video.mp4")` |
| Tracking         | `model.track("video.mp4")`   |
| Unique Count     | `box.id`                     |
| Webcam           | `model(source=0)`            |
| Model Info       | `model.info()`               |
| Benchmark        | `model.benchmark()`          |
| Export           | `model.export()`             |
| JSON Output      | `save_json=True`             |
| Folder Detection | `model("folder/")`           |

---

# ✔ Ready for Real Use

Your model is fully ready for:

* Real-time pothole detection
* Automated road inspection
* Unique pothole counting
* Video analytics systems

---
