
---

# ⭐ **System Requirements (for Pothole Detection Project)**

---

# 🖥 **1. Operating System**

✔ Windows 10 / 11 (64-bit)
✔ Linux (Ubuntu 20.04+)
✔ macOS (Intel or M1/M2)

---

# ⚙️ **2. Hardware Requirements**

## **Minimum (Basic Testing Only)**

* CPU: Intel i5 6th gen
* RAM: 8 GB
* Storage: 5 GB free
* GPU: Not required
* Webcam (optional)

💡 Minimum par video detection slow ho sakta hai.

---

## **Recommended (Smooth Video + Tracking)**

* CPU: Intel i5 10th gen / Ryzen 5 or higher
* RAM: 16 GB
* GPU: NVIDIA 1050 Ti / 1650 / RTX 2060 ya upar
* CUDA Support for fast inference
* SSD (better performance)

💡 GPU recommended for 20–30 FPS real-time tracking.

---

## **High Performance (Real-time + Long Videos)**

* CPU: Intel i7/i9 11th gen / Ryzen 7/9
* RAM: 32 GB
* GPU: RTX 3060 / 3070 / 4060 / 4070
* 16+ GB VRAM ideal
* Fast NVMe SSD

---

# 📦 **3. Software Requirements**

### **Python**

* Python 3.10 – 3.12
  (You are using Python 3.11 which is perfect)

### **Libraries**

```txt
ultralytics
opencv-python
numpy
lap        # ByteTrack ke liye required
torch      # PyTorch
```

Install:

```bash
pip install ultralytics opencv-python numpy lap torch
```

---

# 🔥 **4. GPU Requirements (If using NVIDIA)**

### **CUDA**

* CUDA 11.8 or 12.1 recommended

### **CuDNN**

* CuDNN 8.x

### ✔ GPU Acceleration = 5x faster detection

Without GPU → 15–25ms per frame
With RTX GPU → 3–6ms per frame

---

# 🎥 **5. Video Requirements**

Model ideally supports:

* 720p (best for performance)
* 1080p (okay)
* 4K (slow unless you have strong GPU)

---

# 🌐 **6. Internet Requirements**

Not required except for:

* Benchmark first run (downloads coco8 dataset)
* Installing packages

---

# 📝 **7. Optional Tools**

For deployment:

* Flask / FastAPI
* Postman
* ONNX Runtime
* TFLite Interpreter

---
