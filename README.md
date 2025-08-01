# YOLOv11 Drone Object Detection & Tracking 🚁🔍

This project demonstrates how to use a custom-trained [YOLOv11s](https://github.com/ultralytics/ultralytics) model for real-time object detection and autonomous tracking using a DJI Tello drone. The pipeline includes dataset preparation, model training, and various modes of deployment including webcam, video files, and live drone input.

---

## 📁 Project Structure

```
DroneProject/
├── data/
│   ├── train/
│   │   ├── images/        # Training images
│   │   └── labels/        # YOLO-format labels (.txt)
│   ├── validation/
│   │   ├── images/        # Validation images
│   │   └── labels/        # Validation labels
│   └── config.yaml        # Dataset config for YOLO training
├── train.py               # Model training script
├── drone.py               # Core class for all detection script with drone integration
├── yolo11s.pt             # Pretrained or trained YOLOv11s
├──runs/ 
│  └── detect/
│      └── train/
│      │   └──weights/
│      │      ├──best.pt   # best trained model on our data
│      │      └──last.pt   # last trained model on our data
│      │
│      └──results.png      # Training performance visualization
└── README.md              
```

---

## 1. 🖼️ Dataset Preparation

1. Images were collected and labeled using [LabelImg](https://github.com/tzutalin/labelImg).
2. Annotation format used: **YOLO format** (label files with `.txt` extensions).
3. The object class in this project: `toy`.

The dataset structure:

```
data/
├── train/
│   ├── images/
│   └── labels/
├── validation/
│   ├── images/
│   └── labels/
└── config.yaml
```

**Example `config.yaml`:**

```yaml
path: /DroneProject/data
train: train/images
val: validation/images

names:
  0: toy
```

---

## 2. 🏋️ Model Training

To train a custom model using the Ultralytics `YOLOv11s` implementation:

**`main.py`:**
```python
from ultralytics import YOLO

def main():
    model = YOLO("yolo11s.pt")  # Load pretrained YOLOv11s model
    model.train(
        data="config.yaml",
        epochs=100,
        batch=32,
        device=0
    )

if __name__ == "__main__":
    main()
```

### 📊 Training Results

The training shows significant improvements across loss metrics and high accuracy in evaluation metrics. Below is the visual output from training:

![Training Results](runs\detect\train\results.png)

---

## 3. 🚁 YOLOv11 Integration and Real-Time Use — `drone.py`

A comprehensive class that integrates the trained model and provides multiple modes of prediction and control.

### ✨ Class: `model`

#### ✅ `realTimePredict(device, confidence)`
Uses a connected camera or webcam to run real-time object detection.

#### 🎞️ `videoPredict(videoPath, confidence)`
Reads a video, performs detection frame-by-frame, and saves an annotated output to `out.mp4`.

#### 🔋 `getBattery()`
Connects to the DJI Tello drone and displays the current battery percentage.

#### 🕹️ `realTimeDrone(confidence)`
Runs real-time detection from the drone's camera. You can manually control the drone with your keyboard.

**Keyboard Controls:**
```
T = takeoff        L = land
W/A/S/D = move     R/F = up/down
Q/E = rotate       Z/X/C/V = flips
B = battery        ESC = exit
```

#### 🎯 `pursueObject(confidence)`
Autonomously tracks and follows the most confidently detected object by adjusting drone direction and distance in real-time.

---

## 🧠 Technologies Used

- YOLOv11 (Ultralytics)
- OpenCV (real-time image processing)
- DJITelloPy (Python SDK for DJI Tello)
- pynput (keyboard control for drone navigation)

---

## 🛠️ Installation Requirements

```bash
pip install ultralytics opencv-python djitellopy pynput
```


## 📝 Author

**Aram Sargsyan**