# YOLOv5 Object Detection with Ultralytics

## Overview
This project demonstrates how to use YOLOv5 (You Only Look Once version 5) for real-time object detection using the Ultralytics implementation. YOLOv5 is a state-of-the-art object detection model that can detect multiple objects in images and videos with high accuracy and speed.

## Installation

### Method 1: Using pip
```bash
pip install ultralytics
```

### Method 2: From GitHub
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

## Usage

### Basic Usage
The script `yolo_v5_detect.py` provides a simple interface for object detection:

```bash
# Detect objects in an image
python yolo_v5_detect.py --source path/to/image.jpg

# Use webcam (default)
python yolo_v5_detect.py --source 0

# Use a different model
python yolo_v5_detect.py --model yolov5m.pt

# Adjust confidence threshold
python yolo_v5_detect.py --conf 0.4
```

### Command Line Arguments
- `--source`: Path to image/video or webcam index (default: "0")
- `--model`: Path to model or model name (default: "yolov5s.pt")
- `--conf`: Confidence threshold (default: 0.25)

## Features
- Real-time object detection on images and videos
- Support for webcam input
- Automatic model download
- Bounding box visualization with labels
- Confidence score display
- Result saving functionality

## Model Options
YOLOv5 comes in several sizes:
- `yolov5n.pt`: Nano model (fastest, least accurate)
- `yolov5s.pt`: Small model (default)
- `yolov5m.pt`: Medium model
- `yolov5l.pt`: Large model
- `yolov5x.pt`: Extra large model (slowest, most accurate)

## Custom Training
To train YOLOv5 on a custom dataset:

1. Prepare your dataset in YOLO format:
   - Images and labels in separate directories
   - Labels in YOLO format (normalized coordinates)

2. Create a dataset YAML file:
```yaml
train: path/to/train/images
val: path/to/val/images
nc: number_of_classes
names: ['class1', 'class2', ...]
```

3. Train the model:
```bash
python train.py --img 640 --batch 16 --epochs 100 --data dataset.yaml --weights yolov5s.pt
```

## Output Format
The script generates:
- Console output with detection details (class and confidence)
- Visual output with bounding boxes and labels
- Saved image/video with detections (if input is a file)

## Dependencies
- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics

## Author
Muntasir

## References
- [Ultralytics YOLOv5 GitHub](https://github.com/ultralytics/yolov5)
- [Ultralytics Documentation](https://docs.ultralytics.com) 