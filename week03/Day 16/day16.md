# YOLOv3 Object Detection with OpenCV

## Overview

This project implements real-time object detection using YOLOv3 (You Only Look Once) and OpenCV's DNN module. YOLOv3 is a state-of-the-art object detection algorithm that can detect multiple objects in an image with high accuracy and speed. The implementation uses the pre-trained YOLOv3 model trained on the COCO dataset, which can detect 80 different classes of objects.

## How It Works

### YOLOv3 Architecture
YOLOv3 uses a deep convolutional neural network to detect objects in images. The network divides the input image into a grid and predicts bounding boxes and class probabilities for each grid cell. The key advantages of YOLOv3 include:
- Fast detection speed
- Good accuracy for small objects
- Ability to detect multiple objects simultaneously

### OpenCV DNN Module
The OpenCV DNN module is used to load and run the YOLOv3 model. It provides:
- Efficient loading of pre-trained models
- Hardware acceleration support (CPU/GPU)
- Easy integration with OpenCV's image processing pipeline

## Implementation Details

The script performs the following steps:
1. Loads the YOLOv3 model and configuration
2. Reads class names from COCO dataset
3. Processes input images through the network
4. Applies Non-Max Suppression to filter overlapping detections
5. Draws bounding boxes and labels on detected objects

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- YOLOv3 configuration file (yolov3.cfg)
- YOLOv3 weights file (yolov3.weights)
- COCO class names file (coco.names)

## Installation

```bash
pip install opencv-python numpy
```

## Usage

Run the script with the following command:

```bash
python yolo_opencv_detect.py --image path/to/your/image.jpg
```

Optional arguments:
- `--cfg`: Path to YOLOv3 config file (default: yolov3.cfg)
- `--weights`: Path to YOLOv3 weights file (default: yolov3.weights)
- `--names`: Path to COCO class names file (default: coco.names)
- `--conf`: Confidence threshold (default: 0.5)
- `--nms`: Non-Max Suppression threshold (default: 0.4)

Example:
```bash
python yolo_opencv_detect.py --image test.jpg --conf 0.6 --nms 0.3
```

## Output

The script generates an output image with:
- Bounding boxes around detected objects
- Class labels with confidence scores
- Different colors for different object classes

The output image is saved as 'output_[input_filename]' in the same directory.

## Sample Output

The output image will show:
- Colored bounding boxes around detected objects
- Labels in the format: "class_name: confidence_score"
- Example: "person: 0.95", "car: 0.87", etc.

## Author

Muntasir

## Notes

- The default confidence threshold (0.5) can be adjusted based on your needs
- Higher confidence thresholds will result in fewer but more confident detections
- The NMS threshold helps remove overlapping detections of the same object
- Processing time may vary depending on your hardware and image size 