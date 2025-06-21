# YOLOv4 Real-time Object Detection

## Introduction

YOLOv4 (You Only Look Once version 4) is a state-of-the-art object detection model that improves upon YOLOv3 with several key enhancements:

- Improved backbone network (CSPDarknet53)
- Better feature pyramid network (PANet)
- Enhanced data augmentation techniques
- Improved loss function
- Better training strategies

These improvements result in higher accuracy while maintaining real-time performance, making it ideal for video-based object detection applications.

## Setup Instructions

### Prerequisites

1. Python 3.6 or higher
2. OpenCV with DNN support
3. NumPy

Install required packages:
```bash
pip install opencv-python numpy
```


## Running the Detection Script

The script supports both webcam and video file input. Here are the available command-line arguments:

```bash
python yolo_v4_video.py [--video VIDEO] [--conf_threshold CONF] [--nms_threshold NMS] [--width WIDTH] [--height HEIGHT]
```

Arguments:
- `--video`: Video source (default: '0' for webcam, or path to video file)
- `--conf_threshold`: Confidence threshold (default: 0.5)
- `--nms_threshold`: Non-maximum suppression threshold (default: 0.4)
- `--width`: Input image width (default: 416)
- `--height`: Input image height (default: 416)

Example usage:
```bash
# Use webcam
python yolo_v4_video.py

# Use video file
python yolo_v4_video.py --video path/to/video.mp4

# Adjust detection thresholds
python yolo_v4_video.py --conf_threshold 0.6 --nms_threshold 0.3
```

## Expected Output

The script will:
1. Open a window showing the video feed with real-time object detections
2. Display bounding boxes around detected objects
3. Show class labels and confidence scores
4. Display current FPS in the top-left corner
5. Press 'q' to quit the application

## Performance Considerations

- The default input size (416x416) provides a good balance between speed and accuracy
- For better performance, you can:
  - Reduce the input size (e.g., 320x320)
  - Increase the confidence threshold to reduce false positives
  - Use GPU acceleration if available

## GPU Acceleration (Optional)

If you're using Linux or WSL with an NVIDIA GPU, you can compile and use Darknet directly for better performance:

1. Clone the Darknet repository:
```bash
git clone https://github.com/AlexeyAB/darknet.git
cd darknet
```

2. Modify `Makefile` to enable GPU support:
```makefile
GPU=1
CUDNN=1
OPENCV=1
```

3. Compile Darknet:
```bash
make
```

4. Run detection using Darknet:
```bash
./darknet detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights
```

## Author

Muntasir

## References

- [YOLOv4 Paper](https://arxiv.org/abs/2004.10934)
- [AlexeyAB's Darknet Repository](https://github.com/AlexeyAB/darknet)
- [COCO Dataset](https://cocodataset.org/) 