# Real-Time Object Detection with OpenCV & YOLOv5

**Author:** Muntasir

## Overview
This project demonstrates real-time object detection using YOLOv5 and OpenCV. The system captures video from a webcam, processes each frame using YOLOv5 for object detection, and displays the results in real-time with bounding boxes, class labels, and confidence scores.

## Features
- Real-time object detection using YOLOv5
- Live webcam feed processing
- Bounding box visualization with class labels and confidence scores
- FPS counter display
- Easy quit functionality (press 'q')
- Optional video recording capability

## Requirements
```bash
pip install ultralytics opencv-python numpy
```

## How to Run
1. Ensure you have a working webcam connected to your system
2. Run the script:
```bash
python yolo_opencv_live.py
```
3. Press 'q' to quit the application

## Code Structure
The implementation consists of several key components:

1. **Model Initialization**
   - Uses YOLOv5s pre-trained model
   - Automatically downloads the model if not present

2. **Video Capture**
   - Opens webcam stream using OpenCV
   - Processes frames in real-time

3. **Object Detection**
   - Runs YOLOv5 inference on each frame
   - Extracts bounding boxes, class labels, and confidence scores

4. **Visualization**
   - Draws bounding boxes around detected objects
   - Displays class names and confidence scores
   - Shows current FPS

## Performance Considerations
- The system's performance depends on your hardware capabilities
- YOLOv5s is a smaller model that provides a good balance between speed and accuracy
- FPS counter helps monitor real-time performance

## Sample Output
The system will display a window showing:
- Live webcam feed
- Green bounding boxes around detected objects
- Class labels with confidence scores
- FPS counter in the top-left corner

## Customization Options
1. **Model Selection**
   - Change `yolov5s.pt` to other YOLOv5 variants:
     - `yolov5n.pt` (nano) - faster but less accurate
     - `yolov5m.pt` (medium) - balanced
     - `yolov5l.pt` (large) - more accurate but slower
     - `yolov5x.pt` (xlarge) - most accurate but slowest

2. **Video Recording**
   - Uncomment the video writer code to save the output
   - Adjust resolution and FPS as needed

## Troubleshooting
1. **Webcam Access Issues**
   - Ensure your webcam is properly connected
   - Check if other applications are using the webcam
   - Try changing the camera index (0) to 1 or 2 if multiple cameras are present

2. **Performance Issues**
   - Try using a smaller YOLOv5 model
   - Reduce input resolution
   - Ensure GPU is being utilized if available

## Future Improvements
1. Add support for multiple object tracking
2. Implement custom object detection classes
3. Add support for video file input
4. Implement multi-threading for better performance
5. Add GUI controls for adjusting detection parameters 