# YOLO-based Real-Time People/Object Counter

## Project Overview
This project implements a real-time object detection and counting system using YOLOv5 and OpenCV. The system captures video from a webcam, detects people (or any objects) using YOLO, and maintains a count of unique objects using a simple centroid-based tracking algorithm.

## Features
- Real-time object detection using YOLOv5
- Bounding box visualization with confidence scores
- Object tracking using centroid-based approach
- Live count display
- Video recording capability
- Adjustable confidence threshold
- Clean exit with 'q' key

## Tools & Technologies Used
- Python 3.x
- OpenCV (cv2)
- Ultralytics YOLOv5
- NumPy

## Requirements
```bash
pip install ultralytics opencv-python numpy
```

## How to Run
1. Ensure you have all required dependencies installed
2. Make sure you have a working webcam
3. Run the script:
```bash
python object_counter.py
```
4. Press 'q' to quit the application

## Implementation Details

### Key Components

1. **Object Detection**
   - Uses YOLOv5s model for detection
   - Configurable confidence threshold (default: 0.5)
   - Currently set to detect people (class_id 0 in COCO dataset)

2. **Object Tracking**
   - Implements a simple centroid-based tracking algorithm
   - Maintains unique object IDs
   - Uses distance threshold to associate objects between frames

3. **Visualization**
   - Green bounding boxes around detected objects
   - Confidence score display
   - Real-time count overlay
   - Video recording to 'output.mp4'

### Code Structure
- `ObjectCounter` class handles all functionality
- Methods for initialization, tracking, and main loop
- Clean resource management with proper cleanup

## Results and Insights
The system provides:
- Real-time detection and counting
- Smooth tracking of objects
- Visual feedback through bounding boxes and count display
- Recorded video output for later analysis

## Future Improvements
1. Add support for multiple object classes
2. Implement more sophisticated tracking algorithms
3. Add configuration file for easy parameter adjustment
4. Implement object trajectory visualization
5. Add support for multiple camera inputs

## Author
Muntasir

## License
This project is open source and available under the MIT License. 