# Real-time Object Counting with YOLO and OpenCV

## Overview
This project implements a real-time object detection and counting system using YOLO (You Only Look Once) and OpenCV. The system can detect and count multiple objects in a live video feed from a webcam, displaying bounding boxes, class labels, and running counts for each detected object class.

## Features
- Real-time object detection using YOLOv8
- Live object counting with cumulative counts
- Bounding box visualization with confidence scores
- FPS (Frames Per Second) display
- Clean exit with 'q' key
- Error handling and graceful shutdown

## Requirements
- Python 3.8+
- OpenCV (cv2)
- Ultralytics YOLO
- NumPy

## Installation
1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download YOLO weights (automatically handled on first run):
The system will automatically download the YOLOv8n weights on first run. Alternatively, you can manually download them from the Ultralytics repository.

## Usage
1. Run the object counter:
```bash
python final_object_counter.py
```

2. The program will:
   - Open your webcam feed
   - Display the video with object detections
   - Show running counts for each detected object class
   - Display current FPS

3. To quit:
   - Press 'q' to exit the program
   - The program will clean up resources automatically

## Implementation Details
The project consists of two main components:

1. `ObjectCounter` class:
   - Handles YOLO model initialization
   - Processes frames for object detection
   - Maintains object counts
   - Calculates and displays FPS
   - Draws bounding boxes and labels

2. Main execution loop:
   - Manages video capture
   - Handles user input
   - Ensures proper cleanup on exit

## Performance Considerations
- The system uses YOLOv8n by default for optimal performance
- Confidence threshold is set to 0.5 to balance accuracy and false positives
- FPS calculation is updated every 30 frames for smooth display

## Error Handling
The implementation includes comprehensive error handling for:
- Model initialization failures
- Video capture issues
- Frame processing errors
- Graceful shutdown on errors

## Future Improvements
Potential enhancements could include:
- Support for custom YOLO models
- Multiple video source support
- Object tracking across frames
- Export of counting statistics
- GUI for configuration

## Contributing
Feel free to submit issues and enhancement requests!

## License
[Your chosen license]

## Acknowledgments
- Ultralytics for YOLOv8
- OpenCV team
- Python community 