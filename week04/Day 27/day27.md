# Real-time Object Counting with YOLOv5

This project implements a real-time object detection and counting system using YOLOv5 and OpenCV. The system captures video from a webcam, detects objects using YOLOv5, and displays the count of detected objects in real-time.

## Features

- Real-time object detection using YOLOv5
- Live webcam feed processing
- Bounding box visualization with confidence scores
- Real-time object counting
- Clean, modular code structure

## Requirements

```bash
pip install torch torchvision
pip install opencv-python
pip install ultralytics
```

## Project Structure

- `final_object_counter.py`: Main script containing the object detection and counting implementation
- `day27.md`: This documentation file

## Implementation Details

### ObjectCounter Class

The `ObjectCounter` class handles the core functionality:

1. **Initialization**:
   - Loads YOLOv5 model with pre-trained weights
   - Sets target class and confidence threshold
   - Initializes counting variables

2. **Frame Processing**:
   - Runs YOLO inference on each frame
   - Filters detections by target class and confidence
   - Draws bounding boxes and labels
   - Updates and displays count

### Main Loop

The main loop:
1. Captures frames from webcam
2. Processes each frame through the ObjectCounter
3. Displays the processed frame
4. Handles user input for quitting

## Usage

1. Install the required dependencies
2. Run the script:
   ```bash
   python final_object_counter.py
   ```
3. Press 'q' to quit the application

## Customization

You can modify the following parameters in the code:
- `target_class`: Change the object class to detect (default: "person")
- `confidence_threshold`: Adjust detection confidence threshold (default: 0.5)

## Future Improvements

1. Implement object tracking to avoid double counting
2. Add support for multiple target classes
3. Implement region-based counting
4. Add GUI controls for parameter adjustment
5. Optimize performance for lower-end systems 