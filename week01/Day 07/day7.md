# Real-Time Edge Detection Webcam Application

## Project Overview
This project implements a real-time edge detection application using OpenCV and a webcam. The application captures video from the webcam, processes each frame to detect edges, and displays both the original and edge-detected frames side by side. Users can save frames at any time by pressing a key.

## Features
- Real-time webcam capture
- Edge detection using Canny algorithm
- Side-by-side display of original and processed frames
- Frame capture functionality
- Timestamp-based file naming
- Organized output directory structure

## Technical Details

### OpenCV Techniques Used
1. **Video Capture**
   - `cv2.VideoCapture(0)`: Opens the default webcam
   - `cap.read()`: Reads frames from the webcam

2. **Image Processing**
   - `cv2.cvtColor()`: Converts BGR to grayscale
   - `cv2.GaussianBlur()`: Applies Gaussian blur to reduce noise
   - `cv2.Canny()`: Performs edge detection

3. **Display and I/O**
   - `cv2.imshow()`: Displays frames in windows
   - `cv2.imwrite()`: Saves frames to disk
   - `cv2.waitKey()`: Handles keyboard input
   - `cv2.destroyAllWindows()`: Cleans up resources

### Edge Detection Process
1. Convert frame to grayscale
2. Apply Gaussian blur (kernel size 5x5)
3. Apply Canny edge detection (thresholds: 50, 150)

## How to Run
1. Ensure you have OpenCV installed:
   ```bash
   pip install opencv-python numpy
   ```
2. Run the script:
   ```bash
   python edge_cam.py
   ```

## Controls
- Press 's' to save the current edge-detected frame
- Press 'q' to quit the application

## Output
- Saved frames are stored in the `edge_frames` directory
- Files are named with pattern: `edge_X_YYYYMMDD_HHMMSS.png`
  - X: Sequential counter
  - YYYYMMDD_HHMMSS: Timestamp

## Input/Output
- **Input**: Live webcam feed
- **Output**: 
  - Real-time display of original and edge-detected frames
  - Saved PNG images of edge-detected frames

## Possible Extensions
1. Add FPS counter
2. Implement multiple edge detection algorithms
3. Add adjustable parameters for edge detection
4. Include image filters and effects
5. Add video recording capability
6. Implement motion detection
7. Add GUI controls for parameters

## Author
Muntasir

## Dependencies
- OpenCV (cv2)
- NumPy
- Python 3.x

## Notes
- Ensure webcam is properly connected and accessible
- The application creates an `edge_frames` directory for saved images
- Edge detection parameters can be adjusted in the code for different results 