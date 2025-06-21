# Day 4: Webcam Video Capture with Grayscale Conversion

## Objective
This exercise demonstrates how to capture video from a webcam using OpenCV, apply real-time grayscale conversion, and save the processed video to a file. The goal is to understand the basic concepts of video capture, frame processing, and video writing in OpenCV.

## Code Overview
The script `webcam_grayscale.py` implements the following functionality:
1. Captures video from the default webcam
2. Converts each frame to grayscale in real-time
3. Displays the grayscale video feed
4. Saves the processed video to a file
5. Handles user input to stop recording

## Key Components

### 1. Video Capture
```python
cap = cv2.VideoCapture(0)
```
- Uses `cv2.VideoCapture()` to access the default webcam (index 0)
- Returns a video capture object that manages the webcam stream

### 2. Video Properties
```python
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
```
- Retrieves important video properties: width, height, and frames per second
- These properties are used to configure the output video file

### 3. Video Writer Setup
```python
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height), isColor=False)
```
- Defines the video codec (XVID) for compression
- Creates a VideoWriter object to save the processed frames
- `isColor=False` indicates that we're saving grayscale video

### 4. Frame Processing Loop
```python
while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale Webcam', gray_frame)
    out.write(gray_frame)
```
- Continuously captures frames from the webcam
- Converts each frame to grayscale using `cv2.cvtColor()`
- Displays the processed frame in a window
- Writes the frame to the output video file

### 5. User Interaction
```python
if cv2.waitKey(1) & 0xFF == ord('q'):
    break
```
- Waits for 1ms between frames
- Checks if the 'q' key is pressed to stop recording

### 6. Resource Management
```python
cap.release()
out.release()
cv2.destroyAllWindows()
```
- Properly releases the webcam and video writer resources
- Closes all OpenCV windows

## How to Run
1. Ensure OpenCV is installed (`pip install opencv-python`)
2. Run the script: `python webcam_grayscale.py`
3. Press 'q' to stop recording
4. The processed video will be saved as 'output.avi'

## Key Learnings
- Understanding video capture and frame processing in OpenCV
- Working with grayscale conversion
- Managing video input and output streams
- Handling real-time video processing
- Proper resource management in OpenCV applications 