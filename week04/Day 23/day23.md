# Image Prediction with Keras + OpenCV Webcam Input

## Project Description
This project implements a real-time image prediction pipeline that combines OpenCV for webcam capture with a Keras CNN model for digit recognition. The pipeline captures images from a webcam, preprocesses them to match the model's input requirements, and provides real-time predictions.

## Pipeline Components

### 1. Webcam Capture
- Uses OpenCV's `VideoCapture` to access the webcam feed
- Displays the live feed in a window
- Captures frames when the 's' key is pressed
- Exits when the 'q' key is pressed

### 2. Image Preprocessing
- Converts captured frames to grayscale
- Resizes images to 28x28 pixels (MNIST input size)
- Normalizes pixel values to range [0, 1]
- Reshapes the image to match model input requirements (1, 28, 28, 1)

### 3. Model Architecture
- Uses a simple CNN architecture trained on MNIST dataset
- Architecture:
  - 2 Convolutional layers with ReLU activation
  - 2 MaxPooling layers
  - Flatten layer
  - Dense layers with ReLU and softmax activation
- Input shape: (28, 28, 1)
- Output: 10 classes (digits 0-9)

## Input Format
- Raw webcam input: BGR color format
- Preprocessed input: Grayscale, 28x28 pixels, normalized values
- Model input shape: (1, 28, 28, 1)

## How to Run

1. Ensure you have the required dependencies installed:
   ```bash
   pip install tensorflow opencv-python numpy
   ```

2. Run the script:
   ```bash
   python keras_opencv_predict.py
   ```

3. Usage:
   - Press 's' to capture and predict the current frame
   - Press 'q' to quit the application
   - The prediction and confidence score will be displayed both in the console and on the video feed

## Notes
- The model will be automatically trained on MNIST dataset if no pre-trained model is found
- The trained model is saved as 'model.h5' for future use
- For best results, ensure good lighting and clear digit presentation to the webcam

## Author
Muntasir 