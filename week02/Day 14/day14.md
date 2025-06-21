# Handwritten Digit Recognition with CNN and Streamlit

## Project Overview
This project implements a Convolutional Neural Network (CNN) for recognizing handwritten digits using the MNIST dataset. The model is trained using TensorFlow/Keras and deployed as an interactive web application using Streamlit, allowing users to draw digits and get real-time predictions.

## Model Architecture
The CNN model consists of:
- 3 Convolutional blocks with ReLU activation
- MaxPooling layers for dimensionality reduction
- Dense layers with dropout for classification
- Softmax output layer for 10 digit classes (0-9)

### Model Details:
```python
- Input: 28x28x1 grayscale images
- Conv2D(32, 3x3) → MaxPool2D(2x2)
- Conv2D(64, 3x3) → MaxPool2D(2x2)
- Conv2D(64, 3x3)
- Flatten
- Dense(64) with ReLU
- Dropout(0.5)
- Dense(10) with Softmax
```

## Setup Instructions

### Prerequisites
- Python 3.7+
- pip (Python package installer)

### Installation
1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install tensorflow streamlit streamlit-drawable-canvas opencv-python pandas numpy
```

## Usage Guide

### Training the Model
1. Run the training script:
```bash
python digit_recognizer_cnn.py
```
This will:
- Load and preprocess the MNIST dataset
- Train the CNN model
- Save the trained model to `mnist_cnn_model/`

### Running the Web App
1. Start the Streamlit app:
```bash
streamlit run app.py
```
2. Open your web browser at the URL shown in the terminal (usually http://localhost:8501)

### Using the Web App
1. Draw a digit (0-9) in the canvas
2. Click the "Predict" button
3. View the prediction results and confidence scores
4. Clear the canvas and try another digit

## Features
- Interactive drawing canvas
- Real-time digit recognition
- Confidence score display
- Probability distribution visualization
- Responsive and user-friendly interface

## Dependencies
- tensorflow
- streamlit
- streamlit-drawable-canvas
- opencv-python
- pandas
- numpy

## Author
Muntasir

## Notes
- The model achieves high accuracy on the MNIST test set
- The web app provides an intuitive interface for testing the model
- The drawing canvas supports both mouse and touch input
- The app includes error handling for missing model files 