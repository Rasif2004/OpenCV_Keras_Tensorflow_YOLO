import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Image Detection & Classification",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("Image Detection and Classification")
st.markdown("""
This application performs object detection using YOLOv5 and classification using a CNN model.
Upload an image to see the results!
""")

# Load models
@st.cache_resource
def load_models():
    yolo_model = YOLO('yolov5s.pt')  # Load YOLOv5 small model
    cnn_model = tf.keras.models.load_model('model.h5')  # Load CNN model
    return yolo_model, cnn_model

try:
    yolo_model, cnn_model = load_models()
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Convert PIL Image to numpy array for OpenCV
    image_np = np.array(image)
    
    # Perform YOLO detection
    results = yolo_model(image_np)
    
    # Create a copy of the image for drawing
    output_image = image_np.copy()
    
    # Process each detection
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get YOLO class and confidence
            yolo_class = result.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            
            # Crop the detected object
            roi = image_np[y1:y2, x1:x2]
            
            # Preprocess ROI for CNN
            if roi.size > 0:  # Check if ROI is valid
                roi_resized = cv2.resize(roi, (224, 224))  # Assuming CNN expects 224x224
                roi_preprocessed = roi_resized / 255.0  # Normalize
                roi_batch = np.expand_dims(roi_preprocessed, axis=0)
                
                # Get CNN prediction
                cnn_pred = cnn_model.predict(roi_batch, verbose=0)
                cnn_class = "Class " + str(np.argmax(cnn_pred[0]))  # Assuming CNN outputs class indices
                
                # Draw bounding box and labels
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"YOLO: {yolo_class} ({confidence:.2f})\nCNN: {cnn_class}"
                cv2.putText(output_image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the results
    st.image(output_image, caption="Detection Results", use_column_width=True)
    
    # Display detection information
    st.subheader("Detection Information")
    for result in results:
        boxes = result.boxes
        for box in boxes:
            yolo_class = result.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            st.write(f"Detected: {yolo_class} (Confidence: {confidence:.2f})") 