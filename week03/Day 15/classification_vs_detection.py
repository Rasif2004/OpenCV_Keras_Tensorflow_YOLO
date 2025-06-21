import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import urllib.request
import os

# def download_sample_image():
#     """Download a sample image for testing."""
#     url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/dog.jpg"
#     if not os.path.exists("sample_image.jpg"):
#         urllib.request.urlretrieve(url, "sample_image.jpg")
#     return cv2.imread("sample_image.jpg")

def image_classification(image):
    """
    Perform image classification using MobileNetV2.
    
    Args:
        image: Input image in BGR format (OpenCV format)
    
    Returns:
        predictions: Top 5 class predictions with probabilities
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image to expected input size
    image_resized = cv2.resize(image_rgb, (224, 224))
    
    # Preprocess image for MobileNetV2
    image_array = np.expand_dims(image_resized, axis=0)
    image_array = preprocess_input(image_array)
    
    # Load pre-trained model
    model = MobileNetV2(weights='imagenet')
    
    # Make prediction
    predictions = model.predict(image_array)
    
    # Decode predictions
    decoded_predictions = decode_predictions(predictions, top=5)[0]
    
    return decoded_predictions

def object_detection(image):
    """
    Perform object detection using YOLOv3.
    
    Args:
        image: Input image in BGR format (OpenCV format)
    
    Returns:
        image_with_boxes: Image with bounding boxes and labels
        detections: List of detections (class_id, confidence, bbox)
    """
    # Load YOLO model
    net = cv2.dnn.readNet(
        "yolov3.weights",
        "yolov3.cfg"
    )
    
    # Load COCO class names
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Create blob from image
    blob = cv2.dnn.blobFromImage(
        image, 1/255.0, (416, 416),
        swapRB=True, crop=False
    )
    
    # Set input blob for the network
    net.setInput(blob)
    
    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Run forward pass
    outputs = net.forward(output_layers)
    
    # Process detections
    class_ids = []
    confidences = []
    boxes = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:  # Confidence threshold
                # Bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Draw bounding boxes
    image_with_boxes = image.copy()
    detections = []
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            
            # Draw rectangle and label
            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image_with_boxes, f"{label}: {confidence:.2f}",
                       (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            detections.append((class_ids[i], confidence, (x, y, w, h)))
    
    return image_with_boxes, detections

def main():
    image = cv2.imread("H:\Downloads\opencv_keras_tensorflow_yolo\Init MMR\Week03\Day 15\dogop.jpeg", cv2.IMREAD_COLOR)
    
    # Perform classification
    print("\nPerforming Image Classification:")
    predictions = image_classification(image)
    for _, label, confidence in predictions:
        print(f"{label}: {confidence:.2f}")
    
    # Perform detection
    print("\nPerforming Object Detection:")
    image_with_boxes, detections = object_detection(image)
    
    # Display results
    cv2.imshow("Object Detection", image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 