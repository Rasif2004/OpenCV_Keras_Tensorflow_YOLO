import cv2
import torch
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

class YOLOCNNPipeline:
    def __init__(self, yolo_model_path='yolov5s.pt', cnn_model_path='model.h5'):
        # Initialize YOLO model
        self.yolo_model = YOLO(yolo_model_path)
        
        # Initialize CNN model
        self.cnn_model = load_model(cnn_model_path)
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        
        # FPS calculation variables
        self.prev_time = 0
        self.curr_time = 0
        
    def preprocess_for_cnn(self, img):
        """Preprocess image for CNN input"""
        # Resize to match CNN input shape (assuming 32x32x3)
        img = cv2.resize(img, (32, 32))
        # Normalize
        img = img.astype('float32') / 255.0
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img
    
    def run(self):
        while True:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Calculate FPS
            self.curr_time = time.time()
            fps = 1 / (self.curr_time - self.prev_time)
            self.prev_time = self.curr_time
            
            # YOLO detection
            results = self.yolo_model(frame)
            
            # Process each detection
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Get YOLO class and confidence
                    yolo_class = int(box.cls[0])
                    yolo_conf = float(box.conf[0])
                    
                    # Crop detected object
                    obj_img = frame[y1:y2, x1:x2]
                    
                    # Skip if crop is empty
                    if obj_img.size == 0:
                        continue
                    
                    # CNN classification
                    processed_img = self.preprocess_for_cnn(obj_img)
                    cnn_pred = self.cnn_model.predict(processed_img, verbose=0)
                    cnn_class = np.argmax(cnn_pred)
                    cnn_conf = float(np.max(cnn_pred))
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Display labels
                    label = f'YOLO: {yolo_class} ({yolo_conf:.2f}) | CNN: {cnn_class} ({cnn_conf:.2f})'
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display FPS
            cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('YOLO + CNN Pipeline', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Initialize and run pipeline
    pipeline = YOLOCNNPipeline()
    pipeline.run() 