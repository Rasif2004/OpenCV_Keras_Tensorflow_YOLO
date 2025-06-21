import cv2
import torch
import numpy as np
from collections import defaultdict
import time

class ObjectCounter:
    def __init__(self, target_class="person", confidence_threshold=0.5):
        # Initialize YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.target_class = target_class
        self.confidence_threshold = confidence_threshold
        self.count = 0
        self.previous_detections = set()
        
    def process_frame(self, frame):
        # Run YOLO inference
        results = self.model(frame)
        
        # Get detections for target class
        detections = []
        for det in results.xyxy[0]:  # xyxy format
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            class_name = results.names[int(cls)]
            
            if class_name == self.target_class and conf > self.confidence_threshold:
                detections.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Update count
        self.count = len(detections)
        
        # Display count
        cv2.putText(frame, f"Count: {self.count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Initialize object counter
    counter = ObjectCounter(target_class="person")
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        # Process frame
        processed_frame = counter.process_frame(frame)
        
        # Display frame
        cv2.imshow("Object Counter", processed_frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 