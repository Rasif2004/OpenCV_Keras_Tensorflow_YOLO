import cv2
import numpy as np
from ultralytics import YOLO
import time

class ObjectCounter:
    def __init__(self, model_path='yolov5s.pt', conf_threshold=0.5):
        # Initialize YOLO model
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Error: Could not open webcam")
            
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer
        self.video_writer = cv2.VideoWriter(
            'output.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'),
            30.0,
            (self.frame_width, self.frame_height)
        )
        
        # Initialize tracking variables
        self.previous_centroids = []
        self.object_ids = {}
        self.next_object_id = 0
        
    def calculate_centroid(self, box):
        x1, y1, x2, y2 = box
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def track_objects(self, current_centroids):
        # Simple tracking using centroid distance
        if len(self.previous_centroids) == 0:
            for centroid in current_centroids:
                self.object_ids[self.next_object_id] = centroid
                self.next_object_id += 1
        else:
            # Calculate distances between current and previous centroids
            for current_centroid in current_centroids:
                min_distance = float('inf')
                min_id = None
                
                for obj_id, prev_centroid in self.object_ids.items():
                    distance = np.sqrt(
                        (current_centroid[0] - prev_centroid[0])**2 +
                        (current_centroid[1] - prev_centroid[1])**2
                    )
                    
                    if distance < min_distance and distance < 50:  # Threshold for tracking
                        min_distance = distance
                        min_id = obj_id
                
                if min_id is not None:
                    self.object_ids[min_id] = current_centroid
                else:
                    self.object_ids[self.next_object_id] = current_centroid
                    self.next_object_id += 1
        
        self.previous_centroids = current_centroids
        return len(self.object_ids)
    
    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Run YOLO detection
                results = self.model(frame, conf=self.conf_threshold)[0]
                
                # Process detections
                current_centroids = []
                for result in results.boxes.data.tolist():
                    x1, y1, x2, y2, conf, class_id = result
                    
                    # Filter for person class (class_id 0 in COCO dataset)
                    if int(class_id) == 0:  # 0 is the class ID for person
                        # Draw bounding box
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        
                        # Add label
                        label = f'Person {conf:.2f}'
                        cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Calculate and store centroid
                        centroid = self.calculate_centroid((x1, y1, x2, y2))
                        current_centroids.append(centroid)
                
                # Track objects and get count
                total_count = self.track_objects(current_centroids)
                
                # Display count
                cv2.putText(frame, f'People detected: {total_count}',
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow('Object Counter', frame)
                
                # Save frame to video
                self.video_writer.write(frame)
                
                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            # Cleanup
            self.cap.release()
            self.video_writer.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Create and run object counter
    counter = ObjectCounter(conf_threshold=0.5)
    counter.run() 