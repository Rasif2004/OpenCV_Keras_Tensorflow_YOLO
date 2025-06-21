import cv2
import numpy as np
from ultralytics import YOLO
import time

class ObjectCounter:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize the ObjectCounter with YOLO model.
        
        Args:
            model_path (str): Path to the YOLO model weights file
        """
        try:
            self.model = YOLO(model_path)
            self.class_names = self.model.names
            self.counts = {}
            self.frame_count = 0
            self.fps = 0
            self.start_time = time.time()
        except Exception as e:
            print(f"Error initializing YOLO model: {e}")
            raise

    def process_frame(self, frame):
        """
        Process a single frame and detect objects.
        
        Args:
            frame: Input frame from video source
            
        Returns:
            Processed frame with detections and counts
        """
        try:
            # Run YOLO detection
            results = self.model(frame, conf=0.5)
            
            # Reset counts for this frame
            frame_counts = {}
            
            # Process detections
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get class and confidence
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Get class name
                    class_name = self.class_names[cls]
                    
                    # Update counts
                    if class_name not in frame_counts:
                        frame_counts[class_name] = 0
                    frame_counts[class_name] += 1
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Update total counts
            for class_name, count in frame_counts.items():
                if class_name not in self.counts:
                    self.counts[class_name] = 0
                self.counts[class_name] += count
            
            # Calculate and display FPS
            self.frame_count += 1
            if self.frame_count >= 30:
                self.fps = self.frame_count / (time.time() - self.start_time)
                self.frame_count = 0
                self.start_time = time.time()
            
            # Display counts
            y_offset = 30
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            for class_name, count in self.counts.items():
                y_offset += 30
                cv2.putText(frame, f"{class_name}: {count}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            return frame
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame

def main():
    """
    Main function to run the object counter on webcam feed.
    """
    try:
        # Initialize video capture
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not open video capture device")
        
        # Initialize object counter
        counter = ObjectCounter()
        
        print("Press 'q' to quit")
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break
            
            # Process frame
            processed_frame = counter.process_frame(frame)
            
            # Display frame
            cv2.imshow('Object Counter', processed_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error in main: {e}")
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 