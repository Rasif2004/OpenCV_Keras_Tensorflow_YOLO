import cv2
import time
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description='YOLOv5 Object Detection')
    parser.add_argument('--input', type=str, required=True, help='Input video or image path')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--nms', type=float, default=0.4, help='NMS threshold')
    args = parser.parse_args()

    # Load YOLOv5 model
    model = YOLO('yolov5s.pt')  # Using small model for speed comparison
    
    # Initialize video capture
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Could not open {args.input}")
        return

    # Initialize FPS calculation variables
    fps_list = []
    total_detections = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Start timing
        start_time = time.time()
        
        # Run inference
        results = model(frame, conf=args.conf, iou=args.nms)[0]
        
        # Calculate FPS
        end_time = time.time()
        frame_time = end_time - start_time
        fps = 1 / frame_time if frame_time > 0 else 0
        fps_list.append(fps)
        
        # Process detections
        detections = len(results.boxes)
        total_detections += detections
        frame_count += 1

        # Draw boxes
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{results.names[cls]} {conf:.2f}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("YOLOv5 Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calculate and print performance metrics
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    avg_detections = total_detections / frame_count if frame_count > 0 else 0
    
    print(f"\nPerformance Metrics:")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Total frames processed: {frame_count}")
    print(f"Average detections per frame: {avg_detections:.2f}")
    print(f"Confidence threshold: {args.conf}")
    print(f"NMS threshold: {args.nms}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 