from ultralytics import YOLO
import cv2
import argparse
from pathlib import Path

def detect_objects(model, source, conf_threshold=0.25):
    """
    Perform object detection on an image or video using YOLOv5.
    
    Args:
        model: YOLO model instance
        source: Path to image/video or webcam index (0)
        conf_threshold: Confidence threshold for detections
    """
    # Run inference
    results = model(source, conf=conf_threshold)
    
    # Process results
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get confidence and class
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = result.names[cls]
            
            # Print detection info
            print(f"Detected {class_name} with confidence {conf:.2f}")
            
            # Draw bounding box and label
            if isinstance(source, (str, Path)):  # If source is image/video file
                img = cv2.imread(str(source)) if source.endswith(('.jpg', '.png', '.jpeg')) else None
                if img is not None:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.imshow("Detection", img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    
                    # Save result
                    output_path = Path(source).parent / f"detected_{Path(source).name}"
                    cv2.imwrite(str(output_path), img)
                    print(f"Saved result to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="YOLOv5 Object Detection")
    parser.add_argument("--source", type=str, default="0", help="Path to image/video or webcam index (0)")
    parser.add_argument("--model", type=str, default="yolov5s.pt", help="Path to model or model name")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()
    
    # Load model
    model = YOLO(args.model)
    print(f"Loaded model: {args.model}")
    
    # Run detection
    detect_objects(model, args.source, args.conf)

if __name__ == "__main__":
    main() 