import cv2
import numpy as np
import time
import argparse
from pathlib import Path

def load_yolo():
    """Load YOLOv3 model and classes"""
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

def detect_objects(img, net, output_layers, conf_threshold, nms_threshold):
    """Detect objects in image using YOLOv3"""
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    return boxes, confidences, class_ids, indexes

def main():
    parser = argparse.ArgumentParser(description='YOLOv3 Object Detection with OpenCV DNN')
    parser.add_argument('--input', type=str, required=True, help='Input video or image path')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--nms', type=float, default=0.4, help='NMS threshold')
    args = parser.parse_args()

    # Load YOLO
    net, classes, output_layers = load_yolo()
    
    # Initialize video capture
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Could not open {args.input}")
        return

    # Initialize FPS calculation variables
    frame_count = 0
    total_time = 0
    fps_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Start timing
        start_time = time.time()
        
        # Detect objects
        boxes, confidences, class_ids, indexes = detect_objects(
            frame, net, output_layers, args.conf, args.nms
        )
        
        # Calculate FPS
        end_time = time.time()
        frame_time = end_time - start_time
        fps = 1 / frame_time if frame_time > 0 else 0
        fps_list.append(fps)
        
        # Draw boxes
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("YOLOv3 Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calculate and print average FPS
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    print(f"\nPerformance Metrics:")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Total frames processed: {len(fps_list)}")
    print(f"Confidence threshold: {args.conf}")
    print(f"NMS threshold: {args.nms}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 