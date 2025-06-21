import cv2
import numpy as np
import time
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='YOLOv4 Object Detection')
    parser.add_argument('--video', type=str, default='0',
                        help='Video source (0 for webcam or path to video file)')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--nms_threshold', type=float, default=0.4,
                        help='Non-maximum suppression threshold')
    parser.add_argument('--width', type=int, default=416,
                        help='Width of input image')
    parser.add_argument('--height', type=int, default=416,
                        help='Height of input image')
    return parser.parse_args()

def load_yolo():
    """Load YOLOv4 model and class names."""
    # Load YOLOv4 network
    net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')
    
    # Load class names
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    return net, classes, output_layers

def detect_objects(frame, net, output_layers, conf_threshold, nms_threshold, width, height):
    """Detect objects in the frame using YOLOv4."""
    # Create blob from frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (width, height), swapRB=True, crop=False)
    
    # Set input and forward pass
    net.setInput(blob)
    outputs = net.forward(output_layers)
    
    # Get frame dimensions
    height, width, _ = frame.shape
    
    # Initialize lists for detected objects
    boxes = []
    confidences = []
    class_ids = []
    
    # Process detections
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > conf_threshold:
                # Get bounding box coordinates
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
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    return boxes, confidences, class_ids, indices

def draw_detections(frame, boxes, confidences, class_ids, indices, classes):
    """Draw bounding boxes and labels on the frame."""
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[class_ids[i]]
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label background
            text_size = cv2.getTextSize(label + ' ' + confidence, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x, y - text_size[1] - 10), (x + text_size[0], y), color, -1)
            
            # Draw label text
            cv2.putText(frame, label + ' ' + confidence, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame

def main():
    """Main function to run YOLOv4 object detection."""
    args = parse_args()
    
    # Load YOLOv4 model
    net, classes, output_layers = load_yolo()
    
    # Open video capture
    video_source = 0 if args.video == '0' else args.video
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Initialize FPS calculation variables
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects
        boxes, confidences, class_ids, indices = detect_objects(
            frame, net, output_layers, args.conf_threshold,
            args.nms_threshold, args.width, args.height
        )
        
        # Draw detections
        frame = draw_detections(frame, boxes, confidences, class_ids, indices, classes)
        
        # Calculate and display FPS
        frame_count += 1
        if frame_count >= 30:  # Update FPS every 30 frames
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()
        
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('YOLOv4 Object Detection', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 