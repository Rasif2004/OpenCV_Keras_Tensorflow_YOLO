import cv2
import numpy as np
import argparse

def load_yolo_model(cfg_path, weights_path):
    """
    Load YOLOv3 model using OpenCV's DNN module
    """
    # Load YOLOv3 network
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    
    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    return net, output_layers

def load_class_names(names_path):
    """
    Load COCO class names from file
    """
    with open(names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

def detect_objects(image, net, output_layers, conf_threshold=0.5, nms_threshold=0.4):
    """
    Detect objects in the image using YOLOv3
    """
    height, width, _ = image.shape
    
    # Create blob from image
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    
    # Set input and forward pass
    net.setInput(blob)
    outputs = net.forward(output_layers)
    
    # Initialize lists for detected objects
    boxes = []
    confidences = []
    class_ids = []
    
    # Process each output layer
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
    
    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    return boxes, confidences, class_ids, indices

def draw_detections(image, boxes, confidences, class_ids, indices, classes):
    """
    Draw bounding boxes and labels on the image
    """
    # Generate random colors for each class
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    # Draw boxes and labels
    for i in indices:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = colors[class_ids[i]]
        
        # Draw rectangle and label
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = f"{label}: {confidence:.2f}"
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='YOLOv3 Object Detection')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--cfg', type=str, default='yolov3.cfg', help='Path to YOLOv3 config file')
    parser.add_argument('--weights', type=str, default='yolov3.weights', help='Path to YOLOv3 weights file')
    parser.add_argument('--names', type=str, default='coco.names', help='Path to COCO class names file')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--nms', type=float, default=0.4, help='NMS threshold')
    args = parser.parse_args()
    
    # Load YOLOv3 model
    net, output_layers = load_yolo_model(args.cfg, args.weights)
    
    # Load class names
    classes = load_class_names(args.names)
    
    # Read image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not read image {args.image}")
        return
    
    # Detect objects
    boxes, confidences, class_ids, indices = detect_objects(
        image, net, output_layers, args.conf, args.nms
    )
    
    # Draw detections
    result_image = draw_detections(image, boxes, confidences, class_ids, indices, classes)
    
    # Save output image
    output_path = 'output_' + args.image.split('/')[-1]
    cv2.imwrite(output_path, result_image)
    print(f"Detection results saved to {output_path}")
    
    # Display image (optional)
    cv2.imshow('Object Detection', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 