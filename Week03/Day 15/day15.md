# Image Classification vs Object Detection

**Author:** Muntasir

## Understanding the Difference

### Image Classification
Image classification is the task of assigning a single label to an entire image. It answers the question "What is in this image?" by predicting the most likely class from a predefined set of categories. For example, given an image of a dog, a classification model would output "dog" as the predicted class.

### Object Detection
Object detection is more complex as it involves both classification and localization. It answers two questions:
1. "What objects are in this image?"
2. "Where are these objects located?"

The model outputs:
- Bounding boxes (x, y, width, height) for each detected object
- Class labels for each detected object
- Confidence scores for each detection

## Example Code Overview

The provided code demonstrates both classification and detection using popular deep learning models:

### Classification (MobileNetV2)
- Uses TensorFlow's MobileNetV2 pre-trained on ImageNet
- Processes input image to 224x224 pixels
- Outputs top 5 class predictions with confidence scores

### Detection (YOLOv3)
- Uses OpenCV's DNN module with YOLOv3
- Processes input image to 416x416 pixels
- Outputs bounding boxes, class labels, and confidence scores
- Applies non-maximum suppression to remove overlapping detections

## How Detection Models Work

### Bounding Box Prediction
1. The model divides the input image into a grid
2. For each grid cell, it predicts:
   - Bounding box coordinates (x, y, width, height)
   - Objectness score (probability of containing an object)
   - Class probabilities for each possible class

### Output Processing
1. Filter detections by confidence threshold
2. Apply non-maximum suppression to remove overlapping boxes
3. Scale coordinates to original image dimensions
4. Draw boxes and labels on the image

## Running the Examples

### Prerequisites
```bash
pip install tensorflow opencv-python numpy
```

### Required Files
- YOLOv3 weights: `yolov3.weights`
- YOLOv3 configuration: `yolov3.cfg`
- COCO class names: `coco.names`

### Running the Code
```bash
python classification_vs_detection.py
```

## Further Reading

1. [OpenCV Object Detection Tutorial](https://docs.opencv.org/master/d6/d0f/group__dnn.html)
2. [YOLOv3 Paper](https://arxiv.org/abs/1804.02767)
3. [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
4. [COCO Dataset](https://cocodataset.org/)

## Key Takeaways

1. Classification is simpler but less informative than detection
2. Detection provides spatial information about objects
3. Modern detection models can handle multiple objects in a single image
4. Both approaches use deep learning but with different architectures
5. Detection models are generally more computationally intensive 