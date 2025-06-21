# Real-Time Object Detection and Classification Pipeline

## Architecture Overview

This pipeline combines two powerful deep learning models to create a robust real-time object detection and classification system:

1. **YOLO (You Only Look Once) Detection**
   - Uses YOLOv5 from Ultralytics for fast, real-time object detection
   - Provides bounding boxes and initial class predictions
   - Efficient single-stage detector that processes the entire image at once

2. **CNN Classification**
   - Keras-based CNN model for detailed classification
   - Acts as a verification/refinement step for YOLO detections
   - Provides additional confidence scores and class predictions

## Model Types

- **YOLOv5**: Pre-trained on COCO dataset, capable of detecting 80 common object classes
- **CNN Model**: Custom-trained Keras model (assumed to be saved as `model.h5`)
  - Input shape: 32x32x3 (RGB images)
  - Output: Class probabilities for your specific classification task

## Performance Tips

1. **Optimization**
   - Use GPU acceleration for both YOLO and CNN models
   - Consider using TensorRT for YOLO inference
   - Batch process CNN predictions if multiple objects are detected

2. **Preprocessing**
   - Resize images to match model input requirements
   - Normalize pixel values to [0,1] range
   - Consider using image augmentation for training

3. **Integration**
   - Ensure consistent class mappings between YOLO and CNN
   - Implement confidence thresholds for both models
   - Handle cases where models disagree on classification

## How to Run

1. **Prerequisites**
   ```bash
   pip install ultralytics opencv-python tensorflow torch numpy
   ```

2. **Execution**
   ```bash
   python yolo_cnn_pipeline.py
   ```

3. **Controls**
   - Press 'q' to quit the application
   - FPS counter displayed in top-left corner
   - Bounding boxes show both YOLO and CNN predictions

## Sample Output

The pipeline will display:
- Live video feed with bounding boxes
- YOLO class and confidence score
- CNN class and confidence score
- Real-time FPS counter

## Author
Muntasir

## Notes
- Ensure your webcam is properly connected and accessible
- Adjust confidence thresholds in the code if needed
- Modify preprocessing steps based on your CNN model's requirements
- Consider adding error handling for model loading and inference 