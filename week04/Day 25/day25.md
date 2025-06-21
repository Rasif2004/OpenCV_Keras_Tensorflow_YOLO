# Image Detection and Classification Web App

## Overview
This web application combines YOLOv5 for object detection and a CNN model for classification. Users can upload images through a Streamlit interface and receive detection results with bounding boxes and classification labels.

## Architecture

### Frontend
- Built with **Streamlit** for a simple and intuitive user interface
- Features:
  - Image upload functionality
  - Real-time display of detection results
  - Interactive visualization of bounding boxes and labels

### Backend
- **YOLOv5** for object detection
  - Uses the small model variant (yolov5s.pt)
  - Provides bounding boxes and initial class predictions
- **CNN Model** for detailed classification
  - Processes detected regions of interest (ROIs)
  - Provides additional classification labels

## Libraries Used
- `streamlit`: Web application framework
- `ultralytics`: YOLOv5 implementation
- `tensorflow`: CNN model framework
- `opencv-python`: Image processing
- `numpy`: Numerical operations
- `Pillow`: Image handling

## Setup Instructions

1. Install required packages:
```bash
pip install streamlit ultralytics tensorflow opencv-python numpy Pillow
```

2. Download required models:
   - YOLOv5 model will be automatically downloaded on first run
   - Place your CNN model file (`model.h5`) in the same directory as `app.py`

3. Run the application:
```bash
streamlit run app.py
```

## Usage
1. Open the web interface in your browser (typically at http://localhost:8501)
2. Click "Choose an image..." to upload a JPG, JPEG, or PNG file
3. Wait for processing (detection and classification)
4. View results:
   - Original image with bounding boxes
   - Detection information panel
   - Classification results for each detected object

## Sample Output
[Add screenshot of the application in action here]

## Author
Muntasir 