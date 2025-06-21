# Fine-tuning YOLOv5 on Custom Dataset
Author: Muntasir

## Overview
This guide explains how to fine-tune YOLOv5 on a custom dataset with 1-2 object classes. We'll use the Ultralytics training pipeline for efficient training and evaluation.

## Dataset Structure
The custom dataset should be organized as follows:
```
datasets/custom/
├── images/
│   ├── train/
│   │   └── *.jpg
│   └── val/
│       └── *.jpg
└── labels/
    ├── train/
    │   └── *.txt
    └── val/
        └── *.txt
```

### Label Format
Each image has a corresponding `.txt` file with normalized bounding box annotations:
```
class_id x_center y_center width height
```
- `class_id`: Integer (0-based) class index
- `x_center, y_center`: Normalized center coordinates (0-1)
- `width, height`: Normalized box dimensions (0-1)

## Configuration
The `custom.yaml` file defines dataset parameters:
```yaml
path: datasets/custom
train: images/train
val: images/val
nc: 2  # number of classes
names: ['class1', 'class2']
```

## Training
Two methods to start training:

### 1. Using Python Script
```bash
python train_yolov5_custom.py
```

### 2. Using Command Line
```bash
yolo train model=yolov5s.pt data=custom.yaml epochs=50 imgsz=640
```

## Training Parameters
- **Epochs**: 50 (adjust based on dataset size)
- **Image Size**: 640x640 (standard for YOLOv5)
- **Batch Size**: 16 (adjust based on GPU memory)
- **Model**: YOLOv5s (smallest model, good for small datasets)

## Tips for Small Datasets
1. **Data Augmentation**: Enable all augmentations in training
2. **Transfer Learning**: Use pretrained weights (default)
3. **Early Stopping**: Monitor validation loss
4. **Learning Rate**: Start with default (0.01) and adjust if needed
5. **More Epochs**: Small datasets may need more epochs to converge

## Training Results
Results are saved in `runs/train/exp/`:
- `weights/best.pt`: Best model weights
- `weights/last.pt`: Last checkpoint
- `results.csv`: Training metrics
- `results.png`: Training plots

## Testing the Model
```bash
yolo predict model=runs/train/exp/weights/best.pt source=path/to/test/images
```

## Performance Metrics
Monitor these metrics during training:
- mAP50: Mean Average Precision at IoU=0.5
- mAP50-95: Mean Average Precision at IoU=0.5:0.95
- Precision: True positives / (True positives + False positives)
- Recall: True positives / (True positives + False negatives)

## Troubleshooting
1. **Out of Memory**: Reduce batch size or image size
2. **Poor Performance**: 
   - Check label format
   - Verify class names in YAML
   - Increase epochs for small datasets
3. **Training Instability**:
   - Adjust learning rate
   - Check data quality
   - Verify augmentation settings

## References
- [Ultralytics YOLOv5 GitHub](https://github.com/ultralytics/yolov5)
- [YOLOv5 Train Custom Data Guide](https://docs.ultralytics.com/datasets/custom/) 