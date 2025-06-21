from ultralytics import YOLO
import os
import yaml

def train_yolov5():
    # Load YOLOv5 model
    model = YOLO('yolov5s.pt')  # load pretrained model
    
    # Training arguments
    args = {
        'data': 'custom.yaml',  # path to data config file
        'epochs': 50,           # number of epochs
        'imgsz': 640,          # image size
        'batch': 16,           # batch size
        'device': '',          # cuda device, i.e. 0 or 0,1,2,3 or cpu
        'workers': 8,          # number of worker threads
        'project': 'runs/train',  # save to project/name
        'name': 'exp',         # save to project/name
        'exist_ok': False,     # existing project/name ok, do not increment
        'pretrained': True,    # use pretrained model
        'optimizer': 'SGD',    # optimizer (SGD, Adam, etc.)
        'verbose': True,       # print verbose output
        'seed': 0,            # random seed for reproducibility
        'deterministic': True  # deterministic training
    }
    
    # Start training
    results = model.train(**args)
    
    # Print results
    print(f"Training completed. Results saved to {args['project']}/{args['name']}")
    
    # Save training metrics
    metrics = results.results_dict
    print("\nTraining Metrics:")
    print(f"mAP50: {metrics.get('metrics/mAP50', 'N/A')}")
    print(f"mAP50-95: {metrics.get('metrics/mAP50-95', 'N/A')}")
    print(f"Precision: {metrics.get('metrics/precision', 'N/A')}")
    print(f"Recall: {metrics.get('metrics/recall', 'N/A')}")

if __name__ == "__main__":
    # Verify dataset structure
    required_dirs = [
        'datasets/custom/images/train',
        'datasets/custom/images/val',
        'datasets/custom/labels/train',
        'datasets/custom/labels/val'
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"Warning: Directory {dir_path} does not exist!")
            print("Please ensure your dataset is properly organized.")
            exit(1)
    
    # Verify YAML config
    try:
        with open('custom.yaml', 'r') as f:
            yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading custom.yaml: {e}")
        exit(1)
    
    # Start training
    train_yolov5() 