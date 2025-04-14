import os
import torch
from ultralytics import YOLO

# Check if GPU is available
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using {device.upper()}")

# Training parameters - NOTE THE CHANGED PARAMETER NAMES
config = {
    'data': '/Users/rohanshenoy/Desktop/targetidyolo/dataset.yaml',
    'model': 'yolov8n.pt',  # or 'yolov8n.yaml' to train from scratch
    'epochs': 100,
    'batch': 16,  # Changed from 'batch_size' to 'batch'
    'imgsz': 640,
    'patience': 20,
    'device': device,
    'name': 'black_detection',
    'optimizer': 'auto',
    'lr0': 0.01,
    'weight_decay': 0.0005,
    'augment': True  # Enable data augmentation
}

# Load model
model = YOLO(config['model'])

try:
    # Train the model
    results = model.train(**config)
    
    # Validate
    val_results = model.val()  # Uses same settings as training
    print(f"Validation mAP@0.5: {val_results.box.map:.4f}")
    
    # Export to ONNX
    model.export(format='onnx')
    print(f"Model saved to: {results.save_dir}")
    
except Exception as e:
    print(f"Training failed: {str(e)}")