import os
import torch
from ultralytics import YOLO

# Check if GPU is available
print(f"Using {'GPU' if torch.cuda.is_available() else 'CPU'}")

# Define the model configuration
model_name = 'yolov8n.pt'  # You can use yolov8n, yolov8s, yolov8m, yolov8l, or yolov8x
batch_size = 16
epochs = 100
img_size = 640  # Training image size
data_yaml = '/Users/rohanshenoy/Desktop/targetidyolo/dataset.yaml'
# Load a pre-trained YOLO model
model = YOLO(model_name)

# Train the model on your custom dataset
results = model.train(
    data=data_yaml,
    epochs=epochs,
    batch=batch_size,
    imgsz=img_size,
    patience=20,  # Early stopping patience
    save=True,    # Save the model
    device=0 if torch.cuda.is_available() else 'cpu'
)

# Validate the model
val_results = model.val()
print(f"Validation results: {val_results}")

# Save the trained model
model.export(format='onnx')  # Export to ONNX format (optional)
print("Model training completed and saved.")