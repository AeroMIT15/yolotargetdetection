# YOLOv8 Object Detection Project

![YOLOv8 Logo](https://ultralytics.com/images/logo.png)

This project implements object detection using YOLOv8 to detect black-colored objects. It includes training on custom data and deployment for real-time camera inference.

## Features

- Custom dataset training
- GPU-accelerated training (Kaggle/Colab supported)
- Real-time camera inference
- Model export to ONNX/TensorRT
- Flask web interface option

## Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU (recommended) with CUDA 11.8

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/yolo-black-detection.git
cd yolo-black-detection

# Install dependencies
pip install -r requirements.txt
