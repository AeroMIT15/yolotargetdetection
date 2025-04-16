# YOLOv8 Black Object Detection

![Training Preview](results/black_detection_v1/train_batch0.jpg) 
*Sample training batch with annotations*

## 📊 Training Results

### Performance Metrics

| **Confusion Matrix** | ![Conf Matrix](results/black_detection_v1/confusion_matrix_normalized.png) |

### Learning Curves
<div align="center">
  <img src="results/black_detection_v1/results.png" width="80%">
  <p><em>Training metrics over epochs</em></p>
</div>

## 🔍 Validation Samples

### Labeled vs Predicted
| | Labels | Predictions |
|-|--------|-------------|
| **Batch 0** | ![val_labels0](results/black_detection_v1/val_batch0_labels.jpg) | ![val_pred0](results/black_detection_v1/val_batch0_pred.jpg) |
| **Batch 1** | ![val_labels1](results/black_detection_v1/val_batch1_labels.jpg) | ![val_pred1](results/black_detection_v1/val_batch1_pred.jpg) |

## 🎥 Live Detection Results
![Live Demo](./result%20image.png)


## 📈 Advanced Metrics

<div align="center">
  <img src="results/black_detection_v1/labels_correlogram.jpg" width="45%">
  <img src="results/black_detection_v1/P_curve.png" width="45%">
  <p><em>Left: Label Correlations • Right: Precision Curve</em></p>
</div>

## 🛠️ Training Configuration
```yaml
# Reference: results/black_detection_v1/args.yaml
batch: 16
imgsz: 640
lr0: 0.01
weight_decay: 0.0005
