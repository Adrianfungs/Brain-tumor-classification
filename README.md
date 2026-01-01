# Brain Tumor MRI Classification using Transfer Learning

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=flat&logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green?style=flat&logo=opencv)

## 📌 Project Overview
This project implements a deep learning pipeline to classify brain MRI scans into four clinical categories: **Glioma, Meningioma, Pituitary tumor, and No tumor**. 

The solution leverages **Transfer Learning** with three state-of-the-art architectures (**DenseNet121, VGG16, and EfficientNet-B0**) implemented in PyTorch. A key feature of this project is the robust **Image Preprocessing Pipeline** designed to remove noise and enhance tumor textures before training.

## ⚙️ Key Features
* **Multi-Model Architecture**: Comparative analysis of VGG16, DenseNet121, and EfficientNet-B0.
* **Advanced Preprocessing**:
    * **Contour Cropping**: Automatically detects the largest contour (the brain) to crop out black backgrounds and artifacts.
    * **CLAHE Enhancement**: Applies *Contrast Limited Adaptive Histogram Equalization* to the L-channel (LAB color space) to improve tissue contrast.
* **Two-Stage Training Strategy**:
    * **Stage 1**: Frozen backbone with a custom classifier head trained on high learning rates.
    * **Stage 2**: Unfrozen backbone for fine-tuning with a lower learning rate (`3e-5`) and `ReduceLROnPlateau` scheduling.
* **Device Agnostic**: Automatically detects and utilizes NVIDIA GPUs (CUDA), Mac Silicon (MPS), or CPU.

## 🧪 Methodology

### 1. The Preprocessing Pipeline
Raw MRI scans often contain significant noise and empty space. The `ImagePreprocessor` class handles this:

1.  **Gaussian Blur & Thresholding**: Removes high-frequency noise.
2.  **Erosion & Dilation**: Cleans up small artifacts in the binary mask.
3.  **ROI Extraction**: Finds the extreme points of the largest contour and crops the image.
4.  **CLAHE**: Enhances local contrast to make the tumor boundaries distinct from healthy tissue.

### 2. Model Configuration
All models are initialized with pre-trained ImageNet weights. The classifier heads are modified to match the 4 classes:

| Model | Custom Head Structure |
| :--- | :--- |
| **DenseNet121** | Linear → ReLU → Dropout(0.5) → Linear(4) |
| **VGG16** | Flatten → Linear(4096) → ReLU → Dropout → Linear(256) → Output |
| **EfficientNet** | Dropout(0.3) → Linear → ReLU → Dropout(0.5) → Output |

## 📊 Dataset Structure
The project expects the following directory structure:

```text
/data
    /Training
        /glioma_tumor
        /meningioma_tumor
        /no_tumor
        /pituitary_tumor
    /Testing
        (same structure)
```
## Results Summary

### Internal Test Set Performance
The models were evaluated on a standard test split from the primary dataset (394 images):

| Model | Test Accuracy | Glioma F1 | Meningioma F1 | No Tumor F1 | Pituitary F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **VGG16** | **98.22%** | 0.96 | 0.97 | 1.00 | 1.00 |
| **DenseNet121** | 96.19% | 0.92 | 0.95 | 0.99 | 0.99 |
| **EfficientNet-B0** | 94.92% | 0.90 | 0.95 | 0.96 | 1.00 |

### External Data Validation
To test for real-world generalization, the models were validated against a larger external dataset (7,023 images):

| Model | External Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
| :--- | :--- | :--- | :--- | :--- |
| **VGG16** | **96.34%** | 0.96 | 0.96 | 0.96 |
| **EfficientNet-B0** | 94.69% | 0.95 | 0.95 | 0.95 |
| **DenseNet121** | 94.67% | 0.95 | 0.95 | 0.95 |

## Conclusion
While all models performed exceptionally well, **VGG16** emerged as the top performer, achieving the highest accuracy on both the internal and external datasets. Across all models, "No Tumor" and "Pituitary" classes showed the highest classification reliability, while "Glioma" and "Meningioma" presented the most complexity due to overlapping visual features.

## Source
* Dataset for training: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri
* Dataset for testing: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
