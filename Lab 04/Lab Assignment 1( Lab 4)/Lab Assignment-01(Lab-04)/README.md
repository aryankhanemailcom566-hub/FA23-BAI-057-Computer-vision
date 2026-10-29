<<<<<<< HEAD
# Skin Lesion Boundary Detection Using Canny Edge Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An automated computer vision pipeline designed to detect, delineate, and quantify skin lesion boundaries from dermoscopy images using Gaussian filtering, multi-threshold Canny edge detection, and morphological contour analysis[cite: 5].

---

## 📌 Problem Statement

Accurate delineation of skin lesion boundaries is a crucial step in automated dermoscopic image analysis[cite: 5]. This project evaluates the effectiveness of classical edge detection techniques in isolating lesion contours from surrounding healthy skin, quantifying key geometric properties including **Lesion Area** and **Lesion Perimeter**.

---

## 🚀 Workflow Pipeline

The processing pipeline follows five main steps[cite: 5, 6]:

1. **Image Loading**: Select representative skin lesion images from the ISIC dataset[cite: 5].
2. **Preprocessing**: Convert RGB images to grayscale and apply a $5 \times 5$ Gaussian filter for high-frequency noise reduction[cite: 5, 6].
3. **Canny Edge Detection**: Evaluate hysteresis thresholds ($50\text{--}100$, $100\text{--}200$, $150\text{--}250$) to select optimal edge contours[cite: 5].
4. **Lesion Boundary Extraction**: Apply morphological closing and external contour detection to highlight the outer boundary[cite: 5].
5. **Geometric Quantification**: Calculate total lesion area (pixels) and perimeter (pixels)[cite: 6].
=======
# Skin Cancer Multi-Class Classification & Model Benchmarking

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end empirical study evaluating deep transfer learning architectures and classical machine learning classifiers on the **9-class ISIC Skin Cancer Dataset**. 

This repository provides code, benchmark metrics (Accuracy, Precision, Recall, F1-Score, AUC), and computational efficiency analysis (Parameters, Model Size, FLOPs, and Inference Latency) to identify the optimal model balance between clinical performance and hardware overhead.

---

## 📌 Project Overview

Skin cancer is one of the most common cancers globally, but early detection significantly increases survival rates. Automated computer-aided diagnosis using deep learning can support clinicians in identifying malignant lesions.

This project carries out three comprehensive benchmarking experiments:
1. **Fine-Tuning Convolutional Backbones**: End-to-end training of pre-trained deep learning networks (AlexNet, VGG, ResNet, DenseNet, EfficientNet).
2. **Deep Feature Extraction + ML Classifiers**: Extracting high-dimensional feature representations using ResNet50 and training classical machine learning classifiers (SVM, XGBoost, Random Forest, KNN, etc.).
3. **Computational Efficiency Profiling**: Measuring floating-point operations (FLOPs), parameter footprint, storage requirements, and per-sample inference latency across architectures.
>>>>>>> 13337500a919a3210a98cc7bb2f6743635836c7d

---

## 📊 Experimental Results

<<<<<<< HEAD
### Table 1: Lesion Boundary Measurements Across Dataset Images

| Image | Best Filter | Edge Method | Area (pixels) | Perimeter (pixels) |
| :--- | :--- | :--- | :---: | :---: |
| **Image 1** | Gaussian ($5\times5$)[cite: 6] | Canny ($100\text{--}200$)[cite: 5, 6] | $12,450.50$[cite: 6] | $542.10$[cite: 6] |
| **Image 2** | Gaussian ($5\times5$)[cite: 6] | Canny ($100\text{--}200$)[cite: 5, 6] | $9,820.00$[cite: 6] | $485.30$[cite: 6] |
| **Image 3** | Gaussian ($5\times5$)[cite: 6] | Canny ($100\text{--}200$)[cite: 5, 6] | $15,110.25$[cite: 6] | $612.80$[cite: 6] |
| **Image 4** | Gaussian ($5\times5$)[cite: 6] | Canny ($100\text{--}200$)[cite: 5, 6] | $8,340.75$[cite: 6] | $420.50$[cite: 6] |
| **Image 5** | Gaussian ($5\times5$)[cite: 6] | Canny ($100\text{--}200$)[cite: 5, 6] | $11,290.00$[cite: 6] | $510.40$[cite: 6] |

---

### Table 2: Final Comparison Across Preprocessing & Edge Detection Combinations

| Method | Noise Handling | Edge Quality | Boundary Detection | Overall Performance |
| :--- | :--- | :--- | :--- | :--- |
| **Original + Sobel**[cite: 6] | Poor[cite: 6] | Noisy[cite: 6] | Incomplete[cite: 6] | Poor[cite: 6] |
| **Original + Canny**[cite: 6] | Moderate[cite: 6] | Fair[cite: 6] | Fragmented[cite: 6] | Moderate[cite: 6] |
| **Average + Sobel**[cite: 6] | Moderate[cite: 6] | Blurred[cite: 6] | Weak[cite: 6] | Fair[cite: 6] |
| **Average + Canny**[cite: 6] | Good[cite: 6] | Clean[cite: 6] | Moderate[cite: 6] | Good[cite: 6] |
| **Gaussian + Sobel**[cite: 6] | Good[cite: 6] | Smooth[cite: 6] | Moderate[cite: 6] | Good[cite: 6] |
| **Gaussian + Canny**[cite: 6] | **Excellent**[cite: 6] | **Sharp & Thin**[cite: 6] | **Precise**[cite: 6] | **Best**[cite: 6] |
| **Median + Sobel**[cite: 6] | Good (Impulse)[cite: 6] | Coarse[cite: 6] | Fair[cite: 6] | Fair[cite: 6] |
| **Median + Canny**[cite: 6] | Very Good[cite: 6] | Sharp[cite: 6] | Good[cite: 6] | Very Good[cite: 6] |

---

## ❓ Analytical Insights

1. **Role of Gaussian Filtering**: Gaussian pre-filtering suppresses random skin texture noise that causes false gradient spikes during derivative calculations[cite: 5, 7].
2. **Optimal Canny Threshold**: The medium threshold setting ($100\text{--}200$) provides the optimal balance, preserving continuous outer boundaries without picking up background skin clutter[cite: 5, 7].
3. **Challenges Observed**: Hair artifacts create false linear loops, and low-contrast borders occasionally cause boundary leaks[cite: 5, 7].
4. **Future Improvements**: Pipeline robustness can be enhanced by integrating hair removal algorithms (DullRazor) and processing images in HSV/LAB color spaces[cite: 5, 7].

---

## 🛠️ Usage & Setup

### Requirements
```bash
pip install opencv-python numpy matplotlib pandas kagglehub


Why is Gaussian filtering applied before Canny detection?   
Answer: Edge detection relies on calculating image intensity gradients (first derivatives). High-frequency random noise generates false intensity spikes that trigger false edges. Applying a Gaussian filter smooths out high-frequency noise

 while preserving primary structural boundaries.   How did the three Canny threshold settings affect the result?  
 Answer:Low Threshold ($50\text{–}100$): Detected fine details and textures, but included unwanted background skin noise.   Medium Threshold ($100\text{–}200$): Balanced noise suppression while keeping a continuous 1-pixel boundary.   High Threshold ($150\text{–}250$): Suppressed noise completely, but caused broken and fragmented boundary lines.  
 
  Which threshold produced the best lesion boundary?   
  Answer: The Medium threshold ($100\text{–}200$) produced the clearest boundary. It preserved the outer lesion perimeter without breaking hysteresis connectivity or picking up background skin textures.   
  
  Why are edges useful for detecting skin lesions?  
   Answer: Skin lesions (e.g., melanoma, nevus) typically have distinct irregular borders separating them from surrounding healthy skin. Edge maps highlight these spatial boundaries, enabling automated contour extraction, shape analysis, and area/perimeter calculations.   
   
   What problems did you observe in detecting the lesion boundary?   
   Answer:Presence of Skin Hair: Hair strands produce sharp linear false edges across the lesion.   Low Contrast / Fuzzy Borders: Some benign lesions fade gradually into healthy skin, causing Canny hysteresis to produce broken boundaries.  
   
    How could your method be improved?   Answer:Apply DullRazor hair-removal algorithms before filtering.   Convert images to the HSV or LAB color space (channel $a^*$ or $b^*$) where skin-to-lesion contrast is higher than in grayscale.   Combine Canny edge maps with Active Contours (Snakes) or Otsu Thresholding for robust segmentation. 
=======
### Table 1: Comparison of Transfer Learning Models
*Fine-tuned pre-trained CNN backbones across 9 skin lesion classes.*

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AlexNet** | 60.17 | 62.68 | 60.17 | 60.59 | 91.74 |
| **VGG16** | 59.11 | 57.94 | 59.11 | 57.59 | 90.87 |
| **VGG19** | 59.32 | 59.02 | 59.32 | 56.53 | 89.23 |
| **ResNet18** | 69.28 | 67.46 | 69.28 | 67.22 | 94.24 |
| **ResNet50** | 67.80 | 67.14 | 67.80 | 65.16 | 92.89 |
| **ResNet101** | 69.49 | 68.59 | 69.49 | 67.14 | 93.36 |
| **DenseNet121** | 73.31 | 70.68 | 73.31 | 71.61 | 94.88 |
| **EfficientNet-B0** | 63.35 | 63.44 | 63.35 | 60.08 | 91.10 |

---

### Table 2: Comparison of Different Classifiers
*Evaluation of standard machine learning classifiers trained on deep feature embeddings extracted from ResNet50.*

| Feature Extractor | Classifier | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Deep Features** | Logistic Regression | 56.36 | 56.55 | 56.36 | 56.19 | 88.50 |
| **Deep Features** | Decision Tree | 31.36 | 32.34 | 31.36 | 31.67 | 59.92 |
| **Deep Features** | Random Forest | 50.42 | 49.82 | 50.42 | 45.91 | 84.58 |
| **Deep Features** | K-Nearest Neighbors (KNN) | 48.73 | 50.15 | 48.73 | 47.61 | 81.20 |
| **Deep Features** | Linear SVM | 50.85 | 52.45 | 50.85 | 51.26 | 89.02 |
| **Deep Features** | RBF-SVM | 60.17 | 59.49 | 60.17 | 57.81 | 89.92 |
| **Deep Features** | XGBoost | 56.99 | 54.42 | 56.99 | 55.11 | 87.32 |

---

### Table 3: Computational Efficiency Comparison
*Profiling model complexity, disk storage requirements, and computational cost.*

| Model | Parameters (M) | Model Size (MB) | FLOPs (G) | Inference Time (ms) | Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AlexNet** | 57.04 | 217.60 | 0.71 | 20.95 | 60.17 |
| **VGG16** | 134.30 | 512.31 | 15.47 | 17.54 | 59.11 |
| **VGG19** | 139.61 | 532.57 | 19.63 | 18.64 | 59.32 |
| **ResNet18** | 11.18 | 42.73 | 1.82 | 17.26 | 69.28 |
| **ResNet50** | 23.53 | 90.04 | 4.13 | 18.14 | 67.80 |
| **DenseNet121** | 6.96 | 27.14 | 2.90 | 16.08 | 73.31 |
| **EfficientNet-B0** | 4.02 | 15.69 | 0.41 | 16.63 | 63.35 |

---

## 📂 Dataset Information

The dataset used is the **ISIC Skin Cancer 9 Classes Dataset** downloaded directly via `kagglehub`:
- **Source**: `nodoubttome/skin-cancer9-classesisic`
- **Number of Classes**: 9 distinct skin lesion types (Actinic Keratosis, Basal Cell Carcinoma, Dermatofibroma, Melanoma, Nevus, Pigmented Benign Keratosis, Seborrheic Keratosis, Squamous Cell Carcinoma, Vascular Lesion)
- **Preprocessing**: Resized to 224x224, normalized to ImageNet standard parameters, and augmented with random horizontal/vertical flips.

---

## 🛠️ Project Structure

```text
skin-cancer-classification/
├── main.py              # Complete pipeline (Dataset loading, Fine-tuning, Feature extraction, Profiling)
├── requirements.txt     # Environment dependencies
├── .gitignore           # Ignored files (datasets, model checkpoints, environment settings)
└── README.md            # Project documentation and results
>>>>>>> 13337500a919a3210a98cc7bb2f6743635836c7d
