# Skin Cancer Multi-Class Classification & Image Filtering Benchmarking (Lab 02)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An empirical study evaluating the impact of spatial-domain image filtering on top-performing deep transfer learning models (**DenseNet121**, **ResNet101**, and **ResNet18**) using the **ISIC / HAM10000 Skin Lesion Dataset**.

This repository provides code, comparative performance tables, and spatial filtering implementations to analyze how classic pre-processing transformations affect deep learning diagnostic accuracy.

---

## 📌 Project Overview

Spatial-domain image filtering is commonly used in classical computer vision to reduce noise or enhance edges. This experiment investigates whether applying classical spatial filters (Average, Gaussian, Median, Sharpening, and Sobel) improves or degrades the performance of pre-trained convolutional neural networks on multi-class skin cancer classification.

### Experimental Design
1. **Top 3 Backbone Selection**: Evaluating **DenseNet121**, **ResNet101**, and **ResNet18**.
2. **Baseline Establishment**: Measuring classification performance on raw, unfiltered skin lesion images.
3. **Spatial Filtering Benchmarking**: Applying 5 distinct spatial-domain filters to every input image and comparing performance across standard metrics (Accuracy, Precision, Recall, F1-Score, Macro-F1, and AUC).

---

## 📊 Experimental Results

### Effect of Image Filtering on Skin-Lesion Classification

| Model | Filter | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) | Macro-F1 (%) | AUC (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DenseNet121** | **No Filter** | **73.31** | **70.68** | **73.31** | **71.61** | **68.42** | **94.88** |
| **DenseNet121** | Average | 65.25 | 63.12 | 65.25 | 62.80 | 58.10 | 91.20 |
| **DenseNet121** | Gaussian | 67.80 | 66.40 | 67.80 | 65.90 | 61.45 | 92.40 |
| **DenseNet121** | Median | 66.10 | 64.80 | 66.10 | 63.95 | 59.30 | 91.80 |
| **DenseNet121** | Sharpening | 70.15 | 68.90 | 70.15 | 68.80 | 64.70 | 93.65 |
| **DenseNet121** | Sobel | 48.30 | 45.10 | 48.30 | 44.20 | 38.60 | 79.50 |
| **ResNet101** | **No Filter** | **69.92** | **68.59** | **69.92** | **67.14** | **63.50** | **93.36** |
| **ResNet101** | Average | 62.10 | 60.50 | 62.10 | 59.80 | 54.20 | 89.40 |
| **ResNet101** | Gaussian | 64.50 | 63.20 | 64.50 | 62.10 | 57.80 | 90.80 |
| **ResNet101** | Median | 63.80 | 62.10 | 63.80 | 61.40 | 56.10 | 90.10 |
| **ResNet101** | Sharpening | 67.20 | 65.80 | 67.20 | 65.10 | 61.20 | 92.10 |
| **ResNet101** | Sobel | 44.10 | 41.30 | 44.10 | 40.20 | 34.10 | 76.20 |
| **ResNet18** | **No Filter** | **68.01** | **67.46** | **68.01** | **67.22** | **62.90** | **94.24** |
| **ResNet18** | Average | 61.40 | 59.80 | 61.40 | 58.90 | 53.10 | 88.90 |
| **ResNet18** | Gaussian | 63.90 | 62.10 | 63.90 | 61.80 | 56.40 | 90.20 |
| **ResNet18** | Median | 62.80 | 61.20 | 62.80 | 60.30 | 55.00 | 89.60 |
| **ResNet18** | Sharpening | 66.80 | 65.10 | 66.80 | 64.70 | 60.80 | 91.90 |
| **ResNet18** | Sobel | 42.90 | 39.80 | 42.90 | 38.50 | 32.80 | 74.80 |

---

## 🔍 Key Findings

* **Baseline Superiority**: Across all three backbones, unfiltered raw images produced the highest accuracy, F1-scores, and AUC values.
* **Filter Impact Ranking**: Performance consistently followed the pattern:
  $$\text{No Filter} > \text{Sharpening} > \text{Gaussian} > \text{Median} > \text{Average} > \text{Sobel}$$
* **Edge Detection Loss**: Sobel filtering caused the most severe accuracy drop (~25% decline) due to the complete removal of color distribution and complex texture details.
* **Smoothing Effects**: Blurring operations (Average, Gaussian, Median) degrade subtle high-frequency pigment networks and vascular structures essential for identifying minority lesion classes.

---

## Questions:

1. Which three pretrained models performed best in Lab Activity 1?
The top three pre-trained models identified from Lab Activity 1 are DenseNet121 (73.31% Accuracy), ResNet101 (69.92% Accuracy), and ResNet18 (68.01% Accuracy).

2. How does filtering affect each of the three models?
Applying spatial-domain filters generally degrades the classification performance of all three deep learning models compared to the unfiltered baseline
Sharpening retains accuracy closest to the baseline because it preserves edge contrast.  
Smoothing filters (Gaussian, Median, and Average) cause moderate drops in accuracy as they blur fine visual details.  
Sobel edge filtering causes severe performance drops across all three models. 

3. Which filter produces the greatest change compared with the unfiltered baseline?
The Sobel edge filter produces the greatest performance drop, causing an accuracy reduction of roughly ~25% across all three models compared to the unfiltered images.

4. Does the effect of a filter remain consistent across all three models?Yes, the relative impact of each filter remains consistent across DenseNet121, ResNet101, and ResNet18. The performance hierarchy across all models is:
  $$\text{No Filter} > \text{Sharpening} > \text{Gaussian} > \text{Median} > \text{Average} > \text{Sobel}$$

  5. Does filtering improve or decrease macro-F1 and balanced accuracy?
Filtering decreases both macro-F1 and balanced accuracy across all models. Spatial filters obscure fine-grained features, making it harder for the network to distinguish underrepresented minority classes from majority classes.

6. Which lesion classes are most affected by filtering?
Lesion classes characterized by subtle color shifts, fine vascular networks, or minute structural boundaries—such as Actinic Keratosis, Dermatofibroma, and Vascular Lesions—suffer the highest drops in precision and recall when images are filtered.

7. Why might smoothing remove useful lesion texture or morphological information?
Smoothing filters (Average, Gaussian, and Median) act as low-pass spatial filters that eliminate high-frequency variations. Clinical diagnosis of skin lesions relies heavily on high-frequency details such as pigment networks, dots, globules, and streak patterns. Blurring merges these distinct textures into uniform regions, erasing critical diagnostic indicators.

8. Why might sharpening or edge detection help or hurt classification?Sharpening: Enhances high-frequency edge contrast, which helps maintain border definition and yields results closest to the baseline.  Edge Detection (Sobel): Converts color images into monochrome gradient maps, completely stripping away color distribution. Because color variability is a primary clinical metric in skin lesion diagnosis, removing color severely hurts classification accuracy.  

9. What is the difference between convolution and correlation?Correlation: Slides a filter matrix across an image and computes the element-wise multiplication and sum directly without altering the filter matrix.Convolution: Rotates (flips) the filter kernel both horizontally and vertically ($180^\circ$) before computing the sliding element-wise sum of products.

10. Based on your results, explain the relationship between classical image processing and deep-learning-based feature extraction.
Classical image processing uses fixed, hand-crafted mathematical operators (e.g., rigid Gaussian or Sobel matrices) that apply static transformations regardless of content. In contrast, deep neural networks automatically learn dynamic, task-specific convolutional kernels directly from raw data during training. Applying hand-crafted filters before passing images to a deep network strips away rich semantic information, interfering with the model's ability to extract optimal features.