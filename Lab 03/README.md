# Edge Detection Techniques and Their Impact on Classification Performance (Lab 03)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An empirical study exploring classical spatial-domain edge detection methods (**Sobel**, **Prewitt**, **Laplacian**, **LoG**, and **Canny**), analyzing their sensitivity to artificial noise, and evaluating how edge-only feature representations compare against raw and pre-filtered images for skin lesion classification.

---

## 📌 Project Overview

Edge detection isolates object boundaries by identifying sharp brightness discontinuities. This laboratory investigates whether manually providing edge maps to machine learning models and deep neural networks enhances or degrades multi-class classification performance on the **ISIC 9-Class Skin Cancer Dataset**[cite: 6].

### Key Objectives
1. **Comparative Edge Detection**: Implement first-order (Sobel, Prewitt), second-order (Laplacian, LoG), and multi-stage (Canny) edge detectors[cite: 6].
2. **Noise Sensitivity Analysis**: Evaluate the impact of Gaussian and Salt-and-Pepper noise on edge detectors with and without spatial pre-filtering (Gaussian and Median filters)[cite: 6].
3. **Canny Parameter Tuning**: Analyze threshold combinations and kernel sizes to find optimal edge representations[cite: 6].
4. **Cross-Lab Benchmarking**: Compare classification performance across **Set A (Raw Images - Lab 01)**, **Set B (Filtered Images - Lab 02)**, and **Set C (Edge Maps - Lab 03)** using classical classifiers (SVM, Random Forest, KNN) and deep CNN backbones (ResNet18, ResNet50)[cite: 6].

---

## 📊 Experimental Results

### Table 1: Effect of Noise and Preprocessing on Edge Detection

| Edge Detector | Input Image | Noise Type | Preprocessing | Edge Quality | Noise Sensitivity | Observations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sobel** | Original | None | None | High | Low | Clear lesion boundaries with fine texture details[cite: 6]. |
| **Sobel** | Noisy | Gaussian | None | Poor | High | High false edges across homogenous skin regions[cite: 6]. |
| **Sobel** | Noisy | Salt & Pepper | None | Very Poor | Very High | Impulse noise artifacts produce severe isolated edge spikes[cite: 6]. |
| **Sobel** | Noisy | Gaussian | Gaussian Filter | Moderate | Low | Noise suppressed effectively; edges slightly blurred[cite: 6]. |
| **Sobel** | Noisy | Salt & Pepper | Median Filter | Good | Low | Median filter completely eliminates salt-and-pepper noise artifacts[cite: 6]. |
| **Prewitt** | Original | None | None | High | Low | Similar boundary response to Sobel but slightly weaker diagonal edges[cite: 6]. |
| **Laplacian** | Original | None | None | Moderate | Very High | Second-order derivative accentuates minor intensity changes[cite: 6]. |
| **LoG** | Noisy | Gaussian | Gaussian Filter | Moderate | Moderate | Pre-smoothing suppresses high-frequency noise before derivative computation[cite: 6]. |
| **Canny** | Original | None | Built-in smoothing | Very High | Low | Thin, continuous, 1-pixel wide edge contours of lesion boundaries[cite: 6]. |
| **Canny** | Noisy | Gaussian | Gaussian Filter | Good | Low | Non-maximum suppression and hysteresis preserve primary boundary contours[cite: 6]. |
| **Canny** | Noisy | Salt & Pepper | Median Filter | Good | Low | Median filter removes impulse spots, preventing false hysteresis triggers[cite: 6]. |

---

### Table 2: Canny Parameter Analysis

| Configuration | Low Threshold | High Threshold | Kernel Size | Edge Quality | Number of Detected Edges | Observation |
| :--- | :---: | :---: | :---: | :--- | :---: | :--- |
| **Canny-1** | 30 | 100 | 3×3 | High Detail / Noisy | 4,812 | Captures fine internal textures but includes background noise[cite: 6]. |
| **Canny-2** | 50 | 150 | 3×3 | Optimal / Balanced | 2,945 | Best trade-off: clear continuous lesion boundary without clutter[cite: 6]. |
| **Canny-3** | 100 | 200 | 3×3 | Sparse / Fragmented | 1,120 | High threshold rejects minor details, causing broken boundary lines[cite: 6]. |
| **Canny-4** | 50 | 150 | 5×5 | Smooth / General | 2,310 | Larger 5×5 kernel smooths out fine skin textures before thresholding[cite: 6]. |

---

### Table 3: Cross-Lab Classification Performance Comparison

| Model / Classifier | Accuracy Raw (Lab 1) (%) | Accuracy Filtered (Lab 2) (%) | Accuracy Edge (Lab 3) (%) | Precision (%) | Recall (%) | F1-Score (%) | Training Time (s) | Inference Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM** | 60.17 | 59.40 | 41.20 | 59.49 | 60.17 | 57.81 | 12.45 | 1.85 |
| **Random Forest** | 50.42 | 49.80 | 33.10 | 49.82 | 50.42 | 45.91 | 8.30 | 0.95 |
| **KNN** | 48.73 | 47.90 | 30.50 | 50.15 | 48.73 | 47.61 | 0.05 | 4.20 |
| **CNN Model 1 (ResNet18)** | **68.01** | **66.80** | **42.90** | **67.46** | **68.01** | **67.22** | **45.30** | **17.26** |
| **CNN Model 2 (ResNet50)** | **68.86** | **66.10** | **43.20** | **67.14** | **68.86** | **65.16** | **95.10** | **18.14** |

---

## 🔍 Key Findings & Discussion

1. **Information Loss in Edge Representations**: Passing edge-only images (Set C) to classification models results in a **severe performance drop (~25% decrease in accuracy)**[cite: 6]. Removing color distribution, shading, and internal lesion texture strips away core clinical diagnostic information[cite: 6].
2. **Noise Sensitivity**: Second-order derivative detectors (Laplacian) are significantly more sensitive to high-frequency noise than first-order detectors (Sobel, Prewitt)[cite: 6]. Pre-filtering with Median or Gaussian kernels is essential prior to edge extraction[cite: 6].
3. **Handcrafted vs. Learned Features**: Deep CNNs naturally learn dynamic, data-driven directional edge filters in their early layers during backpropagation[cite: 6]. Supplying static, handcrafted edge maps interferes with this feature learning process and limits overall network capacity[cite: 6].

---

