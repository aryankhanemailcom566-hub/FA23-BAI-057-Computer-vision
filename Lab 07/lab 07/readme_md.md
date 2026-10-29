# Lab 07: Improving HOG-Based Medical Image Classification Using a Genetic Algorithm

This project implements a computer vision pipeline that integrates a **Genetic Algorithm (GA)** for feature selection into a traditional **Histogram of Oriented Gradients (HOG)** and **Support Vector Machine (SVM)** framework to classify medical images from the **MedMNIST (PneumoniaMNIST)** benchmark.

---

## 📌 Project Overview

Traditional HOG feature extraction generates high-dimensional feature vectors that contain redundant spatial gradient information. This project evaluates whether using a Genetic Algorithm to select an optimal subset of HOG features improves classification performance while significantly reducing feature dimensionality.

---

## 🚀 Key Features & Pipeline

1. **Dataset Preprocessing:** Loads the `PneumoniaMNIST` dataset and resizes images to $128 \times 128$ resolution across fixed, identical train/val/test splits.
2. **HOG Feature Extraction:** Computes dense local gradient histograms ($9$ orientations, $8 \times 8$ pixels/cell, $2 \times 2$ cells/block, L2-Hys normalization).
3. **Baseline Model:** Trains an RBF-kernel SVM classifier on all unselected raw HOG features.
4. **Genetic Algorithm Feature Selection:**
   * Binary chromosome representation ($1$ = keep feature, $0$ = drop feature).
   * Fitness function based on Validation Macro-F1 score with a feature count penalty.
   * Tournament selection, single-point crossover, and bit-flip mutation over multiple generations.
5. **Proposed Model Evaluation:** Evaluates the SVM model trained exclusively on GA-selected features against the test set.
6. **Performance Visualization:** Generates plots for GA fitness convergence and feature dimensionality reduction.

---

## ⚙️ Experimental Configuration (Table 1)

| Parameter | Baseline: HOG + SVM | Proposed: GA-HOG + SVM |
| :--- | :--- | :--- |
| **Medical Dataset** | PneumoniaMNIST | PneumoniaMNIST |
| **Image Resolution** | $128 \times 128$ | $128 \times 128$ |
| **HOG Orientations** | 9 | 9 |
| **Pixels per Cell** | $8 \times 8$ | $8 \times 8$ |
| **Cells per Block** | $2 \times 2$ | $2 \times 2$ |
| **Block Normalization** | L2-Hys | L2-Hys |
| **Feature Selection** | None | Genetic Algorithm |
| **Classifier** | SVM (RBF Kernel) | SVM (RBF Kernel) |
| **Data Split** | Fixed Identical Split | Fixed Identical Split |

---

## 🛠 Prerequisites & Installation

To run this project on Google Colab or locally, install the following dependencies:

```bash
pip install medmnist scikit-image scikit-learn matplotlib seaborn pandas opencv-python
```

---

## 💻 How to Run

1. Open **Google Colab** or your local Python environment.
2. Execute the notebook cells sequentially:
   * **Cell 1:** Load data and extract HOG feature matrices.
   * **Cell 2:** Train baseline `HOG + SVM` model.
   * **Cell 3:** Run `GeneticAlgorithmFeatureSelector`.
   * **Cell 4:** Train proposed `GA-HOG + SVM` model and compute evaluation metrics (Table 2).
   * **Cell 5:** Generate fitness convergence and feature reduction plots.

---

## 📊 Outputs & Deliverables

* `hog_medical_samples.png`: Visual comparison of original medical scans and their extracted HOG representations.
* `ga_hog_results.png`: Comparison graphs of GA fitness across generations and feature reduction counts.