# HOG-Based Industrial Defect Detection & Classification

This repository contains the complete implementation for **Lab 05: HOG-Based Industrial Defect Detection and Classification**. The project focuses on building an automated visual inspection system for manufacturing lines using **Histogram of Oriented Gradients (HOG)** features combined with machine learning classifiers (**Support Vector Machines** and **Random Forest**).

Designed to run seamlessly in **Google Colab**, this project includes complete preprocessing, feature extraction, hyperparameter tuning, robustness stress testing, and an industrial decision-control module.

---

## 📌 Features & Capabilities

- **Dataset Setup & Preprocessing:** Handles automated downloading from Kaggle (NEU Surface Defect Database) or fallback synthetic image generation. Resizes and converts images to grayscale.
- **HOG Feature Extraction & Visualization:** Visualizes gradient orientations for defective and non-defective steel surfaces.
- **Multi-Model Evaluation:** Trains and compares SVM (RBF kernel) and Random Forest classifiers.
- **Parameter Sensitivity Analysis:** Compares HOG cell sizes ($4\times4$, $8\times8$, $16\times16$) and orientation bins ($6, 9, 12$).
- **Robustness Stress-Testing:** Evaluates performance degradation under realistic industrial disturbances:
  - Brightness variations ($+40$ offset)
  - Gaussian Noise ($\sigma = 25$)
  - Part Misalignment / Rotation ($15^\circ$)
  - Camera Blur ($5\times5$ Gaussian kernel)
- **Quality-Control Decision Module:** Simulates an automated industrial acceptance/rejection pipeline complete with confidence scores.

---

## 🚀 Getting Started on Google Colab

### 1. Open Colab
1. Navigate to [Google Colab](https://colab.research.google.com/).
2. Create a new notebook.

### 2. Kaggle API Setup (Optional for Real Dataset)
To use the official **NEU Surface Defect Dataset**:
1. Go to your Kaggle Account settings and click **Create New API Token** to download `kaggle.json`.
2. Upload `kaggle.json` to your Colab session files.
3. Run the setup commands embedded in the main script.

> *Note: If `kaggle.json` is not supplied, the script automatically generates a synthetic dataset so that all code blocks execute without error.*

### 3. Run Code
Copy and execute the Python code provided in your notebook to generate visualizations, performance tables, and quality control predictions.

---

## 📊 Summary of Experiments

| Experiment | Target Objective | Key Findings |
| :--- | :--- | :--- |
| **Baseline Classifier** | Compare SVM vs. Random Forest | SVM achieves superior boundary separation on high-dimensional HOG vectors. |
| **Cell Size Tuning** | $4\times4$ vs. $8\times8$ vs. $16\times16$ | $8\times8$ provides optimal balance between feature size and computational efficiency. |
| **Robustness Test** | Evaluate real-world factory noise | Gaussian noise and motion blur degrade recall significantly; brightness variations are handled well due to $L2$-Hys normalization. |

---

## 🛠️ Project Structure

```text
.
├── dataset/                    # Loaded / Generated surface images
│   ├── normal/
│   └── defective/
├── README.md                   # Project instructions
└── report.md                   # Formal technical lab report
```

---

## 📜 License
This project is for educational and research purposes as part of the Computer Vision Laboratory.