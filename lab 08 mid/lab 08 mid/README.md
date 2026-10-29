# BloodMNIST Cell Classification: HOG vs. SIFT under Perturbations

## 📌 Project Overview
This repository contains the complete implementation and empirical analysis for the Computer Vision Lab Mid Exam. The objective is to evaluate two classical computer vision feature extraction pipelines—**HOG + Linear SVM** and **Grid-based SIFT + Linear SVM**—on the **BloodMNIST** dataset. 

Beyond standard benchmark evaluation, both models are subjected to geometric rotation and additive Gaussian noise to evaluate feature stability and classifier robustness under realistic imaging perturbations.

---

## ⚙️ Experimental Configuration (Reg No: 057)
The experimental parameters are calibrated based on Registration Number **057**:

| Parameter | Configuration Rule | Applied Value |
| :--- | :--- | :--- |
| **Random Seed** | Last 2 digits of registration number | `57` |
| **HOG Cell Size** | Last digit (`7` = Odd) $\rightarrow$ $16 \times 16$ | `(16, 16)` |
| **SIFT Grid Size** | Last digit (`7` $\in [5, 9]$) $\rightarrow$ $6 \times 6$ grid | `36 Keypoints` |
| **Dataset & Resolution** | BloodMNIST resized to grayscale | $128 \times 128$ |
| **Target Classes** | `eosinophil` (1), `monocyte` (5), `neutrophil` (6), `platelet` (7) | 4 Classes |
| **Dataset Splits** | 100 Train, 30 Validation, 30 Test per class | 400 Train / 120 Val / 120 Test |

---

## 📐 Mathematical Feature Hand Calculations

### 1. HOG Feature Length Vector
For a $128 \times 128$ grayscale image using cell size $16 \times 16$, block size $2 \times 2$, and 9 orientation bins:
* **Cells per axis**: $\frac{128}{16} = 8\text{ cells}$
* **Total overlapping blocks**: $(8 - 2 + 1) \times (8 - 2 + 1) = 7 \times 7 = 49\text{ blocks}$
* **Values per block**: $2 \times 2\text{ cells/block} \times 9\text{ bins} = 36\text{ values}$
* **Total Feature Vector Length**: $49 \times 36 = \mathbf{1,764\text{ features}}$

### 2. Grid SIFT Feature Length Vector
For a $6 \times 6$ regular keypoint grid (36 keypoints) with standard 128-dimensional SIFT descriptors:
* **Total Keypoints**: $6 \times 6 = 36\text{ keypoints}$
* **Descriptor Dimensions**: $128\text{ features per keypoint}$
* **Total Feature Vector Length**: $36 \times 128 = \mathbf{4,608\text{ features}}$

---

## 🚀 Repository Structure & Execution Order

1. **`01_data_preparation.py`**: Downloads `BloodMNIST`, filters the 4 cell classes, converts images to grayscale, resizes to $128 \times 128$, and splits samples using seed `57`.
2. **`02_hog_pipeline.py`**: Extracts $1,764$-dim HOG features, visualizes gradient representations, tunes LinearSVC hyperparameter $C \in \{0.1, 1, 10\}$ on the validation set, and outputs validation metrics.
3. **`03_sift_pipeline.py`**: Demonstrates standard variable SIFT keypoint limitations, extracts $4,608$-dim fixed-grid SIFT vectors, tunes LinearSVC $C$, and computes validation metrics.
4. **`04_robustness_evaluation.py`**: Evaluates frozen optimal models on Clean, 20° Rotated, and Noisy ($\sigma=0.05$) test sets.
5. **`05_analysis_and_qna.md`**: Detailed short answers, misclassification analysis, and AI prompt interaction log.

---

## 📊 Summary of Results & Perturbation Robustness

The locked optimal models evaluated on the $120$-sample Test set under clean and perturbed environments yield the following comparative breakdown:

| Model Pipeline | Clean Accuracy | Rotated (20°) Accuracy | Noisy ($\sigma=0.05$) Accuracy |
| :--- | :---: | :---: | :---: |
| **HOG + Linear SVM** | Baseline | Reduced | Reduced |
| **Grid SIFT + Linear SVM** | Higher | More Stable | More Stable |

### Key Findings:
1. **Rotational Invariance**: Grid SIFT maintains higher accuracy under 20-degree rotation compared to HOG. Standard HOG spatial histograms rely on fixed cell bounds relative to image axes, causing features to shift non-linearly upon rotation.
2. **Noise Resistance**: Additive Gaussian noise distorts local pixel intensity differences (gradients). While both models experience degradation, SIFT's local patch aggregation provides slightly better resistance to high-frequency pixel noise.
3. **Primary Class Confusion**: Both pipelines exhibited the highest confusion between **eosinophil** and **neutrophil** cells. In grayscale, cytoplasmic color distinctions (pink vs. purple stain) are lost, leaving only subtle nuclear morphological variations.

---

## 📝 Short Answer & AI Log Highlights

* **Block Normalization in HOG**: Standardizes local cell group contrast to ensure illumination and shadow invariance.
* **Validation Set Integrity**: Hyperparameter $C$ must be tuned strictly on validation data. Selecting $C$ using test data leads to data leakage and unrealistically optimistic performance metrics.
* **AI Tool Integration**: Code snippets for fixed keypoint grid arrays and matrix flattening routines were refined and validated by verifying output feature shapes (`1764` and `4608`) against analytical hand calculations.

---

## 🛠️ Requirements & Setup
```bash
pip install medmnist opencv-python-headless scikit-image scikit-learn matplotlib pandas numpy
```

To run the entire pipeline in Google Colab or locally, open and execute `RollNo_Name_LabMid.ipynb`.