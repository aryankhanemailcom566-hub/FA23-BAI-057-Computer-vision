# Comparative Implementation of Machine Learning Regression Algorithms

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-informational)
![Status](https://img.shields.io/badge/Status-Academic%20Project-success)

---

## Table of Contents
- [1. Project Overview](#1-project-overview)
- [2. Objectives](#2-objectives)
- [3. Machine Learning Algorithms](#3-machine-learning-algorithms)
- [4. Dataset Description](#4-dataset-description)
- [5. Technologies and Libraries Used](#5-technologies-and-libraries-used)
- [6. Project Workflow](#6-project-workflow)
- [7. Data Preprocessing](#7-data-preprocessing)
- [8. Train-Test Splitting (80:20)](#8-train-test-splitting-8020)
- [9. Feature Scaling](#9-feature-scaling)
- [10. Model Implementation](#10-model-implementation)
- [11. Evaluation Metrics](#11-evaluation-metrics)
- [12. Comparative Performance Table](#12-comparative-performance-table)
- [13. Actual vs Predicted Visualization](#13-actual-vs-predicted-visualization)
- [14. Strengths and Limitations of All Six Algorithms](#14-strengths-and-limitations-of-all-six-algorithms)
- [15. How to Install and Run the Project](#15-how-to-install-and-run-the-project)
- [16. Requirements and Dependencies](#16-requirements-and-dependencies)
- [17. Results and Discussion](#17-results-and-discussion)
- [18. Final Conclusion](#18-final-conclusion)

---

## 1. Project Overview

This project presents a structured, empirical comparison of six widely used supervised Machine Learning regression algorithms applied to a real-world **House Price Prediction** problem using the **California Housing Dataset**. 

To guarantee a fair, unbiased, and reproducible evaluation, every algorithm is trained and tested under identical experimental conditions:
- **Identical Dataset:** All models operate on the exact same 20,640 samples.
- **Identical Features & Target:** All models use the same 8 input features to predict the continuous target variable (`MedHouseVal`).
- **Identical Data Split:** All models are evaluated on the exact same 80:20 train-test partition initialized with `random_state=42`.

---

## 2. Objectives

- Implement six distinct regression paradigms—ranging from linear and instance-based models to ensemble tree methods and artificial neural networks—on a common benchmark dataset.
- Apply proper preprocessing pipelines, ensuring distance-based and gradient-based models receive standardized inputs without data leakage.
- Evaluate and benchmark all six models using four standard regression metrics: **MAE**, **MSE**, **RMSE**, and **$R^2$ Score**.
- Visualize predictive alignment through **Actual vs. Predicted** scatter plots across all algorithms.
- Analyze the practical trade-offs, strengths, and limitations of each algorithm on tabular numerical data.

---

## 3. Machine Learning Algorithms

1. **Linear Regression:** A parametric statistical model that fits a linear hyperplane to the data by minimizing the residual sum of squares (Ordinary Least Squares) between observed and predicted targets.
2. **Decision Tree Regression:** A non-parametric model that recursively partitions the feature space into rectangular regions using greedy split criteria (e.g., squared error reduction) to predict local target averages.
3. **Random Forest Regression:** An ensemble bagging algorithm that trains multiple independent decision trees on bootstrapped subsets of data and averages their predictions to reduce variance and prevent overfitting.
4. **Neural Network Regression (`MLPRegressor`):** A Multi-Layer Perceptron consisting of interconnected layers of artificial neurons with non-linear activation functions trained via backpropagation to learn complex feature interactions.
5. **XGBoost Regression:** An optimized Extreme Gradient Boosting framework that builds decision trees sequentially, where each new tree corrects the residual errors of prior trees using gradient descent and regularization.
6. **K-Nearest Neighbors Regression (`KNN`):** A non-parametric, instance-based lazy learner that predicts the target value of a query sample by computing the distance-weighted or uniform average of its $k$ closest neighbors in feature space.

---

## 4. Dataset Description

The **California Housing Dataset** (originally derived from the 1990 U.S. Census) is loaded directly via `sklearn.datasets.fetch_california_housing`.

| Dataset Property | Specification |
| :--- | :--- |
| **Dataset Name** | California Housing Dataset |
| **Total Samples** | 20,640 |
| **Input Features** | 8 numerical features |
| **Target Variable** | `MedHouseVal` (Median house value in $100,000s) |
| **Train-Test Split** | 80% Training (16,512 samples) / 20% Testing (4,128 samples) |
| **Random State** | `42` |

### Input Features Breakdown

| Feature Name | Description |
| :--- | :--- |
| **`MedInc`** | Median income for households within a block group (measured in tens of thousands of US Dollars). |
| **`HouseAge`** | Median age of houses within a block group (measured in years). |
| **`AveRooms`** | Average number of rooms per household in the block group. |
| **`AveBedrms`** | Average number of bedrooms per household in the block group. |
| **`Population`** | Total number of people residing within the block group. |
| **`AveOccup`** | Average number of household members (occupancy) per dwelling. |
| **`Latitude`** | Geographic latitude coordinate of the block group (north-south position). |
| **`Longitude`** | Geographic longitude coordinate of the block group (east-west position). |

---

## 5. Technologies and Libraries Used

- **Python (3.8+):** Core programming language.
- **NumPy:** Numerical computing and array operations.
- **Pandas:** Tabular data manipulation, inspection, and metric summarization.
- **Matplotlib:** Data visualization and Actual vs. Predicted scatter plot generation.
- **Scikit-learn:** Dataset loading, preprocessing (`StandardScaler`), pipelines, model implementations, and evaluation metrics.
- **XGBoost:** High-performance gradient boosted decision tree implementation (`XGBRegressor`).

---

## 6. Project Workflow

```text
[Load California Housing Dataset]
               │
               ▼
[Data Inspection & Missing Value Check]
               │
               ▼
[Train-Test Split (80:20, random_state=42)]
               │
       ┌───────┴───────┐
       ▼               ▼
[Scaled Pipeline]  [Unscaled Pipeline]
(Linear, MLP, KNN) (Decision Tree, RF, XGBoost)
       │               │
       └───────┬───────┘
               ▼
[Model Training & Inference on Test Set]
               │
               ▼
[Compute MAE, MSE, RMSE, and R² Score]
               │
               ▼
[Comparative Table & Actual vs. Predicted Plots]
```

---

## 7. Data Preprocessing

1. **Data Loading:** Fetched as a Pandas DataFrame (`as_frame=True`) separating the 8 feature columns ($X$) from the target vector ($y$).
2. **Integrity Verification:** Checked for missing (`NaN`), null, or infinite values across all 20,640 rows and 8 features. The dataset is complete and requires no imputation.
3. **Data Types:** Verified that all input features and the target variable are continuous floating-point values (`float64`), eliminating the need for categorical encoding.

---

## 8. Train-Test Splitting (80:20)

To evaluate generalization performance on unseen data and prevent overfitting, the dataset is partitioned into training and testing subsets before applying any scaling transformations:

- **Training Set (80%):** 16,512 samples used to fit model parameters and scaling statistics.
- **Testing Set (20%):** 4,128 samples held out strictly for final evaluation.
- **Reproducibility:** `random_state=42` is fixed across the split so that every algorithm trains and tests on identical data partitions.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
```

---

## 9. Feature Scaling

Because input features span vastly different numerical ranges (e.g., `Population` reaches thousands while `AveBedrms` is typically between 1 and 2), feature scaling is selectively applied using Scikit-learn `Pipeline` objects:

- **Models Requiring `StandardScaler`:**
  - **Linear Regression:** Ensures well-conditioned coefficient estimation and comparable feature magnitudes.
  - **Neural Network (`MLPRegressor`):** Essential for stable, fast gradient descent convergence across hidden layers.
  - **K-Nearest Neighbors (`KNN`):** Critical because Euclidean distance calculations would otherwise be dominated by large-scale features like `Population`.
- **Models Not Requiring Feature Scaling:**
  - **Decision Tree, Random Forest, and XGBoost:** Tree-based algorithms make threshold splits on individual features independently and are invariant to monotonic feature scaling.

*Note: `StandardScaler` is fitted strictly on `X_train` and applied to `X_test` to prevent data leakage.*

---

## 10. Model Implementation

All six models are instantiated with clean, reproducible configurations:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

models = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Decision Tree": DecisionTreeRegressor(max_depth=15, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Neural Network (MLP)": make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=42)
    ),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1),
    "K-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5, n_jobs=-1))
}
```

---

## 11. Evaluation Metrics

Each model is evaluated on the test set using four standard regression metrics:

1. **Mean Absolute Error (MAE):** Measures the average absolute magnitude of prediction errors in the original target units ($100,000s), treating all errors linearly.
   $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

2. **Mean Squared Error (MSE):** Measures the average of squared prediction errors, penalizing larger outliers more heavily than MAE.
   $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

3. **Root Mean Squared Error (RMSE):** The square root of MSE, returning error to the interpretable units of the target variable while retaining sensitivity to large errors.
   $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

4. **Coefficient of Determination ($R^2$ Score):** Represents the proportion of variance in the target variable explained by the model (closer to `1.0` indicates stronger explanatory power).
   $$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

---

## 12. Comparative Performance Table

> **Note:** Populate the placeholders below with the exact numerical outputs generated after executing the training script or Jupyter Notebook.

| Rank | Regression Algorithm | Feature Scaling | MAE | MSE | RMSE | $R^2$ Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| — | **Linear Regression** | `StandardScaler` | `[MAE_LR]` | `[MSE_LR]` | `[RMSE_LR]` | `[R2_LR]` |
| — | **Decision Tree Regression** | None | `[MAE_DT]` | `[MSE_DT]` | `[RMSE_DT]` | `[R2_DT]` |
| — | **Random Forest Regression** | None | `[MAE_RF]` | `[MSE_RF]` | `[RMSE_RF]` | `[R2_RF]` |
| — | **Neural Network (`MLPRegressor`)** | `StandardScaler` | `[MAE_MLP]` | `[MSE_MLP]` | `[RMSE_MLP]` | `[R2_MLP]` |
| — | **XGBoost Regression** | None | `[MAE_XGB]` | `[MSE_XGB]` | `[RMSE_XGB]` | `[R2_XGB]` |
| — | **K-Nearest Neighbors (`KNN`)** | `StandardScaler` | `[MAE_KNN]` | `[MSE_KNN]` | `[RMSE_KNN]` | `[R2_KNN]` |

---

## 13. Actual vs Predicted Visualization

To visually inspect prediction accuracy, residual spread, and saturation at the dataset's upper target cap (`5.0`), a $2 \times 3$ grid of **Actual vs. Predicted** scatter plots is generated using `Matplotlib`.

- **X-Axis:** Actual Median House Value (`y_test`)
- **Y-Axis:** Predicted Median House Value (`y_pred`)
- **Reference Diagonal ($y = \hat{y}$):** A red dashed diagonal line represents ideal predictions; tighter clustering along this line indicates lower prediction error.

```text
[Insert Generated Visualization Here: e.g., results/actual_vs_predicted.png]
```

---

## 14. Strengths and Limitations of All Six Algorithms

| Algorithm | Key Strengths | Key Limitations |
| :--- | :--- | :--- |
| **Linear Regression** | Fast training and inference; highly interpretable feature weights; strong baseline model. | Assumes strict linear relationships; unable to capture complex non-linear spatial interactions (`Latitude`/`Longitude`). |
| **Decision Tree** | Captures non-linear rules; requires no feature scaling; easy to visualize and interpret. | High variance; prone to overfitting training data without depth or leaf constraints. |
| **Random Forest** | High accuracy on tabular data; robust against overfitting via bagging; handles non-linearities well. | Slower inference and larger memory footprint than single trees; less interpretable ("black-box"). |
| **Neural Network (`MLP`)** | Learns smooth, complex non-linear mappings; flexible architecture. | Sensitive to feature scaling and hyperparameter choices; requires iterative tuning and longer training times. |
| **XGBoost** | State-of-the-art performance on structured tabular datasets; built-in L1/L2 regularization; fast execution. | Requires careful tuning of learning rate and tree depth to avoid overfitting; less intuitive to interpret directly. |
| **K-Nearest Neighbors** | Simple, intuitive, non-parametric; zero training phase required; adapts naturally to local spatial patterns. | Computationally expensive at prediction time on large datasets; degrades in high dimensions and requires scaling. |

---

## 15. How to Install and Run the Project

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/california-housing-regression-comparison.git
cd california-housing-regression-comparison
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Execute the Notebook or Script
- **Using Jupyter Notebook:**
  ```bash
  jupyter notebook
  ```
  Open the project `.ipynb` file and select **Run All Cells**.
- **Using Python Script (if exported to `.py`):**
  ```bash
  python regression_comparison.py
  ```

---

## 16. Requirements and Dependencies

Save the following packages in a `requirements.txt` file in the root of your repository:

```text
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
xgboost>=2.0.0
jupyter>=1.0.0
```

---

## 17. Results and Discussion

> **Instructions:** After executing the implementation and populating Section 12, summarize your empirical findings below using the generated metrics.

- **Overall Best Performer:** `[Insert Top-Performing Model Name]` achieved the lowest RMSE (`[Value]`) and highest $R^2$ score (`[Value]`), demonstrating `[explain why—e.g., strong handling of non-linear geographic and income interactions]`.
- **Linear vs. Non-Linear Models:** Compare the baseline `Linear Regression` score (`[R2_LR]`) against non-linear ensemble and neural models to highlight the importance of non-linear feature interactions in housing valuation.
- **Impact of Feature Scaling & Spatial Features:** Discuss how `KNN` and `MLPRegressor` benefited from `StandardScaler`, particularly when leveraging geographic coordinates (`Latitude` and `Longitude`) alongside economic indicators (`MedInc`).
- **Observation on Target Capping:** Note any visible horizontal clustering near `MedHouseVal = 5.0` in the Actual vs. Predicted plots caused by right-censoring in the original census dataset.

---

## 18. Final Conclusion

This comparative study demonstrates how six distinct regression methodologies perform on a standardized tabular prediction task. By maintaining strict experimental controls—identical dataset, features, `80:20` train-test split (`random_state=42`), and leak-free preprocessing pipelines—the evaluation provides a transparent benchmark across linear, distance-based, neural, and tree-based ensemble paradigms.

Ultimately, selecting an optimal regression algorithm depends on balancing **predictive accuracy**, **training/inference latency**, **preprocessing requirements**, and **model interpretability** for the target deployment environment.
