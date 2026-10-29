# Landmark Recognition System using SIFT Algorithm

An end-to-end Computer Vision application built in Python and OpenCV that recognizes and localizes famous landmarks from query images using the **Scale-Invariant Feature Transform (SIFT)** algorithm. 

The system demonstrates high robustness against variations in scale, rotation, illumination, and viewing perspective by leveraging SIFT feature descriptors, Brute-Force K-NN matching, Lowe's ratio test, and geometric verification using RANSAC Homography.

---

## 📌 Features & Architecture

The application pipeline is divided into six modular tasks:

1. **Landmark Database Management:**
   * Multi-class database organization (Eiffel Tower, Taj Mahal, Colosseum, Statue of Liberty, Great Wall of China).
   * Stores multiple viewing angles and illumination conditions per landmark.

2. **SIFT Feature Extraction:**
   * Scale-space extrema detection, keypoint localization, orientation assignment, and 128-dimensional descriptor extraction.

3. **Query Image Processing:**
   * Preprocessing and descriptor generation on input images.

4. **Feature Matching & Recognition:**
   * Brute-Force Matcher with Euclidean distance ($L_2$ norm).
   * $K$-Nearest Neighbors ($k=2$) matching with Lowe's Ratio Test ($0.75$).
   * Threshold-based decision logic to prevent false positives and reject unknown landmarks.

5. **Landmark Localization:**
   * Perspective transformation estimation via RANSAC Homography.
   * Geometric verification and dynamic bounding polygon projection around detected landmark regions.

6. **Performance Evaluation & Benchmarking:**
   * Performance testing under transformation conditions: Rotation, Scaling, Low Brightness, Perspective Changes, and Unseen/Unknown images.

---

## 🚀 Getting Started on Google Colab

### Prerequisites
Make sure your environment has the following Python libraries installed (pre-installed in Google Colab):
* `opencv-python`
* `numpy`
* `matplotlib`
* `pandas`

### Execution Steps
1. Open [Google Colab](https://colab.research.google.com/).
2. Create a new notebook.
3. Run each task in sequence across separate notebook cells:
   * **Task 1:** Builds the database folder structure and generates reference landmark dataset images.
   * **Task 2:** Extracts and visualizes keypoints for database items.
   * **Task 3:** Loads and processes the query image.
   * **Task 4:** Performs $K$-NN feature matching and computes candidate rankings.
   * **Task 5:** Computes RANSAC Homography and draws the localized bounding polygon.
   * **Task 6:** Executes the benchmarking pipeline across test conditions and generates the performance table.

---

## 📊 Expected Deliverables & Visual Outputs

Upon running the tasks, the system outputs:
* Visualizations of detected keypoints on reference and query images.
* Feature match lines connecting reference keypoints to query keypoints.
* Polygon overlay identifying the exact region of the landmark.
* Performance summary table evaluating execution time and match reliability.

---

## 🛠 Project Structure

```text
├── landmark_database/
│   ├── Eiffel_Tower/
│   ├── Taj_Mahal/
│   ├── Colosseum/
│   ├── Statue_of_Liberty/
│   └── Great_Wall_of_China/
├── README.md
└── landmark_recognition.ipynb
```

---

## 📝 License
Distributed under the MIT License. See `LICENSE` for details.