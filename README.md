# 🤖 Data Science Project

A unified Streamlit application showcasing Supervised and Unsupervised Learning models.

## 📊 Overview

This project implements various Machine Learning algorithms to solve different problems:

1.  **Exam Score Prediction (Regression)**:
    *   **Goal**: Predict student exam scores based on study hours, sleep quality, and other factors.
    *   **Models**: Linear Regression, Decision Tree Regressor, Random Forest Regressor.
    *   **Metric**: R² Score, RMSE.

2.  **Student Grade Prediction (Classification)**:
    *   **Goal**: Classify student grades (A-F) based on their performance metrics.
    *   **Models**: Logistic Regression, KNN, Decision Tree, Random Forest, Naive Bayes.
    *   **Metric**: Accuracy, Confusion Matrix.

3.  **Country Clustering (Unsupervised Learning)**:
    *   **Goal**: Group countries into clusters based on socio-economic indicators (e.g., child mortality, GDP, health).
    *   **Algorithms**: K-Means, DBSCAN, Hierarchical Clustering.
    *   **Visualization**: PCA-reduced scatter plots.

## 🚀 How to Run

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Vibin-007/Data_Science_Project.git
    cd Data_Science_Project
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit App**:
    ```bash
    streamlit run app.py
    ```

## 📂 Project Structure

*   `app.py`: Main Streamlit application file.
*   `Exam_Score_Prediction.csv`: Dataset for Supervised Learning.
*   `Country-data.csv`: Dataset for Unsupervised Learning.
*   `Supervized/`: Contains original Jupyter Notebooks for analysis.
*   `Unsupervised/`: Contains original Jupyter Notebooks for analysis.

## 🛠️ Technologies Used

*   **Python**: Core programming language.
*   **Streamlit**: Web application framework.
*   **Scikit-Learn**: Machine Learning library.
*   **Pandas & NumPy**: Data manipulation.
*   **Matplotlib & Seaborn**: Data visualization.

---
