import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
# Clustering Models
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
# Metrics
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Page Configuration
st.set_page_config(page_title="Data Science Project Hub", layout="wide")

# Title
st.title("🤖 Data Science Model Explorer")
st.markdown("Explore Supervised and Unsupervised Learning models on different datasets.")

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose a Project", [
    "Home", 
    "Supervised Learning (Regression)", 
    "Supervised Learning (Classification)",
    "Unsupervised Learning (Clustering)"
])

# File Paths
REGRESSION_DATA_PATH = r"D:\College\Data Science\Projects\Exam_Score_Prediction.csv"
CLASSIFICATION_DATA_PATH = r"D:\College\Data Science\Projects\Exam_Score_Prediction.csv"
CLUSTERING_DATA_PATH = r"D:\College\Data Science\Projects\Country-data.csv"

# --- HOME PAGE ---
if app_mode == "Home":
    # Fixed deprecation warning: use_column_width=True -> removed or handled differently if needed.
    # Using 'use_column_width="auto"' is often the modern equivalent, or just relying on default behavior.
    st.markdown("""
    ### Welcome to the Project Hub!
    
    This application demonstrates:
    
    1.  **Supervised Learning (Regression)**: Predict student exam scores (Continuous).
    2.  **Supervised Learning (Classification)**: Predict student grades (Categorical A-F).
    3.  **Unsupervised Learning**: Group countries based on socio-economic factors.
    
    👈 **Select a project from the sidebar to get started.**
    """)

# --- SUPERVISED LEARNING (REGRESSION) ---
elif app_mode == "Supervised Learning (Regression)":
    st.header("📈 Exam Score Prediction (Regression)")
    
    try:
        df = pd.read_csv(REGRESSION_DATA_PATH)
        st.success("Dataset Loaded Successfully!")
        
        if st.checkbox("Show Raw Data"):
            st.dataframe(df.head())
            
        # Preprocessing
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].mean())
        
        le = LabelEncoder()
        obj_cols = df.select_dtypes(include='object').columns
        encoders = {}
        for col in obj_cols:
            encoders[col] = LabelEncoder()
            df[col] = encoders[col].fit_transform(df[col])
            
        target = 'exam_score'
        if 'student_id' in df.columns:
            df = df.drop('student_id', axis=1)
            
        X = df.drop(target, axis=1)
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        st.sidebar.subheader("Model Configuration")
        model_choice = st.sidebar.selectbox("Select Regression Model", ["Linear Regression", "Decision Tree", "Random Forest"])
        
        model = None
        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Decision Tree":
            model = DecisionTreeRegressor(random_state=42)
        elif model_choice == "Random Forest":
            est = st.sidebar.slider("Number of Estimators", 10, 200, 100)
            model = RandomForestRegressor(n_estimators=est, random_state=42)
            
        if st.button("Train Regression Model"):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            col1, col2 = st.columns(2)
            col1.metric("R2 Score", f"{r2:.4f}")
            col2.metric("RMSE", f"{rmse:.4f}")
            
            fig, ax = plt.subplots()
            sns.scatterplot(x=y_test, y=y_pred, ax=ax)
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title(f"Actual vs Predicted ({model_choice})")
            st.pyplot(fig)
            
    except FileNotFoundError:
        st.error(f"File not found at {REGRESSION_DATA_PATH}. Please check folder structure.")

# --- SUPERVISED LEARNING (CLASSIFICATION) ---
elif app_mode == "Supervised Learning (Classification)":
    st.header("🏆 Student Grade Prediction (Classification)")
    
    try:
        df = pd.read_csv(CLASSIFICATION_DATA_PATH)
        st.success("Dataset Loaded Successfully!")
        
        # Create Target Variable (Grade)
        def get_grade(score):
            if score >= 90: return 'A'
            elif score >= 80: return 'B'
            elif score >= 70: return 'C'
            elif score >= 60: return 'D'
            else: return 'F'
            
        if 'exam_score' in df.columns:
            df['Grade'] = df['exam_score'].apply(get_grade)
            # Drop features that are directly correlated/leakage (exam_score, student_id)
            # Keeping score makes it trivial, so dropping it is correct for prediction from other features.
            df = df.drop(['exam_score'], axis=1)
        
        if 'student_id' in df.columns:
            df = df.drop('student_id', axis=1)

        if st.checkbox("Show Processed Data"):
            st.dataframe(df.head())
            
        # Preprocessing
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].mean())
                
        # Encoding
        le = LabelEncoder()
        obj_cols = df.select_dtypes(include='object').columns
        for col in obj_cols:
            df[col] = le.fit_transform(df[col])
            
        target = 'Grade'
        X = df.drop(target, axis=1)
        y = df[target]
        
        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        st.sidebar.subheader("Model Configuration")
        model_choice = st.sidebar.selectbox("Select Classification Model", 
                                          ["Logistic Regression", "K-Nearest Neighbors", "Decision Tree", "Random Forest", "Naive Bayes"])
        
        model = None
        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_choice == "K-Nearest Neighbors":
            k = st.sidebar.slider("Neighbors (K)", 1, 15, 5)
            model = KNeighborsClassifier(n_neighbors=k)
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42)
        elif model_choice == "Random Forest":
            est = st.sidebar.slider("Number of Estimators", 10, 200, 100)
            model = RandomForestClassifier(n_estimators=est, random_state=42)
        elif model_choice == "Naive Bayes":
            model = GaussianNB()
            
        if st.button("Train Classification Model"):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            st.metric("Accuracy Score", f"{acc:.4f}")
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_ylabel("Actual")
            ax.set_xlabel("Predicted")
            st.pyplot(fig)

    except FileNotFoundError:
        st.error(f"File not found at {CLASSIFICATION_DATA_PATH}. Please check folder structure.")


# --- UNSUPERVISED LEARNING ---
elif app_mode == "Unsupervised Learning (Clustering)":
    st.header("🌍 Country Clustering Analysis")
    
    try:
        df = pd.read_csv(CLUSTERING_DATA_PATH)
        st.success("Dataset Loaded Successfully!")
        
        if st.checkbox("Show Raw Data"):
            st.dataframe(df.head())
            
        if 'country' in df.columns:
            country_names = df['country']
            X_raw = df.drop('country', axis=1)
        else:
            X_raw = df
            
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        
        st.sidebar.subheader("Clustering Configuration")
        algo = st.sidebar.selectbox("Select Algorithm", ["K-Means", "DBSCAN", "Hierarchical Clustering"])
        
        labels = None
        
        if algo == "K-Means":
            k = st.sidebar.slider("Number of Clusters (K)", 2, 10, 3)
            model = KMeans(n_clusters=k, random_state=42)
            labels = model.fit_predict(X_scaled)
            
        elif algo == "DBSCAN":
            eps = st.sidebar.slider("Epsilon (eps)", 0.1, 5.0, 1.5)
            min_samples = st.sidebar.slider("Min Samples", 2, 10, 3)
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X_scaled)
            
        elif algo == "Hierarchical Clustering":
            k = st.sidebar.slider("Number of Clusters", 2, 10, 3)
            model = AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(X_scaled)
            
        st.subheader("Cluster Visualization")
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='viridis', s=100, ax=ax)
        ax.set_title(f"{algo} Clustering Results (PCA Reduced)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        st.pyplot(fig)
        
        if len(set(labels)) > 1:
            score = silhouette_score(X_scaled, labels)
            st.metric("Silhouette Score", f"{score:.4f}")
        else:
            st.warning("Only one cluster found (or all noise). Silhouette score cannot be calculated.")

    except FileNotFoundError:
        st.error(f"File not found at {CLUSTERING_DATA_PATH}. Please check folder structure.")
