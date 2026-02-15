import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score

# --- Page Configuration ---
st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")

# Custom CSS for Metric Cards
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 25px; }
    div.stMetric {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #d1d5db;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper Function: Preprocessing Raw Data
def preprocess_data(df):
    # 1. Fix TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    # 2. Encode categorical variables
    # We drop 'Churn' before encoding if it's there
    features = df.drop(columns=['Churn'], errors='ignore')
    df_encoded = pd.get_dummies(features, drop_first=True)
    
    # 3. Ensure columns match the training set exactly
    try:
        expected_cols = joblib.load('ML_Assignment_2/model/model_columns.pkl') 
        for col in expected_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        # Reorder and filter
        df_encoded = df_encoded[expected_cols]
    except FileNotFoundError:
        st.error("Error: 'ML_Assignment_2/model/model_columns.pkl' not found. Check your GitHub folder.")
        return None
    
    # 4. Scale numeric columns
    try:
        scaler = joblib.load('ML_Assignment_2/model/scaler.pkl')
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        df_encoded[num_cols] = scaler.transform(df_encoded[num_cols])
    except FileNotFoundError:
        st.error("Error: 'ML_Assignment_2/model/scaler.pkl' not found.")
        return None
        
    return df_encoded

# --- UI Header ---
st.title("📊 Customer Churn Analysis & Prediction")
st.markdown("This app demonstrates 6 ML models using **Actual Raw Data**.")

# --- Sidebar ---
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Raw Test CSV", type="csv")

st.sidebar.header("2. Model Selection")
model_options = {
    "Logistic Regression": "ML_Assignment_2/model/logistic_regression.pkl",
    "Decision Tree": "ML_Assignment_2/model/decision_tree.pkl",
    "kNN": "ML_Assignment_2/model/knn.pkl",
    "Naive Bayes": "ML_Assignment_2/model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "ML_Assignment_2/model/random_forest_(ensemble).pkl",
    "XGBoost (Ensemble)": "ML_Assignment_2/model/xgboost_(ensemble).pkl"
}
selected_model_name = st.sidebar.selectbox("Select a model:", list(model_options.keys()))

if uploaded_file is not None:
    # Load raw data
    df_raw = pd.read_csv(uploaded_file)
    
    if 'Churn' in df_raw.columns:
        # Save actual values for evaluation
        y_test = df_raw['Churn'].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
        
        # Preprocess the raw features
        X_processed = preprocess_data(df_raw)
        
        if X_processed is not None:
            try:
                # Load and Predict
                model = joblib.load(model_options[selected_model_name])
                y_pred = model.predict(X_processed)
                
                # --- Step 6c: Metrics ---
                st.subheader(f"📈 Results: {selected_model_name}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
                c2.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
                c3.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
                c4.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")

                st.divider()

                # --- Step 6d: Visuals ---
                col_left, col_right = st.columns([1, 1])

                with col_left:
                    st.write("**Confusion Matrix**")
                    fig, ax = plt.subplots(figsize=(5, 4))
                    cm = confusion_matrix(y_test, y_pred)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    st.pyplot(fig)

                with col_right:
                    st.write("**Classification Report**")
                    report_dict = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report_dict).transpose()
                    st.dataframe(report_df.style.format(precision=3), use_container_width=True)
            
            except Exception as e:
                st.error(f"Prediction Error: {e}")
    else:
        st.error("The CSV must contain a 'Churn' column for evaluation metrics to work.")
else:
    st.info("👋 Welcome! Please upload your 'telco_raw_test_sample.csv' to see the model in action.")

