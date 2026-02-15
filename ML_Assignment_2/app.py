import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score

# --- Page Configuration ---
st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")

# Custom CSS to make metrics look like "Cards"
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 25px; }
    div.css-1r6slb0.e1tzqz2g1 {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Customer Churn Analysis & Prediction")
st.markdown("This app demonstrates 6 different ML models trained on the Telco Customer Churn dataset.")

# --- Step 6a: Sidebar - Dataset Upload ---
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type="csv")
st.sidebar.info("Upload the 'telco_test_sample.csv' generated in your notebook.")

# --- Step 6b: Sidebar - Model Selection ---
st.sidebar.header("2. Model Selection")
model_options = {
    "Logistic Regression": "ML_Assignment_2/model/logistic_regression.pkl",
    "Decision Tree": "ML_Assignment_2/model/decision_tree.pkl",
    "kNN": "ML_Assignment_2/model/knn.pkl",
    "Naive Bayes": "ML_Assignment_2/model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "ML_Assignment_2/model/random_forest_(ensemble).pkl",
    "XGBoost (Ensemble)": "ML_Assignment_2/model/xgboost_(ensemble).pkl"
}
selected_model_name = st.sidebar.selectbox("Select a model to evaluate:", list(model_options.keys()))

if uploaded_file is not None:
    # Load the data
    df_test = pd.read_csv(uploaded_file)
    
    if 'Churn' in df_test.columns:
        X_test = df_test.drop('Churn', axis=1)
        y_test = df_test['Churn']
        
        # Load the selected model
        try:
            model = joblib.load(model_options[selected_model_name])
            y_pred = model.predict(X_test)
            
            # --- Step 6c: Card Visuals for Evaluation Metrics ---
            st.subheader(f"📈 Performance Metrics: {selected_model_name}")
            c1, c2, c3, c4 = st.columns(4)
            
            c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
            c2.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
            c3.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
            c4.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")

            st.write("---")

            # --- Step 6d: Confusion Matrix and Classification Report ---
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
                # Displaying as a clean table
                st.dataframe(report_df.style.format(precision=3), use_container_width=True)

        except FileNotFoundError:
            st.error(f"Model file not found. Ensure '{model_options[selected_model_name]}' exists in your GitHub repository.")
    else:
        st.warning("The uploaded file must contain a 'Churn' column for evaluation.")
else:

    st.info("Please upload a test CSV file from the sidebar to begin.")
