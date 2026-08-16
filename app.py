import streamlit as st

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction System")

st.markdown("""
## Welcome!

This project predicts whether a telecom customer is likely to churn using Machine Learning.

### Features
- 📈 Exploratory Data Analysis (EDA)
- 🤖 Machine Learning Prediction
- 🌲 Random Forest Classifier
- 📊 Model Performance Dashboard
- 📉 ROC Curve Analysis
- 💡 Business Insights for Customer Retention

---
""")

col1, col2, col3 = st.columns(3)

col1.metric("Dataset", "7,043 Customers")
col2.metric("Best Model", "Random Forest")
col3.metric("ROC-AUC", "0.841")

st.success("Application Loaded Successfully ✅")