import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

st.title("📈 Model Performance")

st.markdown("### Random Forest Classifier Performance")

# Metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "80.20%")
col2.metric("Precision", "66.90%")
col3.metric("Recall", "50.30%")
col4.metric("F1 Score", "57.40%")

st.divider()

st.subheader("Model Comparison")

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy":[0.773,0.749,0.802],
    "Precision":[0.586,0.531,0.669],
    "Recall":[0.492,0.487,0.503],
    "F1 Score":[0.535,0.508,0.574]
})

st.dataframe(results, use_container_width=True)

st.divider()

st.subheader("ROC-AUC Score")

st.metric("ROC-AUC", "0.841")