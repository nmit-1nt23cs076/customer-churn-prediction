import streamlit as st
import pandas as pd
import joblib

st.title("📂 Batch Customer Churn Prediction")

st.write("Upload a CSV file to predict churn for multiple customers.")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    model = joblib.load("models/churn_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    features = joblib.load("models/features.pkl")

    # Save CustomerID if present
    customer_ids = None
    if "CustomerID" in df.columns:
        customer_ids = df["CustomerID"]
        df = df.drop(columns=["CustomerID"])

    # Remove target column if user uploaded the original IBM dataset
    if "Churn Label" in df.columns:
        df = df.drop(columns=["Churn Label"])

    # One-hot encode
    df = pd.get_dummies(df)

    # Match training columns
    df = df.reindex(columns=features, fill_value=0)

    # Scale
    df_scaled = scaler.transform(df)

    # Predict
    predictions = model.predict(df_scaled)
    probabilities = model.predict_proba(df_scaled)[:, 1]

    results = pd.DataFrame({
        "Prediction": [
            "Will Churn" if p == 1 else "Will Stay"
            for p in predictions
        ],
        "Churn Probability": probabilities
    })

    if customer_ids is not None:
        results.insert(0, "CustomerID", customer_ids)

    st.subheader("Prediction Results")
    st.dataframe(results)

    csv = results.to_csv(index=False)

    st.download_button(
        "📥 Download Predictions",
        csv,
        file_name="customer_predictions.csv",
        mime="text/csv"
    )