import streamlit as st
import joblib
import pandas as pd

st.title("🔮 Customer Churn Prediction")

# Load model
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features = joblib.load("models/features.pkl")

st.write("Enter customer details below:")

tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
total = st.slider("Total Charges", 0.0, 9000.0, 1500.0)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

paperless = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

if st.button("Predict Churn"):

    data = {
        "Tenure Months": tenure,
        "Monthly Charges": monthly,
        "Total Charges": total,
        "Contract": contract,
        "Internet Service": internet,
        "Paperless Billing": paperless
    }

    input_df = pd.DataFrame([data])

    # One-hot encode
    input_df = pd.get_dummies(input_df)

    # Match training columns
    input_df = input_df.reindex(columns=features, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")

    st.write(f"Churn Probability: **{probability:.2%}**")