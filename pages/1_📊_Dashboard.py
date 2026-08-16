import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# Page configuration
st.set_page_config(
    page_title="Customer Churn Dashboard",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("../data/telco_customer_churn.csv")
    return df


df = load_data()


# Title
st.title("📊 Customer Churn Analysis Dashboard")


# KPI Cards
total_customers = len(df)
churned_customers = df[df["Churn"] == "Yes"].shape[0]
churn_rate = (churned_customers / total_customers) * 100
avg_charges = df["MonthlyCharges"].mean()


col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", total_customers)
col2.metric("Churned Customers", churned_customers)
col3.metric("Churn Rate", f"{churn_rate:.2f}%")
col4.metric("Average Monthly Charges", f"${avg_charges:.2f}")


# Churn Distribution
st.subheader("Churn Distribution")

churn_count = df["Churn"].value_counts()

fig, ax = plt.subplots()
ax.bar(churn_count.index, churn_count.values)

ax.set_xlabel("Churn")
ax.set_ylabel("Customers")

st.pyplot(fig)


# Contract Analysis
st.subheader("Churn by Contract Type")

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"]
)

st.bar_chart(contract_churn)


# Raw Data
st.subheader("Customer Data Preview")

st.dataframe(df.head(20))