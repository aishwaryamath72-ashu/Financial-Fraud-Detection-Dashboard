import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Financial Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Financial Fraud Detection Dashboard")

# -------------------------------
# Load Model (SAFE - NO TRAINING)
# -------------------------------
model_path = "fraud_model.pkl"

if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
else:
    st.error("❌ fraud_model.pkl not found in repository")
    st.stop()

# -------------------------------
# Load Files
# -------------------------------
df = pd.read_csv("credit_card_fraud_dataset.csv")
feature_df = pd.read_csv("feature_importance.csv")

cm = np.load("confusion_matrix.npy")

with open("metrics.txt", "r") as f:
    metrics = f.readlines()

accuracy = metrics[0].strip()
precision = metrics[1].strip()
recall = metrics[2].strip()
f1 = metrics[3].strip()

# Convert date
df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], dayfirst=True)

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("Fraud Detection")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Dataset", "Prediction"]
)

# ===============================
# DASHBOARD PAGE
# ===============================
if page == "Dashboard":

    st.subheader("📊 Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", accuracy + "%")
    c2.metric("Precision", precision + "%")
    c3.metric("Recall", recall + "%")
    c4.metric("F1 Score", f1 + "%")

    st.markdown("---")

    # Confusion Matrix
    st.subheader("📌 Confusion Matrix")

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Safe", "Fraud"],
                yticklabels=["Safe", "Fraud"], ax=ax)

    st.pyplot(fig)

    # Feature Importance
    st.subheader("⭐ Feature Importance")

    feature_df = feature_df.sort_values(by="Importance", ascending=False)
    st.bar_chart(feature_df.set_index("Feature"))

    # Fraud Distribution
    st.subheader("📊 Fraud Distribution")

    fraud_counts = df["IsFraud"].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(fraud_counts)

    with col2:
        fig, ax = plt.subplots()
        ax.pie(fraud_counts, labels=["Safe", "Fraud"], autopct="%1.1f%%")
        st.pyplot(fig)

# ===============================
# DATASET PAGE
# ===============================
elif page == "Dataset":
    st.subheader("📂 Dataset Preview")
    st.dataframe(df.head(100))

# ===============================
# PREDICTION PAGE
# ===============================
elif page == "Prediction":

    st.subheader("🤖 Fraud Prediction System")

    transaction_id = st.number_input("Transaction ID", value=1)
    amount = st.number_input("Amount", value=100.0)
    merchant_id = st.number_input("Merchant ID", value=101)

    transaction_type = st.selectbox(
        "Transaction Type",
        df["TransactionType"].unique()
    )

    location = st.selectbox(
        "Location",
        df["Location"].unique()
    )

    if st.button("Predict"):

        # Encoding
        transaction_type_code = pd.Categorical(
            [transaction_type],
            categories=df["TransactionType"].unique()
        ).codes[0]

        location_code = pd.Categorical(
            [location],
            categories=df["Location"].unique()
        ).codes[0]

        input_data = pd.DataFrame({
            "TransactionID": [transaction_id],
            "Amount": [amount],
            "MerchantID": [merchant_id],
            "TransactionType": [transaction_type_code],
            "Location": [location_code]
        })

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("🚨 Fraudulent Transaction Detected")
        else:
            st.success("✅ Safe Transaction")