import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_title="Financial Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)
# Load Model
import os

if not os.path.exists("fraud_model.pkl"):
    import model

with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load Model Accuracy
with open("accuracy.txt", "r") as f:
    accuracy = f.read()
    cm = np.load("confusion_matrix.npy")
# Load Dataset
df = pd.read_csv("credit_card_fraud_dataset.csv")
df["TransactionDate"] = pd.to_datetime(
    df["TransactionDate"],
    dayfirst=True
)
feature_df = pd.read_csv("feature_importance.csv")
# Load Model Metrics
with open("metrics.txt", "r") as f:
    metrics = f.readlines()

accuracy = metrics[0].strip()
precision = metrics[1].strip()
recall = metrics[2].strip()
f1 = metrics[3].strip()
# Sidebar
st.sidebar.title(" Fraud Detection")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        " Dashboard",
        " Dataset",
        " Prediction"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Amdox Internship Project")

# Dashboard
if page == " Dashboard":

    st.title("💳 Financial Fraud Detection Dashboard")
    st.write("Professional Machine Learning Dashboard")
    st.subheader("📈 Machine Learning Performance")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Accuracy", accuracy + "%")
    m2.metric("Precision", precision + "%")
    m3.metric("Recall", recall + "%")
    m4.metric("F1 Score", f1 + "%")

    st.markdown("---")

    # -------------------------------
    # Dashboard Filters
    # -------------------------------
    st.sidebar.header("📌 Filters")

    selected_location = st.sidebar.selectbox(
        "Select Location",
        ["All"] + sorted(df["Location"].unique().tolist())
    )

    selected_type = st.sidebar.selectbox(
        "Transaction Type",
        ["All"] + sorted(df["TransactionType"].unique().tolist())
    )

    filtered_df = df.copy()

    if selected_location != "All":
        filtered_df = filtered_df[
            filtered_df["Location"] == selected_location
        ]

    if selected_type != "All":
        filtered_df = filtered_df[
            filtered_df["TransactionType"] == selected_type
        ]

    total = len(filtered_df)
    fraud = filtered_df["IsFraud"].sum()
    safe = total - fraud

    if total > 0:
        rate = round((fraud / total) * 100, 2)
    else:
        rate = 0

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Transactions", total)
    col2.metric("Fraud Cases", fraud)
    col3.metric("Safe Transactions", safe)
    col4.metric("Fraud Rate", f"{rate}%")
    st.success(f"✅ Machine Learning Model Accuracy : {accuracy}%")
    st.markdown("---")

    st.subheader("📊 Confusion Matrix")

    fig, ax = plt.subplots(figsize=(5,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Safe", "Fraud"],
        yticklabels=["Safe", "Fraud"],
        ax=ax
)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)
    st.subheader("⭐ Feature Importance")

    feature_df = feature_df.sort_values(
        by="Importance",
        ascending=False
)

    st.bar_chart(
        feature_df.set_index("Feature")
)

    st.markdown("---")

    # Fraud Distribution
    st.subheader(" Fraud Distribution")

    fraud_counts = filtered_df["IsFraud"].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(fraud_counts)

    with col2:
        fig, ax = plt.subplots(figsize=(5,5))

        ax.pie(
            fraud_counts,
            labels=["Safe", "Fraud"],
            autopct="%1.1f%%",
            startangle=90
        )

        ax.axis("equal")
        st.pyplot(fig)

    st.markdown("---")

    # Amount Analysis
    st.subheader("💰 Transaction Amount Analysis")

    fig = px.histogram(
        filtered_df,
        x="Amount",
        nbins=30,
        title="Transaction Amount Distribution"
)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Transaction Types
    st.subheader("💳 Transaction Types")

    transaction_chart = (
    filtered_df["TransactionType"]
    .value_counts()
    .reset_index()
)

    transaction_chart.columns = ["Transaction Type", "Count"]

    fig = px.bar(
        transaction_chart,
        x="Transaction Type",
        y="Count",
        color="Transaction Type",
        title="Transaction Type Distribution"
)

    st.plotly_chart(fig, use_container_width=True)

    # Location Analysis
    st.subheader("🌍 Top Transaction Locations")

    location_chart = (
    filtered_df["Location"]
    .value_counts()
    .head(10)
    .reset_index()
)

    location_chart.columns = ["Location", "Transactions"]

    fig = px.bar(
        location_chart,
        x="Location",
        y="Transactions",
        color="Transactions",
        title="Top 10 Locations"
)

    st.plotly_chart(fig, use_container_width=True)

    # Search Transactions
    st.subheader("🔍 Search Transactions")

    search = st.text_input("Enter Merchant ID or Location")

    if search:
        search_df = filtered_df[
            filtered_df.astype(str).apply(
                lambda row: row.str.contains(search, case=False).any(),
                axis=1
            )
        ]
        st.dataframe(search_df)
    else:
        st.dataframe(filtered_df.head(20))

    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.markdown("---")

    st.subheader("📈 Fraud Trend Analysis")

    trend_df = (
        filtered_df.groupby("TransactionDate")["IsFraud"]
        .sum()
        .reset_index()
)

    fig = px.line(
        trend_df,
        x="TransactionDate",
        y="IsFraud",
        title="Fraud Cases Over Time",
        markers=True
)

    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📅 Monthly Fraud Analysis")

    monthly = df.copy()

    monthly["Month"] = monthly["TransactionDate"].dt.strftime("%B")

    monthly_chart = (
        monthly.groupby("Month")["IsFraud"]
        .sum()
        .reset_index()
)

    fig = px.bar(
        monthly_chart,
        x="Month",
        y="IsFraud",
        color="IsFraud",
        title="Monthly Fraud Cases"
)

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    st.subheader("🏪 Top Fraud Merchants")

    merchant = (
        df.groupby("MerchantID")["IsFraud"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
)

    fig = px.bar(
        merchant,
        x="MerchantID",
        y="IsFraud",
        color="IsFraud",
        title="Top 10 Fraud Merchants"
)

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    st.subheader("📥 Download Filtered Data")

    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Filtered Data",
        data=csv,
        file_name="fraud_detection_report.csv",
        mime="text/csv"
    )
# -------------------------------
# Dataset Page
# -------------------------------
elif page == " Dataset":

    st.title("📊 Dataset Preview")
    st.dataframe(df)

# -------------------------------
# Prediction Page
# -------------------------------
elif page == " Prediction":

    st.title("🤖 Fraud Prediction System")

    st.write("Enter transaction details below.")

    transaction_id = st.number_input("Transaction ID", value=1)

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0
    )

    merchant_id = st.number_input(
        "Merchant ID",
        value=100
    )

    transaction_type = st.selectbox(
        "Transaction Type",
        sorted(df["TransactionType"].unique())
    )

    location = st.selectbox(
        "Location",
        sorted(df["Location"].unique())
    )

    if st.button("Predict Transaction"):

        # Encode values
        transaction_type_code = (
            pd.Categorical(
                [transaction_type],
                categories=df["TransactionType"].unique()
            ).codes[0]
        )

        location_code = (
            pd.Categorical(
                [location],
                categories=df["Location"].unique()
            ).codes[0]
        )

        input_data = pd.DataFrame({
    "TransactionID": [transaction_id],
    "Amount": [amount],
    "MerchantID": [merchant_id],
    "TransactionType": [transaction_type_code],
    "Location": [location_code]
})

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error(" Fraudulent Transaction Detected")
        else:
            st.success(" Safe Transaction")