import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #1a1a2e);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .header-row {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 4px;
    }

    .header-row svg {
        flex-shrink: 0;
    }

    .header-title {
        background: linear-gradient(90deg, #ff4b4b, #ff9d6c, #ff4b4b);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 3s linear infinite;
        font-weight: 700;
        font-size: 42px;
        line-height: 1.2;
    }

    @keyframes shine {
        to { background-position: 200% center; }
    }

    .stCaption, p, span, label {
        color: #d0d0e0 !important;
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 16px;
        padding: 18px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeInUp 0.6s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-6px) scale(1.02);
        box-shadow: 0 8px 30px rgba(255, 75, 75, 0.3);
        border-color: rgba(255, 75, 75, 0.5);
    }

    div[data-testid="stMetricValue"] {
        color: #ff6b6b;
        font-size: 30px;
        font-weight: 700;
    }

    div[data-testid="stMetricLabel"] {
        color: #a0a0c0 !important;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255,255,255,0.03);
        padding: 8px;
        border-radius: 14px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px 24px;
        color: #d0d0e0;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 75, 75, 0.15);
        border-color: rgba(255, 75, 75, 0.3);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #ff4b4b, #ff6b6b) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
    }

    .stButton > button {
        background: linear-gradient(90deg, #ff4b4b, #ff6b6b);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(255, 75, 75, 0.5);
    }

    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        padding: 10px;
        transition: border-color 0.3s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #ff6b6b;
    }

    .stDataFrame {
        border-radius: 14px;
        overflow: hidden;
        animation: fadeInUp 0.8s ease;
    }

    .stAlert {
        border-radius: 12px;
        animation: fadeInUp 0.5s ease;
    }

    div[data-testid="stNumberInput"] {
        animation: fadeInUp 0.4s ease;
    }

    hr {
        border-color: rgba(255,255,255,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="header-row">'
    '<svg width="42" height="42" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<rect x="2" y="5" width="20" height="14" rx="2.5" fill="url(#grad1)"/>'
    '<rect x="2" y="8" width="20" height="3" fill="#1a1a2e"/>'
    '<rect x="4" y="15" width="6" height="1.6" rx="0.8" fill="white" opacity="0.85"/>'
    '<defs><linearGradient id="grad1" x1="2" y1="5" x2="22" y2="19" gradientUnits="userSpaceOnUse">'
    '<stop stop-color="#ff4b4b"/><stop offset="1" stop-color="#ff9d6c"/>'
    '</linearGradient></defs>'
    '</svg>'
    '<span class="header-title">Credit Card Fraud Detection</span>'
    '</div>',
    unsafe_allow_html=True
)

st.caption("✨ Real-time fraud prediction powered by XGBoost | Trained on 284,807 real anonymized transactions")

model = joblib.load("../models/xgboost_fraud_model.pkl")
feature_columns = joblib.load("../models/feature_columns.pkl")

def preprocess_raw(data):
    scaler = StandardScaler()
    if "Amount" in data.columns:
        data["Amount_scaled"] = scaler.fit_transform(data["Amount"].values.reshape(-1, 1))
        data.drop("Amount", axis=1, inplace=True)
    if "Time" in data.columns:
        data["Time_scaled"] = scaler.fit_transform(data["Time"].values.reshape(-1, 1))
        data.drop("Time", axis=1, inplace=True)
    return data

tab1, tab2, tab3 = st.tabs(["📁  Batch Prediction", "🔢  Single Transaction", "ℹ️  About"])

with tab1:
    st.subheader("Upload Transaction Data")
    uploaded_file = st.file_uploader("Accepts raw or preprocessed CSV files", type=["csv"])

    missing_cols = set()
    if uploaded_file is not None:
        with st.spinner("🔎 Analyzing transactions..."):
            data = pd.read_csv(uploaded_file)

            if "Amount" in data.columns or "Time" in data.columns:
                data = preprocess_raw(data)
                st.info("Raw file detected — automatically scaled Amount & Time columns.")

            missing_cols = set(feature_columns) - set(data.columns)
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                X = data[feature_columns]
                preds = model.predict(X)
                proba = model.predict_proba(X)[:, 1]

                data["Fraud_Prediction"] = preds
                data["Fraud_Probability"] = proba

                fraud_count = int(preds.sum())
                total = len(data)
                fraud_rate = (fraud_count / total) * 100

        if uploaded_file is not None and not missing_cols:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Transactions", f"{total:,}")
            col2.metric("Flagged as Fraud", f"{fraud_count:,}", delta=f"{fraud_rate:.3f}% of total", delta_color="inverse")
            col3.metric("Fraud Rate", f"{fraud_rate:.3f}%")

            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            st.dataframe(
                data.sort_values("Fraud_Probability", ascending=False).head(50).style.background_gradient(
                    subset=["Fraud_Probability"], cmap="Reds"
                ),
                use_container_width=True
            )

            csv_out = data.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️  Download Full Results CSV", csv_out, "predictions.csv", "text/csv")

with tab2:
    st.subheader("Manual Transaction Check")
    st.write("Enter feature values (PCA components V1–V28, plus Amount & Time):")

    input_data = {}
    cols = st.columns(4)
    for i, col_name in enumerate(feature_columns):
        with cols[i % 4]:
            input_data[col_name] = st.number_input(col_name, value=0.0, format="%.5f")

    if st.button("🔍  Predict Now", use_container_width=True):
        with st.spinner("Running model..."):
            input_df = pd.DataFrame([input_data])[feature_columns]
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]

        if pred == 1:
            st.error(f"⚠️  FRAUD DETECTED — Probability: {proba:.4f}")
            st.progress(min(proba, 1.0))
        else:
            st.success(f"✅  Legitimate Transaction — Fraud Probability: {proba:.4f}")
            st.progress(min(proba, 1.0))

with tab3:
    st.subheader("About This Project")
    st.write("""
    This dashboard uses an **XGBoost classifier** trained on the Kaggle
    Credit Card Fraud Detection dataset (284,807 transactions, 492 confirmed frauds).

    **Pipeline:**
    - Preprocessing & feature scaling (Amount, Time)
    - Class imbalance handled via SMOTE oversampling
    - Isolation Forest & Local Outlier Factor tested as baseline anomaly detectors
    - Final model: XGBoost Classifier, evaluated via ROC-AUC, precision, and recall

    Built as part of an internship AI/ML project.
    """)