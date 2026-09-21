from pathlib import Path

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide",
    page_icon="🛡️"
)


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
}

div[data-testid="stMetricValue"] {
    color: #ff6b6b;
    font-size: 30px;
    font-weight: 700;
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
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #ff4b4b, #ff6b6b) !important;
    color: white !important;
}

.stButton > button {
    background: linear-gradient(90deg, #ff4b4b, #ff6b6b);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 28px;
    font-weight: 600;
}

[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.05);
    border: 2px dashed rgba(255, 255, 255, 0.2);
    border-radius: 16px;
    padding: 10px;
}

.stDataFrame {
    border-radius: 14px;
    overflow: hidden;
}

hr {
    border-color: rgba(255,255,255,0.1);
}
</style>
""", unsafe_allow_html=True)


st.markdown(
    '<div class="header-title">💳 Credit Card Fraud Detection</div>',
    unsafe_allow_html=True
)

st.caption(
    "✨ Real-time fraud prediction powered by XGBoost | "
    "Trained on 284,807 real anonymized transactions"
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_fraud_model.pkl"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "models" / "feature_columns.pkl"

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)


def preprocess_raw(data):
    data = data.copy()

    scaler = StandardScaler()

    if "Amount" in data.columns:
        data["Amount_scaled"] = scaler.fit_transform(
            data["Amount"].values.reshape(-1, 1)
        )
        data.drop("Amount", axis=1, inplace=True)

    if "Time" in data.columns:
        data["Time_scaled"] = scaler.fit_transform(
            data["Time"].values.reshape(-1, 1)
        )
        data.drop("Time", axis=1, inplace=True)

    return data


tab1, tab2, tab3 = st.tabs(
    ["📁 Batch Prediction", "🔢 Single Transaction", "ℹ️ About"]
)


# ============================================================
# TAB 1 — BATCH PREDICTION
# ============================================================

with tab1:
    st.subheader("Upload Transaction Data")

    uploaded_file = st.file_uploader(
        "Accepts raw or preprocessed CSV files",
        type=["csv"],
        key="batch_csv_uploader"
    )

    if uploaded_file is not None:

        with st.spinner("🔎 Analyzing transactions..."):

            data = pd.read_csv(uploaded_file)

            if "Amount" in data.columns or "Time" in data.columns:
                data = preprocess_raw(data)

                st.info(
                    "Raw file detected — automatically scaled Amount & Time columns."
                )

            missing_cols = set(feature_columns) - set(data.columns)

            if missing_cols:

                st.error(
                    f"Missing required columns: {missing_cols}"
                )

            else:

                X = data[feature_columns]

                preds = model.predict(X)
                proba = model.predict_proba(X)[:, 1]

                data["Fraud_Prediction"] = preds
                data["Fraud_Probability"] = proba

                fraud_count = int(preds.sum())
                total = len(data)

                fraud_rate = (fraud_count / total) * 100

        if not missing_cols:

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Transactions",
                f"{total:,}"
            )

            col2.metric(
                "Flagged as Fraud",
                f"{fraud_count:,}",
                delta=f"{fraud_rate:.3f}% of total",
                delta_color="inverse"
            )

            col3.metric(
                "Fraud Rate",
                f"{fraud_rate:.3f}%"
            )

            st.markdown("---")

            st.subheader("🎯 Prediction Results")

            st.dataframe(
                data.sort_values(
                    "Fraud_Probability",
                    ascending=False
                ).head(50).style.background_gradient(
                    subset=["Fraud_Probability"],
                    cmap="Reds"
                ),
                use_container_width=True
            )

            csv_out = data.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Full Results CSV",
                csv_out,
                "predictions.csv",
                "text/csv"
            )


# ============================================================
# TAB 2 — SINGLE TRANSACTION
# ============================================================

with tab2:

    st.subheader("Manual Transaction Check")

    st.write(
        "Enter feature values (PCA components V1–V28, plus Amount & Time):"
    )

    input_data = {}

    cols = st.columns(4)

    for i, col_name in enumerate(feature_columns):

        with cols[i % 4]:

            input_data[col_name] = st.number_input(
                col_name,
                value=0.0,
                format="%.5f"
            )

    if st.button(
        "🔍 Predict Now",
        use_container_width=True,
        key="single_transaction_predict"
    ):

        with st.spinner("Running model..."):

            input_df = pd.DataFrame(
                [input_data]
            )[feature_columns]

            pred = model.predict(input_df)[0]

            proba = model.predict_proba(input_df)[0][1]

        if pred == 1:

            st.error(
                f"⚠️ FRAUD DETECTED — Probability: {proba:.4f}"
            )

        else:

            st.success(
                f"✅ Legitimate Transaction — Fraud Probability: {proba:.4f}"
            )

        st.progress(float(min(proba, 1.0)))


# ============================================================
# TAB 3 — ABOUT
# ============================================================

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