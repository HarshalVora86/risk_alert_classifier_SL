"""
Streamlit application for Risk Alert Classification.

Run:
    streamlit run app.py
"""
import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Risk Alert Classifier",
    page_icon="🚨",
    layout="centered",
)


@st.cache_resource
def load_model():
    with open("model/risk_alert_pipeline.pkl", "rb") as file:
        return pickle.load(file)


model = load_model()

st.title("🚨 Risk Alert Classifier")

st.markdown(
    "Enter the customer details below. The saved pipeline applies the same "
    "preprocessing used during training and returns the risk prediction."
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18,
                              max_value=80, value=35, step=1)
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        region = st.selectbox(
            "Region", ["Central", "East", "North", "South", "West"])
        employment_type = st.selectbox(
            "Employment Type",
            ["Retired", "Salaried", "Self-Employed", "Student", "Unemployed"],
        )
        annual_income_inr = st.number_input(
            "Annual Income (INR)", min_value=0.0, max_value=500000.0, value=40000.0, step=1000.0
        )
        credit_score = st.number_input(
            "Credit Score", min_value=300, max_value=900, value=650, step=1
        )
        credit_utilization_ratio = st.slider(
            "Credit Utilization Ratio", min_value=0.0, max_value=1.0, value=0.30, step=0.01
        )
        missed_payments_12m = st.number_input(
            "Missed Payments (12 months)", min_value=0, max_value=12, value=1, step=1
        )
        avg_late_payment_days = st.number_input(
            "Avg Late Payment Days", min_value=0.0, max_value=60.0, value=4.0, step=0.5
        )

    with col2:
        monthly_transaction_count = st.number_input(
            "Monthly Transaction Count", min_value=0, max_value=200, value=40, step=1
        )
        monthly_spend_inr = st.number_input(
            "Monthly Spend (INR)", min_value=0.0, max_value=150000.0, value=20000.0, step=500.0
        )
        cash_advance_count_6m = st.number_input(
            "Cash Advances (6 months)", min_value=0, max_value=10, value=0, step=1
        )
        complaints_last_6m = st.number_input(
            "Complaints (6 months)", min_value=0, max_value=10, value=0, step=1
        )
        failed_login_attempts_3m = st.number_input(
            "Failed Login Attempts (3 months)", min_value=0, max_value=15, value=1, step=1
        )
        account_tenure_months = st.number_input(
            "Account Tenure (months)", min_value=1, max_value=240, value=36, step=1
        )
        debt_balance_inr = st.number_input(
            "Debt Balance (INR)", min_value=0.0, max_value=300000.0, value=30000.0, step=1000.0
        )
        last_txn_days_ago = st.number_input(
            "Days Since Last Transaction", min_value=1, max_value=120, value=20, step=1
        )

    threshold = st.slider(
        "Alert Threshold (lower = catch more risky customers)",
        min_value=0.05, max_value=0.95, value=0.50, step=0.05,
    )

    submitted = st.form_submit_button(
        "🔮 Predict Risk",
        use_container_width=True,
    )

if submitted:
    input_data = pd.DataFrame(
        {
            "gender": [gender],
            "region": [region],
            "employment_type": [employment_type],
            "age": [age],
            "annual_income_inr": [annual_income_inr],
            "credit_score": [credit_score],
            "credit_utilization_ratio": [credit_utilization_ratio],
            "missed_payments_12m": [missed_payments_12m],
            "avg_late_payment_days": [avg_late_payment_days],
            "monthly_transaction_count": [monthly_transaction_count],
            "monthly_spend_inr": [monthly_spend_inr],
            "cash_advance_count_6m": [cash_advance_count_6m],
            "complaints_last_6m": [complaints_last_6m],
            "failed_login_attempts_3m": [failed_login_attempts_3m],
            "account_tenure_months": [account_tenure_months],
            "debt_balance_inr": [debt_balance_inr],
            "last_txn_days_ago": [last_txn_days_ago],
        }
    )

    probability = float(model.predict_proba(input_data)[0, 1])

    st.metric("Probability of High Risk", f"{probability:.1%}")
    if probability >= threshold:
        st.error("🚨 HIGH RISK - send to manual review")
    else:
        st.success("✅ LOW RISK")

    with st.expander("View input data"):
        st.dataframe(input_data, use_container_width=True)

metrics_path = Path("model/metrics.json")
if metrics_path.exists():
    with st.expander("Model performance (test set)"):
        m = json.loads(metrics_path.read_text())
        st.write(
            {k: round(v, 4) if isinstance(v, float) else v
             for k, v in m.items() if k not in ("features", "target")}
        )

st.divider()
st.markdown("### Project Flow")
st.code(
    "User Input → DataFrame → Saved Pipeline → KNN Imputer + StandardScaler "
    "+ OneHotEncoder → Logistic Regression → Risk Probability → Alert"
)
