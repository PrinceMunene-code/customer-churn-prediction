import os
import streamlit as st
import pandas as pd
import joblib

# MUST be first Streamlit call
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_lightgbm_pipeline.pkl")

pipeline = joblib.load(MODEL_PATH)
st.title("📉 Customer Churn Prediction")

st.write(
    "Predict the probability that a customer will churn and identify high-risk customers for retention."
)
st.header("Customer Information")

with st.form("churn_form"):
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)

    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)

    total_charges = st.number_input("Total Charges", min_value=0.0, value=800.0)

    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

    submit = st.form_submit_button("Predict Churn Risk")

if submit:
    input_df = pd.DataFrame(
        [
            {
                "tenure": tenure,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
                "Contract": contract,
                "PaymentMethod": payment_method,
            }
        ]
    )

    churn_proba = pipeline.predict_proba(input_df)[0, 1]

    st.subheader("Prediction Result")
    st.metric("Churn Probability", f"{churn_proba:.2%}")

    if churn_proba >= 0.35:
        st.error("⚠️ High Churn Risk — Retention Action Recommended")
    else:
        st.success("✅ Low Churn Risk")

st.caption("Model: LightGBM | Threshold: 0.35 | Purpose: Customer Retention Targeting")
