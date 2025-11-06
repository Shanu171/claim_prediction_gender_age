# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------
# Load trained model
# -------------------------
MODEL_PATH = "models/best_lgb.pkl"  # update if needed
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Claim Amount Predictor", page_icon="💰", layout="centered")
st.title("💰 Insurance Claim Amount Predictor")

st.markdown("Enter the details below to predict the **expected claim amount**:")

# -------------------------
# User Inputs
# -------------------------
gender = st.selectbox("Select Gender", ["Male", "Female", "Other"])
age = st.number_input("Enter Age", min_value=0, max_value=120, value=30, step=1)

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Claim Amount"):
    input_df = pd.DataFrame({
        "Claimant Gender": [gender],
        "Claimant Age": [age]
    })

    try:
        pred_log = model.predict(input_df)
        predicted_claim = np.expm1(pred_log)[0]
        st.success(f"💵 Predicted Claim Amount: **₹{predicted_claim:,.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
