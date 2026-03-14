import streamlit as st
import joblib
import numpy as np

# Load model & scaler
model = joblib.load("fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💳 Credit Card Fraud Detection App")
st.write("Enter transaction details to check if it may be fraudulent.")

# User input fields
amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
v1 = st.number_input("V1")
v2 = st.number_input("V2")
v3 = st.number_input("V3")
v4 = st.number_input("V4")
v5 = st.number_input("V5")

# (In real dataset there are 28 V features; we’ll limit inputs for simplicity)
features = np.array([[v1, v2, v3, v4, v5] + [0]*(24) + [amount]])
features[:, -1] = scaler.transform(features[:, -1].reshape(-1, 1)).flatten()

if st.button("Predict"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")
