import streamlit as st
import pandas as pd
import joblib

# Load models
clf_model = joblib.load("models/rf_model.pkl")
iso_model = joblib.load("models/iso_model.pkl")
selected_features = joblib.load("models/features.pkl")

st.title("🔐 Hybrid Intrusion Detection System")

st.write("Detects known and unknown network attacks")

# Input fields dynamically create
input_data = {}

st.sidebar.header("Enter Feature Values")

for feature in selected_features:
    input_data[feature] = st.sidebar.number_input(feature, value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Prediction button
if st.button("Predict"):

    clf_pred = clf_model.predict(input_df)[0]
    iso_pred = iso_model.predict(input_df)[0]

    if clf_pred == 1:
        result = "🚨 Known Attack"
    elif iso_pred == -1:
        result = "⚠️ Unknown Attack"
    else:
        result = "✅ Normal Traffic"

    st.subheader(f"Result: {result}")