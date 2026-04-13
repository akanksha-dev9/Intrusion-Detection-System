import streamlit as st
import pandas as pd
import joblib

# Load models
clf_model = joblib.load("models/rf_model.pkl")
iso_model = joblib.load("models/iso_model.pkl")
scaler = joblib.load("models/scaler.pkl")
selected_features = joblib.load("models/features.pkl")
attack_sample = joblib.load("models/sample_attack.pkl")
normal_sample = joblib.load("models/sample_normal.pkl")

st.set_page_config(page_title="NIDS", layout="centered")

st.title("🔐 Hybrid Intrusion Detection System")
st.write("Detects known and unknown network attacks")

if "use_sample" not in st.session_state:
    st.session_state.use_sample = None

# SAMPLE BUTTONS
st.subheader("⚡ Quick Test")

col1, col2 = st.columns(2)

with col1:
    if st.button("🧪 Test Known Attack"):
        st.session_state.use_sample = attack_sample

with col2:
    if st.button("🧪 Test Normal Traffic"):
        st.session_state.use_sample = normal_sample

# Sidebar input
st.sidebar.header("Enter Feature Values")

input_data = {}

for feature in selected_features:
    input_data[feature] = st.sidebar.number_input(feature, value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Override with sample if selected
if st.session_state.use_sample is not None:
    st.info("Using preloaded sample for quick demo 🚀")
    input_df = pd.DataFrame([st.session_state.use_sample])

input_df = input_df[selected_features]

# Apply scaling
input_scaled = scaler.transform(input_df)

# Prediction
if st.button("Predict") or st.session_state.use_sample is not None:

    # Predictions
    clf_pred = clf_model.predict(input_scaled)[0]
    iso_pred = iso_model.predict(input_scaled)[0]

    # Probabilities
    clf_proba = clf_model.predict_proba(input_scaled)[0][1]
    iso_score = iso_model.decision_function(input_scaled)[0]

    # FINAL RESULT
    st.subheader("🔐 Final Verdict")

    if clf_proba > 0.6:
        st.error("🚨 Known Attack Detected")
    elif iso_pred == -1:
        st.warning("⚠️ Potential Unknown Attack")
    else:
        st.success("✅ Normal Traffic")

    # CONFIDENCE BAR
    st.subheader("📊 Detection Confidence")

    st.progress(int(clf_proba * 100))

    if clf_proba > 0.8:
        st.success(f"High confidence attack ({clf_proba:.1%})")
    elif clf_proba > 0.5:
        st.warning(f"Moderate confidence ({clf_proba:.1%})")
    else:
        st.info(f"Low confidence ({clf_proba:.1%})")

    # ANOMALY ANALYSIS
    st.subheader("🧠 Anomaly Analysis")

    if iso_score < -0.2:
        st.error("Highly unusual traffic detected 🚨")
    elif iso_score < 0:
        st.warning("Slightly unusual behavior ⚠️")
    else:
        st.success("Normal behavior ✅")

    # KEY FEATURES
    st.subheader("📌 Key Indicators Used")

    st.write(selected_features[:5])

    # 🧾 EXPLANATION
    st.markdown("""
    ### 🧾 What does this mean?

    - 🚨 **Known Attack** → Matches known malicious patterns  
    - ⚠️ **Unknown Attack** → Unusual behavior detected  
    - ✅ **Normal Traffic** → No threat detected  
    """)

if st.button("🔄 Reset"):
    st.session_state.use_sample = None