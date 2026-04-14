import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time

# =========================
# LOAD MODELS
# =========================
clf_model = joblib.load("models/rf_model.pkl")
iso_model = joblib.load("models/iso_model.pkl")
scaler = joblib.load("models/scaler.pkl")
selected_features = joblib.load("models/features.pkl")  # loads your actual 12 features

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Intrusion Detection System", layout="wide")
st.title("🚨 Intrusion Detection System Dashboard")

# =========================
# SIDEBAR
# =========================
option = st.sidebar.selectbox(
    "Choose Input Method",
    ["Manual Input", "Upload CSV", "Real-Time Simulation"]
)

# =========================
# PREDICT FUNCTION (HYBRID)
# =========================
def predict(data):
    df = data.copy()

    # ensure all 12 features present
    for col in selected_features:
        if col not in df.columns:
            df[col] = 0

    df = df[selected_features]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    df = df.clip(-1e9, 1e9)

    scaled = scaler.transform(df)

    clf_probas = clf_model.predict_proba(scaled)[:, 1]
    iso_scores = iso_model.decision_function(scaled)

    verdicts = []
    for clf_p, iso_s in zip(clf_probas, iso_scores):
        if clf_p > 0.5:
            verdicts.append("🚨 Known Attack")
        elif iso_s < -0.3:
            verdicts.append("⚠️ Unknown Attack")
        else:
            verdicts.append("✅ Normal")

    return verdicts, clf_probas, iso_scores

# =========================
# MANUAL INPUT
# =========================
if option == "Manual Input":
    st.subheader("✍️ Enter Features Manually")

    input_data = {}
    cols = st.columns(3)

    for i, feature in enumerate(selected_features):
        with cols[i % 3]:
            input_data[feature] = st.number_input(feature, value=0.0)

    if st.button("Predict"):
        df = pd.DataFrame([input_data])
        verdicts, probas, scores = predict(df)

        v = verdicts[0]
        if "Known" in v:
            st.error(f"Result: {v}")
        elif "Unknown" in v:
            st.warning(f"Result: {v}")
        else:
            st.success(f"Result: {v}")

        c1, c2 = st.columns(2)
        c1.metric("Classifier Confidence", f"{probas[0]:.1%}")
        c2.metric("Anomaly Score", f"{scores[0]:.3f}")

# =========================
# CSV UPLOAD
# =========================
elif option == "Upload CSV":
    st.subheader("📂 Upload CSV File")

    file = st.file_uploader("Upload your CSV", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()

        st.write("Preview:")
        st.dataframe(df.head())

        if st.button("Run Prediction"):
            verdicts, probas, scores = predict(df)

            df["Verdict"]   = verdicts
            df["CLF_Prob"]  = probas.round(3)
            df["ISO_Score"] = scores.round(3)

            st.success("Prediction complete!")
            st.dataframe(df[selected_features[:4] + ["Verdict", "CLF_Prob", "ISO_Score"]].head(50))

            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results", csv_out, "results.csv", "text/csv")

            # --- charts ---
            st.subheader("📈 Visualization")

            counts = pd.Series(verdicts).value_counts()
            fig, ax = plt.subplots()
            colors = ["#ef4444" if "Known" in l else "#f59e0b" if "Unknown" in l else "#22c55e"
                      for l in counts.index]
            counts.plot(kind="bar", ax=ax, color=colors)
            ax.set_title("Verdict Distribution")
            ax.set_ylabel("Count")
            plt.xticks(rotation=15)
            st.pyplot(fig)

            fig2, ax2 = plt.subplots()
            ax2.hist(probas, bins=30, color="#7b2fff", alpha=0.8)
            ax2.axvline(0.5, color="red", linestyle="--", label="Threshold (0.5)")
            ax2.set_title("Classifier Confidence Distribution")
            ax2.set_xlabel("Attack Probability")
            ax2.set_ylabel("Count")
            ax2.legend()
            st.pyplot(fig2)

# =========================
# REAL-TIME SIMULATION
# =========================
elif option == "Real-Time Simulation":
    st.subheader("⚡ Real-Time Simulation")
    st.caption("Generates random flow data to simulate live traffic predictions.")

    if st.button("Simulate 10 Live Flows"):
        live_data = pd.DataFrame(
            np.random.rand(10, len(selected_features)) * 1000,
            columns=selected_features
        )

        verdicts, probas, scores = predict(live_data)

        live_data["Verdict"]   = verdicts
        live_data["CLF_Prob"]  = probas.round(3)
        live_data["ISO_Score"] = scores.round(3)

        st.dataframe(live_data)

        counts = pd.Series(verdicts).value_counts()

        colors = ["#ef4444" if "Known" in l else "#f59e0b" if "Unknown" in l else "#22c55e"
                  for l in counts.index]
    
        plt.xticks(rotation=15)
 

        st.line_chart(live_data[selected_features[:6]])
