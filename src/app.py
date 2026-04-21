import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time

# Load models
clf_model = joblib.load("models/xgb_model.pkl")  
iso_model = joblib.load("models/iso_model.pkl")
selected_features = joblib.load("models/features.pkl")


# Page config
st.set_page_config(page_title="Intrusion Detection System", layout="wide")
st.title("🚨 Intrusion Detection System Dashboard")

# sidebar
option = st.sidebar.selectbox(
    "Choose Input Method",
    ["Manual Input", "Upload CSV", "Real-Time Simulation"]
)

# Prediction function
def predict(data):
    df = data.copy()

    for col in selected_features:
        if col not in df.columns:
            df[col] = 0

    df = df[selected_features]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    df = df.clip(-1e9, 1e9)

    clf_probas = clf_model.predict_proba(df)[:, 1]
    iso_scores = iso_model.decision_function(df)

    verdicts = []
    for clf_p, iso_s in zip(clf_probas, iso_scores):

        if clf_p > 0.85:
            verdicts.append("🚨 Known Attack")

        elif clf_p > 0.5 and iso_s < -0.4:
            verdicts.append("⚠️ Unknown Attack")

        elif iso_s < -0.5:
            verdicts.append("⚠️ Suspicious")

        else:
            verdicts.append("✅ Normal")

    return verdicts, clf_probas, iso_scores

# Manual input
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

# CSV upload
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

            # charts
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
            ax2.axvline(0.85, color="red", linestyle="--", label="Threshold (0.85)")  # ✅ updated
            ax2.set_title("Classifier Confidence Distribution")
            ax2.set_xlabel("Attack Probability")
            ax2.set_ylabel("Count")
            ax2.legend()
            st.pyplot(fig2)

# real time simulation
elif option == "Real-Time Simulation":
    st.subheader("⚡ Real-Time Live IDS Dashboard")
    st.caption("Shows real-time predictions from live network traffic")

    import os
    placeholder = st.empty()

    while True:
        if os.path.exists("live_data.csv"):
            df = pd.read_csv("live_data.csv")

            if not df.empty:
                df = df.tail(5)

                with placeholder.container():

                    st.markdown("### 📊 Live Traffic Data")
                    st.dataframe(df)

                    latest = df.iloc[-1]

                    if "ATTACK" in latest["Label"]:
                        st.error(f"🚨 ALERT: {latest['Label']}")
                    elif "UNKNOWN" in latest["Label"]:
                        st.warning(f"⚠️ Suspicious Activity Detected")
                    else:
                        st.success("✅ System Normal")

                    st.markdown("### 📉 Confidence Over Time")
                    st.line_chart(df[["CLF_Prob", "ISO_Score"]])

        else:
            st.warning("Waiting for live data... Start IDS script first!")

        time.sleep(2)