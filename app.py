import streamlit as st
import numpy as np
import joblib
from features import extract_features

# -----------------------------
# Load model and scaler
# -----------------------------
model = joblib.load("svm_fatigue_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="EEG Fatigue Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 EEG Fatigue Detection System")
st.markdown(
    "Upload an EEG `.cnt` file to assess fatigue level."
)

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload EEG file",
    type=["cnt"]
)

if uploaded_file is not None:

    st.success("✅ File uploaded successfully")

    if st.button("Predict Fatigue"):

        # Save uploaded file
        with open("temp_eeg_file", "wb") as f:
            f.write(uploaded_file.read())

        # -----------------------------
        # Dataset-specific format fix
        # -----------------------------
        file_name = uploaded_file.name.lower()

        if "fatigue" in file_name:
            features = extract_features(
                "temp_eeg_file",
                data_format="int32"
            )
        else:
            features = extract_features(
                "temp_eeg_file",
                data_format="int16"
            )

        # -----------------------------
        # Prediction
        # -----------------------------
        features_scaled = scaler.transform(features)

        epoch_predictions = model.predict(features_scaled)

        probs = model.predict_proba(features_scaled)

        fatigue_probability = (
            np.mean(probs[:, 1]) * 100
        )

        fatigue_percent = (
            np.mean(epoch_predictions) * 100
        )

        # -----------------------------
        # Results
        # -----------------------------
        st.divider()

        st.subheader("📊 Analysis Results")

        st.metric(
            "Fatigue Probability",
            f"{fatigue_probability:.1f}%"
        )

        st.metric(
            "Fatigue Epochs",
            f"{fatigue_percent:.1f}%"
        )

        st.write("### Fatigue Score")

        st.progress(
            min(int(fatigue_percent), 100)
        )

        # -----------------------------
        # Severity Classification
        # -----------------------------

        if fatigue_percent < 30:

            st.success(
                "🟢 Alert State\n\nNo significant fatigue detected."
            )

        elif fatigue_percent < 60:

            st.warning(
                "🟡 Mild Fatigue Detected\n\nConsider taking a short break."
            )

            try:
                with open("mild_alert.mp3", "rb") as audio_file:
                    st.audio(
                        audio_file.read(),
                        format="audio/mp3",
                        autoplay=True
                    )
            except:
                st.warning(
                    "mild_alert.mp3 not found"
                )

        elif fatigue_percent < 80:

            st.warning(
                "🟠 High Fatigue Detected\n\nRest is strongly recommended."
            )

            try:
                with open("mild_alert.mp3", "rb") as audio_file:
                    st.audio(
                        audio_file.read(),
                        format="audio/mp3",
                        autoplay=True
                    )
            except:
                st.warning(
                    "mild_alert.mp3 not found"
                )

        else:

            st.markdown("""
            <style>
            .blink {
                animation: blinker 1s linear infinite;
                color: red;
                font-size: 36px;
                font-weight: bold;
                text-align: center;
            }

            @keyframes blinker {
                50% { opacity: 0; }
            }
            </style>

            <div class="blink">
            ⚠️ CRITICAL FATIGUE DETECTED ⚠️
            </div>
            """, unsafe_allow_html=True)

            st.error(
                "🔴 Immediate rest is recommended."
            )

            try:
                with open("critical_alert.mp3", "rb") as audio_file:
                    st.audio(
                        audio_file.read(),
                        format="audio/mp3",
                        autoplay=True
                    )
            except:
                st.warning(
                    "critical_alert.mp3 not found"
                )

        # -----------------------------
        # Detailed Results
        # -----------------------------
        with st.expander("🔍 Detailed Results"):

            st.write(
                "Mean Fatigue Probability:",
                round(float(fatigue_probability), 2),
                "%"
            )

            st.write(
                "Fatigue Epoch Percentage:",
                round(float(fatigue_percent), 2),
                "%"
            )

            st.write(
                "Total Epochs Analysed:",
                len(epoch_predictions)
            )
