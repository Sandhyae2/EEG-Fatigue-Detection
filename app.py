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
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="EEG Fatigue Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 EEG Fatigue Detection System")

st.markdown(
    """
    Upload an EEG (.cnt) file and the model will determine
    whether the subject is in a normal or fatigue state.
    """
)

# -----------------------------
# Upload EEG File
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload EEG File",
    type=["cnt"]
)

if uploaded_file is not None:

    st.success("✅ File uploaded successfully")

    if st.button("Predict Fatigue"):

        # -----------------------------
        # Save Uploaded File
        # -----------------------------
        with open("temp_eeg_file", "wb") as f:
            f.write(uploaded_file.read())

        # -----------------------------
        # Dataset-Specific CNT Format
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
        # Feature Scaling
        # -----------------------------
        features_scaled = scaler.transform(features)

        # -----------------------------
        # Predictions
        # -----------------------------
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
            f"{fatigue_probability:.2f}%"
        )

        st.metric(
            "Fatigue Epochs",
            f"{fatigue_percent:.2f}%"
        )

        st.write("### Fatigue Score")

        st.progress(
            int(min(fatigue_percent, 100))
        )

        # -----------------------------
        # Final Decision
        # -----------------------------
        if fatigue_percent < 50:

            st.success(
                "🟢 NORMAL STATE\n\nNo fatigue detected."
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
            ⚠️ FATIGUE DETECTED ⚠️
            </div>
            """, unsafe_allow_html=True)

            st.error(
                "🔴 Fatigue Detected\n\nRest is recommended."
            )

            # -----------------------------
            # Alert Sound
            # -----------------------------
            try:

                with open(
                    "critical_alert.mp3",
                    "rb"
                ) as audio_file:

                    audio_bytes = audio_file.read()

                st.audio(
                    audio_bytes,
                    format="audio/mp3",
                    autoplay=True
                )

            except Exception:

                st.warning(
                    "critical_alert.mp3 not found."
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

            st.write(
                "Predicted Fatigue Epochs:",
                int(np.sum(epoch_predictions))
            )

            st.write(
                "Predicted Normal Epochs:",
                int(
                    len(epoch_predictions)
                    - np.sum(epoch_predictions)
                )
            )
