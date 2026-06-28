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
    page_icon="🧠",
    layout="centered"
)

st.markdown(
    """
    <style>
    .stApp {
    background-image: url("data:image/jpeg;base64,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");
    background-size: cover; 
    background-position: center;
    background-attachment: fixed; 
}

# section[data-testid="stFileUploader"] * {
#     color: white !important;
# }


    </style>
    """,
    unsafe_allow_html=True
)
# st.markdown(
#     "<h1 style='color:#00FFFF; text-align:center;'>🧠 EEG Fatigue Detection System</h1>",
#     unsafe_allow_html=True
# )

# # -----------------------------
# # Title
# # -----------------------------
st.title("🧠 EEG Fatigue Detection System")

st.markdown(
    """
    Upload an EEG (.cnt) file and the model will determine
    whether the subject is in a Normal or Fatigue state.
    """
)

# -----------------------------
# Upload EEG File
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload EEG File",
    type=["cnt"]
)

# -----------------------------
# Prediction Section
# -----------------------------
if uploaded_file is not None:

    st.success("✅ File uploaded successfully")

    if st.button("Analyze EEG"):

        # -----------------------------
        # Save uploaded file
        # -----------------------------
        with open("temp_eeg_file", "wb") as f:
            f.write(uploaded_file.read())

        # -----------------------------
        # Dataset-specific fix
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
        # Scale Features
        # -----------------------------
        features_scaled = scaler.transform(features)

        # -----------------------------
        # Predictions
        # -----------------------------
        epoch_predictions = model.predict(
            features_scaled
        )

        probs = model.predict_proba(
            features_scaled
        )

        fatigue_probability = (
            np.mean(probs[:, 1]) * 100
        )

        fatigue_percent = (
            np.mean(epoch_predictions) * 100
        )

        fatigue_epochs = int(
            np.sum(epoch_predictions)
        )

        normal_epochs = int(
            len(epoch_predictions)
            - fatigue_epochs
        )

        # -----------------------------
        # Results Header
        # -----------------------------
        st.divider()

        st.subheader(
            "📊 Analysis Results"
        )

        # -----------------------------
        # Main Result Cards
        # -----------------------------
        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Fatigue Probability",
                f"{fatigue_probability:.2f}%"
            )

        with col2:

            if fatigue_percent < 50:

                st.metric(
                    "Detection Result",
                    "🟢 NORMAL"
                )

            else:

                st.metric(
                    "Detection Result",
                    "🔴 FATIGUE"
                )

        # -----------------------------
        # Alert Section
        # -----------------------------
        st.divider()

        st.subheader("🚨 Alert Status")

        if fatigue_percent < 50:

            st.success(
                "🟢 Subject is in a Normal State"
            )

        else:

            st.markdown(
                """
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
                """,
                unsafe_allow_html=True
            )

            st.error(
                "🔴 Fatigue Detected. Rest is Recommended."
            )

            # -----------------------------
            # Alarm Sound
            # -----------------------------
            try:

                with open(
                    "critical_alert.mp3",
                    "rb"
                ) as audio_file:

                    audio_bytes = (
                        audio_file.read()
                    )

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
        # Fatigue Score
        # -----------------------------
        st.divider()

        st.subheader(
            "📈 Fatigue Score"
        )

        st.progress(
            int(
                min(
                    fatigue_probability,
                    100
                )
            )
        )

        st.write(
            f"Current Fatigue Probability: "
            f"**{fatigue_probability:.2f}%**"
        )

        # -----------------------------
        # Detailed Results
        # -----------------------------
        st.divider()

        with st.expander(
            "🔍 Detailed Analysis"
        ):

            st.write(
                "Fatigue Probability:",
                f"{fatigue_probability:.2f}%"
            )

            st.write(
                "Fatigue Epoch Percentage:",
                f"{fatigue_percent:.2f}%"
            )

            st.write(
                "Total Epochs Analysed:",
                len(epoch_predictions)
            )

            st.write(
                "Predicted Fatigue Epochs:",
                fatigue_epochs
            )

            st.write(
                "Predicted Normal Epochs:",
                normal_epochs
            )
