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
