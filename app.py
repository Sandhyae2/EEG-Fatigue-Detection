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
    background-position: right top;
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
