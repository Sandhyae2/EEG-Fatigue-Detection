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
    section[data-testid="stFileUploader"] {
        background-image: url("https://static.vecteezy.com/system/resources/previews/014/731/394/non_2x/brain-analysis-interface-vector.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        padding: 15px;
        border-radius: 10px;
}

section[data-testid="stFileUploader"] * {
    color: white !important;
}


/* Make all text white */
h1, h2, h3, h4, h5, h6,
p, label, span, div {
    color: white !important;
}

    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    "<h1 style='color:#FFFFFF; text-align:center;'>🧠 EEG Fatigue Detection System</h1>",
    unsafe_allow_html=True
)

# # -----------------------------
# # Title
# # -----------------------------
# st.title("🧠 EEG Fatigue Detection System")

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
