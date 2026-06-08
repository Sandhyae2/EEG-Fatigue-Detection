import streamlit as st
import numpy as np
import joblib
from features import extract_features

# Load model and scaler
model = joblib.load("svm_fatigue_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("EEG Fatigue Detection System")

uploaded_file = st.file_uploader(
    "Upload EEG file (.cnt or supported format)"
)

if uploaded_file is not None:

    st.success("File uploaded successfully")

    if st.button("Predict"):

        # Save uploaded file
        with open("temp_eeg_file", "wb") as f:
            f.write(uploaded_file.read())

        # Extract epoch-wise features
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

        # st.write("Feature shape:", features.shape)
        # st.write("Scaler expects:", scaler.n_features_in_)

        # # Scale all epochs
        features_scaled = scaler.transform(features)

        # Predict each epoch
        epoch_predictions = model.predict(features_scaled)

        # st.write("Model classes:", model.classes_)

        probs = model.predict_proba(features_scaled)

        st.write(
            "Mean fatigue probability:",
             round(float(np.mean(probs[:,1])), 3)
        )

        # Fatigue percentage
        fatigue_percent = np.mean(epoch_predictions) * 100

        st.write("Fatigue epochs (%):", round(fatigue_percent, 2))

        # Average decision score across epochs
        # decision_scores = model.decision_function(features_scaled)

        # st.write(
        #     "Mean decision score:",
        #     round(float(np.mean(decision_scores)), 2)
        # )

        # Final decision
        if fatigue_percent >= 50:
            st.error("⚠ Fatigue Detected")
        else:
            st.success("✅ Normal State")

        # Optional debug
        # st.write(
        #     "First 20 epoch predictions:",
        #     epoch_predictions[:20]
        # )
