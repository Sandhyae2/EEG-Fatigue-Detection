import streamlit as st
import numpy as np
import joblib
from features import extract_features

# Load model and scaler
model = joblib.load("svm_fatigue_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("EEG Fatigue Detection System")

uploaded_file = st.file_uploader("Upload EEG file (.cnt or supported format)")

if uploaded_file is not None:

    st.success("File uploaded successfully")

    if st.button("Predict"):

        # STEP 1: Save uploaded file temporarily
        with open("temp_eeg_file", "wb") as f:
            f.write(uploaded_file.read())

        # STEP 2: Extract features (IMPORTANT: pass file path)
        features = extract_features("temp_eeg_file")

        # STEP 3: Convert to numpy array
        features = np.array(features).reshape(1, -1)

        # STEP 4: Scale features
        features = scaler.transform(features)

        # STEP 5: Predict
        prediction = model.predict(features)

        # STEP 6: Output result
        if prediction[0] == 1:
            st.error("⚠ Fatigue Detected")
        else:
            st.success("✅ Normal State")
