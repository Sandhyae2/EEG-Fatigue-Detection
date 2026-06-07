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

        # STEP 2: Extract features
        features = extract_features("temp_eeg_file")

        # 🔍 DEBUG 1 (ADD HERE)
        st.write("Raw feature shape:", np.array(features).shape)

       
        # STEP 3: Convert to numpy array
        features = np.array(features)

        st.write("Raw feature shape:", features.shape)

# Check scaler requirements
        st.write("Scaler expects:", scaler.n_features_in_)

# Current reshape
        features = features.reshape(1, -1)

        st.write("After reshape:", features.shape)

# STEP 4: Scale features
        features = scaler.transform(features)
        # STEP 5: Predict
        prediction = model.predict(features)

        # 🔍 DEBUG 3 (ADD HERE - VERY IMPORTANT)
        st.write("Model prediction:", prediction)

        # OPTIONAL DEBUG (HIGHLY RECOMMENDED)
        st.write("Decision score:", model.decision_function(features))

        # STEP 6: Output result
        if prediction[0] == 1:
            st.error("⚠ Fatigue Detected")
        else:
            st.success("✅ Normal State")
