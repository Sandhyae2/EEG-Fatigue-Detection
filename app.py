import numpy as np
import joblib
#from features import extract_features

# load model and scaler
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("EEG Fatigue Detection")

uploaded_file = st.file_uploader("Upload EEG file")

if st.button("Predict"):

    if uploaded_file is not None:

        # STEP 1: extract features only
        features = extract_features(uploaded_file)

        # STEP 2: reshape + scale
        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)

        # STEP 3: predict
        prediction = model.predict(features)

        # STEP 4: output
        if prediction[0] == 1:
            st.error("Fatigue detected")
        else:
            st.success("Normal state")
