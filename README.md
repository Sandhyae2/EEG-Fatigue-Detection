# EEG-Fatigue-Detection

*"Can a machine detect fatigue before a human notices it, through brain activity"🧠?*

This project basically does that. It uses Electroencephalography (EEG) signals and Machine learning to identfy fatigue state through brain activity. 
EEG recordings from normal and fatigue states were preprocessed and transformed into spectral features which were used to train SVM (Support Vector Machine) classifier , capable of distinguishing between normal and fatigue EEG patterns.

# Methods
### 🧠EEG signal preprocessing
   - EEG recordings were loaded from CNT files using MNE (python)
   - Bandpass filter was applied between 1-40 Hz
   - Power line interference were removed using 50 Hz notch filter
   - Resampled EEG recordings to 250 Hz
   - Performed ICA for artifacts reduction
     
### ✂️Epoch segmentation
   - EEG recordings were segmented into 2 second non- overlapping method

### 📈Power spectral density analysis
   - Computed Power Spectral Analysis (PSD) using Multitaper method
   - Analysed frequency components between 1-40 Hz

### 📊Feature extarction
  Extracted  
   - Theta Band Power (4-8 Hz)
   - Alpha Band Power (8-13 Hz)
   - Beta Band Power (13-30 Hz)

### 🤖SVM classifier
   - Combined features from all subjects
   - Generated normal and fatigue state labels
   - Standardized extracted features using StandardScale
   - Trained SVM classifier
   - Recorded the classifications into normal and fatigue state

### 👤Model evaluation using LOSO
   - The model performanance was evaluated using LOSO (Leave One Subject Out Cross Validation)

An interactive streamlit dashboard has been developed to allow users to upload EEG recordings, perform analysis and visualize them.
