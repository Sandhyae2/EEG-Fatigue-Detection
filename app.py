import mne
import numpy as np
from mne.preprocessing import ICA
import os
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVC
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

for f in os.listdir('/content'):
    print(f)

def extract_features(file_path, data_format='int16'):

    raw = mne.io.read_raw_cnt(
        file_path,
        preload=False,
        data_format=data_format,
        recompute_n_samples=True,
        verbose=False
    )

    raw.load_data()

    filtered = raw.copy().filter(
        l_freq=1,
        h_freq=40,
        method='fir'
    )

    filtered.notch_filter(freqs=50)

    resampled = filtered.copy().resample(250)

    ica = ICA(
        n_components=15,
        random_state=42,
        max_iter='auto'
    )

    ica.fit(resampled)

    cleaned = resampled.copy()

    ica.exclude = []

    ica.apply(cleaned)

    epochs = mne.make_fixed_length_epochs(
        cleaned,
        duration=2,
        overlap=0,
        preload=True
    )

    psd = epochs.compute_psd(
        fmin=1,
        fmax=40,
        method='multitaper'
    )

    psd_data = psd.get_data()

    freqs = psd.freqs

    theta_idx = np.where((freqs >= 4) & (freqs < 8))[0]

    alpha_idx = np.where((freqs >= 8) & (freqs < 13))[0]

    beta_idx = np.where((freqs >= 13) & (freqs < 30))[0]

    theta_power = psd_data[:, :, theta_idx].mean(axis=2)

    alpha_power = psd_data[:, :, alpha_idx].mean(axis=2)

    beta_power = psd_data[:, :, beta_idx].mean(axis=2)

    theta_alpha_ratio = (
        theta_power /
        (alpha_power + 1e-10)
    )

    features = np.concatenate([
        theta_power,
        alpha_power,
        beta_power,
        theta_alpha_ratio
    ], axis=1)

    return features

all_features = []
all_labels = []
groups = []

for subject in range(1, 13):

    print(f"\nProcessing Subject {subject}...")

    try:

        # NORMAL EEG
        normal_file = f"/content/S{subject}_Normal state.cnt"

        normal_features = extract_features(
            normal_file,
            data_format='int16'
        )

        normal_labels = np.zeros(
            normal_features.shape[0]
        )

        all_features.append(normal_features)
        all_labels.append(normal_labels)

        # SUBJECT GROUPS
        groups.extend(
            [subject] * normal_features.shape[0]
        )

        print(f"Normal done: {normal_features.shape}")

        # FATIGUE EEG
        fatigue_file = f"/content/S{subject}_Fatigue state.cnt"

        fatigue_features = extract_features(
            fatigue_file,
            data_format='int32'
        )

        fatigue_labels = np.ones(
            fatigue_features.shape[0]
        )

        all_features.append(fatigue_features)
        all_labels.append(fatigue_labels)

        # SUBJECT GROUPS
        groups.extend(
            [subject] * fatigue_features.shape[0]
        )

        print(f"Fatigue done: {fatigue_features.shape}")

    except Exception as e:

        print(f"Error in Subject {subject}:")
        print(e)

X = np.vstack(all_features)

y = np.concatenate(all_labels)

print(X.shape)
print(y.shape)

groups = np.array(groups)

print(groups.shape)
print(np.unique(groups))

## Feature scaling LOSO 

# LOSO object
logo = LeaveOneGroupOut()

# Metric storage
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Store ALL predictions
all_y_test = []
all_y_pred = []
all_y_prob = []

# LOSO loop
for train_idx, test_idx in logo.split(X, y, groups):

    # Train/Test Split
    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    # Feature Scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    # SVM Model
    svm_model = SVC(
        kernel='rbf',
        C=1,
        gamma='scale',
        probability=True
    )

    svm_model.fit(X_train, y_train)

    # Predictions
    y_pred = svm_model.predict(X_test)

    y_prob = svm_model.predict_proba(X_test)[:, 1]

    # Store all fold predictions
    all_y_test.extend(y_test)

    all_y_pred.extend(y_pred)

    all_y_prob.extend(y_prob)

    # Metrics
    acc = accuracy_score(y_test, y_pred)

    prec = precision_score(y_test, y_pred)

    rec = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    accuracies.append(acc)

    precisions.append(prec)

    recalls.append(rec)

    f1_scores.append(f1)

    print(f"Fold Accuracy : {acc:.4f}")

# FINAL METRICS
print("\n===== FINAL LOSO RESULTS =====")

print(f"Mean Accuracy : {np.mean(accuracies):.4f}")

print(f"Mean Precision: {np.mean(precisions):.4f}")

print(f"Mean Recall   : {np.mean(recalls):.4f}")

print(f"Mean F1 Score : {np.mean(f1_scores):.4f}")

##  Confusion Matrix 

# CONFUSION MATRIX
cm = confusion_matrix(
    all_y_test,
    all_y_pred
)

print("\nConfusion Matrix:")

print(cm)

plt.figure(figsize=(5,5))

plt.imshow(cm)

plt.title("LOSO Confusion Matrix")

plt.colorbar()

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.xticks([0,1], ['Normal', 'Fatigue'])

plt.yticks([0,1], ['Normal', 'Fatigue'])

for i in range(2):
    for j in range(2):

        plt.text(
            j,
            i,
            cm[i,j],
            ha='center',
            va='center'
        )

plt.show()

## ROC curve
fpr, tpr, thresholds = roc_curve(
    all_y_test,
    all_y_prob
)

roc_auc = auc(fpr, tpr)

print("\nROC-AUC:", roc_auc)

plt.figure(figsize=(6,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.3f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle='--'
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("LOSO ROC Curve")

plt.legend()

plt.show()

## Multiple classification comparison 

models = {

    "SVM": SVC(
        kernel='rbf',
        C=1,
        gamma='scale'
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "XGBoost": XGBClassifier(
        eval_metric='logloss',
        random_state=42
    )
}

## LOSO for those models 

results = {}

for model_name, model in models.items():

    print(f"\n===== {model_name} =====")

    logo = LeaveOneGroupOut()

    fold_accuracies = []

    for train_idx, test_idx in logo.split(X, y, groups):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        # Scaling
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)

        X_test = scaler.transform(X_test)

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Accuracy
        acc = accuracy_score(y_test, y_pred)

        fold_accuracies.append(acc)

    mean_acc = np.mean(fold_accuracies)

    results[model_name] = mean_acc

    print(f"Mean Accuracy: {mean_acc:.4f}")

print("\n===== MODEL COMPARISON =====")

for model_name, score in results.items():

    print(f"{model_name}: {score:.4f}")
  
## Testing why models are having high values 
print(groups[:50])
print(np.unique(groups))

for train_idx, test_idx in logo.split(X, y, groups):

    print("Train Subjects:",
          np.unique(groups[train_idx]))

    print("Test Subjects:",
          np.unique(groups[test_idx]))

    print("-" * 30)

## Statistical validation 
normal_data = X[y == 0]

fatigue_data = X[y == 1]

print(normal_data.shape)

print(fatigue_data.shape)

normal_mean = np.mean(normal_data, axis=0)

fatigue_mean = np.mean(fatigue_data, axis=0)

print(normal_mean.shape)

print(fatigue_mean.shape)

t_stat, p_values = ttest_ind(
    normal_data,
    fatigue_data,
    axis=0
)

print(p_values.shape)

significant_features = np.sum(p_values < 0.05)

print("Significant Features:")

print(significant_features)

## P value plot 

plt.figure(figsize=(12,5))

plt.plot(p_values)

plt.axhline(
    y=0.05,
    color='r',
    linestyle='--'
)

plt.xlabel("Feature Index")

plt.ylabel("P-value")

plt.title("Feature Significance")

plt.show()


theta_normal = normal_data[:, 0:40].mean()

theta_fatigue = fatigue_data[:, 0:40].mean()

alpha_normal = normal_data[:, 40:80].mean()

alpha_fatigue = fatigue_data[:, 40:80].mean()

beta_normal = normal_data[:, 80:120].mean()

beta_fatigue = fatigue_data[:, 80:120].mean()

ratio_normal = normal_data[:, 120:160].mean()

ratio_fatigue = fatigue_data[:, 120:160].mean()

## Band comparison

bands = [
    "Theta",
    "Alpha",
    "Beta",
    "Theta/Alpha"
]

normal_means = [
    theta_normal,
    alpha_normal,
    beta_normal,
    ratio_normal
]

fatigue_means = [
    theta_fatigue,
    alpha_fatigue,
    beta_fatigue,
    ratio_fatigue
]

x = np.arange(len(bands))

width = 0.35

plt.figure(figsize=(8,5))

plt.bar(
    x - width/2,
    normal_means,
    width,
    label='Normal'
)

plt.bar(
    x + width/2,
    fatigue_means,
    width,
    label='Fatigue'
)

plt.xticks(x, bands)

plt.ylabel("Mean Power")

plt.title("EEG Band Comparison")

plt.legend()

# Log scale improves visibility
plt.yscale('log')

plt.show()

## Feature importance 

# Scale data
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Train RF
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_scaled, y)

feature_importance = rf_model.feature_importances_

print(feature_importance.shape)

plt.figure(figsize=(14,5))

plt.plot(feature_importance)

plt.xlabel("Feature Index")

plt.ylabel("Importance")

plt.title("Feature Importance")

plt.show()

top_indices = np.argsort(
    feature_importance
)[::-1][:10]

print("Top Important Features:\n")

for idx in top_indices:

    print(
        f"Feature {idx}: "
        f"{feature_importance[idx]:.4f}"
    )

top_values = feature_importance[top_indices]

plt.figure(figsize=(10,5))

plt.bar(
    range(len(top_indices)),
    top_values
)

plt.xticks(
    range(len(top_indices)),
    top_indices
)

plt.xlabel("Feature Index")

plt.ylabel("Importance")

plt.title("Top 10 Important Features")

plt.show()

## Topomaps 

raw = mne.io.read_raw_cnt(
    "/content/S1_Normal state.cnt",
    preload=True,
    data_format='int16'
)

channel_names = [
    'Fp1', 'Fp2', 'F3', 'F4',
    'C3', 'C4', 'P3', 'P4',
    'O1', 'O2'
]

info = mne.create_info(
    ch_names=channel_names,
    sfreq=256,
    ch_types='eeg'
)

info.set_montage('standard_1020')

print(raw.info)

theta = theta_normal_map[0::4]
alpha = theta_normal_map[1::4]
beta  = theta_normal_map[2::4]
gamma = theta_normal_map[3::4]

print("Theta :", theta.shape)
print("Alpha :", alpha.shape)
print("Beta  :", beta.shape)
print("Gamma :", gamma.shape)

fig, axes = plt.subplots(2, 2, figsize=(10,10))

# THETA
mne.viz.plot_topomap(
    theta,
    info,
    axes=axes[0,0],
    contours=0,
    show=False
)

axes[0,0].set_title("Theta")

# ALPHA
mne.viz.plot_topomap(
    alpha,
    info,
    axes=axes[0,1],
    contours=0,
    show=False
)

axes[0,1].set_title("Alpha")

# BETA
mne.viz.plot_topomap(
    beta,
    info,
    axes=axes[1,0],
    contours=0,
    show=False
)

axes[1,0].set_title("Beta")

# GAMMA
mne.viz.plot_topomap(
    gamma,
    info,
    axes=axes[1,1],
    contours=0,
    show=False
)

axes[1,1].set_title("Gamma")

plt.tight_layout()

plt.show()

















