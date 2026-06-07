## training model 
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

# ROC CURVE
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




