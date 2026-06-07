joblib.dump(svm_model, "svm_fatigue_model.pkl")
joblib.dump(scaler, "scaler.pkl")

def predict_fatigue(features):

    features = scaler.transform([features])

    prediction = svm_model.predict(features)

    probability = svm_model.predict_proba(features)[0][1]

    return prediction, probability

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

