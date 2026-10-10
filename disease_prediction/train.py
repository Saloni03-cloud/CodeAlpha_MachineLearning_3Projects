import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the breast cancer dataset
dataset = load_breast_cancer(as_frame=True)
X = dataset.data
y = dataset.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train and compare two models
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000)
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42
    ),
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\n{name} accuracy: {accuracy:.4f}")
    print(classification_report(
        y_test, predictions, target_names=dataset.target_names
    ))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Save the best model
joblib.dump(best_model, "disease_prediction_model.joblib")
print(f"Best test accuracy: {best_accuracy:.4f}")
print("Saved disease_prediction_model.joblib")

5. Tap Commit changes….
6. Confirm the commit on the next screen.

This code is for educational practice, not real medical diagnosis.

Do these steps and send me a screenshot after you tap “Commit changes…”.
