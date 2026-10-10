import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.datasets import make_classification

# Create a sample credit-risk dataset
X, y = make_classification(
    n_samples=1000,
    n_features=8,
    n_informative=5,
    n_redundant=1,
    weights=[0.7, 0.3],
    random_state=42
)

columns = [
    "income", "debt", "payment_history", "credit_usage",
    "employment_years", "loan_amount", "savings", "age"
]

data = pd.DataFrame(X, columns=columns)
data["default"] = y

data.to_csv("credit_data.csv", index=False)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data[columns],
    data["default"],
    test_size=0.2,
    random_state=42,
    stratify=data["default"]
)

# Train the model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Credit Scoring Model")
print("--------------------")
print(f"Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save evaluation results
with open("classification_report.txt", "w") as file:
    file.write(f"Accuracy: {accuracy:.3f}\n\n")
    file.write(classification_report(y_test, predictions))

# Save the confusion matrix plot
ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, predictions),
    display_labels=["No Default", "Default"]
).plot()

plt.title("Credit Scoring - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

print("\nSaved credit_data.csv, classification_report.txt,")
print("and confusion_matrix.png")

