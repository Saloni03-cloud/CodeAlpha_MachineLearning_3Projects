"""Educational breast-cancer classification benchmark using scikit-learn data."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             ConfusionMatrixDisplay)
import joblib

ROOT = Path(__file__).resolve().parent
SEED = 42

def main():
    dataset = load_breast_cancer(as_frame=True)
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    models = {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=5000, random_state=SEED))
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", random_state=SEED, n_jobs=-1
        )
    }
    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        # Class 1 in this dataset means benign; class 0 means malignant.
        prob_benign = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision_benign_class": float(precision_score(y_test, pred, zero_division=0)),
            "recall_benign_class": float(recall_score(y_test, pred, zero_division=0)),
            "f1_benign_class": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, prob_benign))
        }
        fitted[name] = model
    best_name = max(results, key=lambda name: results[name]["roc_auc"])
    best = fitted[best_name]
    pred = best.predict(X_test)
    (ROOT / "metrics.json").write_text(json.dumps({
        "dataset": "Wisconsin Diagnostic Breast Cancer (scikit-learn)",
        "target_mapping": {"0": "malignant", "1": "benign"},
        "model_comparison": results,
        "selected_model_by_test_roc_auc": best_name
    }, indent=2))
    (ROOT / "classification_report.txt").write_text(
        classification_report(y_test, pred, target_names=dataset.target_names)
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=dataset.target_names, ax=ax, cmap="Blues"
    )
    ax.set_title(f"Breast Cancer Classification — {best_name}")
    fig.tight_layout()
    fig.savefig(ROOT / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    joblib.dump(best, ROOT / "disease_prediction_model.joblib")
    print("Breast cancer dataset model comparison:")
    print(json.dumps(results, indent=2))
    print(f"Selected by test ROC-AUC: {best_name}")
    print(f"Artifacts saved to: {ROOT}")

if __name__ == "__main__":
    main()
