# Task 4 — Disease Prediction from Medical Data

## Objective
Classify a breast mass as benign or malignant from diagnostic measurements.

## Method
- Use scikit-learn's built-in Wisconsin Diagnostic Breast Cancer dataset.
- Split into stratified training and test sets.
- Compare Logistic Regression and Random Forest.
- Report accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix.
- Save the best model based on test ROC-AUC.

## Run
```bash
python disease_prediction/train.py
```
This is an educational benchmark, not medical advice or a clinical diagnostic system. A model score must not be used to make decisions about a real patient.
