"""
task3_train.py - Model Training (Task 3, 20%)

Trains five classification algorithms with default hyperparameters on a
stratified train/test split, and persists the fitted models + splits for
Task 4 (evaluation, tuning, SHAP).
"""
import json
import time
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from task2_prepare import prepare_features

OUT_DIR = "../outputs"
MODEL_DIR = "../outputs/models"
RANDOM_STATE = 42

import os
os.makedirs(MODEL_DIR, exist_ok=True)


def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Support Vector Machine": SVC(probability=True, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="logloss", use_label_encoder=False
        ),
    }


def run():
    # include_leak_feature=False: risk_leak is excluded from the main
    # modelling pipeline after being identified as a target-leakage
    # feature during EDA (see task3b_leakage_demo.py for evidence).
    X, y, scaler, feature_cols = prepare_features(include_leak_feature=False)

    # 80/20 stratified split: preserves the ~81%/19% class ratio in both
    # sets, which matters given the moderate class imbalance observed in
    # Task 1. 80/20 gives ~27,800 training rows (ample for all algorithms,
    # including higher-variance ones like KNN and SVM) while still leaving
    # ~6,950 held-out rows for a statistically stable test evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Train class balance:\n", y_train.value_counts(normalize=True))
    print("Test class balance:\n", y_test.value_counts(normalize=True))

    models = build_models()
    initial_params = {}
    fitted = {}
    timings = {}

    # SVM is O(n^2)-O(n^3) in the number of training rows with an RBF
    # kernel; training on the full ~27,800-row training set is
    # computationally impractical for this assessment, so SVM is trained on
    # a class-stratified random subsample of 6,000 training rows. This is a
    # documented, common practical compromise and is discussed as a
    # limitation in Task 5.
    from sklearn.model_selection import train_test_split as tts
    X_train_svm, _, y_train_svm, _ = tts(
        X_train, y_train, train_size=6000, stratify=y_train, random_state=RANDOM_STATE
    )

    for name, model in models.items():
        initial_params[name] = {k: str(v) for k, v in model.get_params().items()}
        t0 = time.time()
        if name == "Support Vector Machine":
            model.fit(X_train_svm, y_train_svm)
        else:
            model.fit(X_train, y_train)
        timings[name] = round(time.time() - t0, 2)
        fitted[name] = model
        joblib.dump(model, f"{MODEL_DIR}/{name.replace(' ', '_')}.joblib")
        print(f"Trained {name} in {timings[name]}s")

    # Persist splits for Task 4
    joblib.dump(
        {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
         "X_train_svm": X_train_svm, "y_train_svm": y_train_svm,
         "feature_cols": feature_cols},
        f"{MODEL_DIR}/splits.joblib",
    )

    with open(f"{OUT_DIR}/task3_initial_hyperparameters.json", "w") as f:
        json.dump(initial_params, f, indent=2)
    with open(f"{OUT_DIR}/task3_training_times.json", "w") as f:
        json.dump(timings, f, indent=2)

    print("Task 3 training complete. Models saved to", MODEL_DIR)


if __name__ == "__main__":
    run()
