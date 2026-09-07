"""
task3b_leakage_demo.py - Data Leakage Demonstration

During Task 1 EDA, `risk_leak` was found to correlate ~0.97 with the target
variable, far higher than any genuine behavioural or demographic feature.
This script provides a short, self-contained demonstration of the effect of
including it: a single Logistic Regression trained with `risk_leak`
included reaches (near-)perfect test performance, which is not a realistic
outcome for a credit-default model and is a strong indicator of target
leakage. Results are saved for inclusion in the report's Task 1 / Task 5
discussion.
"""
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from task2_prepare import prepare_features

OUT_DIR = "../outputs"


def run():
    X, y, _, feature_cols = prepare_features(include_leak_feature=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_proba),
    }
    print("Logistic Regression WITH risk_leak included:")
    print(json.dumps(metrics, indent=2))

    with open(f"{OUT_DIR}/task_leakage_demo.json", "w") as f:
        json.dump({"features_used": feature_cols, "metrics": metrics}, f, indent=2)


if __name__ == "__main__":
    run()
