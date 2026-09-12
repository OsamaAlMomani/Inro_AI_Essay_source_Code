
# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import json                     # For saving metrics as JSON
from pathlib import Path        # Object-oriented filesystem paths
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# Import the feature preparation function from task2_prepare
from task2_prepare import prepare_features
from data_prep import ensure_output_dirs, OUTPUTS_DIR

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = str(OUTPUTS_DIR)   # Directory for output files

# Create outputs/ (and every other project output folder) if it doesn't
# exist yet, so this runs standalone on a fresh checkout too.
ensure_output_dirs()


# ---------------------------------------------------------------------------
# Function: run
# ---------------------------------------------------------------------------
def run():
    """Train a simple Logistic Regression with the leaky feature included,
    evaluate it, and save the results as evidence of target leakage."""

    # -----------------------------------------------------------------------
    # Prepare features with the leaky feature included
    # -----------------------------------------------------------------------
    # include_leak_feature=True intentionally includes `risk_leak`
    # to demonstrate the leakage effect.
    X, y, _, feature_cols = prepare_features(include_leak_feature=True)

    # -----------------------------------------------------------------------
    # Train/test split (stratified, same random state as main pipeline)
    # -----------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # -----------------------------------------------------------------------
    # Train a Logistic Regression model
    # -----------------------------------------------------------------------
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # -----------------------------------------------------------------------
    # Generate predictions and predicted probabilities
    # -----------------------------------------------------------------------
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]   # Probability of class 1

    # -----------------------------------------------------------------------
    # Compute evaluation metrics
    # -----------------------------------------------------------------------
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_proba),
    }

    # -----------------------------------------------------------------------
    # Print results to console
    # -----------------------------------------------------------------------
    print("Logistic Regression WITH risk_leak included:")
    print(json.dumps(metrics, indent=2))

    # -----------------------------------------------------------------------
    # Save results to JSON for inclusion in the report
    # -----------------------------------------------------------------------
    with open(f"{OUT_DIR}/task_leakage_demo.json", "w") as f:
        json.dump({"features_used": feature_cols, "metrics": metrics}, f, indent=2)


# ---------------------------------------------------------------------------
# Main block: execute when the script is run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run()