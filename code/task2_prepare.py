"""
task2_prepare.py - Data Preparation (Task 2, 20%)

Produces the final model-ready feature matrix X and target y, and writes
before/after examples to ../outputs for inclusion in the report.
"""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from data_prep import load_raw, clean_dataset, ensure_output_dirs, OUTPUTS_DIR

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = str(OUTPUTS_DIR)
TARGET = "default.payment.next.month"

# Create outputs/ (and every other project output folder) if it doesn't
# exist yet, so this runs standalone on a fresh checkout too.
ensure_output_dirs()


def prepare_features(include_leak_feature: bool = False):
    """Build the model-ready, scaled feature matrix.

    include_leak_feature: if True, `risk_leak` is included in the feature
    set. During EDA (Task 1) risk_leak was found to correlate ~0.97 with
    the target -- far higher than any genuine payment-history or
    demographic variable, and inconsistent with it being an independently
    observable risk score. This is a textbook symptom of target leakage
    (reinforced by the column's own name). Its inclusion is therefore
    reserved for a short, clearly-labelled "leakage demonstration" (see
    task3_leakage_demo.py); the main modelling pipeline in Tasks 3-4
    excludes it to produce a realistic, deployable model.
    """
    raw = load_raw()
    clean, clean_report = clean_dataset(raw)

    before_example = clean[["LIMIT_BAL", "AGE", "BILL_AMT_SUM", "CITY"]].head(5).copy()

    df = clean.copy()

    # ---- Encoding -----------------------------------------------------
    # CITY: 50 near-uniform anonymised labels -> one-hot would add 49 sparse
    # columns; frequency encoding keeps dimensionality low while still
    # letting models exploit any city-level signal.
    city_freq = df["CITY"].value_counts(normalize=True)
    df["CITY_FREQ"] = df["CITY"].map(city_freq)

    # SEX, EDUCATION, MARRIAGE, RISK_RATING are already integer-coded
    # categorical/ordinal variables in the source schema, so no further
    # encoding is required for tree-based or linear models.

    feature_cols = [
        "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
        "BILL_AMT_SUM", "LIMIT_BAL_LOG", "CITY_FREQ", "RISK_RATING",
    ]
    if include_leak_feature:
        feature_cols = feature_cols + ["risk_leak"]

    X = df[feature_cols].copy()
    y = df[TARGET].copy()

    # ---- Scaling --------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    after_example = X_scaled[["LIMIT_BAL", "AGE", "BILL_AMT_SUM"]].head(5).copy()
    after_example.insert(3, "CITY_FREQ", X_scaled["CITY_FREQ"].head(5).values)

    if not include_leak_feature:
        before_example.round(2).to_csv(f"{OUT_DIR}/task2_before_example.csv", index=False)
        after_example.round(3).to_csv(f"{OUT_DIR}/task2_after_example.csv", index=False)

        summary = {
            "n_rows": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "feature_list": feature_cols,
            "class_balance": y.value_counts(normalize=True).round(4).to_dict(),
            "city_unique_values": int(df["CITY"].nunique()),
            "cleaning_report": clean_report,
        }
        with open(f"{OUT_DIR}/task2_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

    print("Feature matrix:", X_scaled.shape)
    print("Class balance:", y.value_counts(normalize=True).round(4).to_dict())
    return X_scaled, y, scaler, feature_cols


if __name__ == "__main__":
    prepare_features()
