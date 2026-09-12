"""
task4_evaluate.py - Model Evaluation & Visualisation (Task 4, 20%)

Evaluates all trained models, produces comparison tables, confusion
matrices, ROC curves, tunes the best model with GridSearchCV, and runs a
SHAP interpretation of the tuned best model.
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
from pathlib import Path
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, classification_report,
)
from sklearn.model_selection import GridSearchCV

from data_prep import ensure_output_dirs, FIGURES_DIR, OUTPUTS_DIR, MODELS_DIR

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = str(FIGURES_DIR)
OUT_DIR = str(OUTPUTS_DIR)
MODEL_DIR = str(MODELS_DIR)

# Create figures/, outputs/, and outputs/models/ if they don't exist yet.
# outputs/models/ should already exist from task3_train.py (that's where
# the trained models this script loads come from), but this makes the
# script robust even if run in isolation on a fresh checkout.
ensure_output_dirs()

sns.set_style("whitegrid")


def load_everything():
    splits_path = Path(MODEL_DIR) / "splits.joblib"
    if not splits_path.exists():
        raise FileNotFoundError(
            f"Could not find {splits_path}. Run task3_train.py first - it "
            "trains the models and saves the train/test splits that this "
            "script (task4_evaluate.py) loads."
        )
    splits = joblib.load(splits_path)
    model_names = [
        "Logistic Regression", "Support Vector Machine", "Decision Tree",
        "Random Forest", "K-Nearest Neighbors", "XGBoost",
    ]
    models = {n: joblib.load(f"{MODEL_DIR}/{n.replace(' ', '_')}.joblib") for n in model_names}
    return splits, models


def evaluate_models(models, X_test, y_test):
    rows = []
    roc_data = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-score": f1_score(y_test, y_pred),
            "AUC-ROC": roc_auc_score(y_test, y_proba),
        })
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr)
    # Rank primarily by AUC-ROC: with ~19% positive class, AUC-ROC is a more
    # threshold- and imbalance-robust ranking criterion than F1-score, which
    # is sensitive to the default 0.5 decision threshold and can be nearly
    # tied between an overfit single tree and a well-generalising ensemble.
    results_df = pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False).reset_index(drop=True)
    return results_df, roc_data


def plot_confusion_matrices(models, X_test, y_test):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, (name, model) in zip(axes.flat, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["No Default", "Default"], yticklabels=["No Default", "Default"])
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/10_confusion_matrices.png", dpi=150)
    plt.close()


def plot_roc_curves(roc_data, results_df):
    fig, ax = plt.subplots(figsize=(8, 7))
    auc_lookup = results_df.set_index("Model")["AUC-ROC"].to_dict()
    for name, (fpr, tpr) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_lookup[name]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - All Models")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/11_roc_curves.png", dpi=150)
    plt.close()


def plot_comparison_bar(results_df):
    metrics = ["Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"]
    fig, ax = plt.subplots(figsize=(11, 6))
    results_df.set_index("Model")[metrics].plot(kind="bar", ax=ax)
    ax.set_title("Model Performance Comparison (default hyperparameters)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/12_metric_comparison_bar.png", dpi=150)
    plt.close()


def tune_best_model(best_name, X_train, y_train, X_test, y_test):
    """Hyperparameter-tune the best-performing model with GridSearchCV."""
    param_grids = {
        "Random Forest": {
            "n_estimators": [100],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        },
        "XGBoost": {
            "n_estimators": [100, 200],
            "max_depth": [3, 6],
            "learning_rate": [0.05, 0.1, 0.2],
        },
        "Decision Tree": {
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
        },
        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
        },
        "K-Nearest Neighbors": {
            "n_neighbors": [3, 5, 7, 11],
            "weights": ["uniform", "distance"],
        },
    }
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from xgboost import XGBClassifier

    base_estimators = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss"),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
    }

    if best_name not in param_grids:
        return None, None

    grid = GridSearchCV(
        base_estimators[best_name], param_grids[best_name],
        scoring="f1", cv=3, n_jobs=1,
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    tuned_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_proba),
    }
    return grid.best_params_, tuned_metrics, best_model


def run_shap(best_model, best_name, X_train, X_test, feature_cols):
    import shap
    sample = X_test.sample(n=min(200, len(X_test)), random_state=42)

    if best_name in ("Random Forest", "Decision Tree", "XGBoost"):
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
    else:
        bg = shap.sample(X_train, 100, random_state=42)
        explainer = shap.KernelExplainer(best_model.predict_proba, bg)
        sv = explainer.shap_values(sample, nsamples=100)
        if isinstance(sv, list):
            shap_values = sv[1]
        elif np.asarray(sv).ndim == 3:
            shap_values = np.asarray(sv)[:, :, 1]
        else:
            shap_values = sv

    plt.figure()
    shap.summary_plot(shap_values, sample, feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/13_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, sample, feature_names=feature_cols, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/14_shap_importance_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols).sort_values(ascending=False)
    return mean_abs_shap


def run():
    splits, models = load_everything()
    X_train, X_test = splits["X_train"], splits["X_test"]
    y_train, y_test = splits["y_train"], splits["y_test"]
    feature_cols = splits["feature_cols"]

    results_df, roc_data = evaluate_models(models, X_test, y_test)
    print("\n=== Model comparison (default hyperparameters) ===")
    print(results_df.round(4).to_string(index=False))
    results_df.round(4).to_csv(f"{OUT_DIR}/task4_model_comparison.csv", index=False)

    plot_confusion_matrices(models, X_test, y_test)
    plot_roc_curves(roc_data, results_df)
    plot_comparison_bar(results_df)

    best_name = results_df.iloc[0]["Model"]
    print(f"\nBest-performing model (by F1): {best_name}")

    # For tuning, use the model trained on the FULL training set (skip SVM
    # if selected, since it used a subsample - substitute the second-best
    # full-data model in that unlikely case)
    if best_name == "Support Vector Machine":
        best_name = results_df.iloc[1]["Model"]
        print(f"(SVM excluded from tuning due to subsample training; tuning {best_name} instead)")

    tune_result = tune_best_model(best_name, X_train, y_train, X_test, y_test)
    best_params, tuned_metrics, tuned_model = tune_result

    before_after = pd.DataFrame({
        "Before Tuning": results_df.set_index("Model").loc[best_name],
        "After Tuning": pd.Series(tuned_metrics),
    })
    print(f"\n=== {best_name}: Before vs After Tuning ===")
    print(before_after.round(4))
    before_after.round(4).to_csv(f"{OUT_DIR}/task4_tuning_before_after.csv")

    with open(f"{OUT_DIR}/task4_best_hyperparameters.json", "w") as f:
        json.dump({"best_model": best_name, "best_params": best_params}, f, indent=2, default=str)

    fig, ax = plt.subplots(figsize=(7, 5))
    before_after.T.plot(kind="bar", ax=ax)
    ax.set_title(f"{best_name}: Performance Before vs After Hyperparameter Tuning")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/15_tuning_before_after.png", dpi=150)
    plt.close()

    # SHAP interpretation of the tuned best model
    mean_abs_shap = run_shap(tuned_model, best_name, X_train, X_test, feature_cols)
    print("\n=== Mean |SHAP value| (feature importance) ===")
    print(mean_abs_shap.head(10))
    mean_abs_shap.round(4).to_csv(f"{OUT_DIR}/task4_shap_importance.csv")

    joblib.dump(tuned_model, f"{MODEL_DIR}/best_tuned_model.joblib")
    print("\nTask 4 evaluation complete.")


if __name__ == "__main__":
    run()
