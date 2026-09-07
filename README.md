# Predictive Modelling Using Machine Learning — Credit Card Default

**Module:** Introduction to Artificial Intelligence (MSc AI, BSBI)
**Student:** Osama Al-Momani
**Assignment:** Practical Skills Assessment — Predictive Modelling Using Machine Learning

This archive contains the full report and the complete, reproducible pipeline
used to produce it.

## Folder structure

```
├── BSBI_ML_Report_Osama_Al-Momani.docx   <- main deliverable (report)
├── BSBI_ML_Report_Osama_Al-Momani.pdf    <- same report as PDF
├── README.md                              <- this file
├── code/                                  <- all Python source + dataset + requirements.txt
├── figures/                                <- every chart referenced in the report (01–15)
└── outputs/                                <- CSV/JSON result files the report's tables were built from
    └── models/                             <- trained models are NOT included (see below); re-run code/ to regenerate
```

## Dataset

`code/Credit_Card.csv` (34,788 rows, 30 columns) is included, as provided via
the assignment brief's linked repository
(https://github.com/abdelDebug/Predictive-Modelling-for-Credit-Card-Default-Using-Machine-Learning).
It is a semicolon-delimited CSV.

## Setup

```bash
cd code
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.12.

## Pipeline / how to reproduce every result and figure in the report

Run the scripts **in this order** from inside the `code/` folder. Each
script reads from/writes to `../outputs` and `../figures` relative to
`code/`, so keep the folder structure intact (the archive already ships
with these folders pre-populated with the exact results used in the report
— re-running will overwrite them with a fresh, equivalent run).

```bash
cd code
python3 data_prep.py            # sanity-checks the cleaning logic (optional, prints a summary)
python3 task1_eda.py            # Task 1: EDA — writes figures/01..09 and outputs/task1_*
python3 task2_prepare.py        # Task 2: cleaning + encoding + scaling — outputs/task2_*
python3 task3_train.py          # Task 3: trains 6 models (leakage-free feature set) — outputs/models/
python3 task3b_leakage_demo.py  # Leakage demonstration (risk_leak included) — outputs/task_leakage_demo.json
python3 task4_evaluate.py       # Task 4: evaluation, GridSearchCV tuning, SHAP — figures/10..15, outputs/task4_*
```

Total runtime on a single-core machine: approximately 3–4 minutes (the
GridSearchCV step on Random Forest is the slowest part).

**Note on `outputs/models/`:** trained model binaries (`.joblib`, ~110 MB in
total, dominated by the Random Forest and its tuned counterpart) are not
included in this archive to keep the submission size manageable. They are
regenerated automatically the first time `task3_train.py` / `task4_evaluate.py`
are run.

## File overview

| File | Purpose |
|---|---|
| `data_prep.py` | Loads the raw CSV and fixes the two data-quality issues discovered during EDA: (1) genuine missing values, (2) a digit-grouping corruption in `risk_leak`/`LIMIT_BAL_LOG` where the decimal point was replaced by grouping periods. `BILL_AMT_SUM` and `LIMIT_BAL_LOG` are recomputed deterministically from their source columns rather than reverse-engineered. |
| `task1_eda.py` | Exploratory Data Analysis: target distribution, descriptive statistics, histograms, boxplots, categorical distributions, correlation analysis. Flags `risk_leak` (r=0.97 with target) as a likely leakage feature. |
| `task2_prepare.py` | Builds the final scaled feature matrix: frequency-encodes `CITY`, retains integer-coded categoricals, applies `StandardScaler`. Has an `include_leak_feature` flag used only by the leakage demo. |
| `task3_train.py` | Trains Logistic Regression, SVM (RBF, 6,000-row subsample), Decision Tree, Random Forest, KNN and XGBoost with default hyperparameters on an 80/20 stratified split. Saves fitted models to `outputs/models/`. |
| `task3b_leakage_demo.py` | Stand-alone script proving that including `risk_leak` produces a trivial, meaningless 100%-accuracy model — the evidence behind excluding it from the main pipeline. |
| `task4_evaluate.py` | Evaluates all six models (accuracy, precision, recall, F1, AUC-ROC), plots confusion matrices and ROC curves, tunes the best model (by AUC-ROC) with `GridSearchCV`, and runs a SHAP (`TreeExplainer`) interpretation of the tuned model. |

## Key findings (see report for full discussion)

- **Data leakage:** `risk_leak` correlates at r=0.97 with the target and produces a
  trivial 100%-accuracy model — it was excluded from all "real" modelling.
- **Best model:** Random Forest, AUC-ROC 0.870 (untuned) / 0.855 (tuned).
- **Top predictors (SHAP):** `PAY_0`, `RISK_RATING`, `PAY_2` — i.e. recent
  repayment behaviour and the bank's own risk classification.

## Notes on reproducibility

All models use `random_state=42`. Minor floating-point differences may occur
across scikit-learn/XGBoost versions, but rankings and conclusions are stable.
