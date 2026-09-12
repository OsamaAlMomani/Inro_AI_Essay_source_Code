# Predictive Modelling Using Machine Learning — Credit Card Default

**Module:** Introduction to Artificial Intelligence (MSc AI, BSBI)
**Student:** Osama Al-Momani
**Assignment:** Practical Skills Assessment — Predictive Modelling Using Machine Learning

This archive contains the report and the complete, reproducible pipeline used to produce it.

---

## 1. Contents

```
├── BSBI_ML_Report_Osama_Al-Momani.docx   <- main deliverable (report)
├── BSBI_ML_Report_Osama_Al-Momani.pdf    <- same report as PDF
├── README.md                              <- this file
├── code/                                  <- Python source, dataset, requirements.txt
├── figures/                               <- every chart referenced in the report (01–15)
└── outputs/                               <- CSV/JSON result files behind the report's tables
    └── models/                            <- trained models (not included — see §5)
```

## 2. Dataset

`code/Credit_Card.csv` — 34,788 rows, 30 columns, semicolon-delimited — as provided via the assignment brief's linked repository:
[Predictive-Modelling-for-Credit-Card-Default-Using-Machine-Learning](https://github.com/abdelDebug/Predictive-Modelling-for-Credit-Card-Default-Using-Machine-Learning)

The dataset ships inside `code/` alongside the scripts. If it's ever renamed or moved, `data_prep.py` will still find it — see §6.

## 3. Setup

```bash
cd code
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.12.

## 4. Running the pipeline

**Quickest way — one command runs everything, in order:**

```bash
cd code
python3 run_all.py
```

This runs all seven steps below back to back and prints progress and timing for each. On a single-core machine it takes roughly 2–4 minutes; the `GridSearchCV` tuning step on Random Forest inside `task4_evaluate.py` is the slowest part.

**Or run each step individually**, in this order, from inside `code/`:

| Step | Command | What it does |
|---|---|---|
| 1 | `python3 data_prep.py` | Sanity-checks the cleaning logic and prints a summary (optional — everything downstream calls this automatically) |
| 2 | `python3 task1_eda.py` | Task 1: EDA — writes `figures/01`–`09` and `outputs/task1_*` |
| 3 | `python3 task1_eda_pytorch.py` | PyTorch-tensor variant of Task 1 — writes `figures_pytorch/` and `outputs_pytorch/` |
| 4 | `python3 task2_prepare.py` | Task 2: cleaning + encoding + scaling — `outputs/task2_*` |
| 5 | `python3 task3_train.py` | Task 3: trains 6 models on the leakage-free feature set — `outputs/models/` |
| 6 | `python3 task3b_leakage_demo.py` | Leakage demonstration (`risk_leak` included) — `outputs/task_leakage_demo.json` |
| 7 | `python3 task4_evaluate.py` | Task 4: evaluation, `GridSearchCV` tuning, SHAP — `figures/10`–`15`, `outputs/task4_*` |

Each script writes to `../outputs` and `../figures` relative to its own location, so it doesn't matter which directory you run it from or where the project folder sits on disk — see §6. The archive ships with `figures/` and `outputs/` already populated with the exact results used in the report; re-running will overwrite them with a fresh, equivalent run.

## 5. Trained models are not included

`outputs/models/*.joblib` (~110 MB in total, dominated by the Random Forest and its tuned counterpart) is excluded from this archive to keep the submission size manageable. Running `task3_train.py` (or `run_all.py`) regenerates all of it automatically before `task4_evaluate.py` needs it.

## 6. Robustness notes

The pipeline is written to run unmodified on any machine, from any folder:

- **No manual folder setup.** Every script creates `figures/`, `outputs/`, and `outputs/models/` itself the first time it runs — you never need to pre-create them.
- **Path-independent.** Every script resolves its paths from its own file location, not the current working directory, so it works whether you run it from inside `code/`, via a full path, or from an IDE's "Run" button.
- **Dataset auto-detection.** `data_prep.py` looks for `Credit_Card.csv` next to the scripts first; if it's been renamed (e.g. a browser download prefixing it with a timestamp), it falls back to any `*.csv` with "credit" and "card" in the name in `code/`, the project root, or a `data/` subfolder — with a clear error message if nothing matches.
- **Helpful failure messages.** For example, running `task4_evaluate.py` before `task3_train.py` prints a direct pointer to the script you need to run first, rather than a raw stack trace.

## 7. File overview

| File | Purpose |
|---|---|
| `run_all.py` | Runs every task script below in order, with progress and timing. |
| `data_prep.py` | Loads the raw CSV and fixes the two data-quality issues found during EDA: (1) genuine missing values, (2) a digit-grouping corruption in `risk_leak`/`LIMIT_BAL_LOG` where the decimal point was replaced by grouping periods. `BILL_AMT_SUM` and `LIMIT_BAL_LOG` are recomputed deterministically from their source columns rather than reverse-engineered. Also defines the shared output-folder setup and CSV auto-detection used by every other script. |
| `task1_eda.py` | Exploratory Data Analysis: target distribution, descriptive statistics, histograms, boxplots, categorical distributions, correlation analysis. Flags `risk_leak` (r=0.97 with target) as a likely leakage feature. |
| `task1_eda_pytorch.py` | Same EDA, recomputed with PyTorch tensors instead of pandas/numpy for the numerical statistics, to demonstrate the tensor-based workflow. Writes to its own `figures_pytorch/`/`outputs_pytorch/` folders so it never overwrites the main EDA outputs. |
| `task2_prepare.py` | Builds the final scaled feature matrix: frequency-encodes `CITY`, retains integer-coded categoricals, applies `StandardScaler`. Has an `include_leak_feature` flag used only by the leakage demo. |
| `task3_train.py` | Trains Logistic Regression, SVM (RBF, 6,000-row subsample), Decision Tree, Random Forest, KNN and XGBoost with default hyperparameters on an 80/20 stratified split. Saves fitted models to `outputs/models/`. |
| `task3b_leakage_demo.py` | Stand-alone script proving that including `risk_leak` produces a trivial, meaningless 100%-accuracy model — the evidence behind excluding it from the main pipeline. |
| `task4_evaluate.py` | Evaluates all six models (accuracy, precision, recall, F1, AUC-ROC), plots confusion matrices and ROC curves, tunes the best model with `GridSearchCV`, and runs a SHAP (`TreeExplainer`) interpretation of the tuned model. |

## 8. Key findings

*(see the report for full discussion)*

- **Data leakage:** `risk_leak` correlates at r=0.97 with the target and produces a trivial 100%-accuracy model — excluded from all "real" modelling.
- **Best model:** Random Forest, AUC-ROC 0.870 (untuned) / 0.855 (tuned).
- **Top predictors (SHAP):** `PAY_0`, `RISK_RATING`, `PAY_2` — i.e. recent repayment behaviour and the bank's own risk classification.

## 9. Reproducibility

All models use `random_state=42`. Minor floating-point differences may occur across scikit-learn/XGBoost versions, but rankings and conclusions are stable.