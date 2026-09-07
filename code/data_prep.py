"""
data_prep.py
------------
Loading and cleaning routines for the Credit_Card.csv dataset used in the
BSBI "Predictive Modelling Using Machine Learning" assignment.

The raw CSV has three distinct data-quality problems that were discovered
during inspection and must be handled before any modelling:

1. Genuine missing values (NaN) in LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
   PAY_AMT1 and PAY_AMT2.
2. A "digit-grouping" corruption in risk_leak and LIMIT_BAL_LOG, where the
   decimal point of the original float was lost and periods were re-inserted
   every three characters (e.g. the true value 1.075675276433290 was stored
   as the string "1.075.675.276.433.290").
3. A small number of risk_leak values stored in European-style scientific
   notation with a comma decimal separator (e.g. "-2,00E+11"), which are
   wildly outside the documented risk_leak range of [-0.39, 1.40] and are
   therefore treated as invalid/corrupted rather than genuine values.

Author: Osama Al-Momani
"""
import numpy as np
import pandas as pd

RAW_PATH = "Credit_Card.csv"


def _fix_grouped_decimal(value):
    """Recover a float whose decimal point was replaced by grouping dots.

    Example: '1.075.675.276.433.290' -> 1.075675276433290
    Returns np.nan for values that cannot be safely recovered (e.g. the
    comma/scientific-notation corruption, which is flagged as invalid data).
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    # Already a clean float
    try:
        return float(s)
    except ValueError:
        pass
    # European scientific notation with comma decimal -> treat as corrupted
    if "e" in s.lower() or "," in s:
        return np.nan
    neg = s.startswith("-")
    body = s[1:] if neg else s
    digits = body.replace(".", "")
    if not digits.isdigit():
        return np.nan
    recovered = float(digits[0] + "." + digits[1:]) if len(digits) > 1 else float(digits)
    recovered = -recovered if neg else recovered
    # risk_leak is documented to lie in [-0.39, 1.40]; discard implausible
    # reconstructions defensively (keeps a small safety margin)
    if not (-2.0 <= recovered <= 2.0):
        return np.nan
    return recovered


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """Read the semicolon-delimited raw CSV exactly as provided."""
    df = pd.read_csv(path, sep=";", low_memory=False)
    return df


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply all documented cleaning steps and return (clean_df, report).

    `report` is a dict capturing before/after statistics used for the
    Data Preparation section of the report (Task 2).
    """
    df = df.copy()
    report = {}

    # ---- Step 1: fix the digit-grouping corruption in risk_leak ----------
    before_bad = df["risk_leak"].apply(
        lambda x: False if _is_clean_float(x) else True
    ).sum()
    df["risk_leak"] = df["risk_leak"].apply(_fix_grouped_decimal)
    after_missing = df["risk_leak"].isna().sum()
    report["risk_leak_corrupted_strings_found"] = int(before_bad)
    report["risk_leak_unrecoverable_set_to_nan"] = int(after_missing)

    # ---- Step 2: recompute BILL_AMT_SUM deterministically -----------------
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    recomputed_sum = df[bill_cols].sum(axis=1)
    mismatch = (df["BILL_AMT_SUM"] - recomputed_sum).abs() > 1.0
    report["bill_amt_sum_mismatches_fixed"] = int(mismatch.sum())
    df["BILL_AMT_SUM"] = recomputed_sum

    # ---- Step 3: recompute LIMIT_BAL_LOG deterministically -----------------
    # (avoids guessing decimal placement for a value that is a pure
    #  transform of LIMIT_BAL: LIMIT_BAL_LOG = ln(LIMIT_BAL))
    report["limit_bal_log_recomputed_rows"] = int(df["LIMIT_BAL_LOG"].notna().sum())
    df["LIMIT_BAL_LOG"] = np.nan  # will be recomputed after LIMIT_BAL is imputed

    # ---- Step 4: handle genuine missing values -----------------------------
    missing_before = df.isna().sum()

    # Numeric columns -> median imputation (robust to skew/outliers)
    for col in ["LIMIT_BAL", "AGE", "PAY_AMT1", "PAY_AMT2"]:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # Categorical-coded numeric columns -> mode imputation
    for col in ["SEX", "EDUCATION", "MARRIAGE"]:
        mode_val = df[col].mode(dropna=True)[0]
        df[col] = df[col].fillna(mode_val)

    # risk_leak (numeric, slightly skewed but no extreme outliers after
    # cleaning) -> median imputation for the small number of unrecoverable
    # corrupted entries
    df["risk_leak"] = df["risk_leak"].fillna(df["risk_leak"].median())

    # Recompute LIMIT_BAL_LOG now that LIMIT_BAL has no missing values
    df["LIMIT_BAL_LOG"] = np.log(df["LIMIT_BAL"].clip(lower=1))

    missing_after = df.isna().sum()
    report["missing_before"] = missing_before[missing_before > 0].to_dict()
    report["missing_after"] = int(missing_after.sum())

    # ---- Step 5: tidy categorical codes ------------------------------------
    # EDUCATION values 0, 5, 6 are undocumented/"unknown" categories in the
    # original UCI credit-card schema -> collapse into a single "Other" (4)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    # MARRIAGE value 0 is undocumented -> collapse into "Other" (3)
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    report["final_shape"] = df.shape
    return df, report


def _is_clean_float(x):
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    raw = load_raw()
    clean, rep = clean_dataset(raw)
    print("Raw shape:", raw.shape)
    print("Clean shape:", clean.shape)
    print("Remaining missing values:", clean.isna().sum().sum())
    for k, v in rep.items():
        print(f"{k}: {v}")
