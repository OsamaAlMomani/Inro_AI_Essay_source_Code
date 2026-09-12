
# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
# Resolve paths relative to this script's own location, not the current
# working directory. This means the script works whether you run it as
# `python data_prep.py` from inside code/, or via a full path / VS Code's
# "Run Python File" button from anywhere else, on any machine.
BASE_DIR = Path(__file__).resolve().parent

# Every output/figure directory used across the whole project, defined once
# here so every script (which imports this module) can create them before
# writing anything. Using parents=True, exist_ok=True makes this safe to
# call every run, on a brand-new checkout, in any folder location.
FIGURES_DIR = BASE_DIR.parent / "figures"
OUTPUTS_DIR = BASE_DIR.parent / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_PYTORCH_DIR = BASE_DIR.parent / "figures_pytorch"
OUTPUTS_PYTORCH_DIR = BASE_DIR.parent / "outputs_pytorch"


def ensure_output_dirs() -> None:
    """Create every directory the pipeline writes to, if it doesn't exist yet.

    Safe to call from any script, any number of times, on a fresh checkout
    with no folders at all. Every downstream task*.py calls this before it
    writes a single file, so the whole pipeline runs on any PC without the
    user having to manually create figures/outputs/outputs/models first.
    """
    for d in (FIGURES_DIR, OUTPUTS_DIR, MODELS_DIR, FIGURES_PYTORCH_DIR, OUTPUTS_PYTORCH_DIR):
        d.mkdir(parents=True, exist_ok=True)


def find_raw_csv() -> Path:
    """Locate the raw credit-card CSV without assuming an exact filename.

    Looks, in order, for:
      1. Credit_Card.csv right next to the scripts (the expected layout).
      2. Any *.csv in the scripts folder whose name contains "credit" and
         "card" (case-insensitive) - covers renamed/downloaded copies such
         as a timestamp-prefixed export.
      3. The same two checks one level up, and in a sibling "data" folder,
         so the script still works if the CSV lives alongside the project
         root instead of inside code/.
    Raises a clear, actionable error if nothing matches, instead of a bare
    FileNotFoundError from pandas.
    """
    candidates_dirs = [BASE_DIR, BASE_DIR.parent, BASE_DIR.parent / "data"]

    exact = BASE_DIR / "Credit_Card.csv"
    if exact.exists():
        return exact

    for d in candidates_dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.csv")):
            name = f.name.lower()
            if "credit" in name and "card" in name:
                return f

    raise FileNotFoundError(
        "Could not find the raw credit-card CSV. Expected 'Credit_Card.csv' "
        f"in {BASE_DIR}, or any CSV with 'credit' and 'card' in its name in "
        f"{BASE_DIR}, {BASE_DIR.parent}, or {BASE_DIR.parent / 'data'}. "
        "Place the dataset in one of those folders, or edit RAW_PATH in "
        "data_prep.py to point at it directly."
    )


# Full path to the raw CSV file, resolved dynamically so this works
# regardless of the exact filename or which machine/folder it runs in.
RAW_PATH = find_raw_csv()


# ---------------------------------------------------------------------------
# Helper function: _fix_grouped_decimal
# ---------------------------------------------------------------------------
def _fix_grouped_decimal(value):
    """Recover a float whose decimal point was replaced by grouping dots.

    Example: '1.075.675.276.433.290' -> 1.075675276433290
    Returns np.nan for values that cannot be safely recovered (e.g. the
    comma/scientific-notation corruption, which is flagged as invalid data).
    """
    # --- Handle missing input immediately ---
    if pd.isna(value):
        return np.nan

    # --- Convert to string and strip whitespace ---
    s = str(value).strip()

    # --- First attempt: try direct float conversion (already clean) ---
    try:
        return float(s)
    except ValueError:
        pass  # Not a simple float, continue to recovery logic

    # --- Detect European scientific notation or comma decimal separator ---
    # These are considered corrupted and are discarded as NaN.
    if "e" in s.lower() or "," in s:
        return np.nan

    # --- Handle negative sign ---
    neg = s.startswith("-")
    body = s[1:] if neg else s

    # --- Remove all dots to get the raw digit string ---
    digits = body.replace(".", "")

    # --- Validate that the remaining string is purely digits ---
    if not digits.isdigit():
        return np.nan

    # --- Reconstruct the float by placing a decimal after the first digit ---
    recovered = float(digits[0] + "." + digits[1:]) if len(digits) > 1 else float(digits)

    # --- Re-apply negative sign if needed ---
    recovered = -recovered if neg else recovered

    # --- Sanity check: risk_leak is documented to lie in [-0.39, 1.40] ---
    # We use a slightly wider safety margin [-2.0, 2.0] to avoid discarding
    # legitimate values due to minor reconstruction inaccuracies.
    if not (-2.0 <= recovered <= 2.0):
        return np.nan

    return recovered


# ---------------------------------------------------------------------------
# Function: load_raw
# ---------------------------------------------------------------------------
def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """Read the semicolon-delimited raw CSV exactly as provided."""
    # The file uses semicolons as separators; low_memory=False avoids mixed-type warnings.
    df = pd.read_csv(path, sep=";", low_memory=False)
    return df


# ---------------------------------------------------------------------------
# Function: clean_dataset
# ---------------------------------------------------------------------------
def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply all documented cleaning steps and return (clean_df, report).

    `report` is a dict capturing before/after statistics used for the
    Data Preparation section of the report (Task 2).
    """
    # Work on a copy to avoid mutating the original DataFrame
    df = df.copy()
    report = {}

    # -----------------------------------------------------------------------
    # Step 1: Fix the digit-grouping corruption in risk_leak
    # -----------------------------------------------------------------------
    # Count how many entries are not clean floats before fixing.
    before_bad = df["risk_leak"].apply(
        lambda x: False if _is_clean_float(x) else True
    ).sum()

    # Apply the recovery function to each value in risk_leak.
    df["risk_leak"] = df["risk_leak"].apply(_fix_grouped_decimal)

    # Count how many values became NaN (unrecoverable or out-of-range).
    after_missing = df["risk_leak"].isna().sum()

    # Record statistics for the report.
    report["risk_leak_corrupted_strings_found"] = int(before_bad)
    report["risk_leak_unrecoverable_set_to_nan"] = int(after_missing)

    # -----------------------------------------------------------------------
    # Step 2: Recompute BILL_AMT_SUM deterministically
    # -----------------------------------------------------------------------
    # The sum of BILL_AMT1..6 should equal BILL_AMT_SUM. We recompute it
    # to correct any inconsistencies.
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    recomputed_sum = df[bill_cols].sum(axis=1)

    # Identify rows where the original sum differs by more than 1.0.
    mismatch = (df["BILL_AMT_SUM"] - recomputed_sum).abs() > 1.0
    report["bill_amt_sum_mismatches_fixed"] = int(mismatch.sum())

    # Overwrite BILL_AMT_SUM with the correct recomputed sum.
    df["BILL_AMT_SUM"] = recomputed_sum

    # -----------------------------------------------------------------------
    # Step 3: Recompute LIMIT_BAL_LOG deterministically
    # -----------------------------------------------------------------------
    # LIMIT_BAL_LOG is a pure transform of LIMIT_BAL: log(LIMIT_BAL).
    # We avoid guessing decimal placement by simply recomputing it later
    # after LIMIT_BAL has been imputed.
    report["limit_bal_log_recomputed_rows"] = int(df["LIMIT_BAL_LOG"].notna().sum())

    # Temporarily set to NaN; will be recomputed after LIMIT_BAL imputation.
    df["LIMIT_BAL_LOG"] = np.nan

    # -----------------------------------------------------------------------
    # Step 4: Handle genuine missing values
    # -----------------------------------------------------------------------
    # Capture missing counts before imputation.
    missing_before = df.isna().sum()

    # --- Numeric columns: median imputation (robust to skew/outliers) ---
    for col in ["LIMIT_BAL", "AGE", "PAY_AMT1", "PAY_AMT2"]:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # --- Categorical-coded numeric columns: mode imputation ---
    for col in ["SEX", "EDUCATION", "MARRIAGE"]:
        mode_val = df[col].mode(dropna=True)[0]
        df[col] = df[col].fillna(mode_val)

    # --- risk_leak: median imputation for the small number of unrecoverable
    #     corrupted entries ---
    df["risk_leak"] = df["risk_leak"].fillna(df["risk_leak"].median())

    # Recompute LIMIT_BAL_LOG now that LIMIT_BAL has no missing values.
    # clip(lower=1) avoids log(0) errors.
    df["LIMIT_BAL_LOG"] = np.log(df["LIMIT_BAL"].clip(lower=1))

    # Capture missing counts after imputation.
    missing_after = df.isna().sum()
    report["missing_before"] = missing_before[missing_before > 0].to_dict()
    report["missing_after"] = int(missing_after.sum())

    # -----------------------------------------------------------------------
    # Step 5: Tidy categorical codes
    # -----------------------------------------------------------------------
    # EDUCATION values 0, 5, 6 are undocumented/"unknown" categories in the
    # original UCI credit-card schema -> collapse into a single "Other" (4).
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

    # MARRIAGE value 0 is undocumented -> collapse into "Other" (3).
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    # Record final shape for the report.
    report["final_shape"] = df.shape

    return df, report


# ---------------------------------------------------------------------------
# Helper function: _is_clean_float
# ---------------------------------------------------------------------------
def _is_clean_float(x):
    """Return True if x can be directly converted to a float without error."""
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Main block: execute when the script is run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Load raw data using the resolved path
    raw = load_raw()

    # Apply all cleaning steps
    clean, rep = clean_dataset(raw)

    # Print a concise summary of the cleaning process
    print("Raw shape:", raw.shape)
    print("Clean shape:", clean.shape)
    print("Remaining missing values:", clean.isna().sum().sum())

    # Print the detailed report dictionary
    for k, v in rep.items():
        print(f"{k}: {v}")