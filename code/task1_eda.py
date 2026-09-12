
# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import json                     # For saving results as JSON
import numpy as np              # Numerical operations
import pandas as pd             # Data manipulation
import matplotlib               # Plotting library
matplotlib.use("Agg")           # Use non-interactive backend (for headless environments)
import matplotlib.pyplot as plt # Plotting interface
import seaborn as sns           # Statistical data visualization
from pathlib import Path        # Object-oriented filesystem paths

# Import cleaning functions from the data_prep module
from data_prep import load_raw, clean_dataset, ensure_output_dirs, FIGURES_DIR, OUTPUTS_DIR

# ---------------------------------------------------------------------------
# Global settings and path configuration
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")      # Set a clean white grid style for plots

# Resolve paths relative to this script's location (works on any machine or
# folder location, since it's derived from __file__, not the cwd).
BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = str(FIGURES_DIR)   # Directory for saving figures
OUT_DIR = str(OUTPUTS_DIR)   # Directory for saving outputs

# Create figures/ and outputs/ (and their siblings) if they don't exist yet,
# so this script runs on a completely fresh checkout with no manual setup.
ensure_output_dirs()

# Target column name for the classification task
TARGET = "default.payment.next.month"


# ---------------------------------------------------------------------------
# Main function: run
# ---------------------------------------------------------------------------
def run():
    # Load raw data and apply cleaning steps
    raw = load_raw()
    df, _ = clean_dataset(raw)

    # Dictionary to accumulate results for later JSON export
    results = {}

    # -----------------------------------------------------------------------
    # 1. Target distribution / class imbalance
    # -----------------------------------------------------------------------
    # Count occurrences of each class in the target variable
    counts = df[TARGET].value_counts().sort_index()
    # Calculate percentages
    pct = (counts / counts.sum() * 100).round(2)

    # Store in results dict
    results["target_counts"] = counts.to_dict()
    results["target_pct"] = pct.to_dict()

    # Create a count plot of the target distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x=TARGET, data=df, ax=ax, hue=TARGET, palette=["#2E7D6B", "#C1440E"], legend=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No Default (0)", "Default (1)"])
    ax.set_title("Distribution of Target Variable")
    # Annotate each bar with count and percentage
    for i, v in enumerate(counts):
        ax.text(i, v + 300, f"{v}\n({pct.iloc[i]}%)", ha="center")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/01_target_distribution.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------------------
    # 2. Descriptive statistics for key numeric features
    # -----------------------------------------------------------------------
    # List of key numeric columns to describe
    key_num = ["LIMIT_BAL", "AGE", "BILL_AMT_SUM", "LIMIT_BAL_LOG", "risk_leak"]
    # Compute descriptive statistics and transpose for readability
    desc = df[key_num].describe().T[["mean", "50%", "std", "min", "max"]]
    desc.columns = ["mean", "median", "std", "min", "max"]  # Rename for clarity
    results["descriptive_stats"] = desc.round(2).to_dict(orient="index")
    # Save to CSV
    desc.round(2).to_csv(f"{OUT_DIR}/task1_descriptive_stats.csv")

    # -----------------------------------------------------------------------
    # 3. Histograms for key numeric features
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    # Iterate over the first 5 subplots and the key numeric columns
    for ax, col in zip(axes.flat, key_num):
        sns.histplot(df[col], kde=True, ax=ax, color="#2E7D6B")
        ax.set_title(f"Distribution of {col}")
    # Turn off the unused 6th subplot
    axes.flat[-1].axis("off")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/02_histograms_numeric.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------------------
    # 4. Boxplots (outlier detection) - by default status
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Create boxplots for three numeric columns, split by target
    for ax, col in zip(axes, ["LIMIT_BAL", "BILL_AMT_SUM", "AGE"]):
        sns.boxplot(x=TARGET, y=col, data=df, ax=ax, hue=TARGET, palette=["#2E7D6B", "#C1440E"], legend=False)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No Default", "Default"])
        ax.set_title(f"{col} by Default Status")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/03_boxplots_by_target.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------------------
    # 5. Categorical feature distributions
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    # Plot countplots for categorical features
    sns.countplot(x="SEX", data=df, ax=axes[0, 0], hue="SEX", palette="viridis", legend=False)
    axes[0, 0].set_title("Sex (1=Male, 2=Female)")
    sns.countplot(x="EDUCATION", data=df, ax=axes[0, 1], hue="EDUCATION", palette="viridis", legend=False)
    axes[0, 1].set_title("Education Level")
    sns.countplot(x="MARRIAGE", data=df, ax=axes[1, 0], hue="MARRIAGE", palette="viridis", legend=False)
    axes[1, 0].set_title("Marital Status")
    sns.countplot(x="RISK_RATING", data=df, ax=axes[1, 1], hue="RISK_RATING", palette="viridis", legend=False)
    axes[1, 1].set_title("Risk Rating (1=Low, 2=Medium, 3=High)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/04_categorical_distributions.png", dpi=150)
    plt.close()

    # CITY has 50 categories -> top 10 by frequency
    fig, ax = plt.subplots(figsize=(10, 5))
    top_cities = df["CITY"].value_counts().head(10)
    sns.barplot(x=top_cities.values, y=top_cities.index, ax=ax, hue=top_cities.index, palette="crest", legend=False)
    ax.set_title("Top 10 Most Frequent CITY Categories")
    ax.set_xlabel("Count")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/05_city_top10.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------------------
    # 6. Correlation with target
    # -----------------------------------------------------------------------
    # Select only numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Compute correlation of each numeric feature with the target
    corr = df[num_cols].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    results["target_correlations"] = corr.round(3).to_dict()
    corr.round(3).to_csv(f"{OUT_DIR}/task1_target_correlations.csv")

    # Plot correlations as horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 9))
    corr.plot(kind="barh", ax=ax, color=["#C1440E" if v > 0 else "#2E7D6B" for v in corr.values])
    ax.set_title("Correlation of Numeric Features with Default (target)")
    ax.invert_yaxis()  # Invert y-axis for better readability
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/06_correlation_with_target.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------------------
    # 7. Full correlation heatmap (subset of most relevant vars)
    # -----------------------------------------------------------------------
    heat_cols = ["LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "BILL_AMT_SUM",
                 "PAY_AMT1", "risk_leak", "RISK_RATING", "LIMIT_BAL_LOG", TARGET]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[heat_cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Correlation Heatmap - Key Features")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/07_correlation_heatmap.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------------------
    # 8. PAY_0..PAY_6 vs default rate
    # -----------------------------------------------------------------------
    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]  # (Not used later, but defined)
    fig, ax = plt.subplots(figsize=(9, 5))
    # Calculate default rate for each value of PAY_0 (most recent repayment status)
    default_rate_pay0 = df.groupby("PAY_0")[TARGET].mean()
    default_rate_pay0.plot(kind="bar", ax=ax, color="#2E7D6B")
    ax.set_title("Default Rate by PAY_0 (Most Recent Repayment Status)")
    ax.set_ylabel("Default Rate")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/08_default_rate_by_pay0.png", dpi=150)
    plt.close()

    # RISK_RATING vs default rate
    fig, ax = plt.subplots(figsize=(6, 5))
    df.groupby("RISK_RATING")[TARGET].mean().plot(kind="bar", ax=ax, color="#C1440E")
    ax.set_title("Default Rate by RISK_RATING")
    ax.set_ylabel("Default Rate")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/09_default_rate_by_risk_rating.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------------------
    # Save results to JSON and print summary
    # -----------------------------------------------------------------------
    with open(f"{OUT_DIR}/task1_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("Task 1 EDA complete.")
    print("Target distribution:", results["target_counts"], results["target_pct"])
    print("\nTop correlations with target:\n", corr.head(8))
    return results


# ---------------------------------------------------------------------------
# Main block: execute when the script is run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run()