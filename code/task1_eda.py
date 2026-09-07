"""
task1_eda.py - Exploratory Data Analysis (Task 1, 20%)
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from data_prep import load_raw, clean_dataset

sns.set_style("whitegrid")
FIG_DIR = "../figures"
OUT_DIR = "../outputs"
TARGET = "default.payment.next.month"


def run():
    raw = load_raw()
    df, _ = clean_dataset(raw)
    results = {}

    # 1. Target distribution / class imbalance
    counts = df[TARGET].value_counts().sort_index()
    pct = (counts / counts.sum() * 100).round(2)
    results["target_counts"] = counts.to_dict()
    results["target_pct"] = pct.to_dict()

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x=TARGET, data=df, ax=ax, palette=["#2E7D6B", "#C1440E"])
    ax.set_xticklabels(["No Default (0)", "Default (1)"])
    ax.set_title("Distribution of Target Variable")
    for i, v in enumerate(counts):
        ax.text(i, v + 300, f"{v}\n({pct.iloc[i]}%)", ha="center")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/01_target_distribution.png", dpi=150)
    plt.close()

    # 2. Descriptive statistics for key numeric features
    key_num = ["LIMIT_BAL", "AGE", "BILL_AMT_SUM", "LIMIT_BAL_LOG", "risk_leak"]
    desc = df[key_num].describe().T[["mean", "50%", "std", "min", "max"]]
    desc.columns = ["mean", "median", "std", "min", "max"]
    results["descriptive_stats"] = desc.round(2).to_dict(orient="index")
    desc.round(2).to_csv(f"{OUT_DIR}/task1_descriptive_stats.csv")

    # 3. Histograms for key numeric features
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, key_num):
        sns.histplot(df[col], kde=True, ax=ax, color="#2E7D6B")
        ax.set_title(f"Distribution of {col}")
    axes.flat[-1].axis("off")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/02_histograms_numeric.png", dpi=150)
    plt.close()

    # 4. Boxplots (outlier detection) - by default status
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, ["LIMIT_BAL", "BILL_AMT_SUM", "AGE"]):
        sns.boxplot(x=TARGET, y=col, data=df, ax=ax, palette=["#2E7D6B", "#C1440E"])
        ax.set_xticklabels(["No Default", "Default"])
        ax.set_title(f"{col} by Default Status")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/03_boxplots_by_target.png", dpi=150)
    plt.close()

    # 5. Categorical feature distributions
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    sns.countplot(x="SEX", data=df, ax=axes[0, 0], palette="viridis")
    axes[0, 0].set_title("Sex (1=Male, 2=Female)")
    sns.countplot(x="EDUCATION", data=df, ax=axes[0, 1], palette="viridis")
    axes[0, 1].set_title("Education Level")
    sns.countplot(x="MARRIAGE", data=df, ax=axes[1, 0], palette="viridis")
    axes[1, 0].set_title("Marital Status")
    sns.countplot(x="RISK_RATING", data=df, ax=axes[1, 1], palette="viridis")
    axes[1, 1].set_title("Risk Rating (1=Low, 2=Medium, 3=High)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/04_categorical_distributions.png", dpi=150)
    plt.close()

    # CITY has 50 categories -> top 10 by frequency
    fig, ax = plt.subplots(figsize=(10, 5))
    top_cities = df["CITY"].value_counts().head(10)
    sns.barplot(x=top_cities.values, y=top_cities.index, ax=ax, palette="crest")
    ax.set_title("Top 10 Most Frequent CITY Categories")
    ax.set_xlabel("Count")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/05_city_top10.png", dpi=150)
    plt.close()

    # 6. Correlation with target
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[num_cols].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    results["target_correlations"] = corr.round(3).to_dict()
    corr.round(3).to_csv(f"{OUT_DIR}/task1_target_correlations.csv")

    fig, ax = plt.subplots(figsize=(8, 9))
    corr.plot(kind="barh", ax=ax, color=["#C1440E" if v > 0 else "#2E7D6B" for v in corr.values])
    ax.set_title("Correlation of Numeric Features with Default (target)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/06_correlation_with_target.png", dpi=150)
    plt.close()

    # 7. Full correlation heatmap (subset of most relevant vars)
    heat_cols = ["LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "BILL_AMT_SUM",
                 "PAY_AMT1", "risk_leak", "RISK_RATING", "LIMIT_BAL_LOG", TARGET]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[heat_cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Correlation Heatmap - Key Features")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/07_correlation_heatmap.png", dpi=150)
    plt.close()

    # 8. PAY_0..PAY_6 vs default rate
    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    fig, ax = plt.subplots(figsize=(9, 5))
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

    with open(f"{OUT_DIR}/task1_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("Task 1 EDA complete.")
    print("Target distribution:", results["target_counts"], results["target_pct"])
    print("\nTop correlations with target:\n", corr.head(8))
    return results


if __name__ == "__main__":
    run()
