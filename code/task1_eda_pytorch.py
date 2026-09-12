"""
task1_eda_pytorch.py - Exploratory Data Analysis (Task 1, 20%)
PyTorch variant: uses torch tensors for the numerical computations
(descriptive statistics and correlations) instead of pandas/numpy,
demonstrating the tensor-based workflow covered in the Introduction to AI
module. Plotting still uses matplotlib/seaborn (PyTorch has no plotting
facility of its own).

Outputs are written to their own dedicated folders (figures_pytorch/ and
outputs_pytorch/) rather than the main figures/outputs/ folders used by
task1_eda.py, so the two versions never overwrite one another and can be
compared side by side.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from data_prep import load_raw, clean_dataset, ensure_output_dirs, FIGURES_PYTORCH_DIR, OUTPUTS_PYTORCH_DIR

sns.set_style("whitegrid")

# Resolve paths relative to this script's own location, not the current
# working directory, so it works whether you run it from inside code/ or
# via a full path / VS Code's "Run Python File" button from anywhere else,
# on any machine.
BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = str(FIGURES_PYTORCH_DIR)
OUT_DIR = str(OUTPUTS_PYTORCH_DIR)
TARGET = "default.payment.next.month"

# Create every project output directory (figures/outputs and their pytorch
# siblings) if they do not exist yet - safe on a completely fresh checkout.
ensure_output_dirs()


def run():
    raw = load_raw()
    df, _ = clean_dataset(raw)
    results = {}

    # ============ 1. Target distribution / class imbalance ============
    counts = df[TARGET].value_counts().sort_index()
    pct = (counts / counts.sum() * 100).round(2)
    results["target_counts"] = counts.to_dict()
    results["target_pct"] = pct.to_dict()

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x=TARGET, data=df, ax=ax, hue=TARGET, palette=["#2E7D6B", "#C1440E"], legend=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No Default (0)", "Default (1)"])
    ax.set_title("Distribution of Target Variable")
    for i, v in enumerate(counts):
        ax.text(i, v + 300, f"{v}\n({pct.iloc[i]}%)", ha="center")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/01_target_distribution.png", dpi=150)
    plt.close()

    # ============ 2. Descriptive statistics using PyTorch ============
    # Note: torch.median() (unlike pandas/numpy) returns the LOWER of the
    # two middle values for an even-sized tensor rather than their average,
    # which would silently disagree with the pandas-based figures already
    # reported elsewhere. torch.quantile(..., 0.5) performs the standard
    # linear-interpolation median instead, matching pandas' definition.
    key_num = ["LIMIT_BAL", "AGE", "BILL_AMT_SUM", "LIMIT_BAL_LOG", "risk_leak"]
    torch_data = torch.tensor(df[key_num].values, dtype=torch.float64)
    mean = torch.mean(torch_data, dim=0)
    median = torch.quantile(torch_data, 0.5, dim=0)
    std = torch.std(torch_data, dim=0, unbiased=True)
    min_val = torch.min(torch_data, dim=0).values
    max_val = torch.max(torch_data, dim=0).values

    desc = pd.DataFrame({
        "mean": mean.numpy(),
        "median": median.numpy(),
        "std": std.numpy(),
        "min": min_val.numpy(),
        "max": max_val.numpy(),
    }, index=key_num)
    results["descriptive_stats"] = desc.round(2).to_dict(orient="index")
    desc.round(2).to_csv(f"{OUT_DIR}/task1_descriptive_stats.csv")

    # ============ 3. Histograms ============
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, key_num):
        sns.histplot(df[col], kde=True, ax=ax, color="#2E7D6B")
        ax.set_title(f"Distribution of {col}")
    axes.flat[-1].axis("off")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/02_histograms_numeric.png", dpi=150)
    plt.close()

    # ============ 4. Boxplots by target ============
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, ["LIMIT_BAL", "BILL_AMT_SUM", "AGE"]):
        sns.boxplot(x=TARGET, y=col, data=df, ax=ax, hue=TARGET, palette=["#2E7D6B", "#C1440E"], legend=False)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No Default", "Default"])
        ax.set_title(f"{col} by Default Status")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/03_boxplots_by_target.png", dpi=150)
    plt.close()

    # ============ 5. Categorical distributions ============
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
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

    # CITY top 10
    fig, ax = plt.subplots(figsize=(10, 5))
    top_cities = df["CITY"].value_counts().head(10)
    sns.barplot(x=top_cities.values, y=top_cities.index, ax=ax, hue=top_cities.index, palette="crest", legend=False)
    ax.set_title("Top 10 Most Frequent CITY Categories")
    ax.set_xlabel("Count")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/05_city_top10.png", dpi=150)
    plt.close()

    # ============ 6. Correlation with target (PyTorch) ============
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    torch_num = torch.tensor(df[num_cols].values, dtype=torch.float64)
    corr_matrix = torch.corrcoef(torch_num.T)
    target_idx = num_cols.index(TARGET)
    corr_values = corr_matrix[target_idx, :].numpy()
    corr_series = pd.Series(corr_values, index=num_cols).drop(TARGET)
    corr_series = corr_series.sort_values(key=abs, ascending=False)
    results["target_correlations"] = corr_series.round(3).to_dict()
    corr_series.round(3).to_csv(f"{OUT_DIR}/task1_target_correlations.csv")

    fig, ax = plt.subplots(figsize=(8, 9))
    corr_series.plot(kind="barh", ax=ax,
                      color=["#C1440E" if v > 0 else "#2E7D6B" for v in corr_series.values])
    ax.set_title("Correlation of Numeric Features with Default (target)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/06_correlation_with_target.png", dpi=150)
    plt.close()

    # ============ 7. Correlation heatmap (PyTorch) ============
    heat_cols = ["LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "BILL_AMT_SUM",
                 "PAY_AMT1", "risk_leak", "RISK_RATING", "LIMIT_BAL_LOG", TARGET]
    torch_heat = torch.tensor(df[heat_cols].values, dtype=torch.float64)
    corr_heat = torch.corrcoef(torch_heat.T).numpy()
    corr_heat_df = pd.DataFrame(corr_heat, index=heat_cols, columns=heat_cols)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_heat_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Correlation Heatmap - Key Features")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/07_correlation_heatmap.png", dpi=150)
    plt.close()

    # ============ 8. Default rates by categorical levels ============
    fig, ax = plt.subplots(figsize=(9, 5))
    default_rate_pay0 = df.groupby("PAY_0")[TARGET].mean()
    default_rate_pay0.plot(kind="bar", ax=ax, color="#2E7D6B")
    ax.set_title("Default Rate by PAY_0 (Most Recent Repayment Status)")
    ax.set_ylabel("Default Rate")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/08_default_rate_by_pay0.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    df.groupby("RISK_RATING")[TARGET].mean().plot(kind="bar", ax=ax, color="#C1440E")
    ax.set_title("Default Rate by RISK_RATING")
    ax.set_ylabel("Default Rate")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/09_default_rate_by_risk_rating.png", dpi=150)
    plt.close()

    # Save results
    with open(f"{OUT_DIR}/task1_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("Task 1 EDA complete (PyTorch version).")
    print(f"Figures written to: {FIG_DIR}")
    print(f"Result files written to: {OUT_DIR}")
    print("Target distribution:", results["target_counts"], results["target_pct"])
    print("\nTop correlations with target:\n", corr_series.head(8))
    return results


if __name__ == "__main__":
    run()
