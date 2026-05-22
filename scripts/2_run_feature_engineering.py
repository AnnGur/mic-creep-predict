"""
Feature Engineering — K. pneumoniae + Meropenem MIC Creep
Vivli AMR Surveillance Challenge 2026

Reads ATLAS raw data, builds feature matrix, time-splits, saves parquet files.

Run:
    python scripts/run_feature_engineering.py
    python scripts/run_feature_engineering.py --data-dir data/raw --out data/processed
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.loader import ATLASLoader
from data.preprocessor import MICPreprocessor
from features.engineer import (
    CARBAPENEMASE_GENES,
    SPECIMEN_KEYWORDS,
    TRAIN_END,
    TEST_START,
    TEST_END,
    build_features,
    build_target,
    map_specimen,
    run_pipeline,
    time_split,
)

SPECIES    = "Klebsiella pneumoniae"
ANTIBIOTIC = "Meropenem"
EUCAST_R   = 8

WOUND_SOURCES = {"Wound", "Abscess", "Skin and Soft Tissue"}
AGE_GROUP_BROAD = {
    "0 - 17":  "Paediatric (0-17)",
    "18 - 30": "Adult (18-60)",
    "31 - 60": "Adult (18-60)",
    "61+":     "Elderly (61+)",
}

sns.set_style("whitegrid")
plt.rcParams.update({"figure.figsize": (12, 5), "figure.dpi": 100})


# ---------------------------------------------------------------------------
# Data loading (same as EDA — reused so both scripts share identical parsing)
# ---------------------------------------------------------------------------

def load_and_parse(data_dir: Path) -> pd.DataFrame:
    loader = ATLASLoader(data_dir)
    df = loader.load(SPECIES, antibiotic=ANTIBIOTIC)

    parsed = df[ANTIBIOTIC].apply(
        lambda x: MICPreprocessor.parse_censored_mic(x) if pd.notna(x) else (None, None)
    )
    df["mic_value"]    = parsed.apply(lambda t: t[0])
    df["mic_operator"] = parsed.apply(lambda t: t[1])
    df = df[df["mic_value"].notna() & (df["mic_value"] > 0)].copy()

    df["mic_log2"]     = np.log2(df["mic_value"])
    df["is_censored"]  = df["mic_operator"].isin([">", "<", ">=", "<="])
    df["is_resistant"] = df["mic_value"] >= EUCAST_R

    df["age_group_broad"] = df["Age Group"].map(AGE_GROUP_BROAD).fillna("Adult (18-60)")
    df["military_proxy"]  = (
        df["Source"].isin(WOUND_SOURCES)
        & (df["Gender"] == "Male")
        & (df["Age Group"].isin(["18 - 30", "31 - 60"]))
    )

    gene_cols_present = [g for g in CARBAPENEMASE_GENES if g in df.columns]
    for gene in gene_cols_present:
        s = df[gene].astype(str).str.strip()
        df[f"{gene}_pos"] = df[gene].notna() & ~s.isin(["", "nan", "0", "None"])

    print(f"  Loaded: {len(df):,} rows | {df['Year'].min()}–{df['Year'].max()}")
    return df


# ---------------------------------------------------------------------------
# Section 1 — Specimen type distribution
# ---------------------------------------------------------------------------

def print_specimen_mapping(df: pd.DataFrame) -> None:
    df = df.copy()
    df["specimen_broad"] = df["Source"].apply(map_specimen)
    counts = df["specimen_broad"].value_counts()
    print(f"\n  {'Category':<15} {'N':>7}  {'%':>5}")
    print("  " + "-" * 30)
    for cat, n in counts.items():
        print(f"  {cat:<15} {n:>7,}  {n/len(df)*100:>5.1f}%")


# ---------------------------------------------------------------------------
# Section 2 — Feature matrix summary
# ---------------------------------------------------------------------------

def print_feature_summary(X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    print(f"\n  Feature matrix: {X_train.shape[1]} columns")
    print(f"  Train rows:     {len(X_train):,}")
    print(f"  Test rows:      {len(X_test):,}")

    core = ["year", "gender_male", "age_paediatric", "age_elderly",
            "military_proxy", "is_censored", "pct_censored_year"]
    spec_cols  = [c for c in X_train.columns if c.startswith("spec_")]
    ctry_cols  = [c for c in X_train.columns if c.startswith("ctry_")]
    gene_cols  = [c for c in X_train.columns if c.endswith("_pos")]

    print(f"\n  Core features ({len(core)}):       {core}")
    print(f"  Specimen dummies ({len(spec_cols)}):    {spec_cols}")
    print(f"  Gene flags ({len(gene_cols)}):         {gene_cols}")
    print(f"  Country dummies ({len(ctry_cols)}): {len(ctry_cols)} columns (one per country)")


# ---------------------------------------------------------------------------
# Section 3 — Censoring rate by year (validation plot)
# ---------------------------------------------------------------------------

def plot_censoring_by_split(df: pd.DataFrame, out: Path) -> None:
    q = df.groupby("Year").agg(
        pct_censored=("is_censored", "mean"),
    ).reset_index()
    q["pct_censored"] *= 100

    fig, ax = plt.subplots()
    train_q = q[q["Year"] <= TRAIN_END]
    test_q  = q[(q["Year"] >= TEST_START) & (q["Year"] <= TEST_END)]

    ax.bar(train_q["Year"], train_q["pct_censored"], color="#1f77b4", alpha=0.7, label="Train")
    ax.bar(test_q["Year"],  test_q["pct_censored"],  color="#d62728", alpha=0.7, label="Test")
    ax.set_ylabel("% Censored MIC values", fontsize=11)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_title("Censoring Rate by Year — Train vs Test Split", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fname = "censoring_by_split.png"
    fig.savefig(out / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {fname}")


# ---------------------------------------------------------------------------
# Section 4 — Target distribution by split
# ---------------------------------------------------------------------------

def plot_target_distribution(train_df: pd.DataFrame, test_df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, subset, label, color in [
        (axes[0], train_df, f"Train ({df['Year'].min()}-{TRAIN_END})", "#1f77b4"),
        (axes[1], test_df,  f"Test ({TEST_START}-{TEST_END})",        "#d62728"),
    ]:
        ax.hist(subset["mic_log2"], bins=40, color=color, alpha=0.8, edgecolor="white", lw=0.3)
        ax.axvline(np.log2(EUCAST_R), color="black", linestyle="--", lw=1.5,
                   label=f"R breakpoint (log2={np.log2(EUCAST_R):.0f})")
        ax.set_xlabel("log2(MIC)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)

    plt.suptitle("Target Distribution (log2 MIC) — Train vs Test",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fname = "target_distribution.png"
    fig.savefig(out / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {fname}")


# ---------------------------------------------------------------------------
# Section 5 — Feature correlation heatmap (core features only)
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(X_train: pd.DataFrame, y_train: pd.Series, out: Path) -> None:
    core = ["year", "gender_male", "age_paediatric", "age_elderly",
            "military_proxy", "is_censored", "pct_censored_year"]
    gene_cols = [c for c in X_train.columns if c.endswith("_pos")]
    cols = core + gene_cols

    subset = X_train[cols].copy()
    subset["log2_mic"] = y_train.values

    corr = subset.corr()

    fig, ax = plt.subplots(figsize=(len(cols) * 0.7 + 2, len(cols) * 0.7 + 2))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, ax=ax, square=True, linewidths=0.5,
                annot_kws={"size": 7})
    ax.set_title("Feature Correlation (core + gene flags) — Train set",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fname = "feature_correlation.png"
    fig.savefig(out / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {fname}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature engineering pipeline — K. pneumoniae Meropenem MIC Creep"
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Directory containing atlas_vivli_2004_2024.csv",
    )
    parser.add_argument(
        "--out", type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Output directory for parquet files (default: data/processed/)",
    )
    parser.add_argument(
        "--reports", type=Path,
        default=PROJECT_ROOT / "reports",
        help="Output directory for diagnostic charts (default: reports/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.reports.mkdir(parents=True, exist_ok=True)
    n = 6

    print(f"[1/{n}] Load & parse ATLAS data")
    global df
    df = load_and_parse(args.data_dir)

    print(f"\n[2/{n}] Specimen type mapping")
    print_specimen_mapping(df)

    print(f"\n[3/{n}] Build features + time split")
    train_df, test_df = time_split(df)
    X_train = build_features(train_df)
    y_train = build_target(train_df)
    X_test  = build_features(test_df)
    y_test  = build_target(test_df)
    print_feature_summary(X_train, X_test)

    print(f"\n[4/{n}] Censoring rate by split")
    plot_censoring_by_split(df, args.reports)

    print(f"\n[5/{n}] Target distribution")
    plot_target_distribution(train_df, test_df, args.reports)

    print(f"\n[6/{n}] Feature correlation heatmap")
    plot_correlation_heatmap(X_train, y_train, args.reports)

    print(f"\nSaving processed data to: {args.out}")
    run_pipeline(df, args.out)

    print("\nDone.")


if __name__ == "__main__":
    main()
