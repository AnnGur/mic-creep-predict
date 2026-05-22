"""Feature engineering for MIC creep prediction."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

CARBAPENEMASE_GENES = ["KPC", "NDM", "OXA", "VIM", "IMP", "GES"]

# Mapping from ATLAS "Source" free-text values to 5 broad specimen categories.
# Matched by substring (case-insensitive); first match wins.
SPECIMEN_KEYWORDS = {
    "wound":       ["wound", "abscess", "skin and soft", "skin", "soft tissue"],
    "blood":       ["blood", "serum"],
    "respiratory": ["bronch", "lavage", "sputum", "respiratory", "tracheal",
                    "endobronch", "thoracentesis", "pleural"],
    "urine":       ["urine", "urinary", "bladder", "catheter urine"],
    "peritoneal":  ["periton", "abdomin", "ascit", "gall"],
}

TRAIN_END   = 2018
TEST_START  = 2019
TEST_END    = 2022


def map_specimen(source: str) -> str:
    if pd.isna(source):
        return "other"
    s = str(source).lower()
    for category, keywords in SPECIMEN_KEYWORDS.items():
        if any(kw in s for kw in keywords):
            return category
    return "other"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from parsed ATLAS dataframe.

    Input df must have columns produced by load_and_parse():
      Year, Gender, Age Group, Source, Country, military_proxy,
      is_censored, mic_log2, and optional {gene}_pos columns.

    Returns feature matrix X (all float/int, no target column).
    """
    X = pd.DataFrame(index=df.index)

    # --- Temporal ---
    X["year"] = df["Year"].astype(int)

    # --- Demographics ---
    X["gender_male"] = (df["Gender"] == "Male").astype(int)
    X["age_paediatric"] = (df["Age Group"] == "0 - 17").astype(int)
    X["age_elderly"]    = (df["Age Group"] == "61+").astype(int)
    # adult (18-60) is the reference category — no column needed

    X["military_proxy"] = df["military_proxy"].astype(int)

    # --- Specimen type (OHE, "other" is reference) ---
    spec = df["Source"].apply(map_specimen)
    spec_dummies = pd.get_dummies(spec, prefix="spec")
    spec_dummies = spec_dummies.drop(columns=["spec_other"], errors="ignore")
    X = pd.concat([X, spec_dummies.set_index(df.index)], axis=1)

    # --- Country (OHE, drop first to avoid multicollinearity) ---
    country_dummies = pd.get_dummies(df["Country"], prefix="ctry", drop_first=True)
    X = pd.concat([X, country_dummies.set_index(df.index)], axis=1)

    # --- Resistance gene flags ---
    for gene in CARBAPENEMASE_GENES:
        col = f"{gene}_pos"
        if col in df.columns:
            X[col] = df[col].astype(int)

    # --- Data quality / censoring ---
    X["is_censored"] = df["is_censored"].astype(int)
    yearly_cens = df.groupby("Year")["is_censored"].mean()
    X["pct_censored_year"] = df["Year"].map(yearly_cens).values

    # Ensure all columns are numeric
    X = X.astype(float)

    return X


def build_target(df: pd.DataFrame) -> pd.Series:
    """Return log2(MIC) as the regression target."""
    return df["mic_log2"].rename("target")


def time_split(
    df: pd.DataFrame,
    train_end: int = TRAIN_END,
    test_start: int = TEST_START,
    test_end: int = TEST_END,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe chronologically — never shuffle.

    Returns (train_df, test_df).
    """
    train = df[df["Year"] <= train_end].copy()
    test  = df[(df["Year"] >= test_start) & (df["Year"] <= test_end)].copy()
    return train, test


def run_pipeline(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Full feature engineering pipeline:
      1. Time-split
      2. Build X, y for train and test
      3. Save to out_dir as parquet

    Never shuffles data.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = time_split(df)

    print(f"  Train: {len(train_df):,} rows  ({df['Year'].min()}–{TRAIN_END})")
    print(f"  Test:  {len(test_df):,} rows  ({TEST_START}–{TEST_END})")

    X_train = build_features(train_df)
    y_train = build_target(train_df)
    X_test  = build_features(test_df)
    y_test  = build_target(test_df)

    print(f"  Feature matrix shape: {X_train.shape[1]} columns")
    print(f"  Features: {list(X_train.columns[:10])} ...")

    X_train.to_parquet(out_dir / "X_train.parquet")
    y_train.to_frame().to_parquet(out_dir / "y_train.parquet")
    X_test.to_parquet(out_dir / "X_test.parquet")
    y_test.to_frame().to_parquet(out_dir / "y_test.parquet")

    print(f"  Saved to: {out_dir}")
