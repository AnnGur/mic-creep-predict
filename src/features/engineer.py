"""Feature engineering for MIC creep prediction."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

from scipy.optimize import curve_fit

CARBAPENEMASE_GENES = ["KPC", "NDM", "OXA", "VIM", "IMP", "GES"]

# Ward/specialty mapping — ICU, Emergency, Surgical, Paediatric ward, and Clinic
# are clinically distinct from the reference group (general medicine + unknown).
WARD_MAP = {
    "Medicine ICU":            "icu",
    "Surgery ICU":             "icu",
    "Pediatric ICU":           "icu",
    "General Unspecified ICU": "icu",
    "Emergency Room":          "emergency",
    "Surgery General":         "surgical",
    "Pediatric General":       "paediatric_ward",
    "Clinic / Office":         "clinic",
    # Medicine General, Nursing Home, None Given, Other → reference "general"
}

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

TRAIN_START = 2008
TRAIN_END   = 2018
TEST_START  = 2019
TEST_END    = 2022


def get_keep_countries(df: pd.DataFrame, min_n: int = 50) -> set:
    """Return set of countries with at least min_n isolates (compute from training set only)."""
    counts = df["Country"].value_counts()
    return set(counts[counts >= min_n].index)


def map_ward(speciality: str) -> str:
    if pd.isna(speciality):
        return "general"
    return WARD_MAP.get(str(speciality), "general")


def map_specimen(source: str) -> str:
    if pd.isna(source):
        return "other"
    s = str(source).lower()
    for category, keywords in SPECIMEN_KEYWORDS.items():
        if any(kw in s for kw in keywords):
            return category
    return "other"


def compute_lag(df_full: pd.DataFrame, df_split: pd.DataFrame) -> pd.Series:
    """
    For each isolate in df_split, return the resistance rate in its country
    in the previous year (lag-1). Lookup is computed from df_full.

    Fallback chain: country + prev_year rate -> global prev_year rate -> global mean.
    """
    rate_dict   = df_full.groupby(["Country", "Year"])["is_resistant"].mean().to_dict()
    global_dict = df_full.groupby("Year")["is_resistant"].mean().to_dict()
    global_mean = float(df_full["is_resistant"].mean())

    vals = [
        rate_dict.get((ctry, yr - 1), global_dict.get(yr - 1, global_mean))
        for ctry, yr in zip(df_split["Country"], df_split["Year"])
    ]
    return pd.Series(vals, index=df_split.index, name="lag1_pct_resistant_ctry", dtype=float)


def build_lag_lookup(df_full: pd.DataFrame) -> dict:
    """
    Build a {country: {year_str: rate}} dict for API inference.
    Includes '_global' key for country-not-found fallback.
    """
    lookup: dict = {}
    for (ctry, yr), rate in df_full.groupby(["Country", "Year"])["is_resistant"].mean().items():
        lookup.setdefault(ctry, {})[str(int(yr))] = round(float(rate), 6)

    lookup["_global"] = {
        str(int(yr)): round(float(r), 6)
        for yr, r in df_full.groupby("Year")["is_resistant"].mean().items()
    }
    lookup["_global_mean"] = round(float(df_full["is_resistant"].mean()), 6)
    return lookup


def _logistic(t: np.ndarray, r: float, t0: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-r * (t - t0)))


def fit_gene_prevalence_curves(train_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """
    Fit logistic growth p(t) = 1/(1+exp(-r*(t-t0))) to each gene's annual prevalence
    using training-period data only. The resulting (r, t0) pairs can safely extrapolate
    to future years without leaking post-2018 information.

    Returns {gene: (r, t0)} for genes with enough data; omits others silently.
    """
    curves: Dict[str, Tuple[float, float]] = {}
    for gene in CARBAPENEMASE_GENES:
        col = f"{gene}_pos"
        if col not in train_df.columns:
            continue
        by_year = train_df.groupby("Year")[col].mean()
        t = by_year.index.values.astype(float)
        p = by_year.values
        if len(t) < 5 or p.max() < 0.01:
            continue
        try:
            popt, _ = curve_fit(
                _logistic, t, p,
                p0=[0.3, t.mean()],
                bounds=([-2.0, 1990.0], [2.0, 2035.0]),
                maxfev=5000,
            )
            r, t0 = float(popt[0]), float(popt[1])
            curves[gene] = (r, t0)
            print(f"    {gene}: r={r:.3f}  t0={t0:.1f}")
        except Exception:
            pass
    return curves


def add_projected_gene_prevalence(
    X: pd.DataFrame,
    df: pd.DataFrame,
    curves: Dict[str, Tuple[float, float]],
) -> pd.DataFrame:
    """
    Append one feature per fitted gene curve: proj_{gene}_prev.

    The value is the logistic-projected population-level gene prevalence for that
    isolate's year. It gives the model an extrapolable forward signal of how likely
    future isolates are to carry the gene - something tree models cannot derive from
    year alone. Per the natural-experiment null result, this feature should only
    strengthen predictions for carriers; do not interpret it as a country-wide MIC
    shift for non-carriers.

    Must be called with curves fitted on training data only.
    """
    if not curves:
        return X
    X = X.copy()
    years = df["Year"].values.astype(float)
    for gene, (r, t0) in curves.items():
        X[f"proj_{gene}_prev"] = _logistic(years, r, t0)
    return X


def build_features(
    df: pd.DataFrame,
    keep_countries: set = None,
    lag: "pd.Series | None" = None,
    gene_curves: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """
    Build feature matrix from parsed ATLAS dataframe.

    Input df must have columns produced by load_and_parse():
      Year, Gender, Age Group, Source, Country, military_proxy,
      is_censored, mic_log2, and optional {gene}_pos columns.

    keep_countries: set of country names to retain as individual OHE columns.
      Countries outside this set are collapsed to "Other". Compute from
      training data via get_keep_countries(); pass the same set for test.
      If None, all countries are encoded individually (legacy behaviour).

    lag: optional Series (same index as df) with lag1_pct_resistant_ctry values.
      Compute via compute_lag(df_full, df_split).

    gene_curves: optional dict of {gene: (r, t0)} from fit_gene_prevalence_curves().
      Must be fitted on training data only; applies to both train and test safely.

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

    # --- Ward / specialty (OHE, "general" is reference) ---
    if "Speciality" in df.columns:
        ward = df["Speciality"].apply(map_ward)
        ward_dummies = pd.get_dummies(ward, prefix="ward")
        ward_dummies = ward_dummies.drop(columns=["ward_general"], errors="ignore")
        X = pd.concat([X, ward_dummies.set_index(df.index)], axis=1)

    # --- Country (OHE) ---
    # Countries with fewer than min_n isolates in the training set are collapsed
    # to "ctry_Other" to reduce OHE noise from sparse entries.
    if keep_countries is not None:
        country_col = df["Country"].apply(
            lambda c: c if c in keep_countries else "Other"
        )
    else:
        country_col = df["Country"]
    country_dummies = pd.get_dummies(country_col, prefix="ctry", drop_first=True)
    X = pd.concat([X, country_dummies.set_index(df.index)], axis=1)

    # --- Resistance gene flags ---
    for gene in CARBAPENEMASE_GENES:
        col = f"{gene}_pos"
        if col in df.columns:
            X[col] = df[col].astype(int)

    # --- Data quality / censoring ---
    # is_censored dropped: it is collinear with low-MIC imputed values and
    # dominated SHAP for both species (artifact, not biology).
    yearly_cens = df.groupby("Year")["is_censored"].mean()
    X["pct_censored_year"] = df["Year"].map(yearly_cens).values

    # --- Lag feature: previous-year country resistance rate ---
    if lag is not None:
        X["lag1_pct_resistant_ctry"] = lag.values

    # --- Gene × year interaction terms ---
    # Captures how each carbapenemase gene's MIC contribution changes over time.
    # NDM and OXA emerged post-2017; KPC plateaued. XGBoost can learn the timing
    # from year alone, but explicit interactions make it transparent in SHAP.
    year_vals = df["Year"].astype(int).values
    for gene in CARBAPENEMASE_GENES:
        col = f"{gene}_pos"
        if col in df.columns:
            X[f"{gene}_x_year"] = df[col].astype(int).values * year_vals

    # --- Projected gene prevalence (fitted on train only, safe to apply to test) ---
    if gene_curves:
        X = add_projected_gene_prevalence(X, df, gene_curves)

    # Ensure all columns are numeric
    X = X.astype(float)

    return X


def build_target(df: pd.DataFrame) -> pd.Series:
    """Return log2(MIC) as the regression target."""
    return df["mic_log2"].rename("target")


def time_split(
    df: pd.DataFrame,
    train_start: int = TRAIN_START,
    train_end: int = TRAIN_END,
    test_start: int = TEST_START,
    test_end: int = TEST_END,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe chronologically - never shuffle.

    Returns (train_df, test_df).
    """
    train = df[(df["Year"] >= train_start) & (df["Year"] <= train_end)].copy()
    test  = df[(df["Year"] >= test_start) & (df["Year"] <= test_end)].copy()
    return train, test


def run_pipeline(df: pd.DataFrame, out_dir: Path, min_country_n: int = 50) -> None:
    """
    Full feature engineering pipeline:
      1. Time-split
      2. Compute keep_countries from training set
      3. Fit gene prevalence curves on training data only
      4. Build X, y for train and test with all features:
         - lag1_pct_resistant_ctry, gene×year interactions, proj_{gene}_prev
      5. Save parquets, keep_countries.json, lag_lookup.json, gene_prevalence_curves.json

    Never shuffles data.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = time_split(df)

    print(f"  Train: {len(train_df):,} rows  ({TRAIN_START}-{TRAIN_END})")
    print(f"  Test:  {len(test_df):,} rows  ({TEST_START}-{TEST_END})")

    keep_countries = get_keep_countries(train_df, min_n=min_country_n)
    n_total = train_df["Country"].nunique()
    n_collapsed = n_total - len(keep_countries)
    print(f"  Countries kept (n>={min_country_n}): {len(keep_countries)}/{n_total} "
          f"({n_collapsed} collapsed to 'Other')")

    # Lag feature: previous-year country resistance rate (uses full df for rate lookup)
    lag_train = compute_lag(df, train_df)
    lag_test  = compute_lag(df, test_df)

    print("  Fitting gene prevalence curves (train only) ...")
    gene_curves = fit_gene_prevalence_curves(train_df)

    X_train = build_features(train_df, keep_countries=keep_countries, lag=lag_train, gene_curves=gene_curves)
    y_train = build_target(train_df)
    X_test  = build_features(test_df,  keep_countries=keep_countries, lag=lag_test,  gene_curves=gene_curves)
    y_test  = build_target(test_df)

    print(f"  Feature matrix shape: {X_train.shape[1]} columns")
    print(f"  Features: {list(X_train.columns[:10])} ...")

    X_train.to_parquet(out_dir / "X_train.parquet")
    y_train.to_frame().to_parquet(out_dir / "y_train.parquet")
    X_test.to_parquet(out_dir / "X_test.parquet")
    y_test.to_frame().to_parquet(out_dir / "y_test.parquet")

    (out_dir / "keep_countries.json").write_text(
        json.dumps(sorted(keep_countries), indent=2)
    )

    lag_lookup = build_lag_lookup(df)
    (out_dir / "lag_lookup.json").write_text(json.dumps(lag_lookup))

    curves_path = out_dir / "gene_prevalence_curves.json"
    curves_path.write_text(json.dumps({g: list(v) for g, v in gene_curves.items()}, indent=2))
    print(f"  Gene curves saved to: {curves_path}")

    print(f"  Saved to: {out_dir}")
