"""
Feature engineering experiments — continent collapse, rolling MIC90, tiered models.
===================================================================================
Opt 6  - Continent collapse:
    Replace 64-80 sparse country OHE with 7 continent OHE columns.
    Reduces sparsity, improves generalisation to low-sample countries.

Opt 7  - + Rolling MIC90:
    Add per-country 3-year rolling MIC90 (training data only, no lookahead).
    Gives XGBoost a local trajectory signal rather than just the global year.

Opt 8  - + Resistant upweight:
    Opt 7 + 3x sample weight for resistant obs (y >= EUCAST R).

Opt 9  - Tiered separate models:
    Train one XGBoost on high-resistance geographies (Europe E, Asia Pacific,
    Middle East) and another on the rest. Route test predictions by tier.

Opt 10 - P(R) Classifier:
    XGBoost binary classifier predicting P(MIC >= 8 mg/L).
    Reported separately with AUC and sensitivity/specificity metrics.

Reference points loaded from saved artefacts (no re-tuning):
    - Baseline: xgb_tuned_{species}.pkl
    - Best prev:  K. pneu = Opt 5 (Tobit feat + upweight), A. bau = baseline

Run:
    .venv/bin/python scripts/pipeline/3d_run_feature_experiments.py --species kpneumoniae
    .venv/bin/python scripts/pipeline/3d_run_feature_experiments.py --species abaumannii
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    roc_auc_score, average_precision_score,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"

EUCAST_R   = 8.0
LOG2_R     = np.log2(EUCAST_R)
LOG2_FLOOR = np.log2(0.06)

SPECIES_MAP = {
    "kpneumoniae": "K. pneumoniae",
    "abaumannii":  "A. baumannii",
}

HIGH_R_CONTINENTS = {"Europe (E)", "Asia Pacific", "Middle East"}

CONTINENT = {
    "United States": "North America", "Canada": "North America", "Mexico": "North America",
    "Guatemala": "North America", "Dominican Republic": "North America",
    "Costa Rica": "North America", "Jamaica": "North America", "Panama": "North America",
    "Brazil": "South America", "Argentina": "South America", "Colombia": "South America",
    "Chile": "South America", "Peru": "South America", "Venezuela": "South America",
    "Ecuador": "South America", "Bolivia": "South America", "Uruguay": "South America",
    "Germany": "Europe (W)", "France": "Europe (W)", "United Kingdom": "Europe (W)",
    "Italy": "Europe (W)", "Spain": "Europe (W)", "Portugal": "Europe (W)",
    "Belgium": "Europe (W)", "Netherlands": "Europe (W)", "Austria": "Europe (W)",
    "Switzerland": "Europe (W)", "Sweden": "Europe (W)", "Norway": "Europe (W)",
    "Denmark": "Europe (W)", "Finland": "Europe (W)", "Ireland": "Europe (W)",
    "Poland": "Europe (E)", "Czech Republic": "Europe (E)", "Hungary": "Europe (E)",
    "Romania": "Europe (E)", "Bulgaria": "Europe (E)", "Serbia": "Europe (E)",
    "Croatia": "Europe (E)", "Slovakia": "Europe (E)", "Slovenia": "Europe (E)",
    "Lithuania": "Europe (E)", "Latvia": "Europe (E)", "Estonia": "Europe (E)",
    "Ukraine": "Europe (E)", "Russia": "Europe (E)", "Greece": "Europe (E)",
    "Turkey": "Europe (E)", "Iceland": "Europe (E)",
    "Saudi Arabia": "Middle East", "Israel": "Middle East", "UAE": "Middle East",
    "Jordan": "Middle East", "Lebanon": "Middle East", "Kuwait": "Middle East",
    "Bahrain": "Middle East", "Oman": "Middle East", "Qatar": "Middle East",
    "Iraq": "Middle East", "Iran": "Middle East", "Egypt": "Middle East",
    "China": "Asia Pacific", "Japan": "Asia Pacific", "South Korea": "Asia Pacific",
    "Taiwan": "Asia Pacific", "India": "Asia Pacific", "Thailand": "Asia Pacific",
    "Malaysia": "Asia Pacific", "Indonesia": "Asia Pacific", "Philippines": "Asia Pacific",
    "Vietnam": "Asia Pacific", "Singapore": "Asia Pacific", "Pakistan": "Asia Pacific",
    "Bangladesh": "Asia Pacific", "Sri Lanka": "Asia Pacific", "Hong Kong": "Asia Pacific",
    "Australia": "Asia Pacific", "New Zealand": "Asia Pacific",
    "South Africa": "Africa", "Morocco": "Africa", "Nigeria": "Africa",
    "Kenya": "Africa", "Tunisia": "Africa", "Algeria": "Africa",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(species: str):
    d = PROJECT_ROOT / "data" / "processed" / species
    X_train = pd.read_parquet(d / "X_train.parquet").reset_index(drop=True)
    y_train = pd.read_parquet(d / "y_train.parquet").squeeze().reset_index(drop=True)
    X_test  = pd.read_parquet(d / "X_test.parquet").reset_index(drop=True)
    y_test  = pd.read_parquet(d / "y_test.parquet").squeeze().reset_index(drop=True)
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)
    return X_train, y_train, X_test, y_test


def load_xgb_params(species: str) -> dict:
    with open(MODELS_DIR / f"xgb_best_params_{species}.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Country / continent helpers
# ---------------------------------------------------------------------------

def reconstruct_country(X: pd.DataFrame) -> pd.Series:
    ctry_cols = [c for c in X.columns if c.startswith("ctry_")]
    ctry_sum  = X[ctry_cols].sum(axis=1)
    country   = X[ctry_cols].idxmax(axis=1).str.replace("ctry_", "", regex=False)
    country[ctry_sum == 0] = "__reference__"
    return country.reset_index(drop=True)


def to_continent(country: pd.Series) -> pd.Series:
    return country.map(CONTINENT).fillna("Other")


def apply_continent_collapse(X_train: pd.DataFrame, X_test: pd.DataFrame,
                             cont_train: pd.Series, cont_test: pd.Series):
    """Replace ctry_ columns with continent OHE."""
    ctry_cols = [c for c in X_train.columns if c.startswith("ctry_")]
    base_tr   = X_train.drop(columns=ctry_cols).reset_index(drop=True)
    base_te   = X_test.drop(columns=ctry_cols).reset_index(drop=True)

    ohe_tr = pd.get_dummies(cont_train, prefix="cont").reset_index(drop=True)
    ohe_te = pd.get_dummies(cont_test,  prefix="cont").reset_index(drop=True)
    ohe_te = ohe_te.reindex(columns=ohe_tr.columns, fill_value=0)

    return (pd.concat([base_tr, ohe_tr], axis=1),
            pd.concat([base_te, ohe_te], axis=1))


# ---------------------------------------------------------------------------
# Rolling MIC90 feature (vectorised via merge, no lookahead)
# ---------------------------------------------------------------------------

def build_mic90_lookup(country: pd.Series, year: pd.Series, y: pd.Series) -> pd.DataFrame:
    """MIC90 per (country, year) from TRAINING data only."""
    df = pd.DataFrame({"country": country.values,
                       "year": year.astype(int).values,
                       "y": y.values})
    return (df.groupby(["country", "year"])["y"]
              .quantile(0.9)
              .reset_index()
              .rename(columns={"y": "mic90"}))


def add_rolling_mic90(country: pd.Series, year: pd.Series,
                      lookup_df: pd.DataFrame, window: int = 3) -> np.ndarray:
    """
    For each obs at year t, average MIC90 of same country over years t-window .. t-1.
    lookup_df must be derived from training data only (no leakage for test set).
    """
    obs = pd.DataFrame({"country": country.values,
                        "year": year.astype(int).values})

    for lag in range(1, window + 1):
        shifted = lookup_df.copy()
        shifted["year"] = shifted["year"] + lag   # mic90[t-lag] is available at year t
        shifted = shifted.rename(columns={"mic90": f"mic90_lag{lag}"})
        obs = obs.merge(shifted, on=["country", "year"], how="left")

    lag_cols = [f"mic90_lag{i}" for i in range(1, window + 1)]
    result   = obs[lag_cols].mean(axis=1).values.copy()

    # Impute NaN (earliest years or unseen countries) with median of known values
    finite = result[np.isfinite(result)]
    if len(finite) > 0:
        result[~np.isfinite(result)] = np.median(finite)
    return result


# ---------------------------------------------------------------------------
# XGBoost training helpers
# ---------------------------------------------------------------------------

def train_xgb(X: np.ndarray, y: np.ndarray, params: dict,
              sample_weight: np.ndarray | None = None) -> xgb.XGBRegressor:
    p = params.copy()
    n_est = int(p.pop("n_estimators", 500))
    model = xgb.XGBRegressor(n_estimators=n_est, objective="reg:squarederror",
                              seed=42, verbosity=0, **p)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def train_xgb_clf(X: np.ndarray, y: np.ndarray, params: dict,
                  sample_weight: np.ndarray | None = None) -> xgb.XGBClassifier:
    p = params.copy()
    n_est = int(p.pop("n_estimators", 500))
    model = xgb.XGBClassifier(n_estimators=n_est, objective="binary:logistic",
                               seed=42, verbosity=0, eval_metric="logloss", **p)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def resistant_weights(y: np.ndarray, weight: float = 3.0) -> np.ndarray:
    w = np.ones(len(y))
    w[y >= LOG2_R] = weight
    return w


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    mae    = mean_absolute_error(y_true, y_pred)
    mask_r = y_true >= LOG2_R
    rmse_r = np.sqrt(mean_squared_error(y_true[mask_r], y_pred[mask_r])) if mask_r.any() else np.nan
    mae_r  = mean_absolute_error(y_true[mask_r], y_pred[mask_r]) if mask_r.any() else np.nan
    return dict(label=label, rmse=rmse, mae=mae,
                rmse_r=rmse_r, mae_r=mae_r, n_r=int(mask_r.sum()))


def evaluate_clf(y_true: np.ndarray, y_prob: np.ndarray, label: str) -> dict:
    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr  = average_precision_score(y_true, y_prob)
    y_pred  = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return dict(label=label, auc_roc=auc_roc, auc_pr=auc_pr,
                sensitivity=sens, specificity=spec,
                n_pos=int(y_true.sum()), n_total=len(y_true))


def print_reg_table(results: list[dict]):
    baseline_rmse_r = results[0]["rmse_r"]
    hdr = f"  {'Model':<42} {'RMSE all':>9} {'MAE all':>8} {'RMSE(R)':>8} {'MAE(R)':>8}  delta(R)"
    print("\n" + hdr)
    print("  " + "-" * 95)
    for r in results:
        delta = r["rmse_r"] - baseline_rmse_r
        sign  = "baseline" if r["label"] == results[0]["label"] else f"{delta:+.4f}"
        print(f"  {r['label']:<42} {r['rmse']:>9.4f} {r['mae']:>8.4f} "
              f"{r['rmse_r']:>8.4f} {r['mae_r']:>8.4f}  {sign}")


def print_clf_table(results: list[dict]):
    hdr = f"  {'Model':<42} {'AUC-ROC':>8} {'AUC-PR':>7} {'Sens':>6} {'Spec':>6}  n_pos/n"
    print("\n" + hdr)
    print("  " + "-" * 85)
    for r in results:
        print(f"  {r['label']:<42} {r['auc_roc']:>8.4f} {r['auc_pr']:>7.4f} "
              f"{r['sensitivity']:>6.3f} {r['specificity']:>6.3f}  "
              f"{r['n_pos']:,}/{r['n_total']:,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", choices=list(SPECIES_MAP), default="kpneumoniae")
    parser.add_argument("--mic90-window", type=int, default=3)
    args = parser.parse_args()

    label = SPECIES_MAP[args.species]
    print(f"\n=== Feature Experiments — {label} + Meropenem ===\n")

    # ---- Load ---------------------------------------------------------------
    print("[1/6] Loading data...")
    X_train, y_train, X_test, y_test = load_data(args.species)
    xgb_params = load_xgb_params(args.species)
    print(f"  Train: {X_train.shape[0]:,} rows x {X_train.shape[1]} features")
    print(f"  Test:  {X_test.shape[0]:,} rows")

    # ---- Baseline -----------------------------------------------------------
    print("\n[2/6] Baseline: original tuned XGBoost...")
    xgb_base  = joblib.load(MODELS_DIR / f"xgb_tuned_{args.species}.pkl")
    res_base  = evaluate(y_test.values, xgb_base.predict(X_test.values),
                         "XGBoost baseline")

    # ---- Country/continent metadata -----------------------------------------
    print("\n[3/6] Building country/continent metadata...")
    country_train = reconstruct_country(X_train)
    country_test  = reconstruct_country(X_test)
    cont_train    = to_continent(country_train)
    cont_test     = to_continent(country_test)

    n_unmapped = (cont_train == "Other").sum()
    print(f"  Countries in train: {country_train.nunique()}")
    print(f"  Unmapped to continent: {n_unmapped:,} obs ({n_unmapped/len(cont_train):.1%})")
    print(f"  Continent distribution (train):")
    for c, n in cont_train.value_counts().items():
        print(f"    {c:<20} {n:>6,}  ({n/len(cont_train):.1%})")

    # ---- Opt 6: Continent collapse ------------------------------------------
    print("\n[4/6] Opt 6: continent OHE collapse...")
    X_tr6, X_te6 = apply_continent_collapse(X_train, X_test, cont_train, cont_test)
    print(f"  Feature count: {X_train.shape[1]} -> {X_tr6.shape[1]} "
          f"(saved {X_train.shape[1] - X_tr6.shape[1]} cols)")
    res_opt6 = evaluate(y_test.values,
                        train_xgb(X_tr6.values, y_train.values, xgb_params).predict(X_te6.values),
                        "Opt 6: continent OHE")

    # ---- Opt 7: + Rolling MIC90 ---------------------------------------------
    print(f"\n[4/6] Opt 7: + rolling MIC90 (window={args.mic90_window} yr)...")
    mic90_lookup = build_mic90_lookup(country_train, X_train["year"], y_train)
    rolling_train = add_rolling_mic90(country_train, X_train["year"],
                                      mic90_lookup, args.mic90_window)
    rolling_test  = add_rolling_mic90(country_test,  X_test["year"],
                                      mic90_lookup, args.mic90_window)

    X_tr7 = X_tr6.copy()
    X_te7 = X_te6.copy()
    X_tr7["rolling_mic90"] = rolling_train
    X_te7["rolling_mic90"] = rolling_test
    print(f"  rolling_mic90 train: mean={rolling_train.mean():.3f}, "
          f"std={rolling_train.std():.3f}, "
          f"NaN_imputed={(~np.isfinite(rolling_train)).sum()}")

    res_opt7 = evaluate(y_test.values,
                        train_xgb(X_tr7.values, y_train.values, xgb_params).predict(X_te7.values),
                        "Opt 7: + rolling MIC90")

    # ---- Opt 8: + Resistant upweight ----------------------------------------
    print("\n[4/6] Opt 8: + resistant upweight (3x)...")
    w_train = resistant_weights(y_train.values, 3.0)
    res_opt8 = evaluate(y_test.values,
                        train_xgb(X_tr7.values, y_train.values, xgb_params,
                                  sample_weight=w_train).predict(X_te7.values),
                        "Opt 8: + upweight 3x")

    # ---- Opt 9: Tiered separate models --------------------------------------
    print("\n[5/6] Opt 9: tiered separate models (high-R vs low-R geography)...")
    high_mask_tr = cont_train.isin(HIGH_R_CONTINENTS).values
    high_mask_te = cont_test.isin(HIGH_R_CONTINENTS).values
    print(f"  High-R tier (train): {high_mask_tr.sum():,} ({high_mask_tr.mean():.1%}) "
          f"  continents: {sorted(HIGH_R_CONTINENTS)}")
    print(f"  Low-R  tier (train): {(~high_mask_tr).sum():,} ({(~high_mask_tr).mean():.1%})")

    # Use same features as Opt 7 (continent + rolling MIC90) for tiered models
    model_high = train_xgb(X_tr7.values[high_mask_tr], y_train.values[high_mask_tr], xgb_params)
    model_low  = train_xgb(X_tr7.values[~high_mask_tr], y_train.values[~high_mask_tr], xgb_params)

    y_pred9 = np.empty(len(y_test))
    y_pred9[high_mask_te]  = model_high.predict(X_te7.values[high_mask_te])
    y_pred9[~high_mask_te] = model_low.predict(X_te7.values[~high_mask_te])
    res_opt9 = evaluate(y_test.values, y_pred9, "Opt 9: tiered models (high/low R)")

    # ---- Opt 10: P(R) Classifier --------------------------------------------
    print("\n[6/6] Opt 10: P(R) binary classifier (XGBoost)...")
    y_bin_train = (y_train.values >= LOG2_R).astype(int)
    y_bin_test  = (y_test.values  >= LOG2_R).astype(int)
    print(f"  Resistant in train: {y_bin_train.sum():,} ({y_bin_train.mean():.1%})")
    print(f"  Resistant in test:  {y_bin_test.sum():,}  ({y_bin_test.mean():.1%})")

    # Train on Opt 7 features (best feature set)
    clf_model = train_xgb_clf(X_tr7.values, y_bin_train, xgb_params)
    y_prob    = clf_model.predict_proba(X_te7.values)[:, 1]
    res_clf   = evaluate_clf(y_bin_test, y_prob, "P(R) classifier (XGBoost)")

    # Also train on original features for comparison
    clf_base  = train_xgb_clf(X_train.values, y_bin_train, xgb_params)
    y_prob_base = clf_base.predict_proba(X_test.values)[:, 1]
    res_clf_base = evaluate_clf(y_bin_test, y_prob_base, "P(R) classifier (baseline features)")

    # ---- Summary ------------------------------------------------------------
    print("\n" + "=" * 95)
    print(f"  REGRESSION COMPARISON — {label}")
    reg_results = [res_base, res_opt6, res_opt7, res_opt8, res_opt9]
    print_reg_table(reg_results)

    print(f"\n  CLASSIFIER COMPARISON — {label}  (task: predict resistant vs susceptible)")
    print_clf_table([res_clf_base, res_clf])

    best_reg = min(reg_results[1:], key=lambda r: r["rmse_r"])
    print(f"\n  Best regression variant (RMSE(R)): {best_reg['label']}")
    print(f"  vs baseline: {res_base['rmse_r']:.4f}  ->  {best_reg['rmse_r']:.4f}  "
          f"(delta {best_reg['rmse_r'] - res_base['rmse_r']:+.4f})\n")


if __name__ == "__main__":
    main()
