"""
Tobit regression for MIC Creep — censoring-corrected parametric model.
======================================================================
Loads the same processed parquets as 3_run_model_training.py and fits
a Tobit model that explicitly models the censoring structure instead of
imputing censored values.

Key output: year coefficient — the censoring-corrected MIC creep rate.
Compare with LR year coefficient to see how much imputation biased it.

Censoring setup (ATLAS Meropenem panel):
  Left  : MIC <= 0.06 mg/L  →  L = log2(0.06) = -4.059
           Detected via: X["is_censored"] == 1
  Right : MIC > 32 mg/L     →  U = log2(32) = 5.0
           Detected via: y > 5.5 (imputed as 64 mg/L → log2=6.0)

Run:
    .venv/bin/python scripts/pipeline/3b_run_tobit.py --species kpneumoniae
    .venv/bin/python scripts/pipeline/3b_run_tobit.py --species abaumannii
"""

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.tobit import TobitRegressor

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

EUCAST_R  = 8.0
LOG2_R    = np.log2(EUCAST_R)         # 3.0
LOG2_CEIL = np.log2(32.0)             # 5.0 — right censoring limit
LOG2_FLOOR = np.log2(0.06)            # -4.059 — left censoring limit

SPECIES_MAP = {
    "kpneumoniae": "K. pneumoniae",
    "abaumannii":  "A. baumannii",
}


def load_data(species: str):
    data_dir = PROJECT_ROOT / "data" / "processed" / species
    X_train = pd.read_parquet(data_dir / "X_train.parquet")
    y_train = pd.read_parquet(data_dir / "y_train.parquet").squeeze()
    X_test  = pd.read_parquet(data_dir / "X_test.parquet")
    y_test  = pd.read_parquet(data_dir / "y_test.parquet").squeeze()
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)
    return X_train, y_train, X_test, y_test


def build_censoring_masks(X: pd.DataFrame, y: pd.Series):
    """Reconstruct left/right censoring masks from is_censored flag and y values."""
    left_mask  = X["is_censored"].astype(bool).values
    # >32 values get clipped to 32 mg/L (not doubled to 64), so right-censored
    # observations sit exactly at log2(32) = 5.0. Use a small tolerance.
    right_mask = (y.values >= LOG2_CEIL - 0.01)
    # Avoid double-counting (a censored floor observation can't also be at ceiling)
    right_mask = right_mask & ~left_mask
    print(f"  Left-censored  (is_censored=1): {left_mask.sum():,}  ({left_mask.mean():.1%})")
    print(f"  Right-censored (y >= 4.99):     {right_mask.sum():,}  ({right_mask.mean():.1%})")
    print(f"  Uncensored:                     {(~left_mask & ~right_mask).sum():,}")
    return left_mask, right_mask


def drop_tobit_incompatible(X: pd.DataFrame) -> pd.DataFrame:
    """
    Drop features that are collinear with the censoring masks.

    is_censored is the left_mask itself — including it as a feature
    causes the optimiser to produce a degenerate -13 coefficient and
    the predictions collapse. pct_censored_year is kept: it controls
    for the 2013-2017 panel methodology shift which is real signal.
    """
    drop_cols = [c for c in ["is_censored"] if c in X.columns]
    if drop_cols:
        print(f"  Dropping Tobit-incompatible features: {drop_cols}")
    return X.drop(columns=drop_cols)


def scale_features(X_train: np.ndarray, X_test: np.ndarray):
    """
    Standardise continuous columns so L-BFGS-B gradients are well-conditioned.
    Binary/OHE columns (all values in {0,1}) are left unchanged.
    Returns scaled arrays + the scaler parameters for reference.
    """
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    # Only scale columns with range > 1 (i.e. non-binary)
    scale_mask = std > 1.0
    std_safe   = np.where(scale_mask, std, 1.0)
    mean_safe  = np.where(scale_mask, mean, 0.0)
    return (
        (X_train - mean_safe) / std_safe,
        (X_test  - mean_safe) / std_safe,
        mean_safe,
        std_safe,
    )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)

    mask_r = y_true >= LOG2_R
    rmse_r = np.sqrt(mean_squared_error(y_true[mask_r], y_pred[mask_r])) if mask_r.any() else np.nan
    mae_r  = mean_absolute_error(y_true[mask_r], y_pred[mask_r]) if mask_r.any() else np.nan
    n_r    = int(mask_r.sum())

    print(f"\n  [{label}]")
    print(f"    RMSE (all)       = {rmse:.4f}")
    print(f"    MAE  (all)       = {mae:.4f}")
    print(f"    RMSE (R subset)  = {rmse_r:.4f}  (n={n_r:,})")
    print(f"    MAE  (R subset)  = {mae_r:.4f}")
    return dict(label=label, rmse=rmse, mae=mae,
                rmse_resistant=rmse_r, mae_resistant=mae_r, n_resistant=n_r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", choices=list(SPECIES_MAP), default="kpneumoniae")
    parser.add_argument("--alpha",   type=float, default=1e-4,
                        help="L2 regularisation strength (default 1e-4)")
    parser.add_argument("--max-iter", type=int, default=2000)
    args = parser.parse_args()

    label = SPECIES_MAP[args.species]
    print(f"\n=== Tobit Regression — {label} + Meropenem ===\n")

    print("[1/4] Loading data...")
    X_train, y_train, X_test, y_test = load_data(args.species)
    print(f"  Train: {X_train.shape[0]:,} rows x {X_train.shape[1]} features")
    print(f"  Test:  {X_test.shape[0]:,} rows")

    print("\n[2/4] Building censoring masks (train)...")
    left_mask, right_mask = build_censoring_masks(X_train, y_train)

    # Drop is_censored AFTER building masks — collinear with left_mask
    X_train = drop_tobit_incompatible(X_train)
    X_test  = drop_tobit_incompatible(X_test)

    # Scale features for well-conditioned gradients
    X_tr_sc, X_te_sc, scale_mean, scale_std = scale_features(
        X_train.values, X_test.values
    )
    feat_names_clean = list(X_train.columns)

    print("\n[3/4] Fitting Tobit model (L-BFGS-B MLE, max_iter={})...".format(args.max_iter))
    model = TobitRegressor(
        lower=LOG2_FLOOR,
        upper=LOG2_CEIL,
        alpha=args.alpha,
        max_iter=args.max_iter,
    )
    model.fit(
        X_tr_sc,
        y_train.values,
        left_censored_mask=left_mask,
        right_censored_mask=right_mask,
    )

    status = "converged" if model.converged_ else "did NOT converge (try --max-iter 5000)"
    print(f"  Optimisation: {status}")
    print(f"  Fitted sigma  = {model.sigma_:.4f} (residual noise in log2 units)")

    # Year coefficient — reported in original (unscaled) units
    year_idx       = feat_names_clean.index("year")
    year_coef_sc   = model.coef_[year_idx]          # coefficient on scaled year
    year_coef_orig = year_coef_sc / scale_std[year_idx]  # back to log2/yr
    fold_change    = 2 ** abs(year_coef_orig) - 1
    direction      = "upward" if year_coef_orig > 0 else "downward"
    print(f"\n  *** Year coefficient (censoring-corrected MIC creep rate) ***")
    print(f"      {year_coef_orig:+.6f} log2 units/yr  (unscaled)")
    print(f"      = {fold_change:.1%} MIC fold-change per year ({direction})")

    # Top 10 coefficients (scaled space — for relative comparison only)
    abs_coef = np.abs(model.coef_)
    top_idx  = np.argsort(abs_coef)[::-1][:10]
    print(f"\n  Top 10 features by |β| (scaled space):")
    print(f"  {'Feature':<28} {'β (scaled)':>12}")
    print("  " + "-" * 42)
    for i in top_idx:
        print(f"  {feat_names_clean[i]:<28} {model.coef_[i]:+12.4f}")

    print("\n[4/4] Evaluating on test set (2019-2022)...")
    y_pred  = model.predict(X_te_sc)
    metrics = evaluate(y_test.values, y_pred, f"Tobit ({label})")

    # Save model + scaler so predict() can be called on raw X later
    artefact = {"model": model, "scale_mean": scale_mean, "scale_std": scale_std,
                "feat_names": feat_names_clean}
    fname = f"tobit_{args.species}.pkl"
    joblib.dump(artefact, MODELS_DIR / fname)
    print(f"\n  Saved -> models/{fname}")

    # Brief comparison reminder
    print("\n--- Interpretation ---")
    print("  Tobit year coeff corrects for ~75-80% censored observations.")
    print("  A negative coeff means: after accounting for gene prevalence and")
    print("  censoring structure, raw temporal drift is negative or flat.")
    print("  Compare with LR year coeff from 3_run_model_training.py.")
    print("  RMSE on resistant subset is the primary performance metric.\n")


if __name__ == "__main__":
    main()
