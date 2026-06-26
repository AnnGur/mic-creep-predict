"""
Tobit-augmented XGBoost experiments.
=====================================
Tests three ways of using Tobit to improve XGBoost on censored MIC data:

  Option 1 - Debiased targets:
    Replace imputed y values for left-censored training obs with Tobit
    latent predictions (E[y*] = Xb). XGBoost trains on better-estimated
    targets rather than the naive panel-floor imputation (half of 0.06).

  Option 2 - Tobit as feature (stacking):
    Add Tobit's latent prediction as an extra column in X. XGBoost learns
    to correct Tobit's linear errors while using it as a censoring-aware
    baseline signal.

  Option 3 - Combined:
    Debiased targets AND Tobit as feature together.

Baseline: original tuned XGBoost loaded from models/.

Run:
    .venv/bin/python scripts/pipeline/3c_run_tobit_xgb.py --species kpneumoniae
    .venv/bin/python scripts/pipeline/3c_run_tobit_xgb.py --species abaumannii
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
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"

EUCAST_R   = 8.0
LOG2_R     = np.log2(EUCAST_R)   # 3.0
LOG2_CEIL  = np.log2(32.0)       # 5.0
LOG2_FLOOR = np.log2(0.06)       # -4.059

SPECIES_MAP = {
    "kpneumoniae": "K. pneumoniae",
    "abaumannii":  "A. baumannii",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data(species: str):
    data_dir = PROJECT_ROOT / "data" / "processed" / species
    X_train = pd.read_parquet(data_dir / "X_train.parquet")
    y_train = pd.read_parquet(data_dir / "y_train.parquet").squeeze()
    X_test  = pd.read_parquet(data_dir / "X_test.parquet")
    y_test  = pd.read_parquet(data_dir / "y_test.parquet").squeeze()
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)
    return X_train, y_train, X_test, y_test


def load_tobit(species: str):
    art = joblib.load(MODELS_DIR / f"tobit_{species}.pkl")
    return art["model"], art["scale_mean"], art["scale_std"], art["feat_names"]


def tobit_latent(X: pd.DataFrame, model, scale_mean, scale_std, feat_names) -> np.ndarray:
    """Return Tobit latent mean E[y*] = Xb for each row of X."""
    Xf  = X[feat_names].values.astype(np.float64)
    Xsc = (Xf - scale_mean) / scale_std
    return model.predict(Xsc)


def build_left_mask(X: pd.DataFrame) -> np.ndarray:
    return X["is_censored"].astype(bool).values


# ---------------------------------------------------------------------------
# XGBoost helpers
# ---------------------------------------------------------------------------

def load_xgb_params(species: str) -> dict:
    with open(MODELS_DIR / f"xgb_best_params_{species}.json") as f:
        return json.load(f)


def train_xgb(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    sample_weight: np.ndarray | None = None,
) -> xgb.XGBRegressor:
    p = params.copy()
    n_estimators = int(p.pop("n_estimators", 500))
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        objective="reg:squarederror",
        seed=42,
        verbosity=0,
        **p,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def resistant_weights(y: np.ndarray, weight: float = 3.0) -> np.ndarray:
    """Upweight resistant observations (y >= LOG2_R) during training."""
    w = np.ones(len(y))
    w[y >= LOG2_R] = weight
    return w


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mask_r = y_true >= LOG2_R
    rmse_r = np.sqrt(mean_squared_error(y_true[mask_r], y_pred[mask_r])) if mask_r.any() else np.nan
    mae_r  = mean_absolute_error(y_true[mask_r], y_pred[mask_r]) if mask_r.any() else np.nan
    return dict(label=label, rmse=rmse, mae=mae,
                rmse_r=rmse_r, mae_r=mae_r, n_r=int(mask_r.sum()))


def print_results(results: list[dict]):
    hdr = f"  {'Model':<38} {'RMSE all':>9} {'MAE all':>8} {'RMSE R':>8} {'MAE R':>8}  n(R)"
    print("\n" + hdr)
    print("  " + "-" * 82)
    baseline = results[0]["rmse_r"]
    for r in results:
        delta = r["rmse_r"] - baseline
        sign  = f"{delta:+.4f}" if r["label"] != results[0]["label"] else "baseline"
        print(f"  {r['label']:<38} {r['rmse']:>9.4f} {r['mae']:>8.4f} "
              f"{r['rmse_r']:>8.4f} {r['mae_r']:>8.4f}  {r['n_r']:,}  ({sign})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", choices=list(SPECIES_MAP), default="kpneumoniae")
    args = parser.parse_args()

    label = SPECIES_MAP[args.species]
    print(f"\n=== Tobit-augmented XGBoost — {label} + Meropenem ===\n")

    # ---- Load ---------------------------------------------------------------
    print("[1/5] Loading data and artifacts...")
    X_train, y_train, X_test, y_test = load_data(args.species)
    tobit_model, scale_mean, scale_std, feat_names = load_tobit(args.species)
    xgb_params = load_xgb_params(args.species)
    print(f"  Train: {X_train.shape[0]:,} rows x {X_train.shape[1]} features")
    print(f"  Test:  {X_test.shape[0]:,} rows")

    # ---- Baseline -----------------------------------------------------------
    print("\n[2/5] Baseline: original tuned XGBoost (saved model, no retraining)...")
    xgb_base = joblib.load(MODELS_DIR / f"xgb_tuned_{args.species}.pkl")
    res_base  = evaluate(y_test.values, xgb_base.predict(X_test.values),
                         "XGBoost tuned (baseline)")

    # ---- Tobit predictions (train + test) -----------------------------------
    X_tr_noic = X_train.drop(columns=["is_censored"], errors="ignore")
    X_te_noic = X_test.drop(columns=["is_censored"], errors="ignore")
    tobit_train = tobit_latent(X_tr_noic, tobit_model, scale_mean, scale_std, feat_names)
    tobit_test  = tobit_latent(X_te_noic, tobit_model, scale_mean, scale_std, feat_names)

    left_mask = build_left_mask(X_train)
    n_left    = left_mask.sum()
    print(f"\n  Left-censored obs in train: {n_left:,} ({left_mask.mean():.1%})")
    print(f"  Tobit latent range for censored obs: "
          f"[{tobit_train[left_mask].min():.3f}, {tobit_train[left_mask].max():.3f}] log2")
    print(f"  Original imputed y for censored obs: "
          f"mean={y_train.values[left_mask].mean():.4f}  "
          f"(floor imputation = {LOG2_FLOOR/2:.4f})")
    print(f"  Tobit latent y for censored obs:     "
          f"mean={tobit_train[left_mask].mean():.4f}")

    # ---- Option 1: debiased targets -----------------------------------------
    print("\n[3/5] Option 1: debiased targets (replace y for censored obs)...")
    y_debiased = y_train.values.copy()
    y_debiased[left_mask] = tobit_train[left_mask]

    res_opt1 = evaluate(
        y_test.values,
        train_xgb(X_train.values, y_debiased, xgb_params).predict(X_test.values),
        "XGBoost + debiased targets (Opt 1)",
    )

    # ---- Option 2: Tobit as feature -----------------------------------------
    print("\n[4/5] Option 2: Tobit prediction as feature (stacking)...")
    X_tr_aug = X_train.copy()
    X_te_aug = X_test.copy()
    X_tr_aug["tobit_pred"] = tobit_train
    X_te_aug["tobit_pred"] = tobit_test

    res_opt2 = evaluate(
        y_test.values,
        train_xgb(X_tr_aug.values, y_train.values, xgb_params).predict(X_te_aug.values),
        "XGBoost + Tobit feature (Opt 2)",
    )

    # ---- Option 3: both combined --------------------------------------------
    print("\n[4b/5] Option 3: debiased targets + Tobit feature combined...")
    res_opt3 = evaluate(
        y_test.values,
        train_xgb(X_tr_aug.values, y_debiased, xgb_params).predict(X_te_aug.values),
        "XGBoost + both combined (Opt 3)",
    )

    # ---- Option 4: resistant upweighting ------------------------------------
    print("\n[5a/5] Option 4: resistant upweighting (3x weight where y >= R)...")
    w_train = resistant_weights(y_train.values, weight=3.0)
    print(f"  Resistant obs in train: {(w_train > 1).sum():,} ({(w_train > 1).mean():.1%})")

    res_opt4 = evaluate(
        y_test.values,
        train_xgb(X_train.values, y_train.values, xgb_params,
                  sample_weight=w_train).predict(X_test.values),
        "XGBoost + resistant upweight (Opt 4)",
    )

    # ---- Option 5: Tobit feature + resistant upweighting --------------------
    print("\n[5b/5] Option 5: Tobit feature + resistant upweighting combined...")
    res_opt5 = evaluate(
        y_test.values,
        train_xgb(X_tr_aug.values, y_train.values, xgb_params,
                  sample_weight=w_train).predict(X_te_aug.values),
        "XGBoost + Tobit feat + upweight (Opt 5)",
    )

    # ---- Summary ------------------------------------------------------------
    print("\n[5/5] Results:")
    results = [res_base, res_opt1, res_opt2, res_opt3, res_opt4, res_opt5]
    print_results(results)

    best = min(results[1:], key=lambda r: r["rmse_r"])
    print(f"\n  Best variant on RMSE(R): {best['label']}")
    print(f"  vs baseline RMSE(R):    {res_base['rmse_r']:.4f}")
    delta = best["rmse_r"] - res_base["rmse_r"]
    print(f"  Delta:                  {delta:+.4f} log2 units\n")


if __name__ == "__main__":
    main()
