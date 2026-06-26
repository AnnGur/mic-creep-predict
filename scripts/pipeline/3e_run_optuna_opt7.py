"""
Optuna hyperparameter tuning on the Opt 7 feature set.
=======================================================
Opt 7 features = continent OHE (7 cols, replaces 64-80 country cols)
               + rolling MIC90 (3-year trailing per country, training data only)

Previous experiments (3d) showed these features reduce RMSE(all) significantly
but hurt RMSE(R) when using old hyperparams tuned for 91 features. This script
re-tunes from scratch for the new 27-feature space.

Outputs:
    models/xgb_opt7_{species}.pkl
    models/xgb_opt7_params_{species}.json

Run:
    .venv/bin/python scripts/pipeline/3e_run_optuna_opt7.py --species kpneumoniae
    .venv/bin/python scripts/pipeline/3e_run_optuna_opt7.py --species abaumannii
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"

EUCAST_R     = 8.0
LOG2_R       = np.log2(EUCAST_R)
RANDOM_STATE = 42
N_TRIALS     = 60

SPECIES_MAP = {
    "kpneumoniae": "K. pneumoniae",
    "abaumannii":  "A. baumannii",
}

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
# Feature engineering
# ---------------------------------------------------------------------------

def load_data(species: str):
    d = PROJECT_ROOT / "data" / "processed" / species
    X_train = pd.read_parquet(d / "X_train.parquet").reset_index(drop=True)
    y_train = pd.read_parquet(d / "y_train.parquet").squeeze().reset_index(drop=True)
    X_test  = pd.read_parquet(d / "X_test.parquet").reset_index(drop=True)
    y_test  = pd.read_parquet(d / "y_test.parquet").squeeze().reset_index(drop=True)
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)
    return X_train, y_train, X_test, y_test


def reconstruct_country(X: pd.DataFrame) -> pd.Series:
    ctry_cols = [c for c in X.columns if c.startswith("ctry_")]
    ctry_sum  = X[ctry_cols].sum(axis=1)
    country   = X[ctry_cols].idxmax(axis=1).str.replace("ctry_", "", regex=False)
    country[ctry_sum == 0] = "__reference__"
    return country.reset_index(drop=True)


def build_opt7_features(X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, window: int = 3):
    """Build Opt 7 feature set: continent OHE + rolling MIC90."""
    ctry_cols = [c for c in X_train.columns if c.startswith("ctry_")]

    country_tr = reconstruct_country(X_train)
    country_te = reconstruct_country(X_test)
    cont_tr    = country_tr.map(CONTINENT).fillna("Other")
    cont_te    = country_te.map(CONTINENT).fillna("Other")

    # Continent OHE
    ohe_tr = pd.get_dummies(cont_tr, prefix="cont").reset_index(drop=True)
    ohe_te = pd.get_dummies(cont_te, prefix="cont").reset_index(drop=True)
    ohe_te = ohe_te.reindex(columns=ohe_tr.columns, fill_value=0)

    base_tr = X_train.drop(columns=ctry_cols).reset_index(drop=True)
    base_te = X_test.drop(columns=ctry_cols).reset_index(drop=True)

    X_tr7 = pd.concat([base_tr, ohe_tr], axis=1)
    X_te7 = pd.concat([base_te, ohe_te], axis=1)

    # Rolling MIC90 — built from training data only (no leakage)
    mic90_df = (pd.DataFrame({"country": country_tr.values,
                               "year": X_train["year"].astype(int).values,
                               "y": y_train.values})
                .groupby(["country", "year"])["y"]
                .quantile(0.9)
                .reset_index()
                .rename(columns={"y": "mic90"}))

    def add_rolling(country, year):
        obs = pd.DataFrame({"country": country.values,
                             "year": year.astype(int).values})
        for lag in range(1, window + 1):
            shifted = mic90_df.copy()
            shifted["year"] = shifted["year"] + lag
            shifted = shifted.rename(columns={"mic90": f"mic90_lag{lag}"})
            obs = obs.merge(shifted, on=["country", "year"], how="left")
        lag_cols = [f"mic90_lag{i}" for i in range(1, window + 1)]
        result   = obs[lag_cols].mean(axis=1).values.copy()
        finite   = result[np.isfinite(result)]
        result[~np.isfinite(result)] = np.median(finite) if len(finite) > 0 else 0.0
        return result

    X_tr7["rolling_mic90"] = add_rolling(country_tr, X_train["year"])
    X_te7["rolling_mic90"] = add_rolling(country_te, X_test["year"])

    return X_tr7, X_te7


# ---------------------------------------------------------------------------
# Optuna tuning (identical search space to 3_run_model_training.py)
# ---------------------------------------------------------------------------

def tune_xgb_optuna(X_train: pd.DataFrame, y_train: pd.Series,
                    n_trials: int) -> dict:
    sample_weight = np.where(y_train.values >= LOG2_R, 3.0, 1.0)

    # Time-aware val split: last 3 years of training data
    val_mask = X_train["year"] >= (X_train["year"].max() - 2)
    X_tr, X_v = X_train[~val_mask], X_train[val_mask]
    y_tr, y_v = y_train[~val_mask], y_train[val_mask]
    sw_tr     = sample_weight[~val_mask]

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 300, 1200, step=100),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma":            trial.suggest_float("gamma", 0.0, 2.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 10.0),
            "tree_method":      "hist",
            "random_state":     RANDOM_STATE,
            "n_jobs":           -1,
            "verbosity":        0,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, sample_weight=sw_tr,
                  eval_set=[(X_v, y_v)], verbose=False)
        return np.sqrt(mean_squared_error(y_v, model.predict(X_v)))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"  Best val RMSE: {study.best_value:.4f}")
    print(f"  Best params:   {study.best_params}")
    return study.best_params


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


def print_table(results: list[dict]):
    baseline = results[0]["rmse_r"]
    hdr = f"  {'Model':<45} {'RMSE all':>9} {'MAE all':>8} {'RMSE(R)':>8} {'MAE(R)':>8}  delta(R)"
    print("\n" + hdr)
    print("  " + "-" * 98)
    for r in results:
        delta = r["rmse_r"] - baseline
        sign  = "baseline" if r["label"] == results[0]["label"] else f"{delta:+.4f}"
        print(f"  {r['label']:<45} {r['rmse']:>9.4f} {r['mae']:>8.4f} "
              f"{r['rmse_r']:>8.4f} {r['mae_r']:>8.4f}  {sign}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_hybrid_features(X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, window: int = 3):
    """Hybrid: keep all original features, append rolling MIC90 only."""
    country_tr = reconstruct_country(X_train)
    country_te = reconstruct_country(X_test)

    mic90_df = (pd.DataFrame({"country": country_tr.values,
                               "year": X_train["year"].astype(int).values,
                               "y": y_train.values})
                .groupby(["country", "year"])["y"]
                .quantile(0.9)
                .reset_index()
                .rename(columns={"y": "mic90"}))

    def add_rolling(country, year):
        obs = pd.DataFrame({"country": country.values,
                             "year": year.astype(int).values})
        for lag in range(1, window + 1):
            shifted = mic90_df.copy()
            shifted["year"] = shifted["year"] + lag
            shifted = shifted.rename(columns={"mic90": f"mic90_lag{lag}"})
            obs = obs.merge(shifted, on=["country", "year"], how="left")
        lag_cols = [f"mic90_lag{i}" for i in range(1, window + 1)]
        result   = obs[lag_cols].mean(axis=1).values.copy()
        finite   = result[np.isfinite(result)]
        result[~np.isfinite(result)] = np.median(finite) if len(finite) > 0 else 0.0
        return result

    X_trh = X_train.copy().reset_index(drop=True)
    X_teh = X_test.copy().reset_index(drop=True)
    X_trh["rolling_mic90"] = add_rolling(country_tr, X_train["year"])
    X_teh["rolling_mic90"] = add_rolling(country_te, X_test["year"])
    return X_trh, X_teh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", choices=list(SPECIES_MAP), default="kpneumoniae")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--mode", choices=["opt7", "hybrid"], default="hybrid",
                        help="opt7: continent collapse + MIC90 | hybrid: all features + MIC90")
    args = parser.parse_args()

    label = SPECIES_MAP[args.species]
    mode_label = "Hybrid (91 features + rolling MIC90)" if args.mode == "hybrid" \
                 else "Opt 7 (continent collapse + rolling MIC90)"
    print(f"\n=== Optuna — {mode_label} — {label} + Meropenem ===\n")

    # ---- Load ---------------------------------------------------------------
    print("[1/5] Loading data...")
    X_train, y_train, X_test, y_test = load_data(args.species)
    print(f"  Train: {X_train.shape[0]:,} rows x {X_train.shape[1]} features (original)")
    print(f"  Test:  {X_test.shape[0]:,} rows")

    # ---- Baseline -----------------------------------------------------------
    print("\n[2/5] Baseline: original tuned XGBoost (saved model)...")
    xgb_base = joblib.load(MODELS_DIR / f"xgb_tuned_{args.species}.pkl")
    res_base = evaluate(y_test.values, xgb_base.predict(X_test.values),
                        f"Baseline XGBoost ({X_train.shape[1]} features)")

    # ---- Build features -----------------------------------------------------
    print(f"\n[3/5] Building {args.mode} feature set...")
    if args.mode == "hybrid":
        X_tr_new, X_te_new = build_hybrid_features(X_train, y_train, X_test)
        model_tag = f"xgb_hybrid_{args.species}"
    else:
        X_tr_new, X_te_new = build_opt7_features(X_train, y_train, X_test)
        model_tag = f"xgb_opt7_{args.species}"

    print(f"  Feature count: {X_train.shape[1]} -> {X_tr_new.shape[1]}")
    print(f"  Added: {[c for c in X_tr_new.columns if c not in X_train.columns]}")

    # Reference: new features with old params (no retuning)
    old_params = json.load(open(MODELS_DIR / f"xgb_best_params_{args.species}.json"))
    p = old_params.copy()
    n_est = int(p.pop("n_estimators", 500))
    xgb_old = xgb.XGBRegressor(n_estimators=n_est, objective="reg:squarederror",
                                seed=RANDOM_STATE, verbosity=0, **p)
    xgb_old.fit(X_tr_new.values, y_train.values)
    res_old = evaluate(y_test.values, xgb_old.predict(X_te_new.values),
                       f"{args.mode} features + old params (untuned)")

    # ---- Optuna -------------------------------------------------------------
    print(f"\n[4/5] Optuna tuning ({args.n_trials} trials)...")
    best_params = tune_xgb_optuna(X_tr_new, y_train, args.n_trials)

    sample_weight = np.where(y_train.values >= LOG2_R, 3.0, 1.0)
    p_new = best_params.copy()
    n_est_new = int(p_new.pop("n_estimators"))
    xgb_new = xgb.XGBRegressor(
        n_estimators=n_est_new,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        **p_new,
    )
    xgb_new.fit(X_tr_new.values, y_train.values, sample_weight=sample_weight)
    res_tuned = evaluate(y_test.values, xgb_new.predict(X_te_new.values),
                         f"{args.mode} features + Optuna retuned")

    joblib.dump(xgb_new, MODELS_DIR / f"{model_tag}.pkl")
    with open(MODELS_DIR / f"{model_tag}_params.json", "w") as f:
        json.dump({**best_params, "n_estimators": n_est_new}, f, indent=2)
    print(f"  Saved -> models/{model_tag}.pkl")

    # ---- Summary ------------------------------------------------------------
    print("\n[5/5] Results:")
    print_table([res_base, res_old, res_tuned])

    delta = res_tuned["rmse_r"] - res_base["rmse_r"]
    print(f"\n  {args.mode} retuned vs baseline RMSE(R): {delta:+.4f} log2 units")
    if delta < 0:
        print("  -> Improvement confirmed.")
    else:
        print("  -> No improvement. Rolling MIC90 does not add signal beyond existing features.")
    print()


if __name__ == "__main__":
    main()
