"""
Model training — K. pneumoniae + Meropenem MIC Creep
=====================================================
Step 1: Random Forest baseline (no tuning)
Step 2: XGBoost Regressor with Optuna hyperparameter search
Step 3: Evaluation — overall RMSE/MAE + resistant-subset RMSE + MIC_90 trend
Step 4: SHAP feature importance
Step 5: Save model artefacts

Inputs (from 2_run_feature_engineering.py):
    data/processed/X_train.parquet
    data/processed/y_train.parquet
    data/processed/X_test.parquet
    data/processed/y_test.parquet

Outputs:
    models/rf_baseline.pkl
    models/xgb_tuned.pkl
    models/feature_names.json
    reports/model_results.md
    reports/shap_summary.png
    reports/shap_beeswarm.png
    reports/mic90_trend_predicted.png
    reports/residuals_by_year.png
    reports/rmse_by_year.png

Run:
    .venv/bin/python scripts/3_run_model_training.py
    .venv/bin/python scripts/3_run_model_training.py --skip-optuna   # fast baseline only
    .venv/bin/python scripts/3_run_model_training.py --n-trials 30   # fewer Optuna trials
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import joblib
import optuna
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "processed"
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS      = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

EUCAST_R      = 8          # mg/L resistance breakpoint
LOG2_R        = np.log2(EUCAST_R)   # = 3.0 in log2 space
RANDOM_STATE  = 42
N_TRIALS_DEFAULT = 60

plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (13, 5)})


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train = pd.read_parquet(DATA_DIR / "X_train.parquet")
    y_train = pd.read_parquet(DATA_DIR / "y_train.parquet").squeeze()
    X_test  = pd.read_parquet(DATA_DIR / "X_test.parquet")
    y_test  = pd.read_parquet(DATA_DIR / "y_test.parquet").squeeze()

    # Align columns (test may miss country dummies if some appeared only in train)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    print(f"  Train: {X_train.shape[0]:,} rows x {X_train.shape[1]} features")
    print(f"  Test:  {X_test.shape[0]:,} rows x {X_test.shape[1]} features")
    print(f"  Target range train: [{y_train.min():.2f}, {y_train.max():.2f}]  "
          f"mean={y_train.mean():.2f}")
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, label: str) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)

    mask_r = y_true >= LOG2_R
    rmse_r = np.sqrt(mean_squared_error(y_true[mask_r], y_pred[mask_r])) if mask_r.any() else np.nan
    mae_r  = mean_absolute_error(y_true[mask_r], y_pred[mask_r]) if mask_r.any() else np.nan
    n_r    = mask_r.sum()

    print(f"\n  [{label}]")
    print(f"    RMSE (all)       = {rmse:.4f}")
    print(f"    MAE  (all)       = {mae:.4f}")
    print(f"    RMSE (R subset)  = {rmse_r:.4f}  (n={n_r:,})")
    print(f"    MAE  (R subset)  = {mae_r:.4f}")

    return {"label": label, "rmse": rmse, "mae": mae,
            "rmse_resistant": rmse_r, "mae_resistant": mae_r, "n_resistant": int(n_r)}


# ---------------------------------------------------------------------------
# Random Forest baseline
# ---------------------------------------------------------------------------

def train_rf(X_train, y_train) -> RandomForestRegressor:
    print("  Fitting Random Forest baseline (200 trees, default params)...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_features="sqrt",
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, MODELS_DIR / "rf_baseline.pkl")
    print("  Saved -> models/rf_baseline.pkl")
    return rf


# ---------------------------------------------------------------------------
# XGBoost + Optuna
# ---------------------------------------------------------------------------

def tune_xgb(X_train, y_train, n_trials: int) -> xgb.XGBRegressor:
    # Resistant isolates get 3x weight so the model doesn't ignore the tail
    sample_weight = np.where(y_train >= LOG2_R, 3.0, 1.0)

    def objective(trial):
        params = {
            "n_estimators":       trial.suggest_int("n_estimators", 300, 1200, step=100),
            "max_depth":          trial.suggest_int("max_depth", 3, 8),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":   trial.suggest_int("min_child_weight", 1, 20),
            "gamma":              trial.suggest_float("gamma", 0.0, 2.0),
            "reg_alpha":          trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda":         trial.suggest_float("reg_lambda", 0.5, 10.0),
            "tree_method":        "hist",
            "random_state":       RANDOM_STATE,
            "n_jobs":             -1,
        }
        model = xgb.XGBRegressor(**params)

        # Time-aware CV: last 3 years of train as internal val
        val_mask  = X_train["year"] >= (X_train["year"].max() - 2)
        X_tr, X_v = X_train[~val_mask], X_train[val_mask]
        y_tr, y_v = y_train[~val_mask], y_train[val_mask]
        sw_tr     = sample_weight[~val_mask]

        model.fit(X_tr, y_tr, sample_weight=sw_tr,
                  eval_set=[(X_v, y_v)], verbose=False)
        pred = model.predict(X_v)
        return np.sqrt(mean_squared_error(y_v, pred))

    print(f"  Running Optuna ({n_trials} trials) — this takes a few minutes...")
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    print(f"  Best val RMSE: {study.best_value:.4f}")
    print(f"  Best params:   {best}")

    # Refit on full train with best params
    xgb_best = xgb.XGBRegressor(**best, tree_method="hist",
                                 random_state=RANDOM_STATE, n_jobs=-1)
    xgb_best.fit(X_train, y_train, sample_weight=sample_weight)
    joblib.dump(xgb_best, MODELS_DIR / "xgb_tuned.pkl")
    print("  Saved -> models/xgb_tuned.pkl")

    with open(MODELS_DIR / "xgb_best_params.json", "w") as f:
        json.dump(best, f, indent=2)

    return xgb_best


# ---------------------------------------------------------------------------
# MIC_90 trend: actual vs predicted
# ---------------------------------------------------------------------------

def plot_mic90_trend(X_test, y_test, rf_pred, xgb_pred) -> None:
    df = X_test[["year"]].copy()
    df["actual"]  = y_test.values
    df["rf_pred"] = rf_pred
    df["xgb_pred"]= xgb_pred

    def mic90_log2(s): return s.quantile(0.90)

    yr = df.groupby("year").agg(
        actual  =("actual",   mic90_log2),
        rf_pred =("rf_pred",  mic90_log2),
        xgb_pred=("xgb_pred", mic90_log2),
    ).reset_index()

    fig, ax = plt.subplots()
    ax.plot(yr["year"], yr["actual"],   "o-",  color="#1f77b4", lw=2, ms=6, label="Actual MIC_90")
    ax.plot(yr["year"], yr["rf_pred"],  "s--", color="#2ca02c", lw=1.5, ms=5, label="RF baseline")
    ax.plot(yr["year"], yr["xgb_pred"], "^-",  color="#d62728", lw=2, ms=5, label="XGBoost tuned")
    ax.axhline(LOG2_R, color="black", linestyle=":", lw=1.5, label="EUCAST R (log2=3)")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("MIC_90 (log2 scale)", fontsize=11)
    ax.set_title("MIC_90 Trend — Actual vs Predicted (Test 2019-2022)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(REPORTS / "mic90_trend_predicted.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> reports/mic90_trend_predicted.png")

    print("\n  MIC_90 (log2) by year — test set:")
    print(f"  {'Year':>4}  {'Actual':>7}  {'RF':>7}  {'XGB':>7}")
    for _, r in yr.iterrows():
        print(f"  {int(r.year):4d}  {r.actual:7.3f}  {r.rf_pred:7.3f}  {r.xgb_pred:7.3f}")


# ---------------------------------------------------------------------------
# Residuals by year
# ---------------------------------------------------------------------------

def plot_residuals(X_test, y_test, xgb_pred) -> None:
    df = X_test[["year"]].copy()
    df["residual"] = y_test.values - xgb_pred

    yr = df.groupby("year")["residual"].agg(["mean", "std", "median"]).reset_index()

    fig, ax = plt.subplots()
    ax.bar(yr["year"], yr["mean"], color="#d62728", alpha=0.7, label="Mean residual")
    ax.errorbar(yr["year"], yr["mean"], yerr=yr["std"], fmt="none",
                ecolor="grey", capsize=4, lw=1.5)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Residual (actual - predicted, log2 MIC)", fontsize=11)
    ax.set_title("XGBoost residuals by year — Test set", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(REPORTS / "residuals_by_year.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> reports/residuals_by_year.png")


# ---------------------------------------------------------------------------
# RMSE by year
# ---------------------------------------------------------------------------

def plot_rmse_by_year(X_test, y_test, rf_pred, xgb_pred) -> None:
    df = X_test[["year"]].copy()
    df["actual"]   = y_test.values
    df["rf_pred"]  = rf_pred
    df["xgb_pred"] = xgb_pred

    rows = []
    for yr, g in df.groupby("year"):
        rows.append({
            "year":     yr,
            "rf_rmse":  np.sqrt(mean_squared_error(g["actual"], g["rf_pred"])),
            "xgb_rmse": np.sqrt(mean_squared_error(g["actual"], g["xgb_pred"])),
        })
    yr_df = pd.DataFrame(rows)

    fig, ax = plt.subplots()
    w = 0.35
    x = np.arange(len(yr_df))
    ax.bar(x - w/2, yr_df["rf_rmse"],  width=w, color="#2ca02c", alpha=0.8, label="RF baseline")
    ax.bar(x + w/2, yr_df["xgb_rmse"], width=w, color="#d62728", alpha=0.8, label="XGBoost tuned")
    ax.set_xticks(x)
    ax.set_xticklabels(yr_df["year"].astype(int))
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("RMSE (log2 MIC)", fontsize=11)
    ax.set_title("RMSE by year — RF vs XGBoost (Test 2019-2022)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(REPORTS / "rmse_by_year.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> reports/rmse_by_year.png")


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def plot_shap(model: xgb.XGBRegressor, X_test: pd.DataFrame) -> None:
    print("  Computing SHAP values (this may take ~1 min)...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary dot plot — top 20 features
    fig, ax = plt.subplots(figsize=(10, 9))
    shap.summary_plot(shap_values, X_test, max_display=20,
                      show=False, plot_type="dot")
    plt.title("SHAP feature importance — XGBoost tuned (test set)",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(REPORTS / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> reports/shap_beeswarm.png")

    # Bar summary
    fig, ax = plt.subplots(figsize=(9, 7))
    shap.summary_plot(shap_values, X_test, max_display=20,
                      show=False, plot_type="bar")
    plt.title("SHAP mean |value| — top 20 features",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(REPORTS / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> reports/shap_summary.png")

    # Top 15 features by mean |SHAP|
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:15]
    print("\n  Top 15 features by mean |SHAP|:")
    print(f"  {'Feature':<28} {'mean|SHAP|':>10}")
    print("  " + "-" * 40)
    for i in top_idx:
        print(f"  {X_test.columns[i]:<28} {mean_abs[i]:10.4f}")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_report(results: list[dict], xgb_params: dict) -> None:
    lines = [
        "# Model Training Results — K. pneumoniae + Meropenem\n",
        f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n",
        "---\n",
        "## Evaluation Metrics\n",
        "| Model | RMSE (all) | MAE (all) | RMSE (R subset) | MAE (R subset) | N resistant |\n",
        "|---|---|---|---|---|---|\n",
    ]
    for r in results:
        lines.append(
            f"| {r['label']} | {r['rmse']:.4f} | {r['mae']:.4f} | "
            f"{r['rmse_resistant']:.4f} | {r['mae_resistant']:.4f} | {r['n_resistant']:,} |\n"
        )

    lines += [
        "\n> **Note**: RMSE on the resistant subset (MIC >= 8 mg/L) is the clinically relevant metric. "
        "The full-set RMSE is dominated by the ~75% of isolates imputed at the censoring floor (log2=-5).\n\n",
        "---\n\n",
        "## MIC_90 Trend — Actual vs Predicted\n\n",
        "![MIC_90 trend predicted](mic90_trend_predicted.png)\n\n",
        "---\n\n",
        "## Residuals by Year\n\n",
        "![Residuals](residuals_by_year.png)\n\n",
        "---\n\n",
        "## RMSE by Year\n\n",
        "![RMSE by year](rmse_by_year.png)\n\n",
        "---\n\n",
        "## SHAP Feature Importance\n\n",
        "![SHAP beeswarm](shap_beeswarm.png)\n\n",
        "![SHAP bar](shap_summary.png)\n\n",
        "---\n\n",
        "## XGBoost Best Hyperparameters (Optuna)\n\n",
        "```json\n",
        json.dumps(xgb_params, indent=2),
        "\n```\n\n",
        "---\n\n",
        "## Next Steps\n\n",
        "- Review SHAP values with domain expert — confirm biological plausibility\n",
        "- Flag `is_censored` in SHAP plots as a data-structure artifact (not biology)\n",
        "- Build FastAPI endpoint: `src/api/main.py`\n",
        "- Push model artefact to Hugging Face Hub\n",
        "- Prepare submission write-up\n",
    ]

    out = REPORTS / "model_results.md"
    out.write_text("".join(lines))
    print(f"  -> reports/model_results.md")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-optuna", action="store_true",
                   help="Run RF baseline only — skip XGBoost Optuna tuning")
    p.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT,
                   help=f"Optuna trial count (default: {N_TRIALS_DEFAULT})")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n = 5 if args.skip_optuna else 6

    print(f"[1/{n}] Load processed data")
    X_train, y_train, X_test, y_test = load_data()

    # Save feature names once for the API to load
    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(list(X_train.columns), f)

    print(f"\n[2/{n}] Random Forest baseline")
    rf = train_rf(X_train, y_train)
    rf_pred  = rf.predict(X_test)
    rf_metrics = compute_metrics(y_test, rf_pred, "RF baseline")

    if args.skip_optuna:
        print("\n  --skip-optuna set: skipping XGBoost tuning.")
        print("\n  Saving dummy XGBoost (same as RF predictions for reports)...")
        xgb_model  = rf          # placeholder so plots still render
        xgb_pred   = rf_pred
        xgb_metrics = rf_metrics
        xgb_params  = {}
        results     = [rf_metrics]
    else:
        print(f"\n[3/{n}] XGBoost + Optuna ({args.n_trials} trials)")
        xgb_model  = tune_xgb(X_train, y_train, n_trials=args.n_trials)
        xgb_pred   = xgb_model.predict(X_test)
        xgb_metrics = compute_metrics(y_test, xgb_pred, "XGBoost tuned")
        with open(MODELS_DIR / "xgb_best_params.json") as f:
            xgb_params = json.load(f)
        results = [rf_metrics, xgb_metrics]

    print(f"\n[{n-2}/{n}] MIC_90 trend + residual plots")
    plot_mic90_trend(X_test, y_test, rf_pred, xgb_pred)
    plot_residuals(X_test, y_test, xgb_pred)
    plot_rmse_by_year(X_test, y_test, rf_pred, xgb_pred)

    print(f"\n[{n-1}/{n}] SHAP")
    if args.skip_optuna:
        print("  (skipped — no XGBoost model)")
    else:
        plot_shap(xgb_model, X_test)

    print(f"\n[{n}/{n}] Write report")
    write_report(results, xgb_params)

    print(f"\nDone. Models -> {MODELS_DIR}  Reports -> {REPORTS}")


if __name__ == "__main__":
    main()
