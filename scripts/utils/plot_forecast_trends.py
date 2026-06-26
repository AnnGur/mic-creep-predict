"""
Forecast trend plots for K. pneumoniae and A. baumannii.
=========================================================
Visualises model performance on the test set (2019-2022) against
the historical training period (2004-2018) for two outputs:

  Row 1 — MIC90 trend: actual vs XGBoost regression predicted
  Row 2 — % Resistant trend: actual vs P(R) classifier predicted

Saves:
    reports/model/forecast_trends.png

Run:
    .venv/bin/python scripts/utils/plot_forecast_trends.py
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports" / "model"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

EUCAST_R   = 8.0
LOG2_R     = np.log2(EUCAST_R)
TRAIN_END  = 2018
TEST_START = 2019
RANDOM_STATE = 42

SPECIES = {
    "kpneumoniae": "K. pneumoniae",
    "abaumannii":  "A. baumannii",
}

COLORS = {
    "actual_train": "#2c7bb6",
    "actual_test":  "#2c7bb6",
    "predicted":    "#d7191c",
    "ci":           "#fdae61",
    "train_bg":     "#f7f7f7",
    "test_bg":      "#fff7ec",
    "breakpoint":   "#d73027",
    "split":        "#636363",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data(species: str):
    d = PROJECT_ROOT / "data" / "processed" / species
    X_train = pd.read_parquet(d / "X_train.parquet").reset_index(drop=True)
    y_train = pd.read_parquet(d / "y_train.parquet").squeeze().reset_index(drop=True)
    X_test  = pd.read_parquet(d / "X_test.parquet").reset_index(drop=True)
    y_test  = pd.read_parquet(d / "y_test.parquet").squeeze().reset_index(drop=True)
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)
    return X_train, y_train, X_test, y_test


def train_classifier(X_train: pd.DataFrame, y_train: pd.Series,
                     species: str) -> xgb.XGBClassifier:
    """Train P(R) binary classifier using regression hyperparams as structural priors."""
    params = json.load(open(MODELS_DIR / f"xgb_best_params_{species}.json"))
    p = params.copy()
    n_est = int(p.pop("n_estimators", 500))
    clf = xgb.XGBClassifier(
        n_estimators=n_est,
        objective="binary:logistic",
        eval_metric="logloss",
        seed=RANDOM_STATE,
        verbosity=0,
        **p,
    )
    y_bin = (y_train.values >= LOG2_R).astype(int)
    clf.fit(X_train.values, y_bin)
    return clf


# ---------------------------------------------------------------------------
# Trend computation
# ---------------------------------------------------------------------------

def mic90_by_year(year: pd.Series, mic: np.ndarray) -> pd.Series:
    return pd.Series(mic, index=year.values).groupby(level=0).quantile(0.9)


def resistant_pct_by_year(year: pd.Series, mic: np.ndarray) -> pd.Series:
    return pd.Series((mic >= LOG2_R).astype(float),
                     index=year.values).groupby(level=0).mean() * 100


def prob_resistant_by_year(year: pd.Series, prob: np.ndarray) -> pd.Series:
    return pd.Series(prob, index=year.values).groupby(level=0).mean() * 100


def bootstrap_ci(year: pd.Series, mic: np.ndarray, func, n_boot: int = 200,
                 q_lo: float = 0.1, q_hi: float = 0.9) -> tuple:
    """Bootstrap confidence interval for a per-year aggregate."""
    rng = np.random.default_rng(42)
    boot_results = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(mic), size=len(mic))
        boot_results.append(func(year, mic[idx]))
    boot_df = pd.concat(boot_results, axis=1)
    return boot_df.quantile(q_lo, axis=1), boot_df.quantile(q_hi, axis=1)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_species_column(axes_mic, axes_res, species: str, label: str):
    print(f"  Processing {label}...")
    X_train, y_train, X_test, y_test = load_data(species)

    # Load regression model
    reg_model = joblib.load(MODELS_DIR / f"xgb_tuned_{species}.pkl")
    y_pred_reg = reg_model.predict(X_test.values)

    # Train classifier
    clf = train_classifier(X_train, y_train, species)
    y_prob = clf.predict_proba(X_test.values)[:, 1]
    y_bin_test = (y_test.values >= LOG2_R).astype(int)
    auc = roc_auc_score(y_bin_test, y_prob)
    print(f"    P(R) AUC-ROC: {auc:.4f}")

    year_tr = X_train["year"]
    year_te = X_test["year"]
    all_years_tr = sorted(year_tr.unique())
    all_years_te = sorted(year_te.unique())

    # ---- MIC90 plot -------------------------------------------------------
    ax = axes_mic

    # Training period background + actual
    ax.axvspan(all_years_tr[0] - 0.4, TRAIN_END + 0.4,
               color=COLORS["train_bg"], zorder=0, label="_nolegend_")
    ax.axvspan(TEST_START - 0.4, all_years_te[-1] + 0.4,
               color=COLORS["test_bg"], zorder=0, label="_nolegend_")

    mic90_tr = mic90_by_year(year_tr, y_train.values)
    mic90_te_actual = mic90_by_year(year_te, y_test.values)
    mic90_te_pred   = mic90_by_year(year_te, y_pred_reg)

    # Bootstrap CI for test predictions
    ci_lo, ci_hi = bootstrap_ci(year_te, y_pred_reg, mic90_by_year)

    ax.plot(mic90_tr.index, mic90_tr.values,
            color=COLORS["actual_train"], lw=2.5, marker="o", ms=5, zorder=3,
            label="Actual (train)")
    ax.plot(mic90_te_actual.index, mic90_te_actual.values,
            color=COLORS["actual_test"], lw=2.5, marker="o", ms=5,
            linestyle="--", zorder=3, label="Actual (test)")
    ax.plot(mic90_te_pred.index, mic90_te_pred.values,
            color=COLORS["predicted"], lw=2.5, marker="s", ms=5, zorder=3,
            label="XGBoost predicted (test)")
    ax.fill_between(ci_lo.index, ci_lo.values, ci_hi.values,
                    color=COLORS["predicted"], alpha=0.15, zorder=2,
                    label="Predicted 80% CI")

    ax.axhline(LOG2_R, color=COLORS["breakpoint"], lw=1.5, ls=":",
               zorder=4, label=f"EUCAST R (MIC=8 mg/L)")
    ax.axvline(TRAIN_END + 0.5, color=COLORS["split"], lw=1.2, ls="--",
               zorder=4, alpha=0.7)

    ax.set_title(label, fontsize=14, fontweight="bold", pad=10)
    ax.set_ylabel("MICₐ₀ (log₂ units)", fontsize=11)
    ax.set_xlabel("")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xticks(sorted(set(all_years_tr) | set(all_years_te)))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    # Annotate train/test split
    ax.text(TRAIN_END + 0.6, ax.get_ylim()[0] + 0.1, "test\nperiod",
            fontsize=7, color=COLORS["split"], va="bottom")

    # ---- Resistant % plot -------------------------------------------------
    ax2 = axes_res

    ax2.axvspan(all_years_tr[0] - 0.4, TRAIN_END + 0.4,
                color=COLORS["train_bg"], zorder=0)
    ax2.axvspan(TEST_START - 0.4, all_years_te[-1] + 0.4,
                color=COLORS["test_bg"], zorder=0)

    res_tr       = resistant_pct_by_year(year_tr, y_train.values)
    res_te_actual = resistant_pct_by_year(year_te, y_test.values)
    res_te_pred   = prob_resistant_by_year(year_te, y_prob)

    ci_lo_r, ci_hi_r = bootstrap_ci(year_te, y_prob, prob_resistant_by_year)

    ax2.plot(res_tr.index, res_tr.values,
             color=COLORS["actual_train"], lw=2.5, marker="o", ms=5, zorder=3,
             label="Actual % resistant (train)")
    ax2.plot(res_te_actual.index, res_te_actual.values,
             color=COLORS["actual_test"], lw=2.5, marker="o", ms=5,
             linestyle="--", zorder=3, label="Actual % resistant (test)")
    ax2.plot(res_te_pred.index, res_te_pred.values,
             color=COLORS["predicted"], lw=2.5, marker="s", ms=5, zorder=3,
             label=f"P(R) classifier mean prob (AUC={auc:.3f})")
    ax2.fill_between(ci_lo_r.index, ci_lo_r.values, ci_hi_r.values,
                     color=COLORS["predicted"], alpha=0.15, zorder=2)

    ax2.axvline(TRAIN_END + 0.5, color=COLORS["split"], lw=1.2, ls="--",
                zorder=4, alpha=0.7)

    ax2.set_ylabel("% Resistant isolates", fontsize=11)
    ax2.set_xlabel("Year", fontsize=11)
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.set_xticks(sorted(set(all_years_tr) | set(all_years_te)))
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(axis="y", alpha=0.3)


def main():
    print("\n=== Forecast Trend Plots — K. pneumoniae + A. baumannii ===\n")

    fig, axes = plt.subplots(
        2, 2,
        figsize=(18, 10),
        gridspec_kw={"hspace": 0.45, "wspace": 0.28},
    )

    for col, (species, label) in enumerate(SPECIES.items()):
        plot_species_column(
            axes_mic=axes[0, col],
            axes_res=axes[1, col],
            species=species,
            label=label,
        )

    # Row labels
    fig.text(0.01, 0.73, "MICₐ₀ trend", va="center", ha="left",
             fontsize=12, fontweight="bold", rotation=90, color="#333333")
    fig.text(0.01, 0.27, "Resistance rate", va="center", ha="left",
             fontsize=12, fontweight="bold", rotation=90, color="#333333")

    fig.suptitle(
        "MIC Creep Prediction — Meropenem — Model Forecast vs Actual (test 2019-2022)\n"
        "Gray background = training period  |  Tan background = test period  |"
        "  Dotted red = EUCAST R breakpoint",
        fontsize=11, y=1.01, color="#333333",
    )

    out = REPORTS_DIR / "forecast_trends.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Saved -> {out.relative_to(PROJECT_ROOT)}")
    plt.close()


if __name__ == "__main__":
    main()
