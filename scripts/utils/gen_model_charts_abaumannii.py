"""
Generate missing A. baumannii model evaluation charts:
  shap_beeswarm_abaumannii.png
  rmse_by_year_abaumannii.png
  residuals_by_year_abaumannii.png

Run: MPLBACKEND=Agg MPLCONFIGDIR=/tmp/.mpl .venv/bin/python scripts/_gen_model_charts_ab.py
"""

import json
import numpy as np
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "model"
MODELS  = ROOT / "models"
REPORTS.mkdir(parents=True, exist_ok=True)
DATA    = ROOT / "data" / "processed" / "abaumannii"
SUFFIX  = "_abaumannii"
EUCAST_R_LOG2 = np.log2(8)


def load_data():
    X_test = pd.read_parquet(DATA / "X_test.parquet")
    y_test = pd.read_parquet(DATA / "y_test.parquet").squeeze()
    return X_test, y_test


def plot_rmse_by_year(model, X_test, y_test):
    years = X_test["year"].values
    preds = model.predict(X_test)
    residuals = preds - y_test.values

    rows = []
    for yr in sorted(np.unique(years)):
        mask = years == yr
        sq_err = (preds[mask] - y_test.values[mask]) ** 2
        rows.append({
            "year": int(yr),
            "rmse_all": float(np.sqrt(sq_err.mean())),
            "rmse_resistant": float(np.sqrt(sq_err[y_test.values[mask] >= EUCAST_R_LOG2].mean()))
            if (y_test.values[mask] >= EUCAST_R_LOG2).any() else np.nan,
        })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["year"], df["rmse_all"], "o-", color="#1f77b4", lw=2, ms=6, label="RMSE (all)")
    ax.plot(df["year"], df["rmse_resistant"], "s--", color="#d62728", lw=2, ms=6, label="RMSE (resistant)")
    ax.axvline(2019, color="gray", linestyle=":", lw=1.5, label="Test period start")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("RMSE (log2 MIC units)", fontsize=11)
    ax.set_title("RMSE by Year - A. baumannii XGBoost (test set 2019-2022)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = REPORTS / f"rmse_by_year{SUFFIX}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.name}")


def plot_residuals_by_year(model, X_test, y_test):
    years = X_test["year"].values
    preds = model.predict(X_test)
    residuals = preds - y_test.values

    df = pd.DataFrame({"year": years, "residual": residuals})
    grouped = df.groupby("year")["residual"]
    yr_vals = sorted(df["year"].unique())
    medians = [grouped.get_group(y).median() for y in yr_vals]
    q25 = [grouped.get_group(y).quantile(0.25) for y in yr_vals]
    q75 = [grouped.get_group(y).quantile(0.75) for y in yr_vals]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(yr_vals, q25, q75, alpha=0.2, color="#1f77b4", label="IQR")
    ax.plot(yr_vals, medians, "o-", color="#1f77b4", lw=2, ms=6, label="Median residual")
    ax.axhline(0, color="black", linestyle="--", lw=1)
    ax.axvline(2019, color="gray", linestyle=":", lw=1.5, label="Test period start")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Residual (pred - actual, log2 MIC)", fontsize=11)
    ax.set_title("Residuals by Year - A. baumannii XGBoost", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = REPORTS / f"residuals_by_year{SUFFIX}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.name}")


def plot_shap_beeswarm(model, X_test):
    import shap
    print("  Computing SHAP values (this may take a minute)...")
    explainer = shap.TreeExplainer(model)
    sample = X_test.sample(min(3000, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(sample)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, sample, show=False, max_display=20,
                      plot_type="dot")
    plt.title("SHAP feature importance - XGBoost tuned (test set) - A. baumannii",
              fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    out = REPORTS / f"shap_beeswarm{SUFFIX}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.name}")


if __name__ == "__main__":
    print("Loading model and data...")
    model = joblib.load(MODELS / "xgb_tuned_abaumannii.pkl")

    feature_names = json.loads((MODELS / "feature_names_abaumannii.json").read_text())
    X_test, y_test = load_data()
    X_test = X_test.reindex(columns=feature_names, fill_value=0)

    print("1/3 RMSE by year...")
    plot_rmse_by_year(model, X_test, y_test)

    print("2/3 Residuals by year...")
    plot_residuals_by_year(model, X_test, y_test)

    print("3/3 SHAP beeswarm...")
    plot_shap_beeswarm(model, X_test)

    print("Done.")
