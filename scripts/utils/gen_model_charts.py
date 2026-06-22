"""
Generate model evaluation charts from saved model + processed parquets.
Produces identical visual style for both species.

Outputs (to reports/model/):
  rmse_by_year_{species}.png        — XGBoost RMSE: all isolates vs resistant
  residuals_by_year_{species}.png   — Median residual + IQR band by year
  shap_beeswarm_{species}.png       — SHAP beeswarm (top 20 features)

Run:
  MPLBACKEND=Agg MPLCONFIGDIR=/tmp/.mpl \\
    .venv/bin/python scripts/utils/gen_model_charts.py --species kpneumoniae
  MPLBACKEND=Agg MPLCONFIGDIR=/tmp/.mpl \\
    .venv/bin/python scripts/utils/gen_model_charts.py --species abaumannii
"""

import argparse
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

EUCAST_R_LOG2 = np.log2(8)

SPECIES_LABELS = {
    "kpneumoniae": "K. pneumoniae",
    "abaumannii":  "A. baumannii",
}


def load_data(species: str):
    data_dir = ROOT / "data" / "processed" / species
    X_test = pd.read_parquet(data_dir / "X_test.parquet")
    y_test = pd.read_parquet(data_dir / "y_test.parquet").squeeze()
    return X_test, y_test


def load_model(species: str):
    model = joblib.load(MODELS / f"xgb_tuned_{species}.pkl")
    feature_names = json.loads((MODELS / f"feature_names_{species}.json").read_text())
    return model, feature_names


def plot_rmse_by_year(model, X_test, y_test, species: str) -> None:
    label = SPECIES_LABELS[species]
    years = X_test["year"].values
    preds = model.predict(X_test)

    rows = []
    for yr in sorted(np.unique(years)):
        mask = years == yr
        sq_err = (preds[mask] - y_test.values[mask]) ** 2
        r_mask = y_test.values[mask] >= EUCAST_R_LOG2
        rows.append({
            "year": int(yr),
            "rmse_all":       float(np.sqrt(sq_err.mean())),
            "rmse_resistant": float(np.sqrt(sq_err[r_mask].mean())) if r_mask.any() else np.nan,
        })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["year"], df["rmse_all"], "o-", color="#1f77b4", lw=2, ms=6, label="RMSE (all isolates)")
    ax.plot(df["year"], df["rmse_resistant"], "s--", color="#d62728", lw=2, ms=6,
            label="RMSE (resistant, MIC ≥ 8 mg/L)")
    ax.axvline(2019, color="gray", linestyle=":", lw=1.5, label="Test period start")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("RMSE (log₂ MIC units)", fontsize=11)
    ax.set_title(f"RMSE by Year — {label} XGBoost (test set 2019-2022)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = REPORTS / f"rmse_by_year_{species}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.name}")


def plot_residuals_by_year(model, X_test, y_test, species: str) -> None:
    label = SPECIES_LABELS[species]
    years = X_test["year"].values
    preds = model.predict(X_test)
    residuals = preds - y_test.values  # positive = model over-predicts

    df = pd.DataFrame({"year": years, "residual": residuals})
    grouped = df.groupby("year")["residual"]
    yr_vals = sorted(df["year"].unique())
    medians = [grouped.get_group(y).median() for y in yr_vals]
    q25 = [grouped.get_group(y).quantile(0.25) for y in yr_vals]
    q75 = [grouped.get_group(y).quantile(0.75) for y in yr_vals]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(yr_vals, q25, q75, alpha=0.2, color="#1f77b4", label="IQR (25th-75th)")
    ax.plot(yr_vals, medians, "o-", color="#1f77b4", lw=2, ms=6, label="Median residual")
    ax.axhline(0, color="black", linestyle="--", lw=1)
    ax.axvline(2019, color="gray", linestyle=":", lw=1.5, label="Test period start")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Residual (predicted - actual, log₂ MIC)", fontsize=11)
    ax.set_title(f"Residuals by Year — {label} XGBoost",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = REPORTS / f"residuals_by_year_{species}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.name}")


def plot_shap_beeswarm(model, X_test, species: str) -> None:
    import shap
    label = SPECIES_LABELS[species]
    print("  Computing SHAP values (this may take a minute)...")
    explainer = shap.TreeExplainer(model)
    sample = X_test.sample(min(3000, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(sample)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, sample, show=False, max_display=20, plot_type="dot")
    plt.title(f"SHAP feature importance — XGBoost tuned (test set) — {label}",
              fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    out = REPORTS / f"shap_beeswarm_{species}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", choices=["kpneumoniae", "abaumannii"],
                        default="abaumannii")
    parser.add_argument("--skip-shap", action="store_true",
                        help="Skip SHAP beeswarm (slow)")
    args = parser.parse_args()

    print(f"Loading model and data for {SPECIES_LABELS[args.species]}...")
    model, feature_names = load_model(args.species)
    X_test, y_test = load_data(args.species)
    X_test = X_test.reindex(columns=feature_names, fill_value=0)

    print("1/3 RMSE by year...")
    plot_rmse_by_year(model, X_test, y_test, args.species)

    print("2/3 Residuals by year...")
    plot_residuals_by_year(model, X_test, y_test, args.species)

    if not args.skip_shap:
        print("3/3 SHAP beeswarm...")
        plot_shap_beeswarm(model, X_test, args.species)
    else:
        print("3/3 SHAP beeswarm skipped.")

    print("Done.")


if __name__ == "__main__":
    main()
