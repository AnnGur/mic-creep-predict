"""
Walk-forward cross-validation for MIC creep prediction.

Trains on expanding windows (2008 -> test_year-1), tests on a single year at a time.
Uses fixed hyperparams from Optuna (no re-tuning per fold).
Produces honest per-fold RMSE / R2 / P(R) calibration vs the single-split baseline.

Run:
    .venv/bin/python scripts/pipeline/5_run_walkforward_cv.py --species kpneumoniae
    .venv/bin/python scripts/pipeline/5_run_walkforward_cv.py --species abaumannii
    .venv/bin/python scripts/pipeline/5_run_walkforward_cv.py --species kpneumoniae --with-consumption
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

MODELS_DIR  = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR    = PROJECT_ROOT / "data" / "processed"

SPECIES_MAP = {
    "kpneumoniae": "Klebsiella pneumoniae",
    "abaumannii":  "Acinetobacter baumannii",
}

EUCAST_R   = 8
LOG2_R     = np.log2(EUCAST_R)
WALK_START = 2014   # first test year (ensures >= 6 training years)
WALK_END   = 2019   # last walk-forward fold


def _reconstruct_country(df: pd.DataFrame) -> pd.Series:
    """Rebuild country name from ctry_* OHE columns. Argentina = drop_first reference."""
    country_cols = [c for c in df.columns if c.startswith("ctry_")]
    country = pd.Series("Argentina", index=df.index)
    for col in country_cols:
        country[df[col] == 1] = col.replace("ctry_", "")
    return country


def _country_stratified_mic90(X_te: pd.DataFrame, pred: np.ndarray,
                               prob_r: np.ndarray) -> float:
    """Hybrid: global ceiling if global P(R) > 10%, else country-stratified ceiling."""
    global_pct_r = prob_r.mean()
    if global_pct_r > 0.10:
        return float(EUCAST_R * 4)

    df = X_te[["year"]].copy()
    df["log2_mic_predicted"] = pred
    df["prob_resistant"]     = prob_r
    df["country"]            = _reconstruct_country(X_te).values

    total_n  = len(df)
    weighted = 0.0
    for _ctry, cg in df.groupby("country"):
        c_pct_r = cg["prob_resistant"].mean()
        if c_pct_r > 0.10:
            c_mic90 = float(EUCAST_R * 4)
        else:
            c_mic90 = float(2 ** np.quantile(cg["log2_mic_predicted"].values, 0.90))
        weighted += (len(cg) / total_n) * c_mic90
    return round(weighted, 4)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all(species: str) -> tuple:
    d = DATA_DIR / species
    X_train = pd.read_parquet(d / "X_train.parquet")
    y_train = pd.read_parquet(d / "y_train.parquet").squeeze()
    X_test  = pd.read_parquet(d / "X_test.parquet")
    y_test  = pd.read_parquet(d / "y_test.parquet").squeeze()
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)
    X_all   = pd.concat([X_train, X_test])
    y_all   = pd.concat([y_train, y_test])
    return X_all, y_all


def load_consumption(species: str) -> pd.DataFrame | None:
    """Load carbapenem DDD country-year lookup if available."""
    path = PROJECT_ROOT / "data" / "external" / "carbapenem_ddd.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # Expected columns: country, year, ddd_per_1000_inh_day
    df.columns = [c.lower().strip() for c in df.columns]
    return df[["country", "year", "ddd_per_1000_inh_day"]].dropna()


def merge_consumption(X: pd.DataFrame, ddd_df: pd.DataFrame) -> pd.DataFrame:
    """Add carbapenem DDD as a feature. Missing -> median imputed."""
    # Country name comes from ctry_* dummies — reconstruct from OHE
    country_cols = [c for c in X.columns if c.startswith("ctry_")]
    country_series = pd.Series("Argentina", index=X.index)  # drop_first reference
    for col in country_cols:
        country_series[X[col] == 1] = col.replace("ctry_", "")

    tmp = X[["year"]].copy()
    tmp["country"] = country_series.values

    merged = tmp.merge(ddd_df, on=["country", "year"], how="left")
    median_ddd = merged["ddd_per_1000_inh_day"].median()
    X = X.copy()
    X["carbapenem_ddd"] = merged["ddd_per_1000_inh_day"].fillna(median_ddd).values
    return X


# ---------------------------------------------------------------------------
# Single fold
# ---------------------------------------------------------------------------

def run_fold(
    X_all: pd.DataFrame,
    y_all: pd.Series,
    test_year: int,
    params: dict,
    ddd_df: pd.DataFrame | None = None,
) -> dict | None:
    mask_train = X_all["year"] < test_year
    mask_test  = X_all["year"] == test_year

    X_tr, y_tr = X_all[mask_train].copy(), y_all[mask_train]
    X_te, y_te = X_all[mask_test].copy(),  y_all[mask_test]

    if len(X_te) == 0:
        return None

    # Add consumption feature if data available
    if ddd_df is not None:
        X_tr = merge_consumption(X_tr, ddd_df)
        X_te = merge_consumption(X_te, ddd_df)

    # Align columns (country dummies may differ between fold subsets)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)

    p     = params.copy()
    n_est = int(p.pop("n_estimators", 500))

    # --- Regression model ---
    reg = xgb.XGBRegressor(
        n_estimators=n_est, objective="reg:squarederror",
        seed=42, verbosity=0, **p,
    )
    reg.fit(X_tr.values, y_tr.values)
    pred = reg.predict(X_te.values)

    # --- P(R) classifier (recency-weighted) ---
    yrs    = X_tr["year"].values
    w      = 0.85 ** (int(yrs.max()) - yrs)
    clf    = xgb.XGBClassifier(
        n_estimators=n_est, objective="binary:logistic",
        seed=42, verbosity=0, **p,
    )
    clf.fit(X_tr.values, (y_tr.values >= LOG2_R).astype(int), sample_weight=w)
    prob_r = clf.predict_proba(X_te.values)[:, 1]

    actual_r   = (y_te.values >= LOG2_R)
    resist_idx = np.where(actual_r)[0]

    rmse_r = float(np.sqrt(mean_squared_error(
        y_te.values[resist_idx], pred[resist_idx]
    ))) if len(resist_idx) else float("nan")

    pct_r_actual = float(actual_r.mean() * 100)
    pct_r_pred   = float(prob_r.mean() * 100)

    mic90_actual      = round(float(2 ** np.quantile(y_te.values, 0.90)), 4)
    mic90_global      = float(EUCAST_R * 4) if pct_r_pred > 10 else \
                        round(float(2 ** np.quantile(pred, 0.90)), 4)
    mic90_stratified  = _country_stratified_mic90(X_te, pred, prob_r)

    return {
        "test_year":            test_year,
        "n_train":              int(len(X_tr)),
        "n_test":               int(len(X_te)),
        "n_resistant":          int(actual_r.sum()),
        "rmse_all":             round(float(np.sqrt(mean_squared_error(y_te, pred))), 4),
        "mae_all":              round(float(mean_absolute_error(y_te, pred)), 4),
        "r2_all":               round(float(r2_score(y_te, pred)), 4),
        "rmse_resistant":       round(rmse_r, 4),
        "mic90_actual":         mic90_actual,
        "mic90_global":         mic90_global,
        "mic90_stratified":     mic90_stratified,
        "pct_r_actual":         round(pct_r_actual, 2),
        "pct_r_predicted":      round(pct_r_pred, 2),
        "pr_error_pp":          round(pct_r_pred - pct_r_actual, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--species", choices=list(SPECIES_MAP), default="kpneumoniae")
    p.add_argument("--with-consumption", action="store_true",
                   help="Add carbapenem DDD covariate (requires data/external/carbapenem_ddd.csv)")
    return p.parse_args()


def print_table(results: list[dict], label: str) -> None:
    print(f"\n  {label}")
    print(f"  {'Year':<6} {'n_train':>8} {'n_test':>7} {'RMSE_all':>9} {'R2':>7} "
          f"{'RMSE_R':>8} {'MIC90_act':>10} {'MIC90_global':>13} {'MIC90_ctry':>11} "
          f"{'%R_act':>7} {'%R_pred':>8} {'PR_err':>7}")
    print("  " + "-" * 115)
    for r in results:
        mic90_g = r.get("mic90_global", r.get("mic90_predicted", float("nan")))
        mic90_s = r.get("mic90_stratified", float("nan"))
        print(
            f"  {r['test_year']:<6} {r['n_train']:>8,} {r['n_test']:>7,} "
            f"{r['rmse_all']:>9.4f} {r['r2_all']:>7.4f} "
            f"{r['rmse_resistant']:>8.4f} "
            f"{r['mic90_actual']:>10.1f} {mic90_g:>13.1f} {mic90_s:>11.1f} "
            f"{r['pct_r_actual']:>6.1f}% {r['pct_r_predicted']:>7.1f}% "
            f"{r['pr_error_pp']:>+7.2f}"
        )

    rmse_vals = [r["rmse_all"] for r in results]
    r2_vals   = [r["r2_all"] for r in results]
    rmse_r    = [r["rmse_resistant"] for r in results if not np.isnan(r["rmse_resistant"])]
    pr_errs   = [r["pr_error_pp"] for r in results]

    print("  " + "-" * 100)
    print(
        f"  {'mean':<6} {'':>8} {'':>7} "
        f"{np.mean(rmse_vals):>9.4f} {np.mean(r2_vals):>7.4f} "
        f"{np.mean(rmse_r):>8.4f} {'':>10} {'':>11} "
        f"{'':>7} {'':>8} {np.mean(pr_errs):>+7.2f}"
    )
    print(
        f"  {'std':<6} {'':>8} {'':>7} "
        f"{np.std(rmse_vals):>9.4f} {np.std(r2_vals):>7.4f} "
        f"{np.std(rmse_r):>8.4f}"
    )


def main() -> None:
    args = parse_args()
    sp   = args.species

    params_path = MODELS_DIR / f"xgb_best_params_{sp}.json"
    if not params_path.exists():
        print(f"No saved hyperparams at {params_path}. Run 3_run_model_training.py first.")
        sys.exit(1)
    params = json.loads(params_path.read_text())
    print(f"Loaded hyperparams for {SPECIES_MAP[sp]}")

    X_all, y_all = load_all(sp)
    print(f"All data: {len(X_all):,} rows, years {int(X_all['year'].min())}-{int(X_all['year'].max())}")

    ddd_df = None
    if args.with_consumption:
        ddd_df = load_consumption(sp)
        if ddd_df is None:
            print("  WARNING: --with-consumption set but data/external/carbapenem_ddd.csv not found.")
            print("  Proceeding without consumption covariate.")
        else:
            print(f"  Consumption data: {len(ddd_df):,} country-year rows loaded.")

    test_years = list(range(WALK_START, WALK_END + 1))
    print(f"\nRunning {len(test_years)} walk-forward folds: test years {test_years[0]}-{test_years[-1]}")

    results_base = []
    results_cons = []

    for yr in test_years:
        sys.stdout.write(f"  Fold test_year={yr}...")
        sys.stdout.flush()
        r_base = run_fold(X_all, y_all, yr, params, ddd_df=None)
        if r_base:
            results_base.append(r_base)
        if ddd_df is not None:
            r_cons = run_fold(X_all, y_all, yr, params, ddd_df=ddd_df)
            if r_cons:
                results_cons.append(r_cons)
        print(" done")

    print_table(results_base, "Baseline model (no consumption covariate)")

    if results_cons:
        print_table(results_cons, "With carbapenem DDD covariate")

        # Delta comparison
        print(f"\n  === Delta: consumption vs baseline ===")
        print(f"  {'Year':<6} {'ΔRMSE_all':>10} {'ΔR2':>8} {'ΔRMSE_R':>9} {'ΔPR_err':>9}")
        print("  " + "-" * 50)
        for b, c in zip(results_base, results_cons):
            print(
                f"  {b['test_year']:<6} "
                f"{c['rmse_all'] - b['rmse_all']:>+10.4f} "
                f"{c['r2_all'] - b['r2_all']:>+8.4f} "
                f"{c['rmse_resistant'] - b['rmse_resistant']:>+9.4f} "
                f"{c['pr_error_pp'] - b['pr_error_pp']:>+9.2f}"
            )
        avg_rmse_delta = np.mean([c["rmse_all"] - b["rmse_all"] for b, c in zip(results_base, results_cons)])
        avg_r2_delta   = np.mean([c["r2_all"] - b["r2_all"] for b, c in zip(results_base, results_cons)])
        print(f"  {'mean':<6} {avg_rmse_delta:>+10.4f} {avg_r2_delta:>+8.4f}")
        if avg_rmse_delta < 0:
            print(f"\n  Consumption covariate IMPROVES RMSE by {abs(avg_rmse_delta):.4f} log2 on average.")
        else:
            print(f"\n  Consumption covariate does NOT improve RMSE ({avg_rmse_delta:+.4f} log2 on average).")

    # Save results
    out = {
        "species":  sp,
        "baseline": results_base,
    }
    if results_cons:
        out["with_consumption"] = results_cons
    fname = REPORTS_DIR / f"walkforward_cv_{sp}.json"
    fname.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {fname}")


if __name__ == "__main__":
    main()
