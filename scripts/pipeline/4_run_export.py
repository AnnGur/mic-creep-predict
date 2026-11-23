"""
Export aggregated data for the API and frontend dashboard.

Reads the trained XGBoost model + processed parquets (local only),
computes aggregated statistics, and saves committable JSON files to reports/.

These JSON files contain NO raw patient data — only year/country-level
aggregates — and are safe to commit to the public repo.

Outputs (all in reports/api/):
    api_{species}_mic90_trend.json      — MIC90 by year (actual + model-predicted + forecast)
    api_{species}_country_stats.json    — resistance rate + MIC90 by country
    api_{species}_censoring_lookup.json — year -> pct_censored (used by live prediction endpoint)
    api_{species}_shap_importance.json  — top 20 SHAP features (skipped if --skip-shap)

Run:
    .venv/bin/python scripts/4_run_export.py             # full
    .venv/bin/python scripts/4_run_export.py --skip-shap # skip slow SHAP import
"""

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
print("Importing libraries...", flush=True)

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

SPECIES_MAP = {
    "kpneumoniae": "Klebsiella pneumoniae",
    "abaumannii":  "Acinetobacter baumannii",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS    = PROJECT_ROOT / "reports" / "api"
REPORTS.mkdir(parents=True, exist_ok=True)

EUCAST_R           = 8
LOG2_R             = np.log2(EUCAST_R)
CREEP_SLOPE_LOG2   = 0.142   # +1.97 mg/L/yr ≈ +0.142 log2/yr at midpoint MIC
PANEL_CEILING_MIC  = 32.0    # ATLAS meropenem panel ceiling
PANEL_CEILING_LOG2 = np.log2(PANEL_CEILING_MIC)

_json_suffix = ""  # set in main() based on --species arg; used by export helpers


def load(data_dir: Path, model_suffix: str) -> tuple:
    print("  Loading parquets and model...")
    X_train = pd.read_parquet(data_dir / "X_train.parquet")
    y_train = pd.read_parquet(data_dir / "y_train.parquet").squeeze()
    X_test  = pd.read_parquet(data_dir / "X_test.parquet")
    y_test  = pd.read_parquet(data_dir / "y_test.parquet").squeeze()
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

    model         = joblib.load(MODELS_DIR / f"xgb_tuned{model_suffix}.pkl")
    feature_names = json.loads((MODELS_DIR / f"feature_names{model_suffix}.json").read_text())

    q90_path = MODELS_DIR / f"xgb_quantile{model_suffix}.pkl"
    q90_model = joblib.load(q90_path) if q90_path.exists() else None
    if q90_model:
        print("  Quantile model (Q0.90) loaded.")

    return X_train, y_train, X_test, y_test, model, feature_names, q90_model


# ---------------------------------------------------------------------------
# 1. MIC90 trend — actual (train), predicted (test), forecast (2023+)
# ---------------------------------------------------------------------------

def train_pr_classifier(X_train: pd.DataFrame, y_train: pd.Series, params: dict) -> xgb.XGBClassifier:
    """Train P(R) binary XGBoost classifier using regression hyperparams as structural priors.

    Recency weighting: years closer to the test window get exponentially higher weight.
    This counteracts dilution from supplementary datasets (e.g. Entasis) that have lower
    resistance rates in overlapping years, and helps calibration near the train/test boundary.
    """
    p = params.copy()
    n_est = int(p.pop("n_estimators", 500))

    years = X_train["year"].values
    max_year = int(years.max())
    # Exponential recency weight: each year back from max_year is discounted by 0.85
    sample_weight = 0.85 ** (max_year - years)

    clf = xgb.XGBClassifier(
        n_estimators=n_est,
        objective="binary:logistic",
        eval_metric="logloss",
        seed=42,
        verbosity=0,
        **p,
    )
    clf.fit(X_train.values, (y_train.values >= LOG2_R).astype(int),
            sample_weight=sample_weight)
    return clf


def _reconstruct_country(df: pd.DataFrame) -> pd.Series:
    """Rebuild country name from ctry_* OHE columns. Argentina is the drop_first reference."""
    country_cols = [c for c in df.columns if c.startswith("ctry_")]
    country = pd.Series("Argentina", index=df.index)
    for col in country_cols:
        country[df[col] == 1] = col.replace("ctry_", "")
    return country


def _country_stratified_mic90(g: pd.DataFrame, pr_col: str = "prob_resistant") -> float:
    """Hybrid MIC90 estimator: global ceiling when P(R) > 10%, else country-stratified.

    When the GLOBAL mean P(R) already exceeds 10%, the population 90th percentile
    mathematically falls in the resistant subpopulation — use the panel ceiling (32).
    Below 10% globally, apply the ceiling correction per country: high-resistance
    countries (Greece, Italy) contribute 32 mg/L while low-resistance countries
    contribute the regression Q0.90. This captures cross-country heterogeneity
    during the transitional period before global threshold is crossed.
    """
    global_pct_r = g[pr_col].mean() if pr_col in g.columns else \
                   (g["log2_mic_predicted"] >= LOG2_R).mean()
    if global_pct_r > 0.10:
        return float(EUCAST_R * 4)   # global threshold — panel ceiling

    country_series = _reconstruct_country(g)
    total_n = len(g)
    weighted = 0.0
    for _ctry, cg in g.groupby(country_series):
        c_pct_r = cg[pr_col].mean() if pr_col in cg.columns else \
                  (cg["log2_mic_predicted"] >= LOG2_R).mean()
        if c_pct_r > 0.10:
            c_mic90 = float(EUCAST_R * 4)
        else:
            c_mic90 = float(2 ** cg["log2_mic_predicted"].quantile(0.90))
        weighted += (len(cg) / total_n) * c_mic90
    return round(weighted, 4)


def _standardised_prevalence(X_train, y_train, X_test, y_test) -> dict:
    """Yearly resistance prevalence (%) re-weighted to a fixed country mix.

    Weights = pooled 2008-2022 country shares. Removes year-to-year changes in
    which countries contribute isolates (A. baumannii test years over-represent
    high-resistance countries). Isolates with no country column set use the
    reference country (see _reconstruct_country).
    """
    X_all = pd.concat([X_train, X_test.reindex(columns=X_train.columns, fill_value=0)],
                      ignore_index=True)
    y_all = np.concatenate([y_train.values, y_test.values])
    df = pd.DataFrame({"year": X_all["year"].astype(int).values,
                       "country": _reconstruct_country(X_all).values,
                       "r": (y_all >= LOG2_R).astype(int)})
    share = df["country"].value_counts(normalize=True)
    by = df.groupby(["year", "country"])["r"].mean().unstack()
    out = {}
    for yr, row in by.iterrows():
        row = row.dropna()
        w = share.reindex(row.index)
        out[int(yr)] = round(float((row * w / w.sum()).sum() * 100), 1)
    return out


def export_mic90_trend(X_train, y_train, X_test, y_test, model, clf=None, q90_model=None,
                       standardised=None) -> None:
    records = {}

    train_df = X_train[["year"]].copy()
    train_df["log2_mic"] = y_train.values
    for yr, g in train_df.groupby("year"):
        records[int(yr)] = {
            "year":          int(yr),
            "actual_mic90":  round(float(2 ** g["log2_mic"].quantile(0.90)), 4),
            "actual_mic50":  round(float(2 ** g["log2_mic"].quantile(0.50)), 4),
            "pct_resistant": round(float((g["log2_mic"] >= LOG2_R).mean() * 100), 2),
            "n":             int(len(g)),
            "source":        "train_actual",
        }

    test_pred = model.predict(X_test)
    test_prob = clf.predict_proba(X_test.values)[:, 1] if clf is not None else None
    test_df = X_test[["year"]].copy()
    test_df["log2_mic_actual"]    = y_test.values
    test_df["log2_mic_predicted"] = test_pred
    if test_prob is not None:
        test_df["prob_resistant"] = test_prob
    for yr, g in test_df.groupby("year"):
        predicted_mic90 = _country_stratified_mic90(g)

        rec = {
            "year":               int(yr),
            "actual_mic90":       round(float(2 ** g["log2_mic_actual"].quantile(0.90)), 4),
            "predicted_mic90":    predicted_mic90,
            "actual_mic50":       round(float(2 ** g["log2_mic_actual"].quantile(0.50)), 4),
            "predicted_mic50":    round(float(2 ** g["log2_mic_predicted"].quantile(0.50)), 4),
            "pct_resistant":      round(float((g["log2_mic_actual"] >= LOG2_R).mean() * 100), 2),
            "pct_resistant_pred": round(float((g["log2_mic_predicted"] >= LOG2_R).mean() * 100), 2),
            "n":                  int(len(g)),
            "source":             "test_actual_and_predicted",
        }
        if clf is not None:
            rec["pct_resistant_prob"] = round(float(g["prob_resistant"].mean() * 100), 2)
        records[int(yr)] = rec

    anchor_log2 = np.log2(records[2022]["predicted_mic90"])
    for yr in range(2023, 2027):
        delta = (yr - 2022) * CREEP_SLOPE_LOG2
        forecast_log2 = min(anchor_log2 + delta, PANEL_CEILING_LOG2)
        records[yr] = {
            "year":                int(yr),
            "forecast_mic90":      round(float(2 ** forecast_log2), 4),
            "forecast_log2_mic90": round(float(forecast_log2), 4),
            "source":              "forecast_extrapolated",
        }

    if standardised:
        for yr, v in standardised.items():
            if yr in records:
                records[yr]["pct_resistant_standardised"] = v

    out = sorted(records.values(), key=lambda r: r["year"])
    fname = f"api{_json_suffix}_mic90_trend.json"
    (REPORTS / fname).write_text(json.dumps(out, indent=2))
    print(f"  -> reports/{fname}  ({len(out)} years)")


# ---------------------------------------------------------------------------
# 2. Country stats
# ---------------------------------------------------------------------------

def export_country_stats(X_test, y_test, model) -> None:
    test_pred = model.predict(X_test)
    df = X_test.copy()
    df["log2_mic_actual"]    = y_test.values
    df["log2_mic_predicted"] = test_pred

    country_cols = [c for c in df.columns if c.startswith("ctry_")]
    df["country"] = "Argentina"
    for col in country_cols:
        df.loc[df[col] == 1, "country"] = col.replace("ctry_", "")

    records = []
    for country, g in df.groupby("country"):
        if len(g) < 10:
            continue
        records.append({
            "country":            country,
            "n":                  int(len(g)),
            "pct_resistant":      round(float((g["log2_mic_actual"] >= LOG2_R).mean() * 100), 2),
            "pct_resistant_pred": round(float((g["log2_mic_predicted"] >= LOG2_R).mean() * 100), 2),
            "mic90_actual":       round(float(2 ** g["log2_mic_actual"].quantile(0.90)), 4),
            "mic90_predicted":    round(float(2 ** g["log2_mic_predicted"].quantile(0.90)), 4),
        })

    records.sort(key=lambda r: r["pct_resistant"], reverse=True)
    fname = f"api{_json_suffix}_country_stats.json"
    (REPORTS / fname).write_text(json.dumps(records, indent=2))
    print(f"  -> reports/{fname}  ({len(records)} countries)")


# ---------------------------------------------------------------------------
# 3. Censoring lookup
# ---------------------------------------------------------------------------

def export_censoring_lookup(X_train, X_test) -> None:
    df = pd.concat([X_train[["year", "pct_censored_year"]],
                    X_test[["year", "pct_censored_year"]]])
    lookup = {int(k): float(v) for k, v in
              df.groupby("year")["pct_censored_year"].first().round(4).items()}
    last_val = lookup[max(lookup)]
    for yr in range(2023, 2031):
        lookup[yr] = last_val
    out = {str(k): v for k, v in sorted(lookup.items())}
    fname = f"api{_json_suffix}_censoring_lookup.json"
    (REPORTS / fname).write_text(json.dumps(out, indent=2))
    print(f"  -> reports/{fname}  ({len(out)} years)")


# ---------------------------------------------------------------------------
# 4. SHAP importance (lazy import — skip if shap hangs)
# ---------------------------------------------------------------------------

def export_shap_importance(model, X_test, feature_names: list) -> None:
    print("  Importing shap (slow on first run — Ctrl+C to skip)...")
    try:
        import shap  # lazy import to avoid blocking main pipeline
    except Exception as e:
        print(f"  shap import failed ({e}), skipping.")
        return

    print("  Computing SHAP values (~1 min)...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    mean_abs    = np.abs(shap_values).mean(axis=0)

    top_idx = np.argsort(mean_abs)[::-1][:20]
    records = [
        {
            "rank":          int(i + 1),
            "feature":       feature_names[idx],
            "mean_abs_shap": round(float(mean_abs[idx]), 5),
            "note":          _shap_note(feature_names[idx]),
        }
        for i, idx in enumerate(top_idx)
    ]
    fname = f"api{_json_suffix}_shap_importance.json"
    (REPORTS / fname).write_text(json.dumps(records, indent=2))
    print(f"  -> reports/{fname}  (top 20 features)")


# ---------------------------------------------------------------------------
# 5. Binary prevalence trend (A. baumannii only)
# ---------------------------------------------------------------------------

def export_prevalence_trend(X_train, y_train, X_test, y_test, clf) -> None:
    """Binary resistance prevalence trend — the correct model for A. baumannii.

    A. baumannii MIC is bimodal (0.25 mg/L susceptible cluster vs 64-128 mg/L
    resistant cluster) with almost nothing in between. Regression on continuous
    MIC is misspecified. This function models the year-level PREVALENCE of
    resistance (P(MIC >= 8)) as the primary output, with:
      - Wilson 95% CI on actual observed prevalence per year
      - P(R) classifier predictions per year (train + test)
      - Logistic trend fit (linear on logit scale) fitted on train years
      - Trend extrapolated to forecast years 2023-2026
    """
    from scipy import stats as scipy_stats

    TRAIN_END_YR = 2018

    # Per-isolate predictions for both splits
    train_df = X_train[["year"]].copy()
    train_df["resistant"] = (y_train.values >= LOG2_R).astype(int)
    train_df["prob_r"]    = clf.predict_proba(X_train.values)[:, 1]
    train_df["split"]     = "train"

    test_df = X_test[["year"]].copy()
    test_df["resistant"] = (y_test.values >= LOG2_R).astype(int)
    test_df["prob_r"]    = clf.predict_proba(X_test.values)[:, 1]
    test_df["split"]     = "test"

    all_df = pd.concat([train_df, test_df])

    def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
        p = k / n
        denom  = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        return max(0.0, center - margin), min(1.0, center + margin)

    year_records = []
    for yr, g in all_df.groupby("year"):
        n  = len(g)
        k  = int(g["resistant"].sum())
        ci_lo, ci_hi = wilson_ci(k, n)
        year_records.append({
            "year":             int(yr),
            "n":                n,
            "n_resistant":      k,
            "pct_resistant":    round(k / n * 100, 2),
            "pct_predicted":    round(float(g["prob_r"].mean() * 100), 2),
            "ci_lo_95":         round(ci_lo * 100, 2),
            "ci_hi_95":         round(ci_hi * 100, 2),
            "split":            g["split"].iloc[0],
        })

    # Logistic trend fitted on TRAIN years only (prospective: no test leakage)
    train_rows = [r for r in year_records if r["split"] == "train"]
    train_yrs  = np.array([r["year"] for r in train_rows])
    train_p    = np.clip([r["pct_resistant"] / 100 for r in train_rows], 0.001, 0.999)
    logit_p    = np.log(train_p / (1 - train_p))

    slope, intercept, r_val, p_val, _ = scipy_stats.linregress(train_yrs, logit_p)

    # Apply trend across all observed + forecast years
    all_years = sorted({r["year"] for r in year_records} | set(range(2023, 2027)))
    trend_by_year = {}
    for yr in all_years:
        logit = intercept + slope * yr
        trend_by_year[yr] = round(float(100 / (1 + np.exp(-logit))), 2)

    for r in year_records:
        r["trend_fitted_pct"] = trend_by_year[r["year"]]

    # Forecast rows
    for yr in range(2023, 2027):
        year_records.append({
            "year":             yr,
            "trend_fitted_pct": trend_by_year[yr],
            "split":            "forecast",
        })

    output = {
        "species":        "Acinetobacter baumannii",
        "primary_metric": True,
        "metric_rationale": (
            "A. baumannii MIC is bimodal (susceptible ~0.25 mg/L vs resistant "
            "64-128 mg/L) with almost no intermediate values. Regression on "
            "continuous MIC is misspecified for this distribution. Resistance "
            "prevalence (P(MIC >= 8)) is the scientifically correct primary output."
        ),
        "trend_model": {
            "type":                  "logistic (linear on logit scale)",
            "fit_years":             f"{int(train_yrs.min())}-{int(train_yrs.max())}",
            "slope_logit_per_year":  round(float(slope), 4),
            "r_squared":             round(float(r_val ** 2), 4),
            "p_value":               round(float(p_val), 6),
            "interpretation":        (
                f"Resistance prevalence {'rising' if slope > 0 else 'falling'} at "
                f"{abs(slope):.4f} logit/yr on training period; "
                f"p={'<0.001' if p_val < 0.001 else f'{p_val:.4f}'}"
            ),
        },
        "years": sorted(year_records, key=lambda r: r["year"]),
    }

    fname = "api_abaumannii_prevalence_trend.json"
    (REPORTS / fname).write_text(json.dumps(output, indent=2))
    print(f"  -> reports/{fname}  ({len(year_records)} years)")
    print(f"  Logistic trend: slope={slope:.4f} logit/yr, R²={r_val**2:.3f}, p={p_val:.4f}")


def _shap_note(feature: str) -> str:
    notes = {
        "KPC_pos":           "KPC carbapenemase — main resistance driver in training period",
        "is_censored":       "DATA ARTIFACT — censored = at panel floor; not a biological signal",
        "OXA_pos":           "OXA-48/OXA-232 carbapenemase — dominant in Europe and Middle East",
        "NDM_pos":           "NDM — fastest-rising mechanism; bypasses avibactam combinations",
        "pct_censored_year": "Surveillance methodology control — partial out panel artifact",
        "year":              "Temporal MIC creep signal",
    }
    if feature in notes:
        return notes[feature]
    if feature.startswith("ctry_"):
        return f"Country effect: {feature[5:]}"
    if feature.endswith("_pos"):
        return f"{feature[:-4]} carbapenemase gene"
    if feature.startswith("spec_"):
        return f"Specimen type: {feature[5:]}"
    return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-shap", action="store_true",
                   help="Skip SHAP importance export (avoids slow IPython import)")
    p.add_argument("--species", choices=list(SPECIES_MAP), default="kpneumoniae",
                   help="Target species (default: kpneumoniae)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    is_abau = args.species == "abaumannii"

    # A. baumannii primary metric is resistance prevalence (bimodal distribution,
    # regression on MIC is misspecified). MIC90 trend is exported as secondary.
    if is_abau:
        n = 3 if args.skip_shap else 4
        step_prevalence, step_mic90, step_country = 2, 3, 3
    else:
        n = 3 if args.skip_shap else 4
        step_prevalence, step_mic90, step_country = None, 2, 3

    model_suffix = f"_{args.species}"
    data_dir = PROJECT_ROOT / "data" / "processed" / args.species

    global _json_suffix
    _json_suffix = f"_{args.species}"

    print(f"[1/{n}] Load data and model  [{SPECIES_MAP[args.species]}]")
    X_train, y_train, X_test, y_test, model, feature_names, q90_model = load(data_dir, model_suffix)
    print(f"  Train: {len(X_train):,} rows  Test: {len(X_test):,} rows")

    params_path = MODELS_DIR / f"xgb_best_params_{args.species}.json"
    clf = None
    if params_path.exists():
        params = json.loads(params_path.read_text())
        print("  Training P(R) classifier...")
        clf = train_pr_classifier(X_train, y_train, params)

    if is_abau and clf is not None:
        print(f"\n[{step_prevalence}/{n}] Resistance prevalence trend  "
              f"[PRIMARY metric for A. baumannii]")
        export_prevalence_trend(X_train, y_train, X_test, y_test, clf)
        print(f"\n[{step_mic90}/{n}] MIC90 trend  [secondary — ceiling-limited]")
    else:
        print(f"\n[{step_mic90}/{n}] MIC90 trend (actual + predicted + forecast)")

    std_prev = _standardised_prevalence(X_train, y_train, X_test, y_test) if is_abau else None
    export_mic90_trend(X_train, y_train, X_test, y_test, model, clf=clf, q90_model=q90_model,
                       standardised=std_prev)

    print(f"\n[{step_country}/{n}] Country stats + censoring lookup")
    export_country_stats(X_test, y_test, model)
    export_censoring_lookup(X_train, X_test)

    if not args.skip_shap:
        print(f"\n[4/{n}] SHAP importance")
        export_shap_importance(model, X_test, feature_names)

    print(f"\nDone. API data -> {REPORTS}")
    print("Next: .venv/bin/python -m uvicorn src.api.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
