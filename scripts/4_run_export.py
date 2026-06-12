"""
Export aggregated data for the API and frontend dashboard.

Reads the trained XGBoost model + processed parquets (local only),
computes aggregated statistics, and saves committable JSON files to reports/.

These JSON files contain NO raw patient data — only year/country-level
aggregates — and are safe to commit to the public repo.

Outputs (all in reports/):
    api_mic90_trend.json     — MIC90 by year (actual + model-predicted + forecast)
    api_country_stats.json   — resistance rate + MIC90 by country
    api_censoring_lookup.json — year -> pct_censored (used by live prediction endpoint)
    api_shap_importance.json — top 20 SHAP features

Run:
    .venv/bin/python scripts/4_run_export.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_DIR  = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS   = PROJECT_ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

EUCAST_R  = 8
LOG2_R    = np.log2(EUCAST_R)
# Linear creep slope from EDA (log2 units per year, fit on 2004-2018)
CREEP_SLOPE_LOG2 = 0.142   # +1.97 mg/L/yr ≈ +0.142 log2/yr at midpoint MIC


def load() -> tuple:
    print("  Loading parquets and model...")
    X_train = pd.read_parquet(DATA_DIR / "X_train.parquet")
    y_train = pd.read_parquet(DATA_DIR / "y_train.parquet").squeeze()
    X_test  = pd.read_parquet(DATA_DIR / "X_test.parquet")
    y_test  = pd.read_parquet(DATA_DIR / "y_test.parquet").squeeze()
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

    model = joblib.load(MODELS_DIR / "xgb_tuned.pkl")
    feature_names = json.loads((MODELS_DIR / "feature_names.json").read_text())

    return X_train, y_train, X_test, y_test, model, feature_names


# ---------------------------------------------------------------------------
# 1. MIC90 trend — actual (train), predicted (test), forecast (2023+)
# ---------------------------------------------------------------------------

def export_mic90_trend(X_train, y_train, X_test, y_test, model) -> None:
    records = {}

    # Actual values from training years
    train_df = X_train[["year"]].copy()
    train_df["log2_mic"] = y_train.values
    for yr, g in train_df.groupby("year"):
        records[int(yr)] = {
            "year":           int(yr),
            "actual_mic90":   round(float(2 ** g["log2_mic"].quantile(0.90)), 4),
            "actual_mic50":   round(float(2 ** g["log2_mic"].quantile(0.50)), 4),
            "pct_resistant":  round(float((g["log2_mic"] >= LOG2_R).mean() * 100), 2),
            "n":              int(len(g)),
            "source":         "train_actual",
        }

    # Actual + predicted for test years
    test_pred = model.predict(X_test)
    test_df = X_test[["year"]].copy()
    test_df["log2_mic_actual"]    = y_test.values
    test_df["log2_mic_predicted"] = test_pred
    for yr, g in test_df.groupby("year"):
        records[int(yr)] = {
            "year":               int(yr),
            "actual_mic90":       round(float(2 ** g["log2_mic_actual"].quantile(0.90)), 4),
            "predicted_mic90":    round(float(2 ** g["log2_mic_predicted"].quantile(0.90)), 4),
            "actual_mic50":       round(float(2 ** g["log2_mic_actual"].quantile(0.50)), 4),
            "predicted_mic50":    round(float(2 ** g["log2_mic_predicted"].quantile(0.50)), 4),
            "pct_resistant":      round(float((g["log2_mic_actual"] >= LOG2_R).mean() * 100), 2),
            "pct_resistant_pred": round(float((g["log2_mic_predicted"] >= LOG2_R).mean() * 100), 2),
            "n":                  int(len(g)),
            "source":             "test_actual_and_predicted",
        }

    # Forecast 2023-2026 — extrapolate from 2022 predicted MIC90 using creep slope
    anchor_log2 = np.log2(records[2022]["predicted_mic90"])
    for yr in range(2023, 2027):
        delta = (yr - 2022) * CREEP_SLOPE_LOG2
        forecast_log2 = anchor_log2 + delta
        records[yr] = {
            "year":              yr,
            "forecast_mic90":    round(float(2 ** forecast_log2), 4),
            "forecast_log2_mic90": round(float(forecast_log2), 4),
            "source":            "forecast_extrapolated",
        }

    out = sorted(records.values(), key=lambda r: r["year"])
    (REPORTS / "api_mic90_trend.json").write_text(json.dumps(out, indent=2))
    print(f"  -> reports/api_mic90_trend.json  ({len(out)} years)")


# ---------------------------------------------------------------------------
# 2. Country stats
# ---------------------------------------------------------------------------

def export_country_stats(X_test, y_test, model) -> None:
    test_pred = model.predict(X_test)
    df = X_test.copy()
    df["log2_mic_actual"]    = y_test.values
    df["log2_mic_predicted"] = test_pred

    # Reconstruct country from OHE columns
    country_cols = [c for c in df.columns if c.startswith("ctry_")]
    # The reference country (drop_first=True in OHE) maps to all zeros
    df["country"] = "Argentina"   # reference country (first alphabetically that was dropped)
    for col in country_cols:
        mask = df[col] == 1
        df.loc[mask, "country"] = col.replace("ctry_", "")

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
    (REPORTS / "api_country_stats.json").write_text(json.dumps(records, indent=2))
    print(f"  -> reports/api_country_stats.json  ({len(records)} countries)")


# ---------------------------------------------------------------------------
# 3. Censoring lookup (needed by live /predict endpoint)
# ---------------------------------------------------------------------------

def export_censoring_lookup(X_train, X_test) -> None:
    df = pd.concat([X_train[["year", "pct_censored_year"]],
                    X_test[["year", "pct_censored_year"]]])
    lookup = (
        df.groupby("year")["pct_censored_year"]
        .first()
        .round(4)
        .to_dict()
    )
    # For future years, use the last observed value
    last_val = lookup[max(lookup)]
    for yr in range(2023, 2031):
        lookup[yr] = last_val

    lookup_str_keys = {str(k): v for k, v in sorted(lookup.items())}
    (REPORTS / "api_censoring_lookup.json").write_text(json.dumps(lookup_str_keys, indent=2))
    print(f"  -> reports/api_censoring_lookup.json  ({len(lookup_str_keys)} years)")


# ---------------------------------------------------------------------------
# 4. SHAP importance
# ---------------------------------------------------------------------------

def export_shap_importance(model, X_test, feature_names: list) -> None:
    print("  Computing SHAP (may take ~1 min)...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    mean_abs    = np.abs(shap_values).mean(axis=0)

    top_n = 20
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    records = [
        {
            "rank":     int(i + 1),
            "feature":  feature_names[idx],
            "mean_abs_shap": round(float(mean_abs[idx]), 5),
            "note":     _shap_note(feature_names[idx]),
        }
        for i, idx in enumerate(top_idx)
    ]

    (REPORTS / "api_shap_importance.json").write_text(json.dumps(records, indent=2))
    print(f"  -> reports/api_shap_importance.json  (top {top_n} features)")


def _shap_note(feature: str) -> str:
    notes = {
        "KPC_pos":          "KPC carbapenemase — main resistance driver in training period",
        "is_censored":      "DATA ARTIFACT — censored = at panel floor; not a biological signal",
        "OXA_pos":          "OXA-48/OXA-232 carbapenemase — dominant in Europe and Middle East",
        "NDM_pos":          "NDM — fastest-rising mechanism; bypasses avibactam combinations",
        "pct_censored_year":"Surveillance methodology control — partial out panel artifact",
        "year":             "Temporal MIC creep signal",
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

def main() -> None:
    n = 4
    print(f"[1/{n}] Load data and model")
    X_train, y_train, X_test, y_test, model, feature_names = load()
    print(f"  Train: {len(X_train):,} rows  Test: {len(X_test):,} rows")

    print(f"\n[2/{n}] MIC90 trend (actual + predicted + forecast)")
    export_mic90_trend(X_train, y_train, X_test, y_test, model)

    print(f"\n[3/{n}] Country-level statistics")
    export_country_stats(X_test, y_test, model)

    print(f"\n[4/{n}] Censoring lookup + SHAP importance")
    export_censoring_lookup(X_train, X_test)
    export_shap_importance(model, X_test, feature_names)

    print(f"\nDone. API data files -> {REPORTS}")
    print("Next: .venv/bin/python -m uvicorn src.api.main:app --reload")


if __name__ == "__main__":
    main()
