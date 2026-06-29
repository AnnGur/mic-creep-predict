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

EUCAST_R         = 8
LOG2_R           = np.log2(EUCAST_R)
CREEP_SLOPE_LOG2 = 0.142   # +1.97 mg/L/yr ≈ +0.142 log2/yr at midpoint MIC

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
    """Train P(R) binary XGBoost classifier using regression hyperparams as structural priors."""
    p = params.copy()
    n_est = int(p.pop("n_estimators", 500))
    clf = xgb.XGBClassifier(
        n_estimators=n_est,
        objective="binary:logistic",
        eval_metric="logloss",
        seed=42,
        verbosity=0,
        **p,
    )
    clf.fit(X_train.values, (y_train.values >= LOG2_R).astype(int))
    return clf


def export_mic90_trend(X_train, y_train, X_test, y_test, model, clf=None, q90_model=None) -> None:
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
        rec = {
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
        if clf is not None:
            rec["pct_resistant_prob"] = round(float(g["prob_resistant"].mean() * 100), 2)
        records[int(yr)] = rec

    anchor_log2 = np.log2(records[2022]["predicted_mic90"])
    for yr in range(2023, 2027):
        delta = (yr - 2022) * CREEP_SLOPE_LOG2
        forecast_log2 = anchor_log2 + delta
        records[yr] = {
            "year":                int(yr),
            "forecast_mic90":      round(float(2 ** forecast_log2), 4),
            "forecast_log2_mic90": round(float(forecast_log2), 4),
            "source":              "forecast_extrapolated",
        }

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
    n = 3 if args.skip_shap else 4

    model_suffix = f"_{args.species}"
    data_dir = PROJECT_ROOT / "data" / "processed" / args.species

    # Module-level sentinel used by json save helpers — set before calling them
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

    print(f"\n[2/{n}] MIC90 trend (actual + predicted + forecast)")
    export_mic90_trend(X_train, y_train, X_test, y_test, model, clf=clf, q90_model=q90_model)

    print(f"\n[3/{n}] Country stats + censoring lookup")
    export_country_stats(X_test, y_test, model)
    export_censoring_lookup(X_train, X_test)

    if not args.skip_shap:
        print(f"\n[4/{n}] SHAP importance")
        export_shap_importance(model, X_test, feature_names)

    print(f"\nDone. API data -> {REPORTS}")
    print("Next: .venv/bin/python -m uvicorn src.api.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
