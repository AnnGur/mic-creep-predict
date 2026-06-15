"""
FastAPI inference endpoint — MIC Creep Prediction
==================================================
Serves predictions and pre-aggregated surveillance data for both
Klebsiella pneumoniae and Acinetobacter baumannii + Meropenem.

Startup behaviour:
  - Dev  (MODEL_SOURCE=local): loads models from disk
  - Prod (MODEL_SOURCE=huggingface): downloads artefacts from HF Hub
  - Pre-computed JSON data (reports/api_*.json) is loaded once at startup
  - Missing A. baumannii artefacts are skipped gracefully (model not yet trained)

Endpoints:
  GET  /health
  GET  /methodology
  GET  /api/trend/mic90          ?species=kpneumoniae|abaumannii
  GET  /api/country-stats        ?species=kpneumoniae|abaumannii
  GET  /api/features/importance  ?species=kpneumoniae|abaumannii
  GET  /api/countries            ?species=kpneumoniae|abaumannii
  POST /api/predict              body includes species field

Run locally:
  .venv/bin/python -m uvicorn src.api.main:app --reload --port 8000
"""

import json
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost  # noqa: F401 — must import at module level before asyncio event loop starts (Mac OpenMP deadlock)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS      = PROJECT_ROOT / "reports"

MODEL_SOURCE = os.getenv("MODEL_SOURCE", "local")
HF_REPO      = os.getenv("HF_MODEL_REPO", "")

EUCAST_R  = 8.0   # mg/L — EUCAST 2024 R breakpoint for both species + meropenem
LOG2_R    = np.log2(EUCAST_R)

SPECIES_MAP = {
    "kpneumoniae": "Klebsiella pneumoniae",
    "abaumannii":  "Acinetobacter baumannii",
}
SPECIES_LABEL = {
    "kpneumoniae": "K. pneumoniae",
    "abaumannii":  "A. baumannii",
}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MIC Creep Prediction API",
    description=(
        "Predicts Minimum Inhibitory Concentration (MIC) drift for "
        "Klebsiella pneumoniae and Acinetobacter baumannii + Meropenem."
    ),
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# State loaded at startup — keyed by species slug
# ---------------------------------------------------------------------------

_models:           dict[str, object]      = {}
_feature_names:    dict[str, list[str]]   = {}
_mic90_trend:      dict[str, list[dict]]  = {}
_country_stats:    dict[str, list[dict]]  = {}
_shap_importance:  dict[str, list[dict]]  = {}
_censoring_lookup: dict[str, dict[int, float]] = {}


def _model_files(species: str) -> tuple[str, str]:
    """Return (model_pkl_name, feature_names_json_name) for a species slug."""
    return f"xgb_tuned_{species}.pkl", f"feature_names_{species}.json"


def _json_prefix(species: str) -> str:
    return f"api_{species}"


@app.on_event("startup")
async def startup_event() -> None:
    for species in SPECIES_MAP:
        model_fname, features_fname = _model_files(species)
        model_path    = MODELS_DIR / model_fname
        features_path = MODELS_DIR / features_fname

        if not model_path.exists():
            print(f"  [{species}] model not found — skipping ({model_path})")
            continue

        if MODEL_SOURCE == "huggingface" and HF_REPO:
            from huggingface_hub import hf_hub_download
            pkl_path  = hf_hub_download(repo_id=HF_REPO, filename=model_fname)
            json_path = hf_hub_download(repo_id=HF_REPO, filename=features_fname)
            _models[species]        = joblib.load(pkl_path)
            _feature_names[species] = json.loads(Path(json_path).read_text())
        else:
            _models[species]        = joblib.load(model_path)
            _feature_names[species] = json.loads(features_path.read_text())

        prefix = _json_prefix(species)

        def _load(fname):
            p = REPORTS / fname
            return json.loads(p.read_text()) if p.exists() else []

        _mic90_trend[species]     = _load(f"{prefix}_mic90_trend.json")
        _country_stats[species]   = _load(f"{prefix}_country_stats.json")
        _shap_importance[species] = _load(f"{prefix}_shap_importance.json")

        raw_cens = _load(f"{prefix}_censoring_lookup.json")
        _censoring_lookup[species] = (
            {int(float(k)): float(v) for k, v in raw_cens.items()}
            if isinstance(raw_cens, dict) else {}
        )

        print(f"  [{species}] loaded — {len(_feature_names[species])} features")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    species:       str   = Field("kpneumoniae", description="'kpneumoniae' or 'abaumannii'")
    year:          int   = Field(..., ge=2004, le=2030, description="Isolation year")
    country:       str   = Field(..., description="Country name (e.g. 'Ukraine')")
    sex:           str   = Field("Male", description="'Male' or 'Female'")
    age_group:     str   = Field("Adult", description="'Paediatric', 'Adult', or 'Elderly'")
    specimen_type: str   = Field("other", description="wound | blood | respiratory | urine | peritoneal | other")
    kpc_pos:       bool  = Field(False)
    ndm_pos:       bool  = Field(False)
    oxa_pos:       bool  = Field(False)
    vim_pos:       bool  = Field(False)
    imp_pos:       bool  = Field(False)
    ges_pos:       bool  = Field(False)


class PredictionResponse(BaseModel):
    species:                str
    log2_mic:               float
    mic_mg_l:               float
    interpretation:         str    # "susceptible" | "intermediate" | "resistant"
    eucast_breakpoint_mg_l: float  = EUCAST_R
    input_echo:             dict


# ---------------------------------------------------------------------------
# Feature vector builder
# ---------------------------------------------------------------------------

def _build_feature_vector(req: PredictionRequest) -> pd.DataFrame:
    feature_names = _feature_names[req.species]
    row = {f: 0 for f in feature_names}

    row["year"]           = req.year
    row["gender_male"]    = 1 if req.sex.lower() == "male" else 0
    row["age_paediatric"] = 1 if req.age_group.lower() == "paediatric" else 0
    row["age_elderly"]    = 1 if req.age_group.lower() == "elderly" else 0

    row["military_proxy"] = int(
        req.specimen_type.lower() == "wound"
        and req.sex.lower() == "male"
        and req.age_group.lower() == "adult"
    )

    spec_col = f"spec_{req.specimen_type.lower()}"
    if spec_col in row:
        row[spec_col] = 1

    ctry_col = f"ctry_{req.country}"
    if ctry_col in row:
        row[ctry_col] = 1

    row["KPC_pos"] = int(req.kpc_pos)
    row["NDM_pos"] = int(req.ndm_pos)
    row["OXA_pos"] = int(req.oxa_pos)
    row["VIM_pos"] = int(req.vim_pos)
    row["IMP_pos"] = int(req.imp_pos)
    row["GES_pos"] = int(req.ges_pos)

    row["is_censored"]       = 0
    row["pct_censored_year"] = _censoring_lookup[req.species].get(req.year, 0.88)

    return pd.DataFrame([row])[feature_names]


def _interpret(log2_mic: float) -> str:
    mic = 2 ** log2_mic
    if mic > EUCAST_R:
        return "resistant"
    if mic > EUCAST_R / 2:
        return "intermediate"
    return "susceptible"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_species(species: str) -> None:
    if species not in SPECIES_MAP:
        raise HTTPException(status_code=422, detail=f"species must be one of {list(SPECIES_MAP)}")
    if species not in _models:
        raise HTTPException(status_code=503,
                            detail=f"Model for {SPECIES_MAP[species]} not loaded. "
                                   "Run scripts/2_run_feature_engineering.py and "
                                   "scripts/3_run_model_training.py with --species "
                                   f"{species} first.")


# ---------------------------------------------------------------------------
# Methodology HTML page
# ---------------------------------------------------------------------------

METHODOLOGY_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Methodology - MIC Creep Prediction</title>
<style>
  *, *::before, *::after { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 860px; margin: 40px auto; padding: 0 20px;
         color: #1a1a2e; line-height: 1.7; font-size: 16px; }
  h1 { color: #16213e; border-bottom: 3px solid #e94560; padding-bottom: 12px;
       font-size: clamp(1.3rem, 4vw, 1.9rem); }
  h2 { color: #16213e; margin-top: 2em; font-size: clamp(1.1rem, 3vw, 1.4rem); }
  h3 { color: #444; font-size: 1rem; }
  .badges { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 4px; }
  .badge { display: inline-block; background: #e94560; color: white;
           border-radius: 4px; padding: 3px 10px; font-size: 0.82em; white-space: nowrap; }
  .badge.green { background: #27ae60; }
  .card { background: #f8f9fa; border-left: 4px solid #e94560;
          padding: 14px 18px; border-radius: 4px; margin: 16px 0; }
  .card.green { border-left-color: #27ae60; }
  .table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 16px 0; }
  table { border-collapse: collapse; width: 100%; min-width: 420px; }
  th { background: #16213e; color: white; padding: 8px 12px; text-align: left;
       font-size: 0.9em; white-space: nowrap; }
  td { padding: 8px 12px; border-bottom: 1px solid #ddd; font-size: 0.9em; }
  tr:hover td { background: #f0f4ff; }
  code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px;
         font-family: "SFMono-Regular", Consolas, monospace; font-size: 0.85em;
         word-break: break-all; }
  .endpoint { font-family: monospace; background: #1a1a2e; color: #e94560;
              padding: 2px 8px; border-radius: 3px; font-size: 0.85em;
              display: inline-block; word-break: break-all; }
  ul { padding-left: 1.4em; }
  li { margin-bottom: 0.4em; }
  footer { margin-top: 48px; font-size: 0.85em; color: #888;
           border-top: 1px solid #ddd; padding-top: 16px; }
  @media (max-width: 600px) {
    body { margin: 20px auto; font-size: 15px; }
    h2 { margin-top: 1.5em; }
    .card { padding: 12px 14px; }
    th, td { padding: 7px 9px; }
    footer { flex-direction: column; gap: 6px; }
  }
</style>
</head>
<body>

<h1>MIC Creep Prediction — Methodology</h1>
<div class="badges">
  <span class="badge green">K. pneumoniae</span>
  <span class="badge green">A. baumannii</span>
</div>

<h2>Research Question</h2>
<p>Can we predict the trajectory of antimicrobial MIC creep for <em>Klebsiella pneumoniae</em>
and <em>Acinetobacter baumannii</em> using global surveillance data, and identify the molecular
and geographic drivers of resistance escalation?</p>

<div class="card">
  <strong>MIC creep</strong> is the gradual upward drift in Minimum Inhibitory Concentration (MIC)
  values across a bacterial population over time. It is a pre-resistance signal that becomes
  clinically critical when the population MIC90 approaches or crosses the EUCAST resistance
  breakpoint. For both species + meropenem, EUCAST 2024 sets <strong>R &gt; 8 mg/L</strong>.
</div>

<h2>Data</h2>
<p><strong>Dataset:</strong> ATLAS (Antimicrobial Testing Leadership and Surveillance),
Pfizer/Vivli. Controlled access via Vivli AMR Surveillance Platform.</p>
<div class="table-wrap"><table>
  <tr><th>Species</th><th>Antibiotic</th><th>Isolates</th><th>Years</th><th>Countries</th></tr>
  <tr><td><em>K. pneumoniae</em></td><td>Meropenem</td><td>89,572</td><td>2004-2022</td><td>66</td></tr>
  <tr><td><em>A. baumannii</em></td><td>Meropenem</td><td>ATLAS cohort</td><td>2004-2022</td><td>66</td></tr>
</table></div>

<h2>Key Finding — MIC Creep is Confirmed</h2>
<div class="card green">
  For <em>K. pneumoniae</em> + meropenem: MIC90 slope = <strong>+1.97 mg/L/yr</strong>,
  R&sup2; = 0.67, p = 6.3 &times; 10&#8315;&sup6;.
  Resistance rate rose from 5% (2007) to 20% (2024) &mdash; a 4&times; increase in 17 years.
</div>
<p>The mechanistic signature is <strong>divergence between MIC50 and MIC90</strong>: MIC50 stays
flat at ~0.03 mg/L for 20 years while MIC90 climbs. This is the fingerprint of a growing
resistant subpopulation, not a population-wide shift.</p>
<p>Carbapenemase gene drivers in <em>K. pneumoniae</em>: KPC dominated 2004-2016, then NDM and OXA rose
sharply from 2017 onward. NDM now peaks at ~9% (2022) and bypasses ceftazidime-avibactam, one of
the last-resort salvage drugs. In <em>A. baumannii</em>, OXA-type carbapenemases dominate throughout.</p>

<h2>Model</h2>
<p>Independent <strong>XGBoost regression models</strong> are trained for each species with
Optuna hyperparameter tuning (60 trials). The target is <code>log2(MIC)</code>.
A <strong>3x sample weight</strong> is applied to resistant isolates (MIC &ge; 8 mg/L) to prevent
the model from ignoring the clinically relevant tail.</p>
<h3>Train/Test Split</h3>
<p>Strictly time-ordered: <strong>Train 2004-2018 | Test 2019-2022</strong>.
No random shuffling (would constitute data leakage).</p>
<h3>Key Features</h3>
<div class="table-wrap"><table>
  <tr><th>Feature</th><th>Type</th><th>Rationale</th></tr>
  <tr><td><code>year</code></td><td>continuous</td><td>Primary creep driver</td></tr>
  <tr><td><code>KPC_pos</code> ... <code>GES_pos</code></td><td>binary</td><td>Carbapenemase gene presence</td></tr>
  <tr><td><code>ctry_*</code></td><td>OHE</td><td>Country of isolation (65 dummies)</td></tr>
  <tr><td><code>spec_*</code></td><td>OHE</td><td>Specimen source (5 categories)</td></tr>
  <tr><td><code>is_censored</code></td><td>binary</td><td>Data-structure artifact (panel floor)</td></tr>
  <tr><td><code>pct_censored_year</code></td><td>float</td><td>Controls for panel methodology changes</td></tr>
  <tr><td><code>military_proxy</code></td><td>binary</td><td>Wound + male + 18-60; combat-wound proxy</td></tr>
</table></div>

<h2>Results — <em>K. pneumoniae</em></h2>
<div class="table-wrap"><table>
  <tr><th>Model</th><th>RMSE (all)</th><th>RMSE (resistant subset)</th><th>MAE (resistant)</th></tr>
  <tr><td>RF baseline</td><td>1.558</td><td>2.869</td><td>2.180</td></tr>
  <tr><td><strong>XGBoost tuned</strong></td><td>1.758</td><td><strong>1.960</strong></td><td><strong>1.127</strong></td></tr>
</table></div>
<p>All metrics in log2 units. <strong>RMSE on the resistant subset is the clinically relevant figure.</strong>
XGBoost improves it by 32% vs the RF baseline. The higher overall RMSE is expected: XGBoost is less
aggressive at fitting the censoring floor, which is the right trade-off.</p>

<h2>Important Caveats</h2>
<ul>
  <li><strong>Panel ceiling artifact:</strong> Post-2018 MIC90 is capped at 32 mg/L by the instrument range.
      The true MIC90 may be 64, 128 mg/L or higher.</li>
  <li><strong><code>is_censored</code> in SHAP plots</strong> is a data-structure variable, not biology.
      Censored isolates are structurally at the MIC floor.</li>
  <li><strong>NDM underrepresented in training:</strong> NDM only became prevalent after 2018 and is therefore
      underweighted in SHAP. Its importance will rise when models are retrained on post-2020 data.</li>
</ul>

<h2>API Endpoints</h2>
<div class="table-wrap"><table>
  <tr><th>Endpoint</th><th>Method</th><th>Description</th></tr>
  <tr><td><span class="endpoint">GET /health</span></td><td>GET</td><td>Service status, loaded models</td></tr>
  <tr><td><span class="endpoint">GET /api/trend/mic90</span></td><td>GET</td><td>MIC90 by year: actual + predicted + forecast. <code>?species=kpneumoniae</code></td></tr>
  <tr><td><span class="endpoint">GET /api/country-stats</span></td><td>GET</td><td>Resistance rate and MIC90 by country. <code>?species=kpneumoniae</code></td></tr>
  <tr><td><span class="endpoint">GET /api/features/importance</span></td><td>GET</td><td>Top 20 SHAP features. <code>?species=kpneumoniae</code></td></tr>
  <tr><td><span class="endpoint">POST /api/predict</span></td><td>POST</td><td>Single-isolate MIC prediction. Include <code>species</code> in body.</td></tr>
  <tr><td><span class="endpoint">GET /api/countries</span></td><td>GET</td><td>Countries known to the model. <code>?species=kpneumoniae</code></td></tr>
</table></div>

<footer>
  Data source: ATLAS (Pfizer/Vivli), controlled access. &nbsp;|&nbsp;
  Model: XGBoost + Optuna, trained on ATLAS 2004-2018. &nbsp;|&nbsp;
  <a href="/docs">Swagger UI</a> &nbsp;|&nbsp;
  <a href="/redoc">ReDoc</a>
</footer>

</body>
</html>"""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": {sp: sp in _models for sp in SPECIES_MAP},
        "features": {sp: len(_feature_names.get(sp, [])) for sp in SPECIES_MAP},
    }


@app.get("/methodology", response_class=HTMLResponse)
def methodology():
    return METHODOLOGY_HTML


@app.get("/api/trend/mic90")
def trend_mic90(species: str = "kpneumoniae", source: Optional[str] = None):
    """
    MIC90 by year: actual (2004-2022) + model predictions (2019-2022) + forecast (2023-2026).
    ?species=kpneumoniae|abaumannii
    Optional ?source=train_actual|test_actual_and_predicted|forecast_extrapolated to filter.
    """
    _require_species(species)
    data = _mic90_trend.get(species, [])
    if source:
        data = [r for r in data if r.get("source") == source]
    return {"species": species, "data": data, "eucast_r_mg_l": EUCAST_R}


@app.get("/api/country-stats")
def country_stats(species: str = "kpneumoniae", min_n: int = 10, sort_by: str = "pct_resistant"):
    """Resistance rate and MIC90 by country (test period 2019-2022). ?species=kpneumoniae|abaumannii"""
    _require_species(species)
    data = [r for r in _country_stats.get(species, []) if r["n"] >= min_n]
    if sort_by in ("pct_resistant", "mic90_actual", "n"):
        data = sorted(data, key=lambda r: r.get(sort_by, 0), reverse=True)
    return {"species": species, "data": data, "period": "2019-2022", "eucast_r_mg_l": EUCAST_R}


@app.get("/api/features/importance")
def feature_importance(species: str = "kpneumoniae"):
    """Top 20 SHAP features from the tuned XGBoost model. ?species=kpneumoniae|abaumannii"""
    _require_species(species)
    return {
        "species": species,
        "data": _shap_importance.get(species, []),
        "note": (
            "is_censored is a data-structure artifact (censored observations are "
            "structurally at the MIC floor), not a biological predictor."
        ),
    }


@app.post("/api/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    """
    Predict MIC for a single isolate given species, year, country, demographics, and gene flags.

    Note: regression model trained on ATLAS 2004-2018. Predictions outside
    this distribution (e.g. novel countries, post-2022) carry higher uncertainty.
    """
    _require_species(req.species)

    X = _build_feature_vector(req)
    log2_pred = float(_models[req.species].predict(X)[0])
    mic_pred  = float(2 ** log2_pred)

    return PredictionResponse(
        species         = req.species,
        log2_mic        = round(log2_pred, 4),
        mic_mg_l        = round(mic_pred, 4),
        interpretation  = _interpret(log2_pred),
        input_echo      = req.model_dump(),
    )


@app.get("/api/countries")
def list_countries(species: str = "kpneumoniae"):
    """List all countries known to the model for the given species."""
    _require_species(species)
    feature_names = _feature_names.get(species, [])
    known = [f.replace("ctry_", "") for f in feature_names if f.startswith("ctry_")]
    known = sorted(["Argentina"] + known)
    return {"species": species, "countries": known, "n": len(known)}
