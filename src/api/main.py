"""
FastAPI inference endpoint — MIC Creep Prediction
==================================================
Serves predictions and pre-aggregated surveillance data for both
Klebsiella pneumoniae and Acinetobacter baumannii + Meropenem.

Startup behaviour:
  - Dev  (MODEL_SOURCE=local): loads models from disk
  - Prod (MODEL_SOURCE=huggingface): downloads artefacts from HF Hub
  - Pre-computed JSON data (reports/api/api_*.json) is loaded once at startup
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
REPORTS      = PROJECT_ROOT / "reports" / "api"

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
_lag_lookup:       dict[str, dict]        = {}  # species -> {country: {year_str: rate}}


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

        if MODEL_SOURCE == "huggingface" and HF_REPO:
            from huggingface_hub import hf_hub_download
            try:
                pkl_path  = hf_hub_download(repo_id=HF_REPO, filename=model_fname)
                json_path = hf_hub_download(repo_id=HF_REPO, filename=features_fname)
            except Exception as exc:
                print(f"  [{species}] HF download failed — skipping: {exc}")
                continue
            _models[species]        = joblib.load(pkl_path)
            _feature_names[species] = json.loads(Path(json_path).read_text())
        else:
            if not model_path.exists():
                print(f"  [{species}] model not found — skipping ({model_path})")
                continue
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

        lag_path = REPORTS / f"{prefix}_lag_lookup.json"
        _lag_lookup[species] = json.loads(lag_path.read_text()) if lag_path.exists() else {}

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

    gene_vals = {
        "KPC_pos": int(req.kpc_pos),
        "NDM_pos": int(req.ndm_pos),
        "OXA_pos": int(req.oxa_pos),
        "VIM_pos": int(req.vim_pos),
        "IMP_pos": int(req.imp_pos),
        "GES_pos": int(req.ges_pos),
    }
    for k, v in gene_vals.items():
        row[k] = v

    row["is_censored"]       = 0
    row["pct_censored_year"] = _censoring_lookup[req.species].get(req.year, 0.88)

    # Lag feature: previous-year country resistance rate
    if "lag1_pct_resistant_ctry" in row:
        lag = _lag_lookup.get(req.species, {})
        ctry_rates = lag.get(req.country, lag.get("_global", {}))
        # Walk backwards from year-1 to find the most recent known rate
        prev_rate = lag.get("_global_mean", 0.5)
        for y in range(req.year - 1, 2003, -1):
            if str(y) in ctry_rates:
                prev_rate = ctry_rates[str(y)]
                break
            if str(y) in lag.get("_global", {}):
                prev_rate = lag["_global"][str(y)]
                break
        row["lag1_pct_resistant_ctry"] = prev_rate

    # Gene × year interaction terms
    for gene, val in gene_vals.items():
        col = f"{gene.replace('_pos', '')}_x_year"
        if col in row:
            row[col] = val * req.year

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
<p><strong>Primary dataset:</strong> ATLAS (Antimicrobial Testing Leadership and Surveillance),
Pfizer/Vivli. Controlled access via Vivli AMR Surveillance Platform.
K. pneumoniae additionally uses SIDERO-WT (2014-2022) and Venatorx (2016-2022) supplementary
surveillance datasets sharing the same MIC panel ceiling.
A. baumannii uses ATLAS only &mdash; supplementary A. bau datasets are clinical trial
enrollment cohorts with selection bias toward treatment-failure isolates.</p>
<div class="table-wrap"><table>
  <tr><th>Species</th><th>Datasets</th><th>Train (2008-2018)</th><th>Test (2019-2022)</th><th>Countries</th></tr>
  <tr><td><em>K. pneumoniae</em></td><td>ATLAS + SIDERO-WT + Venatorx</td><td>64,429</td><td>26,681</td><td>55</td></tr>
  <tr><td><em>A. baumannii</em></td><td>ATLAS only</td><td>20,828</td><td>13,537</td><td>54</td></tr>
</table></div>

<h2>Key Findings</h2>
<div class="card green">
  <strong>K. pneumoniae &mdash; MIC creep confirmed.</strong>
  MIC90 rose from &lt;0.5 mg/L (2008) to 32 mg/L (2018+), crossing the EUCAST R breakpoint around 2018.
  Resistance prevalence increased from ~7% (2008) to ~15% (2022).
  Carbapenemase gene drivers: KPC dominated 2008-2016; NDM and OXA rose sharply from 2017.
</div>
<div class="card">
  <strong>A. baumannii &mdash; resistance switch, not creep.</strong>
  MIC distribution is strongly bimodal (susceptible ~0.25 mg/L; resistant at panel ceiling 32 mg/L).
  MIC90 has been at the ceiling throughout the study period &mdash; continuous drift modelling is not
  applicable. Primary metric: resistance prevalence rose from ~44% (2008) to ~70% (2022),
  logistic trend slope = +0.098 logit/yr (R&sup2;=0.66, p=0.002). Forecast: ~77% by 2026.
  OXA-type carbapenemases dominate throughout.
</div>

<h2>Models</h2>
<p><strong>K. pneumoniae:</strong> XGBoost Regressor + RF baseline, predicting log&#x2082;(MIC).
Resistant isolates weighted 3x. 60 Optuna trials. Train 2008-2018 | Test 2019-2022 (strictly time-ordered).</p>
<div class="table-wrap"><table>
  <tr><th>Model</th><th>RMSE (all)</th><th>RMSE (resistant)</th><th>R&sup2;</th></tr>
  <tr><td>RF baseline</td><td>1.77</td><td>2.95</td><td>0.77</td></tr>
  <tr><td><strong>XGBoost tuned</strong></td><td>2.22</td><td><strong>2.17</strong></td><td>0.63</td></tr>
</table></div>
<p><strong>A. baumannii:</strong> XGBoost Classifier predicting P(R) per isolate, with recency
weighting (w = 0.85^(max_year &minus; year)). Year-level prevalence aggregated from per-isolate
P(R), then fitted with logistic regression on the logit scale. 95% CIs via Wilson score method.
AUC-ROC = 0.845 on the 2019-2022 test period (sensitivity 0.94, specificity 0.50 at a 0.5 threshold).</p>

<h3>Walk-forward Cross-validation (6 folds, test years 2014-2019)</h3>
<p>K. pneumoniae: mean RMSE 1.92 &plusmn; 0.31, R&sup2; 0.56 &plusmn; 0.12.
A. baumannii: mean RMSE 2.44 &plusmn; 0.26, R&sup2; 0.29 &plusmn; 0.09.
MIC90 correctly predicted at ceiling (32 mg/L) in all A. bau folds.</p>

<h3>Country-stratified Hybrid MIC90</h3>
<p>If global P(R) &gt; 10%: MIC90 = panel ceiling (32 mg/L).
If global P(R) &le; 10%: per-country P(R) check; ceiling applied per country where exceeded;
weighted average by sample size. This hybrid correctly produces 32 mg/L for K. pn test years
and outperforms global-only logic in transitional (2014-2018) folds.</p>

<h3>Key Features (83 for K. pn; 76 for A. bau)</h3>
<div class="table-wrap"><table>
  <tr><th>Feature</th><th>Type</th><th>Rationale</th></tr>
  <tr><td><code>year</code></td><td>continuous</td><td>Primary creep / prevalence driver</td></tr>
  <tr><td><code>KPC_pos</code> ... <code>GES_pos</code></td><td>binary</td><td>Carbapenemase gene presence</td></tr>
  <tr><td><code>ctry_*</code></td><td>OHE</td><td>Country (n&lt;50 training isolates collapsed to &ldquo;Other&rdquo;)</td></tr>
  <tr><td><code>spec_*</code></td><td>OHE</td><td>Specimen source (5 categories)</td></tr>
  <tr><td><code>ward_*</code></td><td>OHE</td><td>Hospital setting; top SHAP driver for A. baumannii</td></tr>
  <tr><td><code>pct_censored_year</code></td><td>float</td><td>Controls for panel methodology changes</td></tr>
  <tr><td><code>military_proxy</code></td><td>binary</td><td>Wound + male + 18-60; combat-wound proxy</td></tr>
</table></div>

<h2>Caveats</h2>
<ul>
  <li><strong>Panel ceiling artifact:</strong> ATLAS reports all MIC &ge; 32 mg/L as exactly 32 mg/L.
      True MIC90 may be higher for resistant populations.</li>
  <li><strong>NDM underrepresented in training:</strong> NDM rose sharply post-2018 and is therefore
      underweighted in SHAP values from the 2008-2018 training window.</li>
  <li><strong>A. bau walk-forward CV:</strong> Regression quality is low (R&sup2;=0.29) by design &mdash;
      bimodal distributions are poor targets for regression. Prevalence trend is the correct primary metric.</li>
</ul>

<h2>API Endpoints</h2>
<div class="table-wrap"><table>
  <tr><th>Endpoint</th><th>Method</th><th>Description</th></tr>
  <tr><td><span class="endpoint">GET /health</span></td><td>GET</td><td>Service status, loaded models</td></tr>
  <tr><td><span class="endpoint">GET /api/trend/mic90</span></td><td>GET</td><td>MIC90 by year: actual + predicted + forecast. <code>?species=kpneumoniae|abaumannii</code></td></tr>
  <tr><td><span class="endpoint">GET /api/country-stats</span></td><td>GET</td><td>Resistance rate and MIC90 by country. <code>?species=kpneumoniae|abaumannii</code></td></tr>
  <tr><td><span class="endpoint">GET /api/features/importance</span></td><td>GET</td><td>Top 20 SHAP features. <code>?species=kpneumoniae|abaumannii</code></td></tr>
  <tr><td><span class="endpoint">POST /api/predict</span></td><td>POST</td><td>Single-isolate MIC prediction. Include <code>species</code> in body.</td></tr>
  <tr><td><span class="endpoint">GET /api/countries</span></td><td>GET</td><td>Countries known to the model. <code>?species=kpneumoniae|abaumannii</code></td></tr>
</table></div>

<footer>
  Data source: ATLAS (Pfizer/Vivli), controlled access. &nbsp;|&nbsp;
  Models: XGBoost + Optuna, trained on ATLAS 2008-2018. &nbsp;|&nbsp;
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
