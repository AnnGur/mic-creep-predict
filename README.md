# MIC Creep Prediction — Vivli AMR Challenge 2026

Predicts **MIC Creep** — the gradual year-over-year drift in Minimum Inhibitory
Concentration (MIC) values — using machine learning. MIC creep is a precursor to
full antibiotic resistance that flies under the radar of standard S/I/R reporting
because values remain technically "susceptible" while drifting toward the resistance
threshold.

**Live dashboard**: [mic-creep-predict.vercel.app](https://mic-creep-predict.vercel.app)

---

## Scientific Scope

| Parameter | Value |
|---|---|
| Pathogens | *Klebsiella pneumoniae* · *Acinetobacter baumannii* |
| Antibiotic | Meropenem (carbapenem — last-resort drug) |
| Target variable | log₂(MIC), continuous regression |
| Primary metric | RMSE on resistant subset (MIC ≥ 8 mg/L) + MIC₉₀ trend |
| Training data | ATLAS (Pfizer) via Vivli AMR Register — 127,112 isolates |
| Time split | Train 2004–2018 · Test 2019–2022 (never shuffled) |

---

## Model Results

### K. pneumoniae (89,572 isolates · 81 countries)

| Model | RMSE (all) | MAE (all) | RMSE (resistant) | MAE (resistant) |
|---|---|---|---|---|
| RF baseline | 1.558 | 1.109 | 2.869 | 2.180 |
| **XGBoost tuned** | **1.758** | **1.002** | **1.960** | **1.127** |

MIC₉₀ slope: **+1.97 mg/L/yr** (R²=0.67, p=6.3×10⁻⁶). Resistance: 5% (2007) → 20% (2022).

### A. baumannii (37,540 isolates · 79 countries)

| Model | RMSE (all) | MAE (all) | RMSE (resistant) | MAE (resistant) |
|---|---|---|---|---|
| RF baseline | 1.338 | 0.789 | 0.983 | 0.510 |
| **XGBoost tuned** | **1.379** | **0.707** | **0.748** | **0.270** |

MIC₉₀ at panel ceiling (32 mg/L) since 2005. Resistance: 39% (2006) → 69% (2022).

> All metrics in log₂ MIC units. Resistant subset = isolates with MIC ≥ 8 mg/L (EUCAST 2024 R breakpoint).

---

## Project Structure

```
mic-creep-predict/
│
├── src/                              # Importable Python library
│   ├── data/
│   │   ├── loader.py                 # ATLASLoader — loads and filters raw data
│   │   └── preprocessor.py          # MIC parsing: ">8"->16, "<=0.06"->0.03, log2
│   ├── features/
│   │   └── engineer.py              # build_features(), time_split(), run_pipeline()
│   └── api/
│       └── main.py                  # FastAPI app — predictions + aggregated data
│
├── scripts/
│   ├── pipeline/                     # Run in order to reproduce all results
│   │   ├── 1_run_atlas_eda.py        # EDA charts -> reports/eda/
│   │   ├── 2_run_feature_engineering.py  # Parquets -> data/processed/{species}/
│   │   ├── 3_run_model_training.py   # Train RF + XGBoost -> models/ + reports/model/
│   │   └── 4_run_export.py           # Aggregate predictions -> reports/api/
│   └── utils/                        # Supplementary tools (no raw data required)
│       ├── gen_charts_abaumannii.py  # A. baumannii EDA charts from processed parquets
│       ├── gen_model_charts_abaumannii.py  # A. baumannii model charts from saved model
│       ├── upload_to_huggingface.py  # Push model artifacts to HF Hub
│       └── paediatric_diagnostic.py  # Paediatric MIC₉₀ anomaly diagnostic
│
├── data/                             # All gitignored — local only
│   ├── raw/                          # ATLAS source files (Vivli DUA required)
│   └── processed/
│       ├── kpneumoniae/              # K. pneumoniae feature matrices
│       │   ├── X_train.parquet       # 62,891 rows × 91 features
│       │   ├── X_test.parquet        # 26,681 rows × 91 features
│       │   ├── y_train.parquet       # log2(MIC) targets
│       │   └── y_test.parquet
│       └── abaumannii/               # A. baumannii feature matrices
│           ├── X_train.parquet       # 24,003 rows × 89 features
│           ├── X_test.parquet        # 13,537 rows × 89 features
│           ├── y_train.parquet
│           └── y_test.parquet
│
├── models/                           # Gitignored locally — hosted on Hugging Face Hub
│   ├── xgb_tuned_kpneumoniae.pkl
│   ├── xgb_tuned_abaumannii.pkl
│   ├── feature_names_kpneumoniae.json
│   └── feature_names_abaumannii.json
│
├── reports/
│   ├── eda/                          # Charts from 1_run_atlas_eda.py + gen_charts_abaumannii.py
│   ├── model/                        # Charts + MD from 3_run_model_training.py
│   ├── api/                          # JSON exports from 4_run_export.py (read by FastAPI at startup)
│   └── atlas_eda_analysis.md
│
├── frontend/                         # Next.js dashboard (deployed on Vercel)
│   └── app/
│       ├── page.tsx                  # MIC Trend chart
│       ├── countries/page.tsx        # Country resistance map
│       ├── features/page.tsx         # SHAP feature importance
│       ├── predict/page.tsx          # Single-isolate MIC prediction
│       └── methodology/page.tsx      # Methods documentation
│
├── notebooks/                        # Exploratory analysis (not part of pipeline)
│
├── render.yaml                       # Render.com deployment config
└── requirements.txt
```

---

## Quick Start

### 1. Setup

```bash
git clone <repo-url>
cd mic-creep-predict

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# macOS only
xattr -dr com.apple.quarantine .venv/
```

> **Data**: Obtain the ATLAS dataset from the [Vivli AMR Register](https://www.vivli.org/)
> under a Data Use Agreement. Place the file in `data/raw/`.

### 2. Run the pipeline

Both species use `--species kpneumoniae` (default) or `--species abaumannii`.

```bash
# Step 1 — EDA (outputs to reports/eda/)
.venv/bin/python scripts/pipeline/1_run_atlas_eda.py

# Step 2 — Feature engineering (outputs to data/processed/{species}/)
.venv/bin/python scripts/pipeline/2_run_feature_engineering.py --species kpneumoniae
.venv/bin/python scripts/pipeline/2_run_feature_engineering.py --species abaumannii

# Step 3 — Model training (outputs to models/ and reports/model/)
.venv/bin/python scripts/pipeline/3_run_model_training.py --species kpneumoniae
.venv/bin/python scripts/pipeline/3_run_model_training.py --species abaumannii

# Step 4 — Export aggregated data for API (outputs to reports/api/)
.venv/bin/python scripts/pipeline/4_run_export.py --species kpneumoniae
.venv/bin/python scripts/pipeline/4_run_export.py --species abaumannii

# Run API locally
.venv/bin/python -m uvicorn src.api.main:app --reload --port 8000
```

Step 3 options:
```bash
--skip-optuna      # RF baseline only, ~2 min
--n-trials N       # Optuna trials (default 60)
```

### 3. Supplementary chart generation (no raw data needed)

If models and processed parquets are already available:

```bash
# A. baumannii EDA charts (gene prevalence, specimen source, MIC90 trend)
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/.mpl \
  .venv/bin/python scripts/utils/gen_charts_abaumannii.py

# A. baumannii model evaluation charts (RMSE, residuals, SHAP beeswarm)
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/.mpl \
  .venv/bin/python scripts/utils/gen_model_charts_abaumannii.py
```

---

## Feature Set

| Feature | Type | Notes |
|---|---|---|
| `year` | Continuous | Primary MIC creep driver |
| `gender_male` | Binary | Sex covariate |
| `age_paediatric` | Binary | Age 0–17; adults (18–60) are reference |
| `age_elderly` | Binary | Age 61+ |
| `military_proxy` | Binary | Wound + male + 18–60; combat-infection proxy |
| `spec_*` | OHE (5) | Specimen: wound, blood, respiratory, urine, peritoneal |
| `ctry_*` | OHE | Country of isolation (81 for K. pneu, 79 for A. bau) |
| `KPC_pos … GES_pos` | Binary ×6 | Carbapenemase genes (ATLAS PCR panel) |
| `is_censored` | Binary | Data artifact — isolate at MIC panel floor |
| `pct_censored_year` | Float | Year-level censoring rate; controls panel methodology shifts |

> `is_censored` and `pct_censored_year` are data-structure variables, not biological predictors.
> They are explicitly flagged in all SHAP outputs.

---

## API Endpoints

All endpoints accept `?species=kpneumoniae` (default) or `?species=abaumannii`.

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service status, loaded models |
| `/api/trend/mic90` | GET | MIC₉₀ by year — actual + predicted + forecast |
| `/api/country-stats` | GET | Resistance rate + MIC₉₀ by country |
| `/api/features/importance` | GET | Top 20 SHAP features |
| `/api/countries` | GET | Countries known to the model |
| `/api/predict` | POST | Single-isolate MIC prediction |
| `/methodology` | GET | Methodology HTML page |

Deployed on Render.com (free tier — first request after inactivity takes ~30s to wake).
Model artifacts loaded from [Hugging Face Hub](https://huggingface.co/AnnGur/mic-creep-kpneumoniae) at startup.

---

## Data Security

Raw data is **never committed**. See `.gitignore`.

- `data/` — all raw and processed data, local only
- `*.csv`, `*.xlsx`, `*.parquet` — never committed
- `models/*.pkl` — never committed (hosted on Hugging Face Hub)
- `.env` — credentials, never committed
- Vivli retains data stewardship — ATLAS data cannot be redistributed
- Public repo contains only code and aggregated `reports/api/*.json` (no isolate-level records)

---

## License & Data Use

- **Code**: MIT
- **Data**: Vivli AMR Register — Data Use Agreement required. Raw data must not be redistributed.
