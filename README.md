# MIC Creep Prediction — Vivli AMR Challenge 2026

Predicts **MIC Creep** — the gradual year-over-year drift in Minimum Inhibitory Concentration (MIC) values for *Klebsiella pneumoniae* + Meropenem — using machine learning.

MIC creep is a precursor to full antibiotic resistance that flies under the radar of standard susceptibility reporting because values remain technically "susceptible" while drifting upward.

**Status**: PoC complete — EDA, feature engineering, and model training done on ATLAS dataset.

---

## Scientific Objective

| Parameter | Value |
|-----------|-------|
| Pathogen | *Klebsiella pneumoniae* |
| Antibiotic | Meropenem (carbapenem — last-resort drug) |
| Target variable | log₂(MIC), continuous regression |
| Primary metric | RMSE on resistant subset (MIC ≥ 8 mg/L) + MIC₉₀ trend |
| Training data | ATLAS (Pfizer) via Vivli AMR Register |
| Time split | Train ≤ 2018 · Test 2019–2022 (never shuffled) |

**Result**: MIC creep confirmed — slope +1.97 mg/L/yr, R²=0.67, p=6.3×10⁻⁶.
XGBoost achieves RMSE 1.96 log₂ on resistant isolates vs RF baseline 2.87 (32% improvement).

---

## Project Structure

```
mic-creep-predict/
├── CLAUDE.md                         # Full project specification
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
│
├── data/
│   ├── raw/                          # ATLAS files (gitignored — local only)
│   └── processed/                    # Feature matrices (gitignored)
│       ├── X_train.parquet           # 62,891 rows x 91 features
│       ├── X_test.parquet            # 26,681 rows x 91 features
│       ├── y_train.parquet           # log2(MIC) train targets
│       └── y_test.parquet            # log2(MIC) test targets
│
├── scripts/                          # Run these in order
│   ├── 1_run_atlas_eda.py            # EDA -> 9 charts in reports/
│   ├── 2_run_feature_engineering.py  # Build feature matrix -> data/processed/
│   ├── 3_run_model_training.py       # Train RF + XGBoost -> models/ + reports/
│   └── run_paediatric_diagnostic.py  # Diagnostic for paediatric MIC_90 anomaly
│
├── src/
│   ├── data/
│   │   ├── loader.py                 # ATLASLoader — filters K. pneumoniae + Meropenem
│   │   └── preprocessor.py          # MIC parsing: ">8"->16, "<=0.06"->0.03, log2
│   ├── features/
│   │   └── engineer.py              # build_features(), time_split(), run_pipeline()
│   └── models/
│       └── __init__.py
│
├── notebooks/
│   ├── 04_atlas_eda.ipynb
│   ├── 04b_paediatric_diagnostic.ipynb
│   └── 05_feature_engineering.ipynb
│
├── models/                           # Saved artefacts
│   ├── rf_baseline.pkl
│   ├── xgb_tuned.pkl
│   ├── xgb_best_params.json
│   └── feature_names.json
│
└── reports/                          # All generated outputs
    ├── atlas_eda_analysis.md         # Full EDA writeup
    ├── model_results.md              # Model evaluation report
    ├── model_findings.md             # Summary for stakeholders
    └── *.png                         # Charts
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

# macOS only — clear Gatekeeper quarantine after install
xattr -dr com.apple.quarantine .venv/
```

> **Data**: Obtain ATLAS dataset from the [Vivli AMR Register](https://www.vivli.org/) under a Data Use Agreement. Place the Excel file in `data/raw/`.

### 2. Run the pipeline

Run scripts in order from the project root:

```bash
# Step 1 — EDA (generates 9 charts in reports/)
.venv/bin/python scripts/1_run_atlas_eda.py

# Step 2 — Feature engineering (generates train/test parquet in data/processed/)
.venv/bin/python scripts/2_run_feature_engineering.py

# Step 3 — Model training
# Fast check (RF baseline only, ~2 min):
.venv/bin/python scripts/3_run_model_training.py --skip-optuna

# Full run (RF + XGBoost + Optuna tuning, ~10 min):
.venv/bin/python scripts/3_run_model_training.py --n-trials 40
```

All outputs go to `reports/`. Models saved to `models/`.

---

## Pipeline Details

### Step 1 — EDA (`1_run_atlas_eda.py`)

Loads ATLAS, filters to *K. pneumoniae* + Meropenem, produces:

| Chart | What it shows |
|---|---|
| `mic90_trend_*.png` | MIC90 over time — creep signal + linear fit |
| `mic_violin_by_year.png` | Full distribution shift year by year |
| `geo_analysis.png` | MIC90 and %R by country |
| `mic90_by_age_group.png` | Paediatric / adult / elderly comparison |
| `mic90_military_proxy.png` | Wound/male/18-60 proxy with n-reliability check |
| `gene_prevalence_over_time.png` | KPC / NDM / OXA / VIM / IMP / GES trends |
| `data_quality.png` | Censoring rate + %R by year |
| `specimen_source_mic90.png` | MIC90 by specimen type |

### Step 2 — Feature Engineering (`2_run_feature_engineering.py`)

Builds 91-column feature matrix:

| Group | Features |
|---|---|
| Temporal | `year` |
| Demographics | `gender_male`, `age_paediatric`, `age_elderly` |
| Clinical | `military_proxy`, `spec_wound/blood/resp/urine/peritoneal` |
| Geographic | `ctry_*` (OHE, 73 countries, drop_first) |
| Genes | `KPC_pos`, `NDM_pos`, `OXA_pos`, `VIM_pos`, `IMP_pos`, `GES_pos` |
| Quality | `is_censored`, `pct_censored_year` |

Time split: train <= 2018, test 2019-2022. Never shuffled.

### Step 3 — Model Training (`3_run_model_training.py`)

| Model | RMSE (all) | RMSE (resistant) | MAE (resistant) |
|---|---|---|---|
| RF baseline | 1.558 | 2.869 | 2.180 |
| **XGBoost tuned** | **1.758** | **1.960** | **1.127** |

Resistant subset = isolates with MIC >= 8 mg/L (EUCAST R breakpoint), n=4,305.

Key options:
```bash
--skip-optuna      # RF baseline only (fast)
--n-trials N       # Optuna trial count (default 60)
```

---

## Key Findings

See [reports/model_findings.md](reports/model_findings.md) for the full summary.

**Top SHAP features** (XGBoost, mean |SHAP value|):

| Rank | Feature | Score | Interpretation |
|---|---|---|---|
| 1 | KPC_pos | 0.94 | Strongest driver of high MIC |
| 2 | is_censored | 0.54 | Artifact — censored = at floor = low MIC |
| 3 | OXA_pos | 0.50 | Second carbapenemase family |
| 4 | NDM_pos | 0.44 | Rising threat, not blocked by avibactam |
| 5 | pct_censored_year | 0.33 | Surveillance methodology control |
| 6 | ctry_China | 0.30 | High-resistance country signal |
| 10 | year | 0.07 | Time-trend / creep signal |

> `is_censored` ranks #2 because censored observations are structurally at the MIC floor — it is a data artifact, not a biological predictor. Exclude from domain-expert SHAP presentations.

---

## Data Security

Raw data is **never committed**. See `.gitignore`.

- `/data/` — all raw and processed data (local only)
- `*.csv`, `*.xlsx`, `*.parquet` — never committed
- `.env` — credentials (never committed)
- Vivli retains data stewardship — datasets cannot be redistributed
- Public repo contains only code and aggregated reports

---

## Next Steps

- [ ] SHAP validation with domain expert (biological plausibility check)
- [ ] FastAPI inference endpoint (`src/api/main.py`)
- [ ] Push model artefact to Hugging Face Hub
- [ ] Next.js dashboard for MIC trend visualisation
- [ ] SENTRY dataset integration
- [ ] Write challenge submission

---

## Roadmap Status

| Week | Task | Status |
|------|------|--------|
| 1-2 | Download ATLAS, filter K. pneumoniae, clean MIC formats | Done |
| 3-4 | Feature engineering, EDA, OHE | Done |
| 5-6 | Train RF + XGBoost, time-split validation, Optuna tuning | Done |
| 7 | SHAP values, visualisations, domain expert validation | In progress |
| 8 | Package model, submission, deploy prototype | Not started |

---

## License & Data Use

- **Code**: MIT
- **Data**: Vivli AMR Register — Data Use Agreement required. Raw data must not be redistributed.
