# Project Handoff - MIC Creep Predict
**Vivli AMR Data Challenge 2026**
Last updated: 2026-07-02

---

## What this project does

Predicts MIC Creep - the gradual year-over-year drift in Minimum Inhibitory Concentration (MIC) values - using machine learning on global AMR surveillance data.

Two pathogens, both vs. Meropenem (last-resort carbapenem):

| Pathogen | Problem type | Primary metric |
|---|---|---|
| K. pneumoniae | Creep (continuous upward drift) | MIC90 RMSE, log2 slope/year |
| A. baumannii | Switch (bimodal, prevalence-based) | P(R) logistic trend AUC + slope |

---

## Infrastructure at a glance

| Component | Where | Notes |
|---|---|---|
| Code | github.com/AnnGur/mic-creep-predict (public) | main branch |
| Model artifacts | huggingface.co/AnnGur/mic-creep-kpneumoniae | 6 PKL files + feature JSONs per species |
| Backend API | Render.com (free tier) | FastAPI, auto-deploys from GitHub |
| Frontend | Vercel (free tier) | Next.js, auto-deploys from GitHub on push |
| Raw data | Local only - NEVER committed | Vivli AMR Register DUA |
| Results DB | Not yet set up (Supabase planned) | currently API serves static JSONs |

---

## Setting up on a new machine

### 1. Clone and Python env

```bash
git clone https://github.com/AnnGur/mic-creep-predict.git
cd mic-creep-predict
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Raw data

Data comes from the Vivli AMR Register (data use agreement required). Do NOT commit it.

Expected directory layout (gitignored):

```
data/
  raw/
    ATLAS_Antibiotics.xlsx          # Pfizer/ATLAS - primary dataset
    sentry_abaumannii.csv           # SENTRY - Acinetobacter only (not currently used)
    SIDERO_WT.xlsx                  # Shionogi - K.pn supplement
    Venatorx_VNRX5133.xlsx          # Venatorx - K.pn supplement
  processed/
    kpneumoniae/
      X_train.parquet, X_test.parquet, y_train.parquet, y_test.parquet
      keep_countries.json, lag_lookup.json
    abaumannii/
      (same structure)
```

### 3. Run the full pipeline

```bash
# Step 1 - feature engineering (creates processed parquets)
python scripts/pipeline/2_run_feature_engineering.py
python scripts/pipeline/2_run_feature_engineering.py --atlas-only   # for A.bau

# Step 2 - train models (XGBoost + RF + quantile, 60 Optuna trials each)
python scripts/pipeline/3_run_model_training.py --species kpneumoniae
python scripts/pipeline/3_run_model_training.py --species abaumannii

# Step 3 - export API JSON files
python scripts/pipeline/4_run_export.py --species kpneumoniae
python scripts/pipeline/4_run_export.py --species abaumannii

# Step 4 (optional) - walk-forward CV
python scripts/pipeline/5_run_walkforward_cv.py
```

### 4. Run the API locally

```bash
cd src/api
uvicorn main:app --reload --port 8000
# Test: curl http://localhost:8000/trend/kpneumoniae
```

### 5. Run the frontend locally

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:3000
```

---

## Current model results (as of 2026-07-02)

### K. pneumoniae + Meropenem
- Training data: ATLAS + SIDERO-WT + Venatorx, 2008-2018 (64,429 rows, 93 feature columns)
- Test data: 2019-2022 (26,681 rows)

| Model | RMSE (all) | R2 | RMSE (resistant) |
|---|---|---|---|
| RF baseline | 1.80 | 0.76 | 3.02 |
| XGBoost tuned | 2.22 | 0.63 | 2.29 |
| XGBoost Q0.90 | 4.18 | -0.30 | **1.20** |

RMSE on resistant subset (MIC >= 8 mg/L) is the clinically relevant metric - XGBoost Q0.90 is best here.

Top SHAP features: `lag1_pct_resistant_ctry`, `KPC_pos`, `OXA_pos`, `KPC_x_year`, `NDM_pos`

### A. baumannii + Meropenem
- Training data: ATLAS only (SIDERO/Venatorx/Entasis excluded - trial enrollment selection bias), 2008-2018 (20,828 rows, 84 feature columns). NOTE: the 2026-07-22 retrain was run without `--atlas-only` and silently added 6,181 supplementary rows in 2014-2018 (27,009 rows); rebuilt ATLAS-only and retrained on 2026-11-23. Always pass `--atlas-only` for A. baumannii.
- Test data: 2019-2022

| Model | RMSE (all) | R2 | RMSE (resistant) |
|---|---|---|---|
| RF baseline | 2.51 | 0.33 | 1.62 |
| XGBoost tuned | 2.68 | 0.23 | 0.99 |
| XGBoost Q0.90 | 3.66 | -0.43 | **0.28** |

**Primary metric is the P(R) prevalence classifier**, not RMSE:
- AUC-ROC: 0.845 on 2019-2022 (XGBoost binary classifier per isolate, recency-weighted; sensitivity 0.94, specificity 0.50 at a 0.5 threshold; verified 2026-11-23 with `scripts/pipeline/7_verify_production.py`). An earlier 0.981 figure could not be reproduced on the mixed data, ATLAS-only data, train years or pooled years (all give about 0.84-0.85); its origin is unknown. Tuning the classifier and adding Opt7 features gave no gain (`reports/model/pr_classifier_tuning_abaumannii.md`).
- Logistic trend on logit scale (fit 2008-2018): slope = +0.0976 logit/yr, R2 = 0.659, p = 0.0024 (reports/api/api_abaumannii_prevalence_trend.json). 11 autocorrelated annual points, so the p-value is optimistic.
- Observed: 44% resistant in 2008, a step to 62% in 2012, 57-69% in 2013-2018, then 70.0 / 69.5 / 71.2 / 67.3% in 2019-2022 (flat over the test window; the old note '44% in 2019' was wrong, 44% is 2008). The fitted trend predicts 76.8% for 2022 against 67.3% observed.
- Forecast: the raw-trend extrapolation gives 83% by 2026, but it already overshoots the observed test years, so treat it as an upper-end projection, not a measured creep rate. With a fixed country mix (`reports/model/prevalence_trend_check_abaumannii.md`) the test-year rate is 59-64% and the 2008-2022 slope is +0.028 logit/yr (p 0.034) against +0.078 raw. Part of the raw rise in the test years is which countries contribute isolates.

A. baumannii MIC90 is at the ATLAS panel ceiling (32 mg/L) throughout all years - resistance was established before 2008. The MIC90 regression is secondary; prevalence is the story.

---

## Key architectural decisions

**Train/test split**: 2008-2018 train, 2019-2022 test. NEVER shuffled (time-aware).

**Censored MIC values**: `">32"` -> 64 mg/L (next doubling dilution); `"<=0.25"` -> 0.125 mg/L (half boundary). All log2-transformed.

**Lag feature** (`lag1_pct_resistant_ctry`): previous-year resistance rate in the isolate's country. Computed from full dataset using O(1) dict lookup. Fallback chain: country+prev_year -> global prev_year -> global mean. Saved as `lag_lookup.json` for API inference.

**Gene x year interactions**: KPC, NDM, OXA, VIM, IMP, GES each get a `{gene}_x_year` column = gene_pos * year. NDM/OXA emerged post-2017; KPC plateaued. Top SHAP features.

**Hybrid MIC90**: Global P(R) > 10% -> ceiling 32 mg/L; below 10% -> country-stratified weighted average. Applied post-prediction.

**A. baumannii supplementary datasets removed**: SIDERO-WT and Venatorx were excluded for A.bau because those trials enrolled patients specifically with resistant infections (selection bias). ATLAS-only for A.bau.

**Walk-forward CV**: 6 folds, test years 2014-2019, fixed Optuna hyperparams from held-out validation.

---

## Files that live only on HF Hub (not in git)

Model PKL files are gitignored (`models/*.pkl`). They are stored at:
`huggingface.co/AnnGur/mic-creep-kpneumoniae`

To upload after retraining:
```bash
python scripts/utils/upload_to_huggingface.py
```

The API (Render) downloads them at startup. If you retrain, re-upload to HF Hub and redeploy Render (or it auto-pulls on restart).

---

## Deployment

### API (Render)

- Auto-deploys when `src/api/main.py` changes on main
- Loads models from HF Hub at startup
- Loads `reports/api/api_{species}_*.json` at startup (these ARE committed to git)
- If you change the JSON outputs, commit them and Render will pick them up

### Frontend (Vercel)

- Auto-deploys when anything in `frontend/` changes on main
- ISR cache: 5 minutes (`revalidate: 300` in page.tsx)
- Species switcher: `?species=kpneumoniae` or `?species=abaumannii`

### After retraining locally

```bash
# 1. Export updated API JSONs
python scripts/pipeline/4_run_export.py --species kpneumoniae
python scripts/pipeline/4_run_export.py --species abaumannii

# 2. Copy lag lookups to reports/api/
cp data/processed/kpneumoniae/lag_lookup.json reports/api/api_kpneumoniae_lag_lookup.json
cp data/processed/abaumannii/lag_lookup.json  reports/api/api_abaumannii_lag_lookup.json

# 3. Upload models to HF Hub
python scripts/utils/upload_to_huggingface.py

# 4. Commit and push (triggers Render + Vercel redeploy)
git add reports/api/
git commit -m "Update API outputs after retrain"
git push origin main
```

---

## What still needs to be done (challenge submission)

- [ ] Methodology page: update feature counts (parquets now have 93 K.pn / 84 A.bau columns; the API methodology page in src/api/main.py may still show older counts)
- [ ] Final write-up / submission document
- [ ] SHAP validation with domain expert (biological plausibility check)
- [ ] Supabase setup for aggregated results (optional for MVP)
- [ ] Exclusion of Merck/SMART datasets confirmed in submission notes

---

## Contacts and accounts

- GitHub: github.com/AnnGur
- HF Hub: huggingface.co/AnnGur
- Email: gurina.ann@gmail.com
- Vivli data access: requires DUA login at vivli.org
