# CLAUDE.md — MIC Creep Prediction Project
# Vivli AMR Surveillance Data Challenge 2026

## What This Project Does

Predicts **MIC Creep** — the gradual year-over-year drift in Minimum Inhibitory
Concentration (MIC) values — using machine learning. MIC Creep is a precursor to
full antibiotic resistance that flies under the radar of standard susceptibility
reporting because values remain technically "susceptible" while drifting upward.

This is a Proof of Concept (PoC) submission to the Vivli AMR Data Challenge 2026,
and will serve as the MVP foundation for a subsequent INSCIENCE (Ukrainian national
science funding) grant application.

---

## Scientific Focus

| Parameter        | Value                                                        |
|------------------|--------------------------------------------------------------|
| Pathogen         | Klebsiella pneumoniae (primary) or Acinetobacter baumannii   |
| Antibiotic       | Meropenem (carbapenem class — last-resort drug)              |
| Target variable  | Quantitative MIC value (mg/L), modelled as log₂(MIC)        |
| Primary metric   | MIC₉₀ trend over time + RMSE of regression model            |
| Excluded dataset | Merck/SMART (excluded per challenge rules)                   |

### Vulnerable Group Proxies
- **Military**: isolates from wound/skin/blood samples in males of combat-relevant age
- **Paediatric**: isolates from patients aged 0–17 years

---

## Datasets

- **ATLAS** (Pfizer) — from the Vivli AMR Register. Large, comprehensive,
  strong demographic metadata, good Gram-negative coverage.
- **SENTRY** (JMI Labs) — global surveillance registry, excellent MIC
  standardisation across years.

Both datasets are accessed via the Vivli AMR Register under a data use agreement.

---

## Data Security Rules (CRITICAL)

- Raw data files must NEVER be committed to Git
- Add `/data/` and `*.csv` to `.gitignore` immediately
- Raw isolate-level records must NEVER be exposed publicly
- The public website shows only aggregated/predicted outputs — never raw records
- Credentials and API keys go in `.env` files, never in source code
- Vivli retains data stewardship — datasets cannot be redistributed

---

## Tech Stack

### ML / Data Pipeline (Python)
- `pandas` — data loading and wrangling
- `xgboost` — primary regression model (XGBoost Regressor)
- `scikit-learn` — pipeline wrappers, Random Forest baseline, metrics
- `shap` — model explainability (SHapley Additive exPlanations)
- `optuna` — hyperparameter tuning
- `matplotlib` / `seaborn` — EDA visualisations
- `lightgbm` — optional alternative if performance warrants

### Backend API
- **FastAPI** (Python) — serves model predictions via REST endpoint
- Hosted on **Render.com** (free tier)
- Loads saved model artifact from Hugging Face Hub at startup

### Frontend
- **Next.js** (React) — interactive dashboard for AMR trend visualisation
- **Recharts** or **Plotly.js** — MIC trend charts
- Hosted on **Vercel** (free tier, auto-deploys from GitHub)

### Storage & Infrastructure
| Purpose               | Tool                        | Cost        |
|-----------------------|-----------------------------|-------------|
| Code & versioning     | GitHub (public repo)        | Free        |
| Model artifact        | Hugging Face Hub            | Free        |
| Aggregated results DB | Supabase (PostgreSQL)       | Free tier   |
| Raw data              | Local only — never uploaded | —           |

---

## Model Architecture

### Task Type
Regression — predicting a continuous MIC value (log₂-transformed).

### Train/Test Split — TIME-AWARE (critical)
- Training set: **2004–2018**
- Test set: **2019–2022**
- Data must NEVER be randomly shuffled — this would leak future data into
  training and invalidate all results.

### Features
- `year` — ordinal, primary driver of creep signal
- `country` — One-Hot Encoded
- `age_group` — binned categorical (paediatric / adult / elderly)
- `sex` — binary
- `specimen_type` — wound / blood / urine / respiratory / other
- `infection_type` — hospital-acquired vs community-acquired where available

### Handling Censored MIC Values
MIC data frequently contains censored entries like `">8"` or `"<=0.5"`.
Do NOT drop these or use them as strings.

Recommended approach:
- `">8"` → replace with next doubling dilution (e.g., 16 mg/L)
- `"<=0.5"` → replace with half the boundary value (e.g., 0.25 mg/L)
- Log₂-transform all values after imputation
- Document this decision explicitly in the methodology — reviewers will check

### Evaluation Metrics
- **RMSE** (Root Mean Squared Error) — primary regression metric
- **MAE** (Mean Absolute Error) — secondary
- **MIC₉₀ trend** — year-over-year, reported as descriptive statistic
- **Log₂ slope per year** — quantitative creep rate (from linear fit)
- Optional: classification accuracy against EUCAST breakpoints

---

## SHAP — Model Explainability

SHAP (SHapley Additive exPlanations) explains individual model predictions by
showing how much each feature contributed to the output.

In this project, SHAP answers: "For this predicted MIC value — how much did
`year`, `country`, `age_group`, and `specimen_type` each contribute?"

Usage:
```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

SHAP outputs must be validated by the domain expert (microbiologist/epidemiologist)
for biological plausibility before being included in any report or publication.
If the model weights `age` more heavily than `infection_type`, verify this makes
epidemiological sense.

---

## 8-Week Roadmap

| Week   | IT Lead Tasks                                                          |
|--------|------------------------------------------------------------------------|
| 1–2    | Download ATLAS/SENTRY, filter to K. pneumoniae, clean MIC formats      |
| 3–4    | Feature engineering, One-Hot Encoding, EDA plots by year               |
| 5–6    | Train XGBoost + RF baseline, time-split validation, tuning             |
| 7      | Extract SHAP values, build visualisations, validate with domain expert |
| 8      | Package model, write submission, deploy prototype site                 |

---

## Project Structure (Recommended)

amr-mic-creep/
├── CLAUDE.md                  # This file — Claude Code reads it automatically
├── .gitignore                 # Must include /data/, *.csv, .env
├── .env                       # API keys — never committed
├── data/                      # Raw data — local only, gitignored
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── data/
│   │   ├── loader.py          # Data loading and filtering
│   │   └── preprocessor.py   # MIC cleaning, censored value handling
│   ├── features/
│   │   └── engineer.py       # Feature transformations
│   ├── models/
│   │   ├── train.py          # Training pipeline
│   │   ├── evaluate.py       # Metrics and SHAP
│   │   └── predict.py        # Inference
│   └── api/
│       └── main.py           # FastAPI app
├── frontend/                  # Next.js app
├── models/                    # Saved model artifacts (pushed to HF Hub)
├── reports/                   # Outputs, charts, submission documents
└── requirements.txt

---

## Grant Context (INSCIENCE)

Upon challenge completion, this PoC becomes the MVP for a Ukrainian national
science grant application. The narrative:

> "We have demonstrated the algorithm works for K. pneumoniae + Meropenem on
> 100,000+ isolates. Funding will scale this to all WHO priority pathogens,
> add automated retraining, and integrate with Ukraine's national epidemiological
> surveillance infrastructure."

The grant application can state that work is already underway — de-risking the
proposal for reviewers.

---

## Key Roles

- **IT Lead**: Data pipeline, model training, API, frontend, infrastructure
- **Domain Expert** (co-lead): Biological interpretation, group definitions,
  report writing, SHAP validation, grant narrative

---

## Challenge Rules Reminder

- Merck/SMART datasets: EXCLUDED
- Code must be open-source (public GitHub repo required)
- Raw data must not be redistributed
- Results and methodology must be reproducible from the submitted code