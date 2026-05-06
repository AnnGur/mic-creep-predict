# MIC Creep Prediction — Vivli AMR Challenge 2026

Predicts **MIC Creep** (year-over-year drift in Minimum Inhibitory Concentration values) using machine learning. MIC creep is a precursor to full antibiotic resistance that flies under the radar of standard susceptibility reporting.

This is a Proof-of-Concept submission to the **Vivli AMR Surveillance Data Challenge 2026** and will serve as the MVP foundation for a subsequent INSCIENCE (Ukrainian national science funding) grant application.

---

## 🎯 Scientific Objective

| Parameter | Value |
|-----------|-------|
| **Pathogen** | *Klebsiella pneumoniae* (primary) or *Acinetobacter baumannii* |
| **Antibiotic** | Meropenem (carbapenem class — last-resort drug) |
| **Target Variable** | Quantitative MIC value (mg/L), modeled as log₂(MIC) |
| **Primary Metric** | MIC₉₀ trend over time + RMSE of regression model |
| **Data Source** | ATLAS (Pfizer) & SENTRY (JMI Labs) via Vivli AMR Register |

### Vulnerable Group Proxies

- **Military**: Isolates from wound/skin/blood samples in males of combat-relevant age
- **Paediatric**: Isolates from patients aged 0–17 years

---

## 📁 Project Structure

```
mic-creep-predict/
├── CLAUDE.md                          # Full project specification
├── README.md                          # This file
├── requirements.txt                   # Python dependencies (locked versions)
├── .gitignore                         # Git ignore rules (data/ protected)
│
├── data/
│   └── raw/                           # Raw datasets (gitignored)
│       ├── Klebsiella_pneumoniae_BVBRC_genome_amr.csv
│       ├── Klebsiella_pneumoniae_isolates.tsv
│       ├── Acinetobacter_baumannii_BVBRC_genome_amr.csv
│       └── Acinetobacter_baumannii_isolates.tsv
│
├── notebooks/                         # Exploratory analysis & development
│   ├── 01_data_exploration.ipynb      # Initial data QA
│   ├── 02_eda_temporal_trends.ipynb   # Year × country coverage heatmap
│   └── 03_geographic_antibiotic_profile.ipynb  # Resistance patterns by region
│
└── src/                               # Production code
    ├── __init__.py
    │
    ├── data/
    │   ├── __init__.py
    │   ├── schema.py                  # Data schema definitions & validation
    │   ├── loader.py                  # Load & join BVBRC AMR + isolate data
    │   └── preprocessor.py            # MIC parsing, censoring, log2 transform
    │
    ├── features/
    │   └── __init__.py                # Feature engineering (scaffolded)
    │
    └── models/
        └── __init__.py                # Model training & evaluation (scaffolded)
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9+
- Virtual environment (recommended)

### 2. Setup

```bash
# Clone repository
git clone <repo-url>
cd mic-creep-predict

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install Jupyter
pip install jupyter
```

### 3. Run Notebooks

```bash
jupyter notebook notebooks/
```

Open:
- **01_data_exploration.ipynb** — Overview of raw datasets
- **02_eda_temporal_trends.ipynb** — Temporal & geographic coverage
- **03_geographic_antibiotic_profile.ipynb** — Antibiotic resistance patterns

---

## 🔧 Data Pipeline

### Overview

The data pipeline loads BVBRC genome-level AMR annotations and joins them with isolate metadata to enable temporal and geographic analysis.

```
┌─────────────────────────────────────────────────────────┐
│  BVBRC Genome AMR Records                               │
│  (Antibiotic, Measurement Value, Phenotype S/I/R)      │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ Join on Genome ID
                     ↓
┌────────────────────────────────────────────────────────┐
│  BVBRC Isolate Metadata                                │
│  (Isolate Create Date, Location, Source)              │
└────────────────────┬───────────────────────────────────┘
                     │
                     ↓
         ┌───────────────────────┐
         │ Extract Temporal Data │ (year)
         └───────────┬───────────┘
                     │
         ┌───────────↓───────────┐
         │ Extract Geographic    │ (country)
         └───────────┬───────────┘
                     │
         ┌───────────↓──────────────┐
         │ Parse & Clean MIC Values │
         │ - Handle censoring       │
         │ - Validate ranges        │
         └───────────┬──────────────┘
                     │
         ┌───────────↓──────────────┐
         │ Log₂ Transform MIC       │
         │ mic_log2 = log₂(mic_val) │
         └───────────┬──────────────┘
                     │
                     ↓
         ┌───────────────────────────┐
         │ Ready for Modeling        │
         │ Features: year, country,  │
         │           age_group, sex  │
         │ Target: mic_log2          │
         └───────────────────────────┘
```

### Key Modules

**`src/data/loader.py`** — Loading & Joining
```python
from src.data.loader import BVBRCDataLoader
from pathlib import Path

loader = BVBRCDataLoader(Path('data/raw'))
joined = loader.load_and_join("Klebsiella pneumoniae")
```

**`src/data/preprocessor.py`** — Cleaning & Transformation
```python
from src.data.preprocessor import MICPreprocessor

# Parse censored MIC (e.g., ">8" → 16)
value, op = MICPreprocessor.parse_censored_mic(">8")

# Log₂-transform
log2_mic = MICPreprocessor.log2_transform_mic(value)

# End-to-end cleaning
cleaned = MICPreprocessor.clean_mic_dataframe(
    df,
    antibiotic="Meropenem",
    min_year=2004,
    max_year=2022
)
```

**`src/data/schema.py`** — Data Definitions
```python
from src.data.schema import Pathogen, PhenotypeCategory, ProcessedDataSchema

# Feature columns for modeling
features = ProcessedDataSchema.FEATURE_COLUMNS
# ['year', 'country', 'age_group', 'sex', 'specimen_type', 'infection_type']

target = ProcessedDataSchema.TARGET_COLUMN
# 'mic_log2'
```

---

## 📊 Exploratory Notebooks

### Notebook 1: Data Exploration (`01_data_exploration.ipynb`)

Load raw datasets and inspect structure:
- BVBRC AMR dataset (columns, data types, sample records)
- Isolate metadata (temporal coverage, geographic distribution)
- Antibiotic availability
- Measurement Value statistics

**Output**: Overview of dataset size, quality, and temporal range.

### Notebook 2: Temporal Trends (`02_eda_temporal_trends.ipynb`)

Analyze year-over-year coverage:
- **Heatmap**: Isolates by year × country (top 10 countries)
- **Time series**: Records per year
- **Creep signal detection**: (placeholder for when real MIC data arrives)

**Output**: Visualization of surveillance coverage and temporal trends.

### Notebook 3: Geographic Antibiotic Profile (`03_geographic_antibiotic_profile.ipynb`)

Resistance patterns by region:
- **Antibiotic distribution**: Coverage by antibiotic class
- **Resistance phenotype by country**: Stacked bar chart (S/I/R)
- **Regional hotspots**: High-burden resistance zones

**Output**: Geographic patterns for targeting surveillance and intervention.

---

## ⚠️ Data Security & Compliance

**CRITICAL — Do NOT commit:**
- `/data/` directory (all raw data)
- `*.csv`, `*.tsv` files
- `.env` files with credentials

**Compliance rules (per challenge):**
- ✓ Raw data is local-only (never uploaded)
- ✓ Public repository shows only code (no data)
- ✓ Aggregated/predicted outputs only (no isolate-level records)
- ✓ Credentials in `.env` (gitignored)
- ✓ Vivli retains data stewardship — datasets cannot be redistributed

See `.gitignore` for rules.

---

## 🛠️ Model Architecture (Upcoming)

### Task Type
**Regression** — Predicting continuous log₂(MIC) values.

### Train/Test Split — TIME-AWARE (Critical)
- **Training**: 2004–2018
- **Testing**: 2019–2022
- ⚠️ Data must NEVER be randomly shuffled (leaks future info into training)

### Features
- `year` — ordinal, primary driver of creep signal
- `country` — One-Hot Encoded
- `age_group` — binned (paediatric / adult / elderly)
- `sex` — binary
- `specimen_type` — wound / blood / urine / respiratory / other
- `infection_type` — hospital-acquired vs community-acquired

### Handling Censored MIC Values
MIC data frequently contains censored entries:
- `">8"` → impute as next doubling dilution (16 mg/L)
- `"<=0.5"` → impute as half the boundary (0.25 mg/L)
- All values log₂-transformed **after** imputation
- Methodology documented explicitly (reviewers will verify)

### Evaluation Metrics
- **RMSE** (Root Mean Squared Error) — primary
- **MAE** (Mean Absolute Error) — secondary
- **MIC₉₀ trend** — year-over-year slope + descriptive stats
- **Log₂ slope/year** — quantitative creep rate

### Model Explainability (SHAP)
```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

**CRITICAL**: SHAP outputs must be validated by domain expert for biological plausibility before publication.

---

## 📋 8-Week Roadmap

| Week | Task | Status |
|------|------|--------|
| 1–2 | Download ATLAS/SENTRY, filter to K. pneumoniae, clean MIC formats | Scaffolded ✓ |
| 3–4 | Feature engineering, One-Hot Encoding, EDA plots by year | In progress |
| 5–6 | Train XGBoost + RF baseline, time-split validation, tuning | Not started |
| 7 | Extract SHAP values, build visualisations, validate with domain expert | Not started |
| 8 | Package model, write submission, deploy prototype site | Not started |

---

## 🌐 Deployment Pipeline (Future)

### Backend API
- **FastAPI** (Python) — REST endpoint for model predictions
- Hosted on **Render.com** (free tier)
- Loads saved model from Hugging Face Hub at startup

### Frontend Dashboard
- **Next.js** (React) — Interactive MIC trend visualisation
- **Recharts** or **Plotly.js** — Time-series charts
- Hosted on **Vercel** (free tier, auto-deploy from GitHub)

### Storage & Infrastructure
| Purpose | Tool | Cost |
|---------|------|------|
| Code & versioning | GitHub (public) | Free |
| Model artifact | Hugging Face Hub | Free |
| Aggregated results DB | Supabase (PostgreSQL) | Free tier |
| Raw data | Local only (gitignored) | — |

---

## 📚 Dependencies

See `requirements.txt` for full list. Key packages:

### Data Processing
- `pandas==2.2.0` — data manipulation
- `numpy==1.24.3` — numerical computing

### Machine Learning
- `scikit-learn==1.4.2` — pipeline, RF baseline, metrics
- `xgboost==2.0.3` — primary model
- `lightgbm==4.0.0` — optional alternative
- `optuna==3.1.1` — hyperparameter tuning
- `shap==0.44.0` — explainability

### Visualization
- `matplotlib==3.8.3`
- `seaborn==0.13.1`
- `plotly==5.18.0`

### Deployment
- `fastapi==0.109.0` — REST API
- `uvicorn==0.27.0` — ASGI server
- `pydantic==2.5.3` — data validation

---

## 🔍 What's Inside `src/data/`

### `schema.py`
Data type definitions and validation:
```python
class Pathogen(str, Enum):
    KLEBSIELLA_PNEUMONIAE = "Klebsiella pneumoniae"
    ACINETOBACTER_BAUMANNII = "Acinetobacter baumannii"

class PhenotypeCategory(str, Enum):
    SUSCEPTIBLE = "S"
    INTERMEDIATE = "I"
    RESISTANT = "R"

class ProcessedDataSchema:
    FEATURE_COLUMNS = ['year', 'country', 'age_group', 'sex', ...]
    TARGET_COLUMN = 'mic_log2'
```

### `loader.py`
Load and join BVBRC data:
```python
loader = BVBRCDataLoader(data_dir)
amr_df = loader.load_amr_data("Klebsiella pneumoniae")
isolates_df = loader.load_isolate_data("Klebsiella pneumoniae")
joined = loader.join_amr_isolates(amr_df, isolates_df)
```

### `preprocessor.py`
MIC data cleaning and transformation:
```python
# Parse censored MIC strings
value, op = MICPreprocessor.parse_censored_mic(">8")

# Log₂-transform
log2_val = MICPreprocessor.log2_transform_mic(value)

# Extract country and year
country = MICPreprocessor.parse_location_to_country("USA:CA")
year = MICPreprocessor.extract_year_from_date("2020-10-21T03:02:19Z")

# End-to-end pipeline
cleaned = MICPreprocessor.clean_mic_dataframe(df, antibiotic="Meropenem")
```

---

## 📖 References & Further Reading

- **CLAUDE.md** — Full technical specification (in this repo)
- **Vivli AMR Register** — Data source (https://www.vivli.org/)
- **BVBRC (Bacterial & Viral Bioinformatics Resource Center)** — Genome annotations (https://www.bvbrc.org/)
- **EUCAST & CLSI Breakpoints** — Resistance definitions (for validation)
- **SHAP Documentation** — Model explainability (https://shap.readthedocs.io/)

---

## 👥 Contributing

This is a proof-of-concept project for a Ukrainian national science grant application (INSCIENCE). Contributions are welcome:

1. **Data cleaning** — Improve MIC parsing, handle edge cases
2. **Feature engineering** — Develop demographic/specimen proxies
3. **Model optimization** — Test alternative algorithms (LightGBM, CatBoost)
4. **Visualization** — Improve SHAP plots, dashboard UX
5. **Documentation** — Expand methodology section

---

## ⚖️ License & Data Use Agreement

- **Code**: [Specify license — e.g., MIT, Apache 2.0]
- **Data**: Vivli AMR Register (Data Use Agreement required)
  - Raw data must NOT be redistributed
  - Aggregated outputs only
  - Challenge rules compliance mandatory

---

## 🤝 Support & Contact

For questions:
- Check CLAUDE.md for technical details
- Review notebook comments for workflow examples
- Refer to docstrings in `src/data/` modules

---

**Last Updated**: May 4, 2026  
**Status**: Proof-of-Concept (PoC) — Awaiting ATLAS/SENTRY data
