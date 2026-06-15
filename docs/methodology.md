# Methodology — MIC Creep Prediction for *K. pneumoniae* and *A. baumannii*
**Vivli AMR Surveillance Challenge 2026**
**Generated**: 2026-06-14

---

## 1. Research Question

Can we predict the trajectory of antimicrobial MIC creep for *Klebsiella pneumoniae* and *Acinetobacter baumannii* using global surveillance data, and identify the molecular and geographic drivers of resistance escalation?

MIC creep is the gradual upward drift in Minimum Inhibitory Concentration (MIC) values across a bacterial population over time - a pre-resistance signal that becomes clinically critical when the population MIC90 approaches or crosses the EUCAST resistance breakpoint. For both *K. pneumoniae* and *A. baumannii* with meropenem, the EUCAST 2024 resistance threshold is **8 mg/L** (R > 8 mg/L).

Both pathogens are WHO Critical Priority pathogens for carbapenem resistance. *K. pneumoniae* is the dominant cause of carbapenem-resistant nosocomial infections globally; *A. baumannii* is the primary pathogen in trauma and combat-wound infections, making it directly relevant to the military/conflict surveillance component of this project. The same modeling pipeline runs for both species - predictions and SHAP analysis are available for each independently via the `species` parameter in all API endpoints.

---

## 2. Data Source

**Dataset**: ATLAS (Antimicrobial Testing Leadership and Surveillance), Pfizer/Vivli
**Access**: Vivli AMR Surveillance Platform (controlled access)
**Species covered**: *K. pneumoniae* + Meropenem; *A. baumannii* + Meropenem
**Records**: 89,572 isolates (*K. pneumoniae*), additional cohort for *A. baumannii*; 2004-2022
**Geographic coverage**: 66 countries across 6 continents

ATLAS records patient-level MIC measurements alongside metadata: country, year, sex, age group, specimen source, and genotypic carbapenemase gene results (KPC, NDM, OXA, VIM, IMP, GES) where available.

**Gene families by species**: In *K. pneumoniae*, KPC historically dominated (2004-2018), with NDM and OXA rising after 2017. In *A. baumannii*, OXA-23/OXA-40/OXA-58 class carbapenemases dominate, with NDM as the secondary mechanism - the same OXA gene column in ATLAS captures both species' relevant variants.

---

## 3. Data Quality — Censoring

The most significant data quality issue is **left-censoring**: MIC dilution panels have a fixed lowest concentration (0.06 mg/L for meropenem). Any isolate with a true MIC at or below this floor is reported as `<=0.06` rather than an exact value. Approximately **80-90%** of observations fall at this floor for both species, because most isolates remain highly susceptible.

**Imputation approach**: Censored observations were assigned a value of `0.03 mg/L` (half the floor), following standard microbiological convention. Log2 of 0.03 gives approximately -5.06.

**Panel ceiling artifact**: Post-2018, MIC90 saturates at **32 mg/L** (the panel upper limit). Isolates with true MIC above 32 are reported as `>32`, which we impute as 32. The observed plateau in MIC90 after 2018 is therefore a **measurement ceiling artifact**, not a biological plateau. True MIC90 may be 64, 128, or higher. The linear slope estimate underestimates the true rate of resistance escalation for this reason.

**Censoring rate shift**: A critical methodological artifact is the censoring rate drop from ~85% to ~25-30% during 2013-2017, followed by a return to ~88-90% in 2019-2022. This is almost certainly a change in the ATLAS panel dilution range, not a biological change. This artifact must be controlled in modeling (see Section 5.2).

---

## 4. MIC Creep Confirmation — *K. pneumoniae*

Before modeling, we confirmed that MIC creep is statistically significant in the ATLAS dataset for *K. pneumoniae* + meropenem:

| Metric | Value |
|--------|-------|
| MIC90 slope (linear scale) | +1.97 mg/L/yr |
| R2 (linear trend, 2004-2018) | 0.67 |
| p-value | 6.3 x 10^-6 |
| Resistance rate 2007 to 2024 | 5% to 20% (4x increase) |

The mechanistic signature is **divergence between MIC50 and MIC90**: MIC50 stays flat at ~0.03 mg/L for 20 years (the susceptible majority is unchanged), while MIC90 climbs steadily. This is the epidemiological fingerprint of a growing resistant subpopulation, not a population-wide shift.

Carbapenemase gene prevalence in *K. pneumoniae* shows three distinct eras:
- 2004-2016: KPC-dominated (~5-6%), NDM and OXA near zero
- 2017-2020: OXA rises sharply to ~6%, NDM begins climbing
- 2021-2024: NDM overtakes KPC, peaks ~9% (2022); OXA ~8%

The NDM rise is particularly significant: NDM-carrying isolates are resistant to ceftazidime-avibactam, one of the last-resort salvage drugs, because NDM is a metallo-beta-lactamase not inhibited by avibactam.

In *A. baumannii*, OXA-type carbapenemases have dominated throughout the entire surveillance period - this is the expected pattern given the distinct plasmid ecology of *A. baumannii* relative to Enterobacterales.

---

## 5. Feature Engineering

### 5.1 Train/Test Split

The split is **strictly time-ordered** - no random shuffling:
- **Train**: 2004-2018 (70% of dataset)
- **Test**: 2019-2022 (30% of dataset)

Shuffling would constitute **data leakage**: any future observation included in training would teach the model about epidemiological events it could not have seen in deployment. All hyperparameter tuning was performed on the training set only. The same split boundaries apply to both *K. pneumoniae* and *A. baumannii*.

### 5.2 Feature Set

| Feature | Type | Rationale |
|---------|------|-----------|
| `year` | continuous | Primary creep driver |
| `gender_male` | binary | 1 = Male; sex is a surveillance covariate |
| `age_paediatric` | binary | Age Group = 0-17; adults (18-60) are reference |
| `age_elderly` | binary | Age Group = 61+; adults are reference |
| `military_proxy` | binary | Wound/abscess + male + age 18-60; proxy for combat-related infections per project design |
| `spec_*` | OHE (5 levels) | Specimen source: wound, blood, respiratory, urine, peritoneal; "other" is reference |
| `ctry_*` | OHE (65 levels) | Country of isolation; Argentina is reference (first alphabetically, drop_first=True) |
| `KPC_pos` ... `GES_pos` | binary | Carbapenemase gene presence: KPC, NDM, OXA, VIM, IMP, GES |
| `is_censored` | binary | 1 if observation was reported with censoring operator; captures measurement floor artifact |
| `pct_censored_year` | float | Year-level censoring rate; controls for panel methodology changes across years |

The feature set is identical for both species. Country dummy sets will differ (country coverage varies by species in ATLAS), but the construction logic is the same.

`is_censored` and `pct_censored_year` are **data-structure variables**, not biological predictors. They are flagged explicitly in all SHAP plots.

### 5.3 Target Variable

**Target**: `log2(mic_value)` - continuous regression.

Log2 scale is standard in microbiology because MIC dilution panels report in doubling dilutions (0.06, 0.12, 0.25, 0.5, 1, 2, 4, 8, 16, 32 mg/L). One log2 unit = one dilution step = the minimum detectable difference between MIC measurements.

---

## 6. Modeling Approach

### 6.1 Baseline Model

**Random Forest Regressor** (scikit-learn, n_estimators=200) trained on the full feature matrix with **3x sample weighting on resistant isolates** (MIC >= 8 mg/L).

Sample weighting is required because the target distribution is severely bimodal: ~75% of observations cluster at log2 = -5 (censoring floor), with a near-empty middle and a resistant tail. Without upweighting, a model minimizing MSE would learn to predict the floor for all observations.

### 6.2 Primary Model

**XGBoost Regressor** (xgboost >= 1.7) with Optuna hyperparameter optimization (60 trials by default, objective: RMSE on last 3 training years as time-aware internal validation split).

XGBoost was chosen over linear models because:
1. `year` and `pct_censored_year` are highly collinear (r = +0.61), which degrades linear model performance but does not affect tree-based models
2. The bimodal target benefits from the non-parametric split-finding of trees
3. Gene x country interaction effects (e.g. NDM in India vs NDM in Greece) are naturally captured without manual interaction terms

Independent models are trained for each species - the *K. pneumoniae* model is not used to predict *A. baumannii* MICs and vice versa.

**Best hyperparameters for *K. pneumoniae*** (Optuna, 40 trials):
```
n_estimators: 500    max_depth: 7    learning_rate: 0.0106
subsample: 0.879     colsample_bytree: 0.695    min_child_weight: 3
gamma: 1.633         reg_alpha: 1.085           reg_lambda: 6.996
```

### 6.3 Evaluation Strategy

The primary evaluation metric is **RMSE on the resistant subset** (isolates with true MIC >= 8 mg/L). Overall RMSE is reported for completeness but is dominated by the ~75% of test observations at the censoring floor and has limited clinical meaning.

Secondary metric: **MIC90 predicted vs actual by year** (2019-2022).

---

## 7. Results — *K. pneumoniae* + Meropenem

### 7.1 Model Performance

| Model | RMSE all | RMSE resistant | MAE resistant | N resistant |
|-------|----------|----------------|---------------|-------------|
| RF baseline | 1.558 | 2.869 | 2.180 | 4,305 |
| **XGBoost tuned** | **1.758** | **1.960** | **1.127** | 4,305 |

All metrics in log2 units. XGBoost improves resistant-subset RMSE by **32%** over the RF baseline.

### 7.2 MIC90 Trend - Predicted vs Actual

| Year | Actual (log2) | RF | XGBoost |
|------|---------------|----|---------| 
| 2019 | 5.00 | 2.58 | 3.66 |
| 2020 | 5.00 | 2.80 | 3.87 |
| 2021 | 5.00 | 3.11 | 3.99 |
| 2022 | 5.00 | 2.87 | 3.92 |

Actual MIC90 = log2(5) = ~32 mg/L in all test years due to the panel ceiling artifact. Both models correctly predict an upward trend. XGBoost gets within approximately 1 log2 dilution of the ceiling.

### 7.3 Key Driver Analysis (SHAP)

| Rank | Feature | Mean |SHAP| | Note |
|------|---------|------------|------|
| 1 | KPC_pos | 0.943 | KPC carbapenemase - dominant in training period 2004-2018 |
| 2 | is_censored | 0.543 | Data-structure artifact - not a biological predictor |
| 3 | OXA_pos | 0.504 | OXA carbapenemase - dominant in Europe and Middle East |
| 4 | NDM_pos | 0.438 | NDM - bypasses avibactam combinations; underrepresented in train |
| 5 | pct_censored_year | 0.331 | Surveillance methodology control |
| 6 | ctry_China | 0.304 | High-resistance country |
| 7 | ctry_India | 0.160 | NDM origin country |
| 10 | year | 0.071 | Temporal creep signal |

KPC ranks above NDM because the training period (2004-2018) is KPC-dominated. NDM only became prevalent after 2018 and is underrepresented in training. This is a data recency limitation, not a model error.

For *A. baumannii*, OXA-type genes are expected to rank highest in SHAP given their dominance in that species throughout the surveillance period.

---

## 8. Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|-----------|
| Panel ceiling at 32 mg/L | MIC90 saturates post-2018; true values unknown | Flag as right-censored; trend analysis valid 2004-2018 only |
| Bimodal target (75% at floor) | RMSE dominated by floor spike | 3x sample weight on resistant isolates; report RMSE separately for susceptible vs resistant |
| Covariate shift in censoring | Train has 25-87% censoring variation; test locked at 88-90% | `pct_censored_year` feature explicitly controls for this |
| NDM underrepresented in train | Reduces model sensitivity to fastest-rising mechanism | Include post-2020 data when retraining; stated as known limitation |
| Military proxy is broad | "wound + male + 18-60" captures 2,813 isolates (2.8%); many are not combat-related | Proxy is stated as approximate; acceptable for proof-of-concept |
| Paediatric MIC90 artifact | Flat at 0.12 mg/L 2006-2016, then jumps to 32 in 2019 | Driven by censoring floor (91.7% at floor pre-2017) + small n; excluded from primary trend |

---

## 9. Reproducibility

Run each script with `--species kpneumoniae` or `--species abaumannii`:

| Script | Command | Output |
|--------|---------|--------|
| Feature engineering | `python scripts/2_run_feature_engineering.py --species kpneumoniae` | `data/processed/X_train.parquet` etc. |
| | `python scripts/2_run_feature_engineering.py --species abaumannii` | `data/processed/abaumannii/X_train.parquet` etc. |
| Model training | `python scripts/3_run_model_training.py --species kpneumoniae` | `models/xgb_tuned.pkl` |
| | `python scripts/3_run_model_training.py --species abaumannii` | `models/xgb_tuned_abaumannii.pkl` |
| Data export | `python scripts/4_run_export.py --species kpneumoniae` | `reports/api_mic90_trend.json` etc. |
| | `python scripts/4_run_export.py --species abaumannii` | `reports/api_abaumannii_mic90_trend.json` etc. |
| API | `.venv/bin/python -m uvicorn src.api.main:app --reload --port 8000` | Serves both species automatically |

**Environment**: Python 3.12, xgboost >= 1.7, scikit-learn >= 1.4, optuna >= 3.5, shap >= 0.44. See `requirements.txt` for pinned versions.

---

## 10. API

A FastAPI inference service serves predictions and pre-aggregated surveillance data for both species. All endpoints accept a `species` parameter (`kpneumoniae` or `abaumannii`).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service status, loaded models |
| `/methodology` | GET | This methodology page (HTML) |
| `/api/trend/mic90` | GET | Historical + forecast MIC90 by year (`?species=kpneumoniae`) |
| `/api/country-stats` | GET | Resistance rate and MIC90 by country (`?species=kpneumoniae`) |
| `/api/features/importance` | GET | Top 20 SHAP features (`?species=kpneumoniae`) |
| `/api/predict` | POST | Single-isolate MIC prediction (include `species` in body) |
| `/api/countries` | GET | Countries known to the model (`?species=kpneumoniae`) |

---

*Figures referenced in this document are in the `reports/` directory. Source: ATLAS dataset, Pfizer/Vivli, controlled access via Vivli AMR Surveillance Platform.*
