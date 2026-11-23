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
**Records**: *K. pneumoniae*: 84,016 ATLAS isolates with a meropenem MIC in 2008-2022, plus 7,094 supplementary isolates (SIDERO-WT, Venatorx) in the 2014-2018 training years, 91,110 modelled rows in total. *A. baumannii*: 34,365 ATLAS isolates, ATLAS only, no supplementary data. Years modelled: 2008-2022.
**Data provenance**: the ATLAS file labels each isolate with a study (TEST, ATLAS, Inform); there is no SMART label, and Merck/SMART data are not used. TEST and Inform are believed to be Pfizer-run studies; confirm against the Vivli data dictionary before submission. A retrain on 2026-07-22 had accidentally added 6,181 supplementary rows (2014-2018) to the *A. baumannii* training data; this was corrected on 2026-11-23 by rebuilding with `--atlas-only` and retraining.
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

Source: `reports/model/model_results_{species}.md` (retrain of 2026-07). Test set 2019-2022; resistant = MIC >= 8 mg/L (4,305 *K. pneumoniae* and 9,415 *A. baumannii* test isolates).

*K. pneumoniae* + Meropenem (93 feature columns):

| Model | RMSE all | RMSE resistant | MAE resistant |
|-------|----------|----------------|---------------|
| RF baseline | 1.772 | 2.947 | 1.995 |
| **XGBoost tuned** (3x resistant weight) | 2.224 | **2.172** | 1.350 |
| XGBoost Q0.90 | 4.224 | 1.161 | 0.656 |

*A. baumannii* + Meropenem (84 feature columns):

| Model | RMSE all | RMSE resistant | MAE resistant |
|-------|----------|----------------|---------------|
| RF baseline | 2.509 | 1.623 | 1.310 |
| **XGBoost tuned** (3x resistant weight) | 2.684 | **0.995** | 0.687 |
| XGBoost Q0.90 | 3.657 | 0.275 | 0.099 |

All metrics in log2 units. Overall RMSE is dominated by the large share of isolates imputed at the censoring floor, so the resistant-subset RMSE is the clinically relevant metric. The RF baseline has lower overall RMSE; the 3x-weighted XGBoost trades overall RMSE for accuracy on resistant isolates. For *A. baumannii* the primary reported output is the resistance-prevalence trend (binary P(R) classifier with a logistic trend fit, exported to `api_abaumannii_prevalence_trend.json`), not MIC regression.

### 7.2 MIC90 Trend - Predicted vs Actual (*K. pneumoniae*)

| Year | Actual MIC90 (mg/L) | Predicted MIC90 (mg/L) | % resistant actual | % resistant predicted |
|------|---------------------|------------------------|--------------------|-----------------------|
| 2018 (train) | 16 | | 11.5 | |
| 2019 | 32 | 32 | 14.9 | 13.0 |
| 2020 | 32 | 32 | 15.2 | 14.2 |
| 2021 | 32 | 32 | 18.1 | 15.8 |
| 2022 | 32 | 32 | 16.1 | 15.6 |

Actual MIC90 reaches the 32 mg/L panel ceiling from 2019, so MIC90 cannot rise further in the data and is right-censored. The hybrid MIC90 post-processing (global P(R) above 10% gives the 32 mg/L ceiling, otherwise a country-stratified weighted average; implemented in `4_run_export.py`) reproduces the ceiling in all test years. Resistance prevalence rose from 3.3% (2008) to 11.5% (2018) and 14.9-18.1% in 2019-2022, and the model tracks it with a small underestimate in 2021.

### 7.3 Key Driver Analysis (SHAP, *K. pneumoniae*)

Source: `reports/api/api_kpneumoniae_shap_importance.json`.

| Rank | Feature | Mean abs SHAP | Note |
|------|---------|---------------|------|
| 1 | lag1_pct_resistant_ctry | 0.752 | Previous-year resistance rate in the country |
| 2 | KPC_pos | 0.605 | KPC carbapenemase, main driver in training period |
| 3 | KPC_x_year | 0.285 | KPC by year interaction |
| 4 | OXA_pos | 0.275 | OXA-48-like, dominant in Europe and Middle East |
| 5 | NDM_pos | 0.254 | NDM, fastest-rising mechanism |
| 6 | OXA_x_year | 0.182 | |
| 7 | NDM_x_year | 0.173 | |
| 8 | ctry_China | 0.147 | High-resistance country |
| 9 | ctry_India | 0.122 | NDM origin country |
| 13 | year | 0.065 | Temporal creep signal |

The top feature is an autoregressive country-level lag, so part of the signal is persistence of local resistance rather than isolate biology. KPC ranks above NDM because the training period (2008-2018) is KPC-dominated; NDM became prevalent later and is underrepresented in training, which is a data recency limitation, not a model error. `is_censored` is no longer a feature. These rankings still require validation by the domain expert for biological plausibility before publication.

For *A. baumannii* (`api_abaumannii_shap_importance.json`) the ranking is led by `lag1_pct_resistant_ctry`, then `year`, `ward_icu`, `age_paediatric` and `age_elderly`; OXA gene flags do not appear in the top 12, although OXA-type carbapenemases are the dominant mechanism in this species. This needs expert review (see `docs/shap_validation_checklist.md`).

### 7.4 Additional Experiments (both species)

Script: `scripts/pipeline/6_run_improvements.py`. Same features, hyperparameters and time split as the production models (train 2008-2018, test 2019-2022). Full tables: `reports/model/improvements_{species}.md`.

Production `xgb_tuned` is trained with 3x sample weight on resistant isolates (MIC >= 8 mg/L), so the "resistant x3" row is the production recipe. The "unweighted" row is a reference without that weighting.

*K. pneumoniae* (RMSE in log2 units, resistant = MIC >= 8 mg/L):

| Variant | RMSE all | RMSE resistant | Sensitivity (R) |
|---------|----------|----------------|-----------------|
| Unweighted XGBoost | 1.77 | 2.90 | 0.75 |
| Recency weighting | 1.78 | 2.85 | 0.74 |
| LightGBM | 1.78 | 2.88 | 0.74 |
| NDM/OXA weighting | 1.79 | 2.86 | 0.74 |
| Resistant x3 (production recipe) | 2.22 | 2.17 | 0.83 |
| Tobit debiased targets | 2.11 | 3.12 | 0.72 |
| Tobit as stacked feature | 2.45 | 5.50 | 0.15 |

Findings:
- Resistant weighting is the only change that materially improves the clinically relevant subset, at the cost of overall RMSE. Recency weighting and LightGBM give differences too small to justify changing the pipeline.
- Tobit variants did not help. About 67% of K. pneumoniae training targets are left-censored, and both uses of Tobit (debiased targets, stacked feature) scored worse than the unweighted reference. The Tobit year coefficient is conditional on the gene-by-year interaction features and is not reported as a creep rate.
- Censoring bounds for Tobit are approximate: the imputed floor varies by panel, and a single left bound per species was used (K. pneumoniae log2 0.06 mg/L, A. baumannii 0.5 mg/L).
- EUCAST classification (R / not R at 8 mg/L): the dedicated P(R) classifier for *K. pneumoniae* reaches AUC 0.975, sensitivity 0.834 and specificity 0.984 on 2019-2022.
- Subgroups (resistant RMSE, resistant x3 model): *K. pneumoniae* military proxy 1.90, paediatric 2.40, other 2.16. Subgroup MIC90 slopes are in the generated report. The military proxy is small (775 test isolates) and approximate.

*A. baumannii* (ATLAS-only data, resistant RMSE on 2019-2022): unweighted XGBoost 1.67, recency weighting 1.69, NDM/OXA weighting 1.68, LightGBM 1.61, Tobit debiased targets 1.46, Tobit as a stacked feature 1.48, resistant x3 (production recipe) **0.995** with sensitivity 0.94. The Tobit variants beat the unweighted reference (14.6% of training targets are left-censored) but remain far from the production recipe, so Tobit was not adopted for either species. The Tobit year coefficient (-0.10 log2/yr against +0.13 naive) is conditional on the interaction features and is not reported as a creep rate. The experiment's P(R) classifier reached AUC 0.842 (0.845 with the export recipe). Subgroups (resistant RMSE, production recipe): paediatric 1.67, military proxy 1.37, other 0.93.

Opt 7 feature set (continent one-hot encoding plus a 3-year trailing per-country MIC90) for *A. baumannii*, re-tuned with Optuna on the ATLAS-only data (`3e_run_optuna_opt7.py`): resistant RMSE 1.003 against 0.995 for production, overall RMSE 2.688 against 2.684, so no improvement. An earlier gain (1.045 against 1.135) came from the pre-rebuild data that wrongly included supplementary datasets and did not hold. With the old hyperparameters the feature set was clearly worse (1.66). For the P(R) classifier the Opt 7 features gave AUC 0.847 against 0.845, also no meaningful gain. Not promoted.

P(R) classifier tuning for *A. baumannii* (`8_tune_pr_classifier.py`, 50 Optuna trials, validation on training years 2016-2018, test untouched during tuning), ATLAS-only data: the tuned classifier was worse than the production recipe on test (AUC 0.837 against 0.845, log-loss 0.448 against 0.437, yearly prevalence error 4.4 against 2.5 percentage points). A validation-chosen threshold (0.61) trades sensitivity for specificity (0.91 / 0.59 against 0.94 / 0.50 at 0.5); it is a choice, not an improvement. An earlier run on the pre-rebuild mixed data suggested tuning halves the prevalence error; that did not replicate, so the production recipe is kept. The 2016-2018 validation years are few and sit in a different prevalence regime than the test years, so tuning on them may not transfer. An AUC near 0.84 appears to be the ceiling for the current feature set.

Robustness of the *A. baumannii* prevalence trend (`9_check_prevalence_trend.py`, ATLAS-only data, `reports/model/prevalence_trend_check_abaumannii.md`): observed prevalence steps from 44% (2011) to 62% (2012), reaches 69% in 2018, and is 70.0 / 69.5 / 71.2 / 67.3% in 2019-2022. Re-weighting each year to a fixed country mix (pooled 2008-2022 shares) gives 63.6 / 61.2 / 62.1 / 58.8% for 2019-2022, so the high raw test-year level is partly a change in which countries contribute isolates. The logit-linear slope is +0.078/yr raw over 2008-2022 (p 0.0001) but +0.028/yr standardised (p 0.034), and 2012-2022 alone is +0.032/yr (p 0.060). The 2011-2012 step is partly composition (Turkey and Mexico enter in 2012, and only 30 countries contribute in 2011) but also appears within countries (for example Germany 13% to 49%, Spain 55% to 73%, United States 36% to 49%), which may reflect a change in surveillance sampling rather than epidemiology; this is unresolved. The p-values use 11 to 15 autocorrelated annual points and are optimistic. The exported raw-trend fit (2008-2018, +0.0976 logit/yr, R2 0.659, p 0.0024) predicts 76.8% for 2022 against 67.3% observed and extrapolates to 83% for 2026, so the forecast should be read as an upper-end projection, not a measured creep rate. Isolates with no country column set are assigned the reference country used in the export (Argentina), and rare countries are pooled.

Feature and structure experiments (`3d_run_feature_experiments.py`, evaluated against production `xgb_tuned`): continent one-hot encoding, a rolling country MIC90, and tiered models (separate models for high- and low-resistance geographies) all lowered overall RMSE for *K. pneumoniae* (1.79 to 1.82 against 2.22) but made resistant-subset RMSE clearly worse (2.85 to 2.95 against 2.17), so none was adopted. The same pattern held for *A. baumannii* (resistant RMSE 1.5 to 2.3 against 1.135), but that run used the pre-rebuild mixed data and was not repeated on the ATLAS-only data; the gap is large enough that the direction is unlikely to change.

Not available in the local data, so not tested: SENTRY, post-2020 NDM-rich data, and a carbapenem consumption file (`data/external/carbapenem_ddd.csv`). ATLAS has no hospital/community-acquired flag, so the ward/specialty features serve as the infection setting proxy.

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
