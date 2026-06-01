# Model Findings — K. pneumoniae + Meropenem MIC Creep
**Dataset**: ATLAS (Pfizer/Vivli), 89,572 isolates, 2004–2022  
**Generated**: 2026-06-01

---

## Core Result

**MIC creep is confirmed and statistically significant.**

| Metric | Value |
|---|---|
| MIC90 slope | +1.97 mg/L/yr |
| R² (linear trend) | 0.67 |
| p-value | 6.3×10⁻⁶ |
| Resistance rate 2007 → 2024 | 5% → 20% (4× increase) |
| Dominant mechanism post-2020 | NDM + OXA (~8-9% each) |

---

## Model Performance

### Train / Test Split (time-aware — never shuffled)

| Split | Rows | Years |
|---|---|---|
| Train | 62,891 | 2004–2018 |
| Test | 26,681 | 2019–2022 |

### Results

| Model | RMSE all | RMSE resistant | MAE resistant | N resistant |
|---|---|---|---|---|
| RF baseline | 1.558 | 2.869 | 2.180 | 4,305 |
| **XGBoost tuned** | **1.758** | **1.960** | **1.127** | 4,305 |

All metrics in log₂ units. **RMSE on the resistant subset is the clinically relevant figure** — the overall RMSE is dominated by the ~75% of isolates imputed at the censoring floor (log₂ = −5).

XGBoost improves resistant-subset RMSE by **32%** over the RF baseline (2.87 → 1.96).

### MIC90 Trend — Predicted vs Actual (Test 2019–2022)

| Year | Actual (log₂) | RF | XGBoost |
|---|---|---|---|
| 2019 | 5.00 | 2.58 | 3.66 |
| 2020 | 5.00 | 2.80 | 3.87 |
| 2021 | 5.00 | 3.11 | 3.99 |
| 2022 | 5.00 | 2.87 | 3.92 |

Actual MIC90 = log₂(5) = 32 mg/L in all test years — this is the **panel ceiling artifact** (values right-censored at 32; true MIC90 may be higher). Both models correctly predict an upward trend. XGBoost gets within ~1 log₂ dilution of the ceiling.

### XGBoost Best Hyperparameters (Optuna, 40 trials)

```
n_estimators:     500
max_depth:        7
learning_rate:    0.0106
subsample:        0.879
colsample_bytree: 0.695
min_child_weight: 3
gamma:            1.633
reg_alpha:        1.085
reg_lambda:       6.996
```

Internal validation: RMSE 1.808 on last 3 training years (2016–2018).

---

## SHAP — What Drives Predicted MIC?

Top 15 features by mean |SHAP value| on the test set:

| Rank | Feature | Mean |SHAP| | Biological meaning |
|---|---|---|---|
| 1 | KPC_pos | 0.943 | KPC carbapenemase — strongest resistance driver in train period |
| 2 | is_censored | 0.543 | **Artifact** — censored = at panel floor = structurally low MIC |
| 3 | OXA_pos | 0.504 | OXA-48/OXA-232 class — dominant in Europe/Middle East |
| 4 | NDM_pos | 0.438 | NDM — rising threat, bypasses avibactam combinations |
| 5 | pct_censored_year | 0.331 | Methodology control — partial out surveillance artifact |
| 6 | ctry_China | 0.304 | High-resistance country |
| 7 | ctry_India | 0.160 | NDM origin country, high baseline resistance |
| 8 | ctry_Italy | 0.116 | KPC-endemic country |
| 9 | ctry_Greece | 0.077 | KPC-endemic country |
| 10 | year | 0.071 | Temporal creep signal |
| 11 | ctry_Brazil | 0.036 | |
| 12 | VIM_pos | 0.034 | Metallo-beta-lactamase (minor contribution) |
| 13-15 | ctry_Israel/Russia/France | ~0.02 | |

### Key interpretations

**KPC dominates training signal.** KPC was the primary carbapenemase in 2004–2018 (the training period), and ATLAS captured many KPC-positive isolates with exact (uncensored) MIC values during the 2012–2018 low-censoring window. The model learned the strong KPC → high MIC association from this signal-rich period.

**NDM ranks below KPC despite biological importance.** NDM overtook KPC in 2021–2022 (the test period), meaning it was relatively rare in the training set. This is a data recency problem, not a model error. With SENTRY data or retraining on post-2018 data, NDM importance will rise.

**`is_censored` is #2 — flag as artifact.** Censored observations are structurally at the MIC floor (because the panel can't measure below 0.06 mg/L). The model correctly uses this as a signal, but domain experts must be told it is a data-structure variable — not biology. Do not show it in a clinical SHAP plot without this explanation.

**`year` ranks only #10.** The model is primarily differentiating resistance by gene presence and country, not by time. This is expected given the extreme bimodal target distribution — the annual shift is small relative to the gene/country effect.

---

## Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Panel ceiling at 32 mg/L | MIC90 saturates post-2018; true values unknown | Flag as right-censored; trend analysis valid 2004–2018 only |
| Bimodal target (75% at floor) | RMSE dominated by floor spike; model misses resistant tail | 3× sample weight on resistant isolates during training |
| Covariate shift in censoring | Train has 25–87% censoring variation; test is locked at 88–90% | `pct_censored_year` feature explicitly controls for this |
| NDM underrepresented in train | Reduces model sensitivity to the fastest-rising mechanism | Include post-2020 data when retraining; note in submission |
| Military proxy is broad | "wound + male + 18-60" captures 2,813 isolates (2.8%); many are not combat-related | Proxy is stated as approximate; acceptable for PoC |
| Paediatric MIC90 artifact | Flat at 0.12 mg/L 2006–2016, then jumps to 32 in 2019 | Driven by censoring floor (91.7% at floor pre-2017) + small n; flagged in diagnostic notebook |

---

## Next Steps

1. **Domain expert review** — Validate SHAP top features for biological plausibility. Confirm that KPC > NDM ordering is expected for a 2004–2018 training period.
2. **FastAPI endpoint** — Serve predictions via REST API for the submission dashboard.
3. **SENTRY integration** — Second dataset to validate generalisability across labs.
4. **Submission write-up** — Methodology section using findings from `atlas_eda_analysis.md` and this document.
