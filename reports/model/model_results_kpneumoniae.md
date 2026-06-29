# Model Training Results — K. pneumoniae + Meropenem
**Generated**: 2026-06-29 11:38
---
## Evaluation Metrics
| Model | RMSE (all) | MAE (all) | R2 (all) | RMSE (R subset) | MAE (R subset) | N resistant |
|---|---|---|---|---|---|---|
| Linear Regression (interpretable) | 1.9029 | 1.2443 | 0.7309 | 3.7509 | 2.8603 | 4,305 |
| RF baseline | 1.7782 | 1.1143 | 0.7650 | 2.9972 | 2.2592 | 4,305 |
| XGBoost tuned | 1.8958 | 1.3910 | 0.7329 | 2.3595 | 1.4673 | 4,305 |

> **Note**: RMSE on the resistant subset (MIC >= 8 mg/L) is the clinically relevant metric. The full-set RMSE is dominated by the ~75% of isolates imputed at the censoring floor (log2=-5).

---

## MIC_90 Trend — Actual vs Predicted

![MIC_90 trend predicted](mic90_trend_predicted.png)

---

## Residuals by Year

![Residuals](residuals_by_year.png)

---

## RMSE by Year

![RMSE by year](rmse_by_year.png)

---

## SHAP Feature Importance

![SHAP beeswarm](shap_beeswarm.png)

![SHAP bar](shap_summary.png)

---

## XGBoost Best Hyperparameters (Optuna)

```json
{
  "n_estimators": 300,
  "max_depth": 6,
  "learning_rate": 0.01356199603274235,
  "subsample": 0.9007938638565945,
  "colsample_bytree": 0.8384908435118392,
  "min_child_weight": 9,
  "gamma": 1.2783409290641892,
  "reg_alpha": 2.5154431178772767,
  "reg_lambda": 9.538685217866622
}
```

---

## Next Steps

- Review SHAP values with domain expert — confirm biological plausibility
- Flag `is_censored` in SHAP plots as a data-structure artifact (not biology)
- Build FastAPI endpoint: `src/api/main.py`
- Push model artefact to Hugging Face Hub
- Prepare submission write-up
