# Model Training Results — A. baumannii + Meropenem
**Generated**: 2026-06-29 11:39
---
## Evaluation Metrics
| Model | RMSE (all) | MAE (all) | R2 (all) | RMSE (R subset) | MAE (R subset) | N resistant |
|---|---|---|---|---|---|---|
| Linear Regression (interpretable) | 2.6335 | 2.0439 | 0.2610 | 1.6541 | 1.3672 | 9,415 |
| RF baseline | 2.6129 | 2.0735 | 0.2725 | 1.7346 | 1.4500 | 9,415 |
| XGBoost tuned | 2.7286 | 1.7518 | 0.2067 | 1.1275 | 0.7045 | 9,415 |

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
  "n_estimators": 1200,
  "max_depth": 7,
  "learning_rate": 0.029744597173878992,
  "subsample": 0.7375653847768968,
  "colsample_bytree": 0.739030572726955,
  "min_child_weight": 7,
  "gamma": 1.9144185603877224,
  "reg_alpha": 4.738179660045682,
  "reg_lambda": 5.764295442201324
}
```

---

## Next Steps

- Review SHAP values with domain expert — confirm biological plausibility
- Flag `is_censored` in SHAP plots as a data-structure artifact (not biology)
- Build FastAPI endpoint: `src/api/main.py`
- Push model artefact to Hugging Face Hub
- Prepare submission write-up
