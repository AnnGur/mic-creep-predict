# Model Training Results — A. baumannii + Meropenem
**Generated**: 2026-06-14 20:59
---
## Evaluation Metrics
| Model | RMSE (all) | MAE (all) | RMSE (R subset) | MAE (R subset) | N resistant |
|---|---|---|---|---|---|
| RF baseline | 1.3383 | 0.7891 | 0.9832 | 0.5100 | 9,415 |
| XGBoost tuned | 1.3789 | 0.7074 | 0.7481 | 0.2698 | 9,415 |

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
  "n_estimators": 400,
  "max_depth": 5,
  "learning_rate": 0.04402558816902642,
  "subsample": 0.6186035194493451,
  "colsample_bytree": 0.728696503325334,
  "min_child_weight": 2,
  "gamma": 1.6316117995624455,
  "reg_alpha": 3.034361067969938,
  "reg_lambda": 4.172226195617825
}
```

---

## Next Steps

- Review SHAP values with domain expert — confirm biological plausibility
- Flag `is_censored` in SHAP plots as a data-structure artifact (not biology)
- Build FastAPI endpoint: `src/api/main.py`
- Push model artefact to Hugging Face Hub
- Prepare submission write-up
