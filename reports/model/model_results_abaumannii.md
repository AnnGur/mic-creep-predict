# Model Training Results — A. baumannii + Meropenem
**Generated**: 2026-06-29 16:09
---
## Evaluation Metrics
| Model | RMSE (all) | MAE (all) | R2 (all) | RMSE (R subset) | MAE (R subset) | N resistant |
|---|---|---|---|---|---|---|
| RF baseline | 2.6073 | 2.0890 | 0.2756 | 1.7039 | 1.4544 | 9,415 |
| XGBoost tuned | 2.7136 | 1.7549 | 0.2154 | 1.0735 | 0.7010 | 9,415 |
| XGBoost Q0.90 | 3.6710 | 2.0228 | -0.4360 | 0.2642 | 0.0524 | 9,415 |

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
  "n_estimators": 700,
  "max_depth": 5,
  "learning_rate": 0.05505703610991669,
  "subsample": 0.806507299808196,
  "colsample_bytree": 0.6121114505101882,
  "min_child_weight": 6,
  "gamma": 0.30013530521423654,
  "reg_alpha": 0.41440331074275605,
  "reg_lambda": 5.510155656648423
}
```

---

## Next Steps

- Review SHAP values with domain expert — confirm biological plausibility
- Build FastAPI endpoint: `src/api/main.py`
- Push model artefact to Hugging Face Hub
- Prepare submission write-up
