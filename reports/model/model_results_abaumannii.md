# Model Training Results — A. baumannii + Meropenem
**Generated**: 2026-06-30 20:58
---
## Evaluation Metrics
| Model | RMSE (all) | MAE (all) | R2 (all) | RMSE (R subset) | MAE (R subset) | N resistant |
|---|---|---|---|---|---|---|
| RF baseline | 2.6077 | 2.1520 | 0.2754 | 1.8527 | 1.6182 | 9,415 |
| XGBoost tuned | 2.6924 | 1.7579 | 0.2276 | 1.1874 | 0.7789 | 9,415 |
| XGBoost Q0.90 | 3.8857 | 2.6189 | -0.6089 | 1.0295 | 0.8334 | 9,415 |

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
  "n_estimators": 900,
  "max_depth": 4,
  "learning_rate": 0.10411630546830548,
  "subsample": 0.7838411618943655,
  "colsample_bytree": 0.5917385974767692,
  "min_child_weight": 14,
  "gamma": 0.011458223408450985,
  "reg_alpha": 4.3861625696641635,
  "reg_lambda": 1.1668364649416088
}
```

---

## Next Steps

- Review SHAP values with domain expert — confirm biological plausibility
- Build FastAPI endpoint: `src/api/main.py`
- Push model artefact to Hugging Face Hub
- Prepare submission write-up
