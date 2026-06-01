# Model Training Results — K. pneumoniae + Meropenem
**Generated**: 2026-06-01 11:01
---
## Evaluation Metrics
| Model | RMSE (all) | MAE (all) | RMSE (R subset) | MAE (R subset) | N resistant |
|---|---|---|---|---|---|
| RF baseline | 1.5578 | 0.8582 | 2.8685 | 2.1804 | 4,305 |
| XGBoost tuned | 1.7580 | 1.0629 | 1.9596 | 1.1273 | 4,305 |

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
  "n_estimators": 500,
  "max_depth": 7,
  "learning_rate": 0.010614000612708266,
  "subsample": 0.8787664449976689,
  "colsample_bytree": 0.6954383239281635,
  "min_child_weight": 3,
  "gamma": 1.6334622790192184,
  "reg_alpha": 1.0854105137735537,
  "reg_lambda": 6.995684167780037
}
```

---

## Next Steps

- Review SHAP values with domain expert — confirm biological plausibility
- Flag `is_censored` in SHAP plots as a data-structure artifact (not biology)
- Build FastAPI endpoint: `src/api/main.py`
- Push model artefact to Hugging Face Hub
- Prepare submission write-up
