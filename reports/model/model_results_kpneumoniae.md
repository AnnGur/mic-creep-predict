# Model Training Results — K. pneumoniae + Meropenem
**Generated**: 2026-06-29 16:09
---
## Evaluation Metrics
| Model | RMSE (all) | MAE (all) | R2 (all) | RMSE (R subset) | MAE (R subset) | N resistant |
|---|---|---|---|---|---|---|
| RF baseline | 1.7750 | 1.1265 | 0.7659 | 3.0528 | 2.2768 | 4,305 |
| XGBoost tuned | 1.8998 | 1.4221 | 0.7318 | 2.4086 | 1.4916 | 4,305 |
| XGBoost Q0.90 | 3.1533 | 2.5031 | 0.2611 | 1.4452 | 0.8929 | 4,305 |

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
  "n_estimators": 600,
  "max_depth": 4,
  "learning_rate": 0.010612878172829193,
  "subsample": 0.7146163616857919,
  "colsample_bytree": 0.6902785247460157,
  "min_child_weight": 6,
  "gamma": 0.165830636381099,
  "reg_alpha": 3.779781487252998,
  "reg_lambda": 7.758354287759749
}
```

---

## Next Steps

- Review SHAP values with domain expert — confirm biological plausibility
- Build FastAPI endpoint: `src/api/main.py`
- Push model artefact to Hugging Face Hub
- Prepare submission write-up
