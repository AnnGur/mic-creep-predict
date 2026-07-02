# Model Training Results — K. pneumoniae + Meropenem
**Generated**: 2026-07-02 20:57
---
## Evaluation Metrics
| Model | RMSE (all) | MAE (all) | R2 (all) | RMSE (R subset) | MAE (R subset) | N resistant |
|---|---|---|---|---|---|---|
| RF baseline | 1.7978 | 1.2172 | 0.7598 | 3.0215 | 2.1308 | 4,305 |
| XGBoost tuned | 2.2172 | 1.6801 | 0.6347 | 2.2936 | 1.4171 | 4,305 |
| XGBoost Q0.90 | 4.1833 | 3.2303 | -0.3005 | 1.2047 | 0.7199 | 4,305 |

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
  "max_depth": 7,
  "learning_rate": 0.016802861627862372,
  "subsample": 0.7694514135270807,
  "colsample_bytree": 0.5297888932659223,
  "min_child_weight": 14,
  "gamma": 0.3323394155194935,
  "reg_alpha": 0.9531058072037039,
  "reg_lambda": 9.40864404244111
}
```

---

## Next Steps

- Review SHAP values with domain expert — confirm biological plausibility
- Build FastAPI endpoint: `src/api/main.py`
- Push model artefact to Hugging Face Hub
- Prepare submission write-up
