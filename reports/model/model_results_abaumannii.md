# Model Training Results — A. baumannii + Meropenem
**Generated**: 2026-07-02 20:57
---
## Evaluation Metrics
| Model | RMSE (all) | MAE (all) | R2 (all) | RMSE (R subset) | MAE (R subset) | N resistant |
|---|---|---|---|---|---|---|
| RF baseline | 2.5022 | 1.9381 | 0.3328 | 1.6513 | 1.3388 | 9,415 |
| XGBoost tuned | 2.6878 | 1.7473 | 0.2302 | 1.0268 | 0.6882 | 9,415 |
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
  "n_estimators": 400,
  "max_depth": 7,
  "learning_rate": 0.03604888939743829,
  "subsample": 0.956751202451233,
  "colsample_bytree": 0.803356138277042,
  "min_child_weight": 5,
  "gamma": 0.00504768374825719,
  "reg_alpha": 3.4230998099385426,
  "reg_lambda": 5.496445946643274
}
```

---

## Next Steps

- Review SHAP values with domain expert — confirm biological plausibility
- Build FastAPI endpoint: `src/api/main.py`
- Push model artefact to Hugging Face Hub
- Prepare submission write-up
