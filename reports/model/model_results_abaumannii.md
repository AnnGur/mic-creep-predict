# Model Training Results — A. baumannii + Meropenem
**Generated**: 2026-06-29 13:38
---
## Evaluation Metrics
| Model | RMSE (all) | MAE (all) | R2 (all) | RMSE (R subset) | MAE (R subset) | N resistant |
|---|---|---|---|---|---|---|
| RF baseline | 2.6040 | 2.0972 | 0.2775 | 1.7142 | 1.4718 | 9,415 |
| XGBoost tuned | 2.7061 | 1.7476 | 0.2197 | 1.0860 | 0.7012 | 9,415 |
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
  "n_estimators": 1100,
  "max_depth": 5,
  "learning_rate": 0.0541292449276475,
  "subsample": 0.6644987581833566,
  "colsample_bytree": 0.6233969548871495,
  "min_child_weight": 7,
  "gamma": 0.8374708167373068,
  "reg_alpha": 2.655232677257683,
  "reg_lambda": 7.773111477453876
}
```

---

## Next Steps

- Review SHAP values with domain expert — confirm biological plausibility
- Build FastAPI endpoint: `src/api/main.py`
- Push model artefact to Hugging Face Hub
- Prepare submission write-up
