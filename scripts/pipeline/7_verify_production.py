"""
Verify saved production artefacts directly against the test set.
Reports, per species: RMSE / RMSE(resistant) of xgb_tuned, xgb_hybrid,
xgb_quantile, and the AUC / sensitivity / specificity of the P(R) classifier
exactly as 4_run_export.py trains it (recency-weighted, production params).
Output: reports/model/production_verification.json

Run: .venv/bin/python scripts/pipeline/7_verify_production.py
"""
import importlib.util, json, sys, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, mean_squared_error, roc_auc_score

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import xgboost as xgb


def train_pr_classifier(X_train, y_train, params):
    """Same recipe as 4_run_export.train_pr_classifier (copied to avoid its heavy
    module-level imports): recency weight 0.85 ** (max_year - year)."""
    p = params.copy()
    n_est = int(p.pop("n_estimators", 500))
    w = 0.85 ** (int(X_train["year"].max()) - X_train["year"].values)
    clf = xgb.XGBClassifier(n_estimators=n_est, objective="binary:logistic",
                            eval_metric="logloss", seed=42, verbosity=0, **p)
    clf.fit(X_train.values, (y_train.values >= 3.0).astype(int), sample_weight=w)
    return clf

out = {}
for s in ["kpneumoniae", "abaumannii"]:
    d = ROOT / "data/processed" / s
    Xtr = pd.read_parquet(d / "X_train.parquet")
    ytr = pd.read_parquet(d / "y_train.parquet").squeeze().values
    Xte = pd.read_parquet(d / "X_test.parquet").reindex(columns=Xtr.columns, fill_value=0)
    y = pd.read_parquet(d / "y_test.parquet").squeeze().values
    r = y >= 3
    out[s] = {"n_train": len(Xtr), "n_test": len(Xte), "n_features": Xtr.shape[1]}
    for n in ["xgb_tuned", "xgb_hybrid", "xgb_quantile"]:
        f = ROOT / "models" / f"{n}_{s}.pkl"
        if not f.exists():
            continue
        try:
            p = joblib.load(f).predict(Xte.values)
        except ValueError as e:      # stale artefact trained on another feature set
            out[s][n] = {"skipped": str(e).splitlines()[0][:90]}
            print(s, n, "SKIPPED:", out[s][n]["skipped"])
            continue
        out[s][n] = {"rmse": float(mean_squared_error(y, p) ** .5),
                     "rmse_resistant": float(mean_squared_error(y[r], p[r]) ** .5)}
        print(s, n, out[s][n])
    params = json.loads((ROOT / "models" / f"xgb_best_params_{s}.json").read_text())
    clf = train_pr_classifier(Xtr, pd.Series(ytr), params)
    prob = clf.predict_proba(Xte.values)[:, 1]
    yt = r.astype(int); pred = (prob >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    out[s]["pr_classifier_export_recipe"] = {
        "auc": float(roc_auc_score(yt, prob)), "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp))}
    print(s, "P(R)", out[s]["pr_classifier_export_recipe"])
(ROOT / "reports/model/production_verification.json").write_text(json.dumps(out, indent=2))
print("Wrote reports/model/production_verification.json")
