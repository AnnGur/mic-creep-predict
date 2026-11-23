"""
Compare the current A. baumannii processed data (ATLAS + supplementary datasets
in training years) against an ATLAS-only rebuild, using the production P(R)
classifier recipe (regression params, recency decay 0.85) on the same test years.

Build the ATLAS-only data first (writes to a separate folder, nothing overwritten):
    .venv/bin/python scripts/pipeline/2_run_feature_engineering.py \
        --species abaumannii --atlas-only --out data/processed_atlas_only
Run: .venv/bin/python scripts/pipeline/10_compare_atlas_only.py
Output: reports/model/atlas_only_comparison_abaumannii.json / .md
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent.parent
params = json.loads((ROOT / "models/xgb_best_params_abaumannii.json").read_text())
out = {}
for name, d in [("current (with supplementary)", ROOT / "data/processed/abaumannii"),
                ("atlas_only", ROOT / "data/processed_atlas_only/abaumannii")]:
    Xtr = pd.read_parquet(d / "X_train.parquet"); ytr = pd.read_parquet(d / "y_train.parquet").squeeze().values
    Xte = pd.read_parquet(d / "X_test.parquet").reindex(columns=Xtr.columns, fill_value=0)
    yte = pd.read_parquet(d / "y_test.parquet").squeeze().values
    p = dict(params); n = int(p.pop("n_estimators", 500))
    w = 0.85 ** (int(Xtr["year"].max()) - Xtr["year"].values)
    clf = xgb.XGBClassifier(n_estimators=n, objective="binary:logistic", eval_metric="logloss",
                            seed=42, verbosity=0, **p)
    clf.fit(Xtr.values, (ytr >= 3).astype(int), sample_weight=w)
    prob = clf.predict_proba(Xte.values)[:, 1]; yt = (yte >= 3).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, (prob >= .5).astype(int), labels=[0, 1]).ravel()
    by = pd.DataFrame({"y": yt, "p": prob, "yr": Xte["year"].values}).groupby("yr").mean()
    tr = pd.DataFrame({"y": (ytr >= 3).astype(int), "yr": Xtr["year"].values}).groupby("yr")["y"].agg(["mean", "size"])
    out[name] = dict(n_train=len(Xtr), n_test=len(Xte), auc=float(roc_auc_score(yt, prob)),
                     logloss=float(log_loss(yt, np.clip(prob, 1e-6, 1 - 1e-6))),
                     sensitivity=float(tp / (tp + fn)), specificity=float(tn / (tn + fp)),
                     prevalence_mae_pp=float((by["y"] - by["p"]).abs().mean() * 100),
                     train_prevalence_by_year={int(k): [round(float(v["mean"]) * 100, 1), int(v["size"])] for k, v in tr.iterrows()},
                     test_prevalence_by_year={int(k): [round(float(v.y) * 100, 1), round(float(v.p) * 100, 1)] for k, v in by.iterrows()})
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[name].items() if not k.endswith("year")})
rep = ROOT / "reports/model"
(rep / "atlas_only_comparison_abaumannii.json").write_text(json.dumps(out, indent=2))
L = ["# A. baumannii: current data vs ATLAS-only", "", "| Dataset | n train | n test | AUC | Log-loss | Sens | Spec | Prev MAE (pp) |", "|---|---|---|---|---|---|---|---|"]
for k, v in out.items():
    L.append(f"| {k} | {v['n_train']:,} | {v['n_test']:,} | {v['auc']:.4f} | {v['logloss']:.4f} | {v['sensitivity']:.3f} | {v['specificity']:.3f} | {v['prevalence_mae_pp']:.2f} |")
L += ["", "Train prevalence by year, % resistant [n]:", "", "| Year | current | atlas_only |", "|---|---|---|"]
a, b = out["current (with supplementary)"]["train_prevalence_by_year"], out["atlas_only"]["train_prevalence_by_year"]
for y in sorted(a):
    L.append(f"| {y} | {a[y][0]} [{a[y][1]}] | {b.get(y, ['-', '-'])[0]} [{b.get(y, ['-', '-'])[1]}] |")
(rep / "atlas_only_comparison_abaumannii.md").write_text("\n".join(L) + "\n")
print("Wrote atlas_only_comparison")
