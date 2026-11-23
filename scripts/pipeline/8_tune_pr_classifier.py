"""
Tune the P(R) classifier (resistance prevalence model) for A. baumannii.
=======================================================================
The production classifier (4_run_export.train_pr_classifier) reuses the
REGRESSION hyperparameters and a fixed recency decay of 0.85. On 2019-2022
it reaches AUC 0.844 / sensitivity 0.90 / specificity 0.60 (A. baumannii).

This script tunes the classifier for its own objective, with a strictly
time-aware inner validation split INSIDE the training years:
    inner-train 2008-2015, validation 2016-2018  (test 2019-2022 is untouched
    during tuning). Objective: validation log-loss (calibrated probabilities
    matter because year-level prevalence is the headline output).
Tuned: XGBoost structure + regularisation, scale_pos_weight, recency decay.

Variants compared ONCE on the test set:
    P0  production recipe (regression params, decay 0.85)
    P1  tuned params + tuned decay
    P2  P1 with decision threshold chosen on validation (Youden's J)
    P3  P0 + Opt7 features (continent OHE + rolling MIC90)
    P4  P1 + Opt7 features
Metrics: AUC-ROC, AUC-PR, log-loss, Brier, sens/spec, and the mean absolute
error (percentage points) of predicted yearly prevalence vs observed.

Outputs:
    models/pr_clf_params_{species}.json     (tuned params; NOT wired into the API)
    reports/model/pr_classifier_tuning_{species}.json / .md

Run: .venv/bin/python scripts/pipeline/8_tune_pr_classifier.py --species abaumannii
"""
import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, log_loss, roc_auc_score, roc_curve)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
LOG2_R = 3.0
VAL_START = 2016

spec = importlib.util.spec_from_file_location(
    "fx", ROOT / "scripts/pipeline/3d_run_feature_experiments.py")
fx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fx)


def weights(years, decay):
    return decay ** (int(years.max()) - years)


def make_clf(params, n_estimators=None):
    p = dict(params)
    n = int(n_estimators or p.pop("n_estimators", 500))
    p.pop("n_estimators", None)
    return xgb.XGBClassifier(n_estimators=n, objective="binary:logistic",
                             eval_metric="logloss", seed=42, verbosity=0, **p)


def evaluate(yt, prob, years, thr=0.5):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    df = pd.DataFrame({"y": yt, "p": prob, "yr": years})
    by = df.groupby("yr").agg(obs=("y", "mean"), pred=("p", "mean"))
    return dict(
        auc=float(roc_auc_score(yt, prob)),
        auc_pr=float(average_precision_score(yt, prob)),
        logloss=float(log_loss(yt, np.clip(prob, 1e-6, 1 - 1e-6))),
        brier=float(brier_score_loss(yt, prob)),
        threshold=float(thr),
        sensitivity=float(tp / (tp + fn)), specificity=float(tn / (tn + fp)),
        prevalence_mae_pp=float((by["obs"] - by["pred"]).abs().mean() * 100),
        prevalence_by_year={int(k): dict(obs=round(float(v.obs) * 100, 1),
                                         pred=round(float(v.pred) * 100, 1))
                            for k, v in by.iterrows()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default="abaumannii",
                    choices=["kpneumoniae", "abaumannii"])
    ap.add_argument("--n-trials", type=int, default=50)
    args = ap.parse_args()
    s = args.species

    Xtr, ytr, Xte, yte = fx.load_data(s)
    prod_params = fx.load_xgb_params(s)
    yr_tr, yr_te = Xtr["year"].values, Xte["year"].values
    ctr, cte = (ytr.values >= LOG2_R).astype(int), (yte.values >= LOG2_R).astype(int)
    print(f"{s}: train {Xtr.shape} test {Xte.shape}; prevalence train "
          f"{ctr.mean():.1%} test {cte.mean():.1%}")

    inner = yr_tr < VAL_START
    Xi, Xv = Xtr.values[inner], Xtr.values[~inner]
    ci, cv = ctr[inner], ctr[~inner]
    print(f"  inner-train {inner.sum():,} (<{VAL_START}), validation {(~inner).sum():,}")

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 30),
            gamma=trial.suggest_float("gamma", 0.0, 3.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
            scale_pos_weight=trial.suggest_float("scale_pos_weight", 0.5, 2.0))
        decay = trial.suggest_float("decay", 0.80, 1.0)
        m = make_clf(params)
        m.fit(Xi, ci, sample_weight=weights(yr_tr[inner], decay))
        return log_loss(cv, np.clip(m.predict_proba(Xv)[:, 1], 1e-6, 1 - 1e-6))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args.n_trials)
    best = dict(study.best_params)
    decay = best.pop("decay")
    print(f"  best validation log-loss {study.best_value:.4f}; decay {decay:.3f}")
    print(f"  params {best}")

    # production reference on validation for context
    m0v = make_clf(prod_params)
    m0v.fit(Xi, ci, sample_weight=weights(yr_tr[inner], 0.85))
    p0v = m0v.predict_proba(Xv)[:, 1]
    print(f"  validation: production recipe log-loss "
          f"{log_loss(cv, np.clip(p0v, 1e-6, 1 - 1e-6)):.4f} AUC {roc_auc_score(cv, p0v):.4f}")

    # Youden threshold from validation predictions of the tuned model
    mv = make_clf(best)
    mv.fit(Xi, ci, sample_weight=weights(yr_tr[inner], decay))
    fpr, tpr, thr = roc_curve(cv, mv.predict_proba(Xv)[:, 1])
    youden_thr = float(thr[np.argmax(tpr - fpr)])
    print(f"  Youden threshold (validation): {youden_thr:.3f}")

    # Opt7 features (continent + rolling MIC90 from training years only)
    ctry_tr, ctry_te = fx.reconstruct_country(Xtr), fx.reconstruct_country(Xte)
    X7tr, X7te = fx.apply_continent_collapse(Xtr, Xte, fx.to_continent(ctry_tr),
                                             fx.to_continent(ctry_te))
    lk = fx.build_mic90_lookup(ctry_tr, Xtr["year"], ytr)
    X7tr["rolling_mic90"] = fx.add_rolling_mic90(ctry_tr, Xtr["year"], lk, 3)
    X7te["rolling_mic90"] = fx.add_rolling_mic90(ctry_te, Xte["year"], lk, 3)

    def fit_eval(name, params, dec, Xa, Xb, thr=0.5):
        m = make_clf(params)
        m.fit(Xa.values, ctr, sample_weight=weights(yr_tr, dec))
        r = evaluate(cte, m.predict_proba(Xb.values)[:, 1], yr_te, thr)
        print(f"  {name:<34} AUC {r['auc']:.4f} logloss {r['logloss']:.4f} "
              f"sens {r['sensitivity']:.3f} spec {r['specificity']:.3f} "
              f"prevMAE {r['prevalence_mae_pp']:.2f}pp")
        return r

    res = {}
    res["P0 production recipe"] = fit_eval("P0 production recipe", prod_params, 0.85, Xtr, Xte)
    res["P1 tuned"] = fit_eval("P1 tuned", best, decay, Xtr, Xte)
    res["P2 tuned + Youden threshold"] = fit_eval("P2 tuned + Youden threshold", best,
                                                  decay, Xtr, Xte, youden_thr)
    res["P3 production + Opt7 features"] = fit_eval("P3 production + Opt7 features",
                                                    prod_params, 0.85, X7tr, X7te)
    res["P4 tuned + Opt7 features"] = fit_eval("P4 tuned + Opt7 features", best,
                                               decay, X7tr, X7te)

    out = dict(species=s, n_trials=args.n_trials, val_split=f"train<{VAL_START}",
               best_val_logloss=float(study.best_value), decay=decay,
               youden_threshold=youden_thr, params=best, results=res)
    (ROOT / "models" / f"pr_clf_params_{s}.json").write_text(
        json.dumps(dict(params=best, decay=decay, youden_threshold=youden_thr), indent=2))
    rep = ROOT / "reports" / "model"
    (rep / f"pr_classifier_tuning_{s}.json").write_text(json.dumps(out, indent=2))

    L = [f"# P(R) classifier tuning - {s}", "",
         f"Optuna {args.n_trials} trials, validation = training years >= {VAL_START} "
         "(test 2019-2022 untouched during tuning). Test metrics below.", "",
         "| Variant | AUC | AUC-PR | Log-loss | Brier | Sens | Spec | Prevalence MAE (pp) |",
         "|---|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        L.append(f"| {k} | {r['auc']:.4f} | {r['auc_pr']:.4f} | {r['logloss']:.4f} | "
                 f"{r['brier']:.4f} | {r['sensitivity']:.3f} | {r['specificity']:.3f} | "
                 f"{r['prevalence_mae_pp']:.2f} |")
    L += ["", "Prevalence by test year (observed % / predicted %), P0 vs P1:", "",
          "| Year | Observed | P0 | P1 |", "|---|---|---|---|"]
    for yr, v in res["P0 production recipe"]["prevalence_by_year"].items():
        L.append(f"| {yr} | {v['obs']} | {v['pred']} | "
                 f"{res['P1 tuned']['prevalence_by_year'][yr]['pred']} |")
    (rep / f"pr_classifier_tuning_{s}.md").write_text("\n".join(L) + "\n")
    print(f"Wrote reports/model/pr_classifier_tuning_{s}.md")


if __name__ == "__main__":
    main()
