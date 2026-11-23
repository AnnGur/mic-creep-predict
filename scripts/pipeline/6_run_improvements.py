"""
Consolidated improvement experiments (items missed in earlier sessions).
=======================================================================
All experiments use the production feature matrix and the production
XGBoost hyperparameters (models/xgb_best_params_{species}.json), the same
time-aware split (train 2008-2018, test 2019-2022), and are compared against
the production-equivalent baseline in ONE table.

  A. Unweighted XGBoost (production params, NO resistant weighting).
     NOTE: production xgb_tuned trains with 3x weight on resistant isolates,
     i.e. row G "resistant x3 weight" is the production recipe, not row A.
  B. Tobit debiased targets   - left/right-censored training targets replaced
                                by Tobit latent predictions (censoring-aware)
  C. Tobit as stacked feature - Tobit latent prediction added as a column
  D. Recency weighting        - later training years weighted higher
  E. NDM/OXA emergence weight - recency weighting + extra weight for isolates
                                carrying fast-rising genes (NDM, OXA)
  F. LightGBM                 - alternative learner (optional in CLAUDE.md)
  G. Opt7-style blend         - average of A and the resistant-upweighted model
  H. EUCAST classification    - R/not-R accuracy, sensitivity, specificity,
                                3-class S/I/R accuracy derived from regression,
                                plus a dedicated P(R) XGBoost classifier
  I. Subgroups                - paediatric, military proxy, rest: RMSE, MAE,
                                resistant RMSE, observed vs predicted log2 slope
  J. Tobit year coefficient   - censoring-corrected creep rate (log2 MIC/year)

Note on infection_type: ATLAS has no hospital/community-acquired flag. The
ward/specialty one-hot features (ICU, surgical, ...) are the available proxy.

Outputs:
    reports/model/improvements_{species}.json
    reports/model/improvements_{species}.md

Run:
    .venv/bin/python scripts/pipeline/6_run_improvements.py --species kpneumoniae
    .venv/bin/python scripts/pipeline/6_run_improvements.py --species abaumannii
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (confusion_matrix, mean_absolute_error,
                             mean_squared_error, roc_auc_score)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.tobit import TobitRegressor

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "model"

EUCAST_R = 8.0
LOG2_R = float(np.log2(EUCAST_R))      # 3.0 : project convention, MIC >= 8 is resistant
LOG2_S = 1.0                           # EUCAST S breakpoint 2 mg/L
RIGHT_CENSOR_LOG2 = 5.5                # imputed >32 mg/L -> 64 mg/L (log2 = 6)
# Left-censoring: "<=X" values were imputed as X/2, so the panel floor varies.
# (imputed-value cutoff, censoring bound L) per species, from the observed
# target distribution: K.pn floors -5.06/-6.06/... (L = log2 0.06);
# A.bau floors at -2.0 (<=0.5 imputed 0.25) and below (L = log2 0.5).
# Imputed values sitting at the next level up (e.g. -4.06 for K.pn) are
# ambiguous with true measurements and are treated as uncensored.
LEFT_RULE = {"kpneumoniae": (-5.0, -4.0589), "abaumannii": (-2.0, -1.0)}


def load(species):
    d = PROJECT_ROOT / "data" / "processed" / species
    Xtr = pd.read_parquet(d / "X_train.parquet").reset_index(drop=True)
    ytr = pd.read_parquet(d / "y_train.parquet").squeeze().reset_index(drop=True)
    Xte = pd.read_parquet(d / "X_test.parquet").reset_index(drop=True)
    yte = pd.read_parquet(d / "y_test.parquet").squeeze().reset_index(drop=True)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
    with open(MODELS_DIR / f"xgb_best_params_{species}.json") as f:
        params = json.load(f)
    return Xtr, ytr, Xte, yte, params


def fit_xgb(X, y, params, w=None):
    p = params.copy()
    n = int(p.pop("n_estimators", 500))
    m = xgb.XGBRegressor(n_estimators=n, objective="reg:squarederror",
                         seed=42, verbosity=0, **p)
    m.fit(X, y, sample_weight=w)
    return m


def fit_lgbm(X, y, params, w=None):
    import lightgbm as lgb
    p = params
    m = lgb.LGBMRegressor(
        n_estimators=int(p.get("n_estimators", 500)),
        learning_rate=p.get("learning_rate", 0.05),
        max_depth=int(p.get("max_depth", 6)),
        subsample=p.get("subsample", 0.8), subsample_freq=1,
        colsample_bytree=p.get("colsample_bytree", 0.8),
        reg_alpha=p.get("reg_alpha", 0.0), reg_lambda=p.get("reg_lambda", 1.0),
        random_state=42, verbose=-1)
    m.fit(X, y, sample_weight=w)
    return m


def metrics(y, p):
    r = y >= LOG2_R
    return dict(
        rmse=float(np.sqrt(mean_squared_error(y, p))),
        mae=float(mean_absolute_error(y, p)),
        rmse_r=float(np.sqrt(mean_squared_error(y[r], p[r]))) if r.any() else None,
        n=int(len(y)), n_r=int(r.sum()))


def eucast_class(y, p):
    """Binary R/not-R and 3-class S/I/R metrics from continuous predictions."""
    yt, yp = (y >= LOG2_R).astype(int), (p >= LOG2_R).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    cls = lambda v: np.where(v <= LOG2_S, 0, np.where(v < LOG2_R, 1, 2))
    return dict(
        accuracy_R=float((yt == yp).mean()),
        sensitivity_R=float(tp / (tp + fn)) if tp + fn else None,
        specificity_R=float(tn / (tn + fp)) if tn + fp else None,
        accuracy_SIR=float((cls(y) == cls(p)).mean()),
        pred_pct_R=float(yp.mean() * 100), obs_pct_R=float(yt.mean() * 100))


def slope(years, vals):
    if len(np.unique(years)) < 2:
        return None
    return float(np.polyfit(years, vals, 1)[0])


def yearly_mic90(years, vals):
    df = pd.DataFrame({"y": years, "v": vals})
    return df.groupby("y")["v"].quantile(0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", choices=["kpneumoniae", "abaumannii"],
                    default="kpneumoniae")
    species = ap.parse_args().species

    Xtr, ytr, Xte, yte, params = load(species)
    ytr_v, yte_v = ytr.values.astype(float), yte.values.astype(float)
    yr_tr, yr_te = Xtr["year"].values, Xte["year"].values
    print(f"{species}: train {Xtr.shape}, test {Xte.shape}")

    results, preds = {}, {}

    def record(name, p):
        results[name] = metrics(yte_v, p)
        results[name]["eucast"] = eucast_class(yte_v, p)
        preds[name] = p
        r = results[name]
        print(f"  {name:<34} RMSE {r['rmse']:.3f}  RMSE(R) {r['rmse_r']:.3f}  "
              f"acc(R) {r['eucast']['accuracy_R']:.3f}  "
              f"sens {r['eucast']['sensitivity_R']:.3f}")

    # A. baseline
    record("A unweighted XGB", fit_xgb(Xtr.values, ytr_v, params).predict(Xte.values))

    # Tobit (needed for B, C, J). Standardise features for L-BFGS stability.
    left = ytr_v <= LEFT_RULE[species][0] + 1e-6
    right = ytr_v >= RIGHT_CENSOR_LOG2
    tobit = None
    try:
        sc = StandardScaler().fit(Xtr.values)
        Ztr, Zte = sc.transform(Xtr.values), sc.transform(Xte.values)
        tobit = TobitRegressor(alpha=1.0)
        tobit.lower = LEFT_RULE[species][1] if left.any() else None
        tobit.upper = 5.0 if right.any() else None
        tobit.fit(Ztr, ytr_v, left_censored_mask=left, right_censored_mask=right)
        print(f"  Tobit converged={tobit.converged_}  sigma={tobit.sigma_:.3f}  "
              f"left={left.mean():.1%}  right={right.mean():.1%}")
    except Exception as e:                      # keep the rest of the run alive
        print(f"  Tobit failed: {e}")

    if tobit is not None:
        lat_tr, lat_te = tobit.predict(Ztr), tobit.predict(Zte)
        # B. debiased targets
        yb = ytr_v.copy()
        yb[left] = np.minimum(lat_tr[left], tobit.lower)
        yb[right] = np.maximum(lat_tr[right], tobit.upper)
        record("B Tobit debiased targets",
               fit_xgb(Xtr.values, yb, params).predict(Xte.values))
        # C. Tobit as stacked feature
        record("C Tobit as feature",
               fit_xgb(np.column_stack([Xtr.values, lat_tr]), ytr_v, params)
               .predict(np.column_stack([Xte.values, lat_te])))
        # J. year coefficient in original (unscaled) units
        yi = list(Xtr.columns).index("year")
        results["J_tobit_year_coef_log2_per_year"] = float(
            tobit.coef_[yi] / sc.scale_[yi])
        ols = np.polyfit(yr_tr, ytr_v, 1)[0]
        results["J_naive_imputed_year_slope"] = float(ols)
        print(f"  Tobit year coef {results['J_tobit_year_coef_log2_per_year']:+.4f} "
              f"vs naive slope {ols:+.4f} log2/yr")

    # D. recency weighting
    span = max(yr_tr.max() - yr_tr.min(), 1)
    w_rec = 1.0 + 2.0 * (yr_tr - yr_tr.min()) / span
    record("D recency weighting",
           fit_xgb(Xtr.values, ytr_v, params, w_rec).predict(Xte.values))

    # E. recency + fast-rising gene weight (NDM/OXA emerged late)
    w_gene = w_rec.copy()
    for g in ("NDM_pos", "OXA_pos"):
        if g in Xtr.columns:
            w_gene[Xtr[g].values == 1] *= 2.0
    record("E recency + NDM/OXA weight",
           fit_xgb(Xtr.values, ytr_v, params, w_gene).predict(Xte.values))

    # F. LightGBM
    try:
        record("F LightGBM", fit_lgbm(Xtr.values, ytr_v, params).predict(Xte.values))
    except Exception as e:
        print(f"  LightGBM skipped: {e}")

    # G. blend of baseline and resistant-upweighted model
    w_res = np.where(ytr_v >= LOG2_R, 3.0, 1.0)
    p_up = fit_xgb(Xtr.values, ytr_v, params, w_res).predict(Xte.values)
    record("G resistant x3 (= production recipe)", p_up)
    record("G blend (A + x3 weight)", 0.5 * preds["A unweighted XGB"] + 0.5 * p_up)

    # H. dedicated P(R) classifier (all species, incl. K. pneumoniae)
    p = params.copy()
    n = int(p.pop("n_estimators", 500))
    clf = xgb.XGBClassifier(n_estimators=n, objective="binary:logistic", seed=42,
                            verbosity=0, eval_metric="logloss", **p)
    clf.fit(Xtr.values, (ytr_v >= LOG2_R).astype(int), sample_weight=w_rec)
    prob = clf.predict_proba(Xte.values)[:, 1]
    yt = (yte_v >= LOG2_R).astype(int)
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    results["H_PR_classifier"] = dict(
        auc_roc=float(roc_auc_score(yt, prob)),
        sensitivity=float(tp / (tp + fn)), specificity=float(tn / (tn + fp)),
        accuracy=float((yt == pred).mean()),
        obs_pct_R_by_year={int(k): float(v) for k, v in
                           pd.Series(yt).groupby(yr_te).mean().items()},
        pred_mean_PR_by_year={int(k): float(v) for k, v in
                              pd.Series(prob).groupby(yr_te).mean().items()})
    h = results["H_PR_classifier"]
    print(f"  P(R) classifier: AUC {h['auc_roc']:.3f} sens {h['sensitivity']:.3f} "
          f"spec {h['specificity']:.3f}")

    # I. subgroup evaluation with the best model on RMSE(R) among regressors
    cand = [k for k in results if k[0] in "ABCDEFG" and isinstance(results[k], dict)
            and results[k].get("rmse_r")]
    best = min(cand, key=lambda k: results[k]["rmse_r"])
    print(f"  Best by RMSE(R): {best}")
    pb = preds[best]
    groups = {}
    masks = {"all": np.ones(len(yte_v), bool)}
    if "age_paediatric" in Xte:
        masks["paediatric"] = Xte["age_paediatric"].values == 1
    if "military_proxy" in Xte:
        masks["military_proxy"] = Xte["military_proxy"].values == 1
    if "age_paediatric" in Xte and "military_proxy" in Xte:
        masks["other"] = ~(masks["paediatric"] | masks["military_proxy"])
    for g, m in masks.items():
        if m.sum() < 30:
            groups[g] = dict(n=int(m.sum()), note="too few isolates")
            continue
        o, q = yearly_mic90(yr_te[m], yte_v[m]), yearly_mic90(yr_te[m], pb[m])
        groups[g] = dict(
            **metrics(yte_v[m], pb[m]),
            eucast=eucast_class(yte_v[m], pb[m]),
            obs_mic90_slope=slope(o.index.values, o.values),
            pred_mic90_slope=slope(q.index.values, q.values))
        print(f"  subgroup {g:<15} n={m.sum():>6} RMSE {groups[g]['rmse']:.3f} "
              f"RMSE(R) {groups[g]['rmse_r'] if groups[g]['rmse_r'] is None else round(groups[g]['rmse_r'], 3)}")
    results["I_subgroups"] = dict(model=best, groups=groups)

    # write outputs
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / f"improvements_{species}.json", "w") as f:
        json.dump(results, f, indent=2)

    base = results["A unweighted XGB"]["rmse_r"]
    lines = [f"# Improvement experiments - {species}", "",
             "Same features, params and time split as production. "
             "RMSE(R) is on isolates with MIC >= 8 mg/L.", "",
             "| Model | RMSE | RMSE(R) | d RMSE(R) | Acc R | Sens R | Spec R | Acc S/I/R |",
             "|---|---|---|---|---|---|---|---|"]
    for k, r in results.items():
        if k[0] in "ABCDEFG" and isinstance(r, dict) and "eucast" in r:
            e = r["eucast"]
            lines.append(f"| {k} | {r['rmse']:.3f} | {r['rmse_r']:.3f} | "
                         f"{r['rmse_r'] - base:+.3f} | {e['accuracy_R']:.3f} | "
                         f"{e['sensitivity_R']:.3f} | {e['specificity_R']:.3f} | "
                         f"{e['accuracy_SIR']:.3f} |")
    lines += ["", f"P(R) classifier: AUC {h['auc_roc']:.3f}, sensitivity "
              f"{h['sensitivity']:.3f}, specificity {h['specificity']:.3f}.", ""]
    if "J_tobit_year_coef_log2_per_year" in results:
        lines.append(f"Tobit (censoring-corrected) creep rate: "
                     f"{results['J_tobit_year_coef_log2_per_year']:+.4f} log2 MIC/yr; "
                     f"naive imputed-target slope: "
                     f"{results['J_naive_imputed_year_slope']:+.4f} log2 MIC/yr.")
    lines += ["", f"## Subgroups (model: {best})", "",
              "| Group | n | RMSE | RMSE(R) | Obs MIC90 slope | Pred MIC90 slope |",
              "|---|---|---|---|---|---|"]
    for g, v in groups.items():
        if "note" in v:
            lines.append(f"| {g} | {v['n']} | too few | | | |")
        else:
            f2 = lambda x: "n/a" if x is None else f"{x:.3f}"
            lines.append(f"| {g} | {v['n']} | {v['rmse']:.3f} | {f2(v['rmse_r'])} | "
                         f"{f2(v['obs_mic90_slope'])} | {f2(v['pred_mic90_slope'])} |")
    (REPORTS_DIR / f"improvements_{species}.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote reports/model/improvements_{species}.md")


if __name__ == "__main__":
    main()
