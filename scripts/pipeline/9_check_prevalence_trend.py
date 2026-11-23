"""
Robustness check of the A. baumannii resistance-prevalence trend.
  1. Raw yearly prevalence P(MIC >= 8), 2008-2022 (train + test pooled).
  2. Logit-linear trend fitted on 2008-2018 vs 2008-2022 vs 2012-2022.
  3. Country-standardised prevalence: yearly rates re-weighted to a fixed
     country mix (pooled 2008-2022 shares) to test whether composition
     changes (not resistance) drive the trend.
  4. The 2011 -> 2012 step: country shares and within-country prevalence.
Output: reports/model/prevalence_trend_check_{species}.json / .md
Run: .venv/bin/python scripts/pipeline/9_check_prevalence_trend.py --species abaumannii
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
ap = argparse.ArgumentParser(); ap.add_argument("--species", default="abaumannii")
s = ap.parse_args().species
d = ROOT / "data/processed" / s
X = pd.concat([pd.read_parquet(d / "X_train.parquet"), pd.read_parquet(d / "X_test.parquet")
               .reindex(columns=pd.read_parquet(d / "X_train.parquet").columns, fill_value=0)],
              ignore_index=True)
y = pd.concat([pd.read_parquet(d / "y_train.parquet").squeeze(),
               pd.read_parquet(d / "y_test.parquet").squeeze()], ignore_index=True)
cc = [c for c in X.columns if c.startswith("ctry_")]
country = X[cc].idxmax(axis=1).str.replace("ctry_", "", regex=False)
country[X[cc].sum(axis=1) == 0] = "Argentina"          # drop_first reference (see export)
df = pd.DataFrame({"year": X["year"].astype(int), "country": country, "r": (y.values >= 3).astype(int)})

yr = df.groupby("year")["r"].agg(["mean", "size"]).rename(columns={"mean": "raw", "size": "n"})

def logit_fit(sub):
    p = sub["raw"].clip(1e-4, 1 - 1e-4)
    lg = np.log(p / (1 - p)); lr = stats.linregress(sub.index.values, lg.values)
    return dict(slope=float(lr.slope), r2=float(lr.rvalue ** 2), p=float(lr.pvalue), n_years=len(sub))

# country-standardised prevalence
share = df["country"].value_counts(normalize=True)
cy = df.groupby(["year", "country"])["r"].mean().unstack()
std = {}
for y_, row in cy.iterrows():
    w = share.reindex(row.dropna().index); std[y_] = float((row.dropna() * w / w.sum()).sum())
yr["standardised"] = pd.Series(std)

out = dict(
    yearly={int(k): dict(n=int(v.n), raw_pct=round(v.raw * 100, 1),
                         standardised_pct=round(v.standardised * 100, 1)) for k, v in yr.iterrows()},
    trend_2008_2018=logit_fit(yr.loc[2008:2018]),
    trend_2008_2022=logit_fit(yr.loc[2008:2022]),
    trend_2012_2022=logit_fit(yr.loc[2012:2022]),
    trend_2008_2022_standardised=logit_fit(yr.loc[2008:2022].assign(raw=yr["standardised"])),
)
# 2011 vs 2012 step
a, b = df[df.year == 2011], df[df.year == 2012]
step = {}
for c in sorted(set(a.country) | set(b.country)):
    ca, cb = a[a.country == c], b[b.country == c]
    if len(ca) + len(cb) >= 40:
        step[c] = dict(n2011=len(ca), n2012=len(cb),
                       pct2011=None if len(ca) < 10 else round(ca.r.mean() * 100, 1),
                       pct2012=None if len(cb) < 10 else round(cb.r.mean() * 100, 1))
out["step_2011_2012_by_country"] = step
out["n_countries"] = {int(k): int(v) for k, v in df.groupby("year")["country"].nunique().items()}
rep = ROOT / "reports/model"
(rep / f"prevalence_trend_check_{s}.json").write_text(json.dumps(out, indent=2))
L = [f"# Prevalence trend check - {s}", "", "| Year | n | Raw % | Country-standardised % | Countries |", "|---|---|---|---|---|"]
for k, v in out["yearly"].items():
    L.append(f"| {k} | {v['n']} | {v['raw_pct']} | {v['standardised_pct']} | {out['n_countries'][k]} |")
L += ["", "| Logit trend | slope/yr | R2 | p | years |", "|---|---|---|---|---|"]
for k in ["trend_2008_2018", "trend_2008_2022", "trend_2012_2022", "trend_2008_2022_standardised"]:
    t = out[k]; L.append(f"| {k} | {t['slope']:+.4f} | {t['r2']:.3f} | {t['p']:.4f} | {t['n_years']} |")
L += ["", "## 2011 -> 2012 by country", "", "| Country | n 2011 | n 2012 | % R 2011 | % R 2012 |", "|---|---|---|---|---|"]
for c, v in step.items():
    L.append(f"| {c} | {v['n2011']} | {v['n2012']} | {v['pct2011']} | {v['pct2012']} |")
(rep / f"prevalence_trend_check_{s}.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
print("Wrote prevalence_trend_check")
