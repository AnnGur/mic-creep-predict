"""
Generate A. baumannii charts from processed parquet data.
Run: .venv/bin/python scripts/_gen_charts_ab.py

Outputs (all in reports/):
  mic90_trend_abaumannii_meropenem.png
  gene_prevalence_over_time_abaumannii.png
  specimen_source_mic90_abaumannii.png
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "eda"
API_DIR = ROOT / "reports" / "api"
DATA    = ROOT / "data" / "processed" / "abaumannii"
REPORTS.mkdir(parents=True, exist_ok=True)
EUCAST_R = 8.0
SPECIES_LABEL = "A. baumannii"


# ---------------------------------------------------------------------------
# Helper: simple linear regression without scipy
# ---------------------------------------------------------------------------

def polyfit_line(x, y):
    m, b = np.polyfit(x, y, 1)
    yhat = m * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return m, b, r2


# ---------------------------------------------------------------------------
# 1. MIC90 trend (from aggregated JSON)
# ---------------------------------------------------------------------------

def plot_mic90_trend():
    raw = json.loads((API_DIR / "api_abaumannii_mic90_trend.json").read_text())
    years, mic90, mic50, ns = [], [], [], []
    for r in raw:
        if "actual_mic90" in r and "n" in r:
            years.append(r["year"])
            mic90.append(r["actual_mic90"])
            mic50.append(r.get("actual_mic50", float("nan")))
            ns.append(r["n"])

    years = np.array(years, dtype=float)
    mic90 = np.array(mic90, dtype=float)
    mic50 = np.array(mic50, dtype=float)
    ns    = np.array(ns, dtype=float)
    slope90, int90, r2_90 = polyfit_line(years, mic90)
    slope50, int50, r2_50 = polyfit_line(years, mic50)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(years, mic90, "o-", color="#d62728", lw=2.5, ms=6, label="MIC_90", zorder=3)
    ax1.plot(years, mic50, "s--", color="#1f77b4", lw=1.5, ms=5, label="MIC_50", alpha=0.8, zorder=3)
    ax1.plot(years, int90 + slope90 * years, "-", color="#d62728", alpha=0.25, lw=2)
    ax1.plot(years, int50 + slope50 * years, "-", color="#1f77b4", alpha=0.25, lw=2)
    ax1.axhline(EUCAST_R, color="black", linestyle=":", lw=1.5, label=f"EUCAST R ({EUCAST_R} mg/L)")
    ax1.set_yscale("log", base=2)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax1.set_ylabel("Meropenem MIC (mg/L)", fontsize=11)
    ax1.set_xlabel("Year", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title(
        f"{SPECIES_LABEL} Meropenem MIC_90 Trend 2004-2022\n"
        f"MIC_90 slope: {slope90:+.3f} mg/L/yr  (R²={r2_90:.2f})",
        fontsize=12, fontweight="bold",
    )
    ax2 = ax1.twinx()
    ax2.bar(years, ns, alpha=0.10, color="grey", width=0.7)
    ax2.set_ylabel("Isolates per year", fontsize=10, color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    out = REPORTS / "mic90_trend_abaumannii_meropenem.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.name}")


# ---------------------------------------------------------------------------
# 2. Gene prevalence over time (from X_train + X_test parquets)
# ---------------------------------------------------------------------------

def plot_gene_prevalence():
    Xtr = pd.read_parquet(DATA / "X_train.parquet")
    Xte = pd.read_parquet(DATA / "X_test.parquet")
    X = pd.concat([Xtr, Xte], ignore_index=True)

    genes = [g for g in ["KPC_pos", "NDM_pos", "OXA_pos", "VIM_pos", "IMP_pos", "GES_pos"] if g in X.columns]
    if not genes:
        print("  (no gene columns found - skipping gene prevalence chart)")
        return

    gene_yr = X.groupby("year")[genes].mean().reset_index()

    fig, ax = plt.subplots(figsize=(11, 5))
    palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
    for gene, color in zip(genes, palette):
        label = gene.replace("_pos", "")
        ax.plot(gene_yr["year"], gene_yr[gene] * 100, "o-", label=label,
                color=color, lw=2, ms=5)

    ax.set_ylabel("% Isolates with Gene", fontsize=11)
    ax.set_xlabel("Year", fontsize=11)
    ax.legend(title="Carbapenemase Gene", fontsize=9, loc="upper left")
    ax.set_title(
        f"Carbapenemase Gene Prevalence Over Time — {SPECIES_LABEL} (ATLAS)",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    out = REPORTS / "gene_prevalence_over_time_abaumannii.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.name}")


# ---------------------------------------------------------------------------
# 3. Specimen source MIC90 (from X + y parquets)
# ---------------------------------------------------------------------------

def plot_specimen_source():
    Xtr = pd.read_parquet(DATA / "X_train.parquet")
    Xte = pd.read_parquet(DATA / "X_test.parquet")
    ytr = pd.read_parquet(DATA / "y_train.parquet").squeeze()
    yte = pd.read_parquet(DATA / "y_test.parquet").squeeze()

    X = pd.concat([Xtr, Xte], ignore_index=True)
    y = pd.concat([ytr, yte], ignore_index=True)

    spec_cols = [c for c in X.columns if c.startswith("spec_")]
    if not spec_cols:
        print("  (no spec_ columns - skipping specimen chart)")
        return

    mic_mg_l = 2 ** y.values
    is_resistant = mic_mg_l >= EUCAST_R

    rows = []
    for col in spec_cols:
        mask = X[col].values == 1
        if mask.sum() < 50:
            continue
        label = col.replace("spec_", "").capitalize()
        rows.append({
            "source": label,
            "n": int(mask.sum()),
            "mic90": float(np.quantile(mic_mg_l[mask], 0.90)),
            "pct_resistant": float(is_resistant[mask].mean() * 100),
        })

    # Add "Other/unspecified" for isolates not in any spec_ column
    assigned = X[spec_cols].any(axis=1)
    mask_other = ~assigned.values
    if mask_other.sum() >= 50:
        rows.append({
            "source": "Other",
            "n": int(mask_other.sum()),
            "mic90": float(np.quantile(mic_mg_l[mask_other], 0.90)),
            "pct_resistant": float(is_resistant[mask_other].mean() * 100),
        })

    rows.sort(key=lambda r: r["mic90"], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    sources = [r["source"] for r in rows]
    mic90s  = [r["mic90"] for r in rows]
    norm = plt.Normalize(min(mic90s), max(mic90s))
    colors = plt.cm.RdYlGn_r(norm(mic90s))
    ax.barh(sources, mic90s, color=colors, alpha=0.9)
    ax.axvline(EUCAST_R, color="black", linestyle="--", lw=1.5, label="R breakpoint")
    for i, r in enumerate(rows):
        ax.text(r["mic90"] + 0.3, i,
                f"n={r['n']:,}  ({r['pct_resistant']:.0f}%R)",
                va="center", fontsize=9)
    ax.set_xlabel("MIC_90 (mg/L)", fontsize=11)
    ax.set_title(
        f"MIC_90 by Specimen Source — {SPECIES_LABEL} Meropenem (ATLAS)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = REPORTS / "specimen_source_mic90_abaumannii.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.name}")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Imports done. Starting chart generation...")
    print("1/3 MIC90 trend...", flush=True)
    plot_mic90_trend()
    print("2/3 Gene prevalence...", flush=True)
    plot_gene_prevalence()
    print("3/3 Specimen source...", flush=True)
    plot_specimen_source()
    print("Done.")
