"""
Generate EDA charts from processed parquets — identical style for both species.

Outputs (to reports/eda/):
  gene_prevalence_over_time_{species}.png  — carbapenemase gene % by year
  specimen_source_mic90_{species}.png      — MIC90 by specimen source (top 10)

Run:
  MPLBACKEND=Agg MPLCONFIGDIR=/tmp/.mpl \\
    .venv/bin/python scripts/utils/gen_eda_charts.py --species kpneumoniae
  MPLBACKEND=Agg MPLCONFIGDIR=/tmp/.mpl \\
    .venv/bin/python scripts/utils/gen_eda_charts.py --species abaumannii
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "eda"
REPORTS.mkdir(parents=True, exist_ok=True)

EUCAST_R = 8.0
GENE_PALETTE = {
    "KPC": "#e41a1c",
    "NDM": "#377eb8",
    "OXA": "#4daf4a",
    "VIM": "#984ea3",
    "IMP": "#ff7f00",
    "GES": "#a65628",
}
SPECIES_LABELS = {
    "kpneumoniae": "K. pneumoniae",
    "abaumannii":  "A. baumannii",
}


def load_combined(species: str):
    data_dir = ROOT / "data" / "processed" / species
    Xtr = pd.read_parquet(data_dir / "X_train.parquet")
    Xte = pd.read_parquet(data_dir / "X_test.parquet")
    ytr = pd.read_parquet(data_dir / "y_train.parquet").squeeze()
    yte = pd.read_parquet(data_dir / "y_test.parquet").squeeze()
    X = pd.concat([Xtr, Xte], ignore_index=True)
    y = pd.concat([ytr, yte], ignore_index=True)
    return X, y


def plot_gene_prevalence(species: str) -> None:
    label = SPECIES_LABELS[species]
    X, _ = load_combined(species)

    gene_cols = [f"{g}_pos" for g in GENE_PALETTE if f"{g}_pos" in X.columns]
    if not gene_cols:
        print("  (no gene columns found - skipping)")
        return

    gene_yr = X.groupby("year")[gene_cols].mean().reset_index()

    fig, ax = plt.subplots(figsize=(11, 5))
    for col in gene_cols:
        gene = col.replace("_pos", "")
        ax.plot(gene_yr["year"], gene_yr[col] * 100, "o-",
                label=gene, color=GENE_PALETTE[gene], lw=2, ms=5)

    ax.set_ylabel("% Isolates with Gene", fontsize=11)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(title="Carbapenemase Gene", fontsize=9, loc="upper left")
    ax.set_title(f"Carbapenemase Gene Prevalence Over Time — {label} (ATLAS)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = REPORTS / f"gene_prevalence_over_time_{species}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.name}")


def plot_specimen_source(species: str, top_n: int = 10) -> None:
    label = SPECIES_LABELS[species]
    X, y = load_combined(species)

    mic_mg_l = 2 ** y.values
    is_resistant = mic_mg_l >= EUCAST_R
    spec_cols = [c for c in X.columns if c.startswith("spec_")]

    rows = []
    for col in spec_cols:
        mask = X[col].values == 1
        if mask.sum() < 50:
            continue
        rows.append({
            "source": col.replace("spec_", "").capitalize(),
            "n": int(mask.sum()),
            "mic90": float(np.quantile(mic_mg_l[mask], 0.90)),
            "pct_resistant": float(is_resistant[mask].mean() * 100),
        })

    assigned = X[spec_cols].any(axis=1)
    mask_other = ~assigned.values
    if mask_other.sum() >= 50:
        rows.append({
            "source": "Other",
            "n": int(mask_other.sum()),
            "mic90": float(np.quantile(mic_mg_l[mask_other], 0.90)),
            "pct_resistant": float(is_resistant[mask_other].mean() * 100),
        })

    rows.sort(key=lambda r: (r["mic90"], r["pct_resistant"]), reverse=True)
    rows = rows[:top_n]

    sources = [r["source"] for r in rows]
    mic90s  = [r["mic90"] for r in rows]
    resists = [r["pct_resistant"] for r in rows]

    # Color by %R when MIC90 is uniform (saturated at ceiling), else by MIC90
    mic90_range = max(mic90s) - min(mic90s)
    if mic90_range < 1.0:
        norm = plt.Normalize(min(resists), max(resists))
        colors = plt.cm.RdYlGn_r(norm(resists))
        color_note = " (color = %R — MIC90 saturated at ceiling for all sources)"
    else:
        norm = plt.Normalize(min(mic90s), max(mic90s))
        colors = plt.cm.RdYlGn_r(norm(mic90s))
        color_note = ""

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(sources, mic90s, color=colors, alpha=0.9)
    ax.axvline(EUCAST_R, color="black", linestyle="--", lw=1.5, label="R breakpoint (8 mg/L)")
    for i, r in enumerate(rows):
        x_pos = r["mic90"] + max(mic90s) * 0.01
        ax.text(x_pos, i,
                f"n={r['n']:,}  ({r['pct_resistant']:.0f}%R)",
                va="center", fontsize=9)
    ax.set_xlabel(f"MIC₉₀ (mg/L){color_note}", fontsize=10)
    ax.set_title(f"MIC₉₀ by Specimen Source — {label} Meropenem (ATLAS)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = REPORTS / f"specimen_source_mic90_{species}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", choices=["kpneumoniae", "abaumannii"],
                        default="abaumannii")
    args = parser.parse_args()

    print(f"Generating EDA charts for {SPECIES_LABELS[args.species]}...")
    print("1/2 Gene prevalence over time...")
    plot_gene_prevalence(args.species)
    print("2/2 Specimen source MIC90...")
    plot_specimen_source(args.species)
    print("Done.")


if __name__ == "__main__":
    main()
