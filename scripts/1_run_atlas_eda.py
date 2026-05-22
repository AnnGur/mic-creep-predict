"""
ATLAS EDA — K. pneumoniae + Meropenem MIC Creep
Vivli AMR Surveillance Challenge 2026

Run:
    python scripts/run_atlas_eda.py
    python scripts/run_atlas_eda.py --data-dir /path/to/raw --reports /path/to/reports
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display required

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.loader import ATLASLoader
from data.preprocessor import MICPreprocessor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPECIES = "Klebsiella pneumoniae"
ANTIBIOTIC = "Meropenem"
EUCAST_R = 8  # EUCAST 2024 R breakpoint for K. pneumoniae Meropenem (mg/L)

WOUND_SOURCES = {"Wound", "Abscess", "Skin and Soft Tissue"}
CARBAPENEMASE_GENES = ["KPC", "NDM", "OXA", "VIM", "IMP", "GES"]

AGE_ORDER = ["0 - 17", "18 - 30", "31 - 60", "61+"]
AGE_GROUP_BROAD = {
    "0 - 17": "Paediatric (0–17)",
    "18 - 30": "Adult (18–60)",
    "31 - 60": "Adult (18–60)",
    "61+": "Elderly (61+)",
}

sns.set_style("whitegrid")
plt.rcParams.update({"figure.figsize": (13, 5), "figure.dpi": 100})


# ---------------------------------------------------------------------------
# Data loading & parsing
# ---------------------------------------------------------------------------

def load_and_parse(data_dir: Path) -> pd.DataFrame:
    """Load ATLAS, filter to species+antibiotic, parse censored MICs."""
    loader = ATLASLoader(data_dir)
    df = loader.load(SPECIES, antibiotic=ANTIBIOTIC)

    print(f"  Raw rows loaded:  {len(df):,}")
    print(f"  Year range:       {df['Year'].min()} – {df['Year'].max()}")
    print(f"  Countries:        {df['Country'].nunique()}")

    parsed = df[ANTIBIOTIC].apply(
        lambda x: MICPreprocessor.parse_censored_mic(x) if pd.notna(x) else (None, None)
    )
    df["mic_value"] = parsed.apply(lambda t: t[0])
    df["mic_operator"] = parsed.apply(lambda t: t[1])

    df = df[df["mic_value"].notna() & (df["mic_value"] > 0)].copy()
    df["mic_log2"] = np.log2(df["mic_value"])
    df["is_censored"] = df["mic_operator"].isin([">", "<", ">=", "<="])
    df["is_resistant"] = df["mic_value"] >= EUCAST_R

    df["age_group_broad"] = df["Age Group"].map(AGE_GROUP_BROAD).fillna("Adult (18–60)")

    df["military_proxy"] = (
        df["Source"].isin(WOUND_SOURCES)
        & (df["Gender"] == "Male")
        & (df["Age Group"].isin(["18 - 30", "31 - 60"]))
    )

    gene_cols_present = [g for g in CARBAPENEMASE_GENES if g in df.columns]
    for gene in gene_cols_present:
        s = df[gene].astype(str).str.strip()
        df[f"{gene}_pos"] = df[gene].notna() & ~s.isin(["", "nan", "0", "None"])

    print(f"  Valid rows:       {len(df):,}")
    print(f"  Censored:         {df['is_censored'].sum():,} ({df['is_censored'].mean()*100:.1f}%)")
    print(f"  Resistant (≥{EUCAST_R}): {df['is_resistant'].sum():,} ({df['is_resistant'].mean()*100:.1f}%)")
    print(f"  Military proxy:   {df['military_proxy'].sum():,} ({df['military_proxy'].mean()*100:.1f}%)")

    return df


def _yearly_stats(df: pd.DataFrame) -> pd.DataFrame:
    yearly = (
        df.groupby("Year")
        .agg(
            n=("mic_value", "count"),
            mic50=("mic_value", lambda x: x.quantile(0.50)),
            mic90=("mic_value", lambda x: x.quantile(0.90)),
            pct_resistant=("is_resistant", "mean"),
        )
        .reset_index()
    )
    yearly["pct_resistant"] *= 100
    return yearly


# ---------------------------------------------------------------------------
# Section 1 — Year-by-year overview table
# ---------------------------------------------------------------------------

def print_yearly_table(df: pd.DataFrame) -> None:
    yearly = _yearly_stats(df)
    print(f"\n{'Year':>4}  {'N':>6}  {'MIC_50':>6}  {'MIC_90':>6}  {'%R':>5}")
    print("-" * 36)
    for _, r in yearly.iterrows():
        print(f"{int(r.Year):4d}  {int(r.n):6,}  {r.mic50:6.2f}  {r.mic90:6.2f}  {r.pct_resistant:5.1f}%")


# ---------------------------------------------------------------------------
# Section 2 — MIC_90 trend (main result chart)
# ---------------------------------------------------------------------------

def plot_mic90_trend(df: pd.DataFrame, out: Path) -> None:
    yearly = _yearly_stats(df)

    slope90, int90, r90, p90, _ = linregress(yearly["Year"], yearly["mic90"])
    slope50, int50, r50, p50, _ = linregress(yearly["Year"], yearly["mic50"])

    fig, ax1 = plt.subplots()

    ax1.plot(yearly["Year"], yearly["mic90"], "o-", color="#d62728", lw=2.5, ms=6, label="MIC_90", zorder=3)
    ax1.plot(yearly["Year"], yearly["mic50"], "s--", color="#1f77b4", lw=1.5, ms=5, label="MIC_50", alpha=0.8, zorder=3)
    ax1.plot(yearly["Year"], int90 + slope90 * yearly["Year"], "-", color="#d62728", alpha=0.25, lw=2)
    ax1.plot(yearly["Year"], int50 + slope50 * yearly["Year"], "-", color="#1f77b4", alpha=0.25, lw=2)
    ax1.axhline(EUCAST_R, color="black", linestyle=":", lw=1.5, label=f"EUCAST R ({EUCAST_R} mg/L)")

    ax1.set_yscale("log", base=2)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax1.set_ylabel("Meropenem MIC (mg/L)", fontsize=11)
    ax1.set_xlabel("Year", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title(
        f"K. pneumoniae Meropenem MIC_90 Trend 2004–2024\n"
        f"MIC_90 slope: {slope90:+.3f} mg/L/yr  (R²={r90**2:.2f}, p={p90:.2e})",
        fontsize=12,
        fontweight="bold",
    )

    ax2 = ax1.twinx()
    ax2.bar(yearly["Year"], yearly["n"], alpha=0.10, color="grey", width=0.7)
    ax2.set_ylabel("Isolates per year", fontsize=10, color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    fname = "mic90_trend_kpneumoniae_meropenem.png"
    fig.savefig(out / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {fname}")
    print(f"     MIC_90 slope: {slope90:+.4f} mg/L/yr | R²={r90**2:.3f} | p={p90:.2e}")
    print(f"     MIC_50 slope: {slope50:+.4f} mg/L/yr | R²={r50**2:.3f} | p={p50:.2e}")


# ---------------------------------------------------------------------------
# Section 3 — MIC distribution by year (violin)
# ---------------------------------------------------------------------------

def plot_violin(df: pd.DataFrame, out: Path) -> None:
    all_years = sorted(df["Year"].unique())
    plot_years = [y for y in all_years if y % 2 == 0]
    df_vln = df[df["Year"].isin(plot_years)].copy()

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.violinplot(
        data=df_vln,
        x="Year",
        y="mic_log2",
        inner="quartile",
        palette="RdYlGn_r",
        density_norm="width",
        ax=ax,
    )
    ax.axhline(
        np.log2(EUCAST_R), color="black", linestyle="--", lw=1.5,
        label=f"R breakpoint (log2 = {np.log2(EUCAST_R):.0f})",
    )
    ax.set_ylabel("log2(MIC) [log2 mg/L]", fontsize=11)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_title(
        "Meropenem MIC Distribution by Year — K. pneumoniae (ATLAS)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9)
    plt.xticks(rotation=45)
    plt.tight_layout()
    fname = "mic_violin_by_year.png"
    fig.savefig(out / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {fname}")


# ---------------------------------------------------------------------------
# Section 4 — Geographic analysis
# ---------------------------------------------------------------------------

def plot_geographic(df: pd.DataFrame, out: Path) -> None:
    country_stats = (
        df.groupby("Country")
        .agg(
            n=("mic_value", "count"),
            mic90=("mic_value", lambda x: x.quantile(0.90)),
            pct_resistant=("is_resistant", "mean"),
        )
        .reset_index()
    )
    country_stats = country_stats[country_stats["n"] >= 50].copy()
    top_mic90 = country_stats.sort_values("mic90", ascending=False).head(30)
    top_resist = country_stats.sort_values("pct_resistant", ascending=False).head(30)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    axes[0].barh(top_mic90["Country"], top_mic90["mic90"], color="#d62728", alpha=0.8)
    axes[0].axvline(EUCAST_R, color="black", linestyle="--", lw=1.2, label="R breakpoint")
    axes[0].set_xlabel("MIC_90 (mg/L)", fontsize=11)
    axes[0].set_title("MIC_90 by Country (top 30, ≥50 isolates)", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)

    axes[1].barh(top_resist["Country"], top_resist["pct_resistant"] * 100, color="#ff7f0e", alpha=0.8)
    axes[1].set_xlabel("% Resistant (MIC ≥ 8 mg/L)", fontsize=11)
    axes[1].set_title("% Resistant by Country (top 30)", fontsize=11, fontweight="bold")

    plt.suptitle("K. pneumoniae Meropenem — Geographic Analysis (ATLAS)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = "geo_analysis.png"
    fig.savefig(out / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {fname}")


# ---------------------------------------------------------------------------
# Section 5 — MIC_90 by age group over time
# ---------------------------------------------------------------------------

def plot_age_groups(df: pd.DataFrame, out: Path) -> None:
    age_stats = (
        df[df["Age Group"].isin(AGE_ORDER)]
        .groupby("Age Group")
        .agg(
            n=("mic_value", "count"),
            mic50=("mic_value", lambda x: x.quantile(0.50)),
            mic90=("mic_value", lambda x: x.quantile(0.90)),
            pct_resistant=("is_resistant", "mean"),
        )
        .reindex(AGE_ORDER)
    )
    print(f"\n  Age group summary:\n{age_stats.to_string()}")

    palette = sns.color_palette("Set2", len(AGE_ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = range(len(age_stats))

    axes[0].bar(x, age_stats["mic90"], color=palette)
    axes[0].axhline(EUCAST_R, linestyle="--", color="black", lw=1.2, label="R breakpoint")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(age_stats.index)
    axes[0].set_ylabel("MIC_90 (mg/L)")
    axes[0].set_title("MIC_90 by Age Group")
    axes[0].legend(fontsize=8)

    axes[1].bar(x, age_stats["pct_resistant"] * 100, color=palette)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(age_stats.index)
    axes[1].set_ylabel("% Resistant")
    axes[1].set_title("% Resistant by Age Group")

    plt.suptitle("K. pneumoniae Meropenem — Age Group Analysis (ATLAS)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    # Over-time panel
    group_yearly = (
        df.groupby(["Year", "age_group_broad"])["mic_value"]
        .quantile(0.90)
        .reset_index()
    )
    group_yearly.columns = ["Year", "Group", "MIC90"]

    colors = {
        "Paediatric (0–17)": "#e377c2",
        "Adult (18–60)": "#1f77b4",
        "Elderly (61+)": "#ff7f0e",
    }
    fig2, ax2 = plt.subplots()
    for grp, gdf in group_yearly.groupby("Group"):
        ax2.plot(gdf["Year"], gdf["MIC90"], "o-", label=grp, color=colors.get(grp, "grey"), lw=2, ms=5)
    ax2.axhline(EUCAST_R, color="black", linestyle=":", lw=1.5, label="EUCAST R breakpoint")
    ax2.set_yscale("log", base=2)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax2.set_ylabel("MIC_90 (mg/L)", fontsize=11)
    ax2.set_xlabel("Year", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_title("MIC_90 by Age Group Over Time — K. pneumoniae Meropenem (ATLAS)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    fname1 = "age_group_bar.png"
    fname2 = "mic90_by_age_group.png"
    fig.savefig(out / fname1, dpi=150, bbox_inches="tight")
    fig2.savefig(out / fname2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plt.close(fig2)
    print(f"  → {fname1}")
    print(f"  → {fname2}")


# ---------------------------------------------------------------------------
# Section 6 — Military proxy
# ---------------------------------------------------------------------------

MIN_N_MILITARY = 50  # minimum isolates per year for a reliable MIC_90 estimate

def plot_military_proxy(df: pd.DataFrame, out: Path) -> None:
    mil = df[df["military_proxy"]]
    mil_n   = mil.groupby("Year")["mic_value"].count().reset_index().rename(columns={"mic_value": "n"})
    mil_mic = mil.groupby("Year")["mic_value"].quantile(0.90).reset_index().rename(columns={"mic_value": "mic90"})
    mil_yr  = mil_n.merge(mil_mic, on="Year")

    gen_yr = df[~df["military_proxy"]].groupby("Year")["mic_value"].quantile(0.90).reset_index()

    reliable   = mil_yr[mil_yr["n"] >= MIN_N_MILITARY]
    unreliable = mil_yr[mil_yr["n"] <  MIN_N_MILITARY]

    print(f"  Military proxy n per year (threshold={MIN_N_MILITARY}):")
    for _, r in mil_yr.iterrows():
        flag = "" if r.n >= MIN_N_MILITARY else "  ← low n, greyed out"
        print(f"    {int(r.Year)}: n={int(r.n)}{flag}")

    fig, ax = plt.subplots()

    ax.plot(gen_yr["Year"], gen_yr["mic_value"], "o-", color="#1f77b4", lw=2, ms=5, label="General population")

    # Reliable proxy years — solid red
    if not reliable.empty:
        ax.plot(reliable["Year"], reliable["mic90"], "s", color="#d62728", ms=7, zorder=4)
        # Connect only consecutive reliable years with a line
        ax.plot(reliable["Year"], reliable["mic90"], "s--", color="#d62728", lw=1.8,
                label=f"Military proxy (n >= {MIN_N_MILITARY})")

    # Unreliable years — grey markers, no line, small
    if not unreliable.empty:
        ax.scatter(unreliable["Year"], unreliable["mic90"],
                   marker="s", color="grey", s=30, zorder=3, alpha=0.5,
                   label=f"Military proxy (n < {MIN_N_MILITARY}, unreliable)")

    ax.axhline(EUCAST_R, color="black", linestyle=":", lw=1.5, label="EUCAST R breakpoint")
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_ylabel("MIC_90 (mg/L)", fontsize=11)
    ax.set_xlabel("Year", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_title("Military Proxy vs General Population — MIC_90 Trend", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fname = "mic90_military_proxy.png"
    fig.savefig(out / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {fname}")


# ---------------------------------------------------------------------------
# Section 7 — Specimen source
# ---------------------------------------------------------------------------

def plot_specimen_source(df: pd.DataFrame, out: Path) -> None:
    source_stats = (
        df.groupby("Source")
        .agg(
            n=("mic_value", "count"),
            mic90=("mic_value", lambda x: x.quantile(0.90)),
            pct_resistant=("is_resistant", "mean"),
        )
        .reset_index()
    )
    source_stats = source_stats[source_stats["n"] >= 100].sort_values("mic90", ascending=False)

    fig, ax = plt.subplots(figsize=(14, 7))
    norm = plt.Normalize(source_stats["mic90"].min(), source_stats["mic90"].max())
    colors_bar = plt.cm.RdYlGn_r(norm(source_stats["mic90"]))
    ax.barh(source_stats["Source"], source_stats["mic90"], color=colors_bar, alpha=0.9)
    ax.axvline(EUCAST_R, color="black", linestyle="--", lw=1.5, label="R breakpoint")
    for i, (_, row) in enumerate(source_stats.iterrows()):
        ax.text(
            row["mic90"] + 0.15, i,
            f"n={int(row['n']):,}  ({row['pct_resistant']*100:.0f}%R)",
            va="center", fontsize=8,
        )
    ax.set_xlabel("MIC_90 (mg/L)", fontsize=11)
    ax.set_title(
        "MIC_90 by Specimen Source (≥100 isolates) — K. pneumoniae Meropenem (ATLAS)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    fname = "specimen_source_mic90.png"
    fig.savefig(out / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {fname}")


# ---------------------------------------------------------------------------
# Section 8 — Carbapenemase gene prevalence over time
# ---------------------------------------------------------------------------

def plot_resistance_genes(df: pd.DataFrame, out: Path) -> None:
    gene_cols = [g for g in CARBAPENEMASE_GENES if g in df.columns]
    if not gene_cols:
        print("  (no carbapenemase gene columns present — skipping)")
        return

    pos_cols = [f"{g}_pos" for g in gene_cols]
    gene_yr = df.groupby("Year")[pos_cols].mean().reset_index()
    gene_yr.columns = ["Year"] + gene_cols

    fig, ax = plt.subplots()
    for gene, color in zip(gene_cols, sns.color_palette("tab10", len(gene_cols))):
        ax.plot(gene_yr["Year"], gene_yr[gene] * 100, "o-", label=gene, color=color, lw=2, ms=5)
    ax.set_ylabel("% Isolates with Gene", fontsize=11)
    ax.set_xlabel("Year", fontsize=11)
    ax.legend(title="Carbapenemase Gene", fontsize=9)
    ax.set_title(
        "Carbapenemase Gene Prevalence Over Time — K. pneumoniae (ATLAS)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fname = "gene_prevalence_over_time.png"
    fig.savefig(out / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {fname}")

    res_df = df[df["is_resistant"]]
    print(f"\n  Among {len(res_df):,} resistant isolates (MIC ≥ {EUCAST_R} mg/L):")
    for gene in gene_cols:
        pct = res_df[f"{gene}_pos"].mean() * 100
        print(f"    {gene:<6}: {pct:.1f}% carry the gene")


# ---------------------------------------------------------------------------
# Section 9 — Data quality
# ---------------------------------------------------------------------------

def plot_data_quality(df: pd.DataFrame, out: Path) -> None:
    quality = (
        df.groupby("Year")
        .agg(
            n_total=("mic_value", "count"),
            n_censored=("is_censored", "sum"),
            n_resistant=("is_resistant", "sum"),
        )
        .reset_index()
    )
    quality["pct_censored"] = quality["n_censored"] / quality["n_total"] * 100
    quality["pct_resistant"] = quality["n_resistant"] / quality["n_total"] * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(quality["Year"], quality["pct_censored"], color="#aec7e8", alpha=0.9)
    axes[0].set_ylabel("% Censored MIC values", fontsize=11)
    axes[0].set_xlabel("Year", fontsize=11)
    axes[0].set_title("Censoring Rate Over Time", fontsize=11, fontweight="bold")

    axes[1].plot(quality["Year"], quality["pct_resistant"], "o-", color="#d62728", lw=2, ms=5)
    axes[1].set_ylabel("% Resistant (MIC ≥ 8 mg/L)", fontsize=11)
    axes[1].set_xlabel("Year", fontsize=11)
    axes[1].set_title("Resistance Rate Over Time", fontsize=11, fontweight="bold")

    plt.suptitle(
        "Data Quality & Resistance Overview — K. pneumoniae Meropenem (ATLAS)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fname = "data_quality.png"
    fig.savefig(out / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {fname}")
    print(f"\n  Overall censoring rate:  {quality['pct_censored'].mean():.1f}%")
    print(f"  Overall resistance rate: {quality['pct_resistant'].mean():.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ATLAS EDA — K. pneumoniae + Meropenem MIC Creep"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Directory containing atlas_vivli_2004_2024.csv (default: data/raw/)",
    )
    parser.add_argument(
        "--reports",
        type=Path,
        default=PROJECT_ROOT / "reports",
        help="Output directory for PNG charts (default: reports/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.reports.mkdir(parents=True, exist_ok=True)

    n = 10

    print(f"[1/{n}] Load & parse data")
    print(f"  data-dir: {args.data_dir}")
    df = load_and_parse(args.data_dir)

    print(f"\n[2/{n}] Year-by-year overview table")
    print_yearly_table(df)

    print(f"\n[3/{n}] MIC_90 trend")
    plot_mic90_trend(df, args.reports)

    print(f"\n[4/{n}] Violin by year")
    plot_violin(df, args.reports)

    print(f"\n[5/{n}] Geographic analysis")
    plot_geographic(df, args.reports)

    print(f"\n[6/{n}] Age group analysis")
    plot_age_groups(df, args.reports)

    print(f"\n[7/{n}] Military proxy")
    plot_military_proxy(df, args.reports)

    print(f"\n[8/{n}] Specimen source")
    plot_specimen_source(df, args.reports)

    print(f"\n[9/{n}] Resistance gene prevalence")
    plot_resistance_genes(df, args.reports)

    print(f"\n[10/{n}] Data quality")
    plot_data_quality(df, args.reports)

    print(f"\nDone. Charts saved to: {args.reports}")


if __name__ == "__main__":
    main()
