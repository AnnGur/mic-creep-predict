"""
Paediatric MIC diagnostic — why is MIC_90 flat 2007-2017?

Tests 4 hypotheses:
  H1: n too small → MIC_90 unreliable
  H2: geographic bias → only low-resistance countries report paediatric data
  H3: censoring floor → 0.125 mg/L is the panel floor, all values hit it
  H4: true biology → paediatric isolates genuinely more susceptible

Run:
    python scripts/run_paediatric_diagnostic.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.loader import ATLASLoader
from data.preprocessor import MICPreprocessor

EUCAST_R = 8
REPORTS  = PROJECT_ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams.update({"figure.figsize": (13, 5), "figure.dpi": 100})


def load(data_dir: Path) -> pd.DataFrame:
    loader = ATLASLoader(data_dir)
    df = loader.load("Klebsiella pneumoniae", antibiotic="Meropenem")
    parsed = df["Meropenem"].apply(
        lambda x: MICPreprocessor.parse_censored_mic(x) if pd.notna(x) else (None, None)
    )
    df["mic_value"]    = parsed.apply(lambda t: t[0])
    df["mic_operator"] = parsed.apply(lambda t: t[1])
    df = df[df["mic_value"].notna() & (df["mic_value"] > 0)].copy()
    df["mic_log2"]     = np.log2(df["mic_value"])
    df["is_censored"]  = df["mic_operator"].isin([">", "<", ">=", "<="])
    return df


# ---------------------------------------------------------------------------
# H1 — Is n too small to trust MIC_90?
# ---------------------------------------------------------------------------

def h1_sample_size(paeds: pd.DataFrame) -> None:
    yr = paeds.groupby("Year").agg(
        n       = ("mic_value", "count"),
        mic50   = ("mic_value", lambda x: x.quantile(0.50)),
        mic90   = ("mic_value", lambda x: x.quantile(0.90)),
        pct_cens= ("is_censored", "mean"),
    ).reset_index()
    yr["pct_cens"] *= 100

    print(f"\n{'Year':>4}  {'N':>5}  {'MIC_50':>7}  {'MIC_90':>7}  {'%Cens':>6}")
    print("-" * 42)
    for _, r in yr.iterrows():
        flag = "  <- LOW N" if r.n < 50 else ""
        print(f"{int(r.Year):4d}  {int(r.n):5,}  {r.mic50:7.3f}  {r.mic90:7.3f}  {r.pct_cens:6.1f}%{flag}")

    fig, ax1 = plt.subplots()
    ax1.plot(yr["Year"], yr["mic90"], "o-", color="#e377c2", lw=2, ms=5, label="MIC_90 (paeds)")
    ax1.plot(yr["Year"], yr["mic50"], "s--", color="#aec7e8", lw=1.5, ms=4, label="MIC_50 (paeds)")
    ax1.axhline(EUCAST_R, color="black", linestyle=":", lw=1.5, label="EUCAST R")
    ax1.set_yscale("log", base=2)
    ax1.set_ylabel("MIC (mg/L)", fontsize=11)
    ax1.set_xlabel("Year", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title("H1: Paediatric MIC_90 with sample size — low-n years flagged",
                  fontsize=12, fontweight="bold")
    ax2 = ax1.twinx()
    ax2.bar(yr["Year"], yr["n"], alpha=0.15, color="grey", width=0.7)
    ax2.axhline(50, color="grey", linestyle="--", lw=1, alpha=0.6, label="n=50 threshold")
    ax2.set_ylabel("N isolates per year", fontsize=10, color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")
    plt.tight_layout()
    fig.savefig(REPORTS / "paeds_h1_sample_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> paeds_h1_sample_size.png")


# ---------------------------------------------------------------------------
# H2 — Geographic bias
# ---------------------------------------------------------------------------

def h2_geography(paeds: pd.DataFrame) -> None:
    early = paeds[paeds["Year"].between(2004, 2016)]
    late  = paeds[paeds["Year"].between(2017, 2024)]

    early_ctry = early["Country"].value_counts().head(20)
    late_ctry  = late["Country"].value_counts().head(20)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].barh(early_ctry.index[::-1], early_ctry.values[::-1], color="#aec7e8")
    axes[0].set_title("Paediatric by country\n2004-2016 (flat MIC_90)", fontweight="bold")
    axes[0].set_xlabel("N isolates")

    axes[1].barh(late_ctry.index[::-1], late_ctry.values[::-1], color="#d62728", alpha=0.8)
    axes[1].set_title("Paediatric by country\n2017-2024 (jump period)", fontweight="bold")
    axes[1].set_xlabel("N isolates")

    plt.suptitle("H2: Geographic composition — before vs after the 2019 jump",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(REPORTS / "paeds_h2_geography.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> paeds_h2_geography.png")

    new = set(late["Country"].unique()) - set(early["Country"].unique())
    print(f"\n  Countries appearing only post-2017 ({len(new)}): {sorted(new)}")

    # MIC_90 by country for early period
    print("\n  MIC_90 by country (2004-2016, top contributors):")
    ctry_mic = (
        early.groupby("Country")
        .agg(n=("mic_value", "count"), mic90=("mic_value", lambda x: x.quantile(0.90)))
        .query("n >= 10")
        .sort_values("n", ascending=False)
        .head(15)
    )
    for _, r in ctry_mic.iterrows():
        print(f"    {r.name:<30} n={int(r.n):>4}  MIC_90={r.mic90:.3f}")


# ---------------------------------------------------------------------------
# H3 — Is 0.125 the censoring floor?
# ---------------------------------------------------------------------------

def h3_censoring_floor(paeds: pd.DataFrame) -> None:
    flat = paeds[paeds["Year"].between(2007, 2016)]

    print(f"\n  Raw value distribution (paediatric 2007-2016, top 15):")
    print(flat["Meropenem"].value_counts().head(15).to_string())
    print(f"\n  MIC_90         = {flat['mic_value'].quantile(0.90):.4f} mg/L")
    print(f"  % censored     = {flat['is_censored'].mean()*100:.1f}%")
    print(f"  % at or below 0.125 = {(flat['mic_value'] <= 0.125).mean()*100:.1f}%")
    print(f"  % at 0.03125 (floor) = {(flat['mic_value'] == 0.03125).mean()*100:.1f}%")

    fig, ax = plt.subplots()
    ax.hist(flat["mic_log2"], bins=30, color="#aec7e8", edgecolor="white", lw=0.5)
    ax.axvline(np.log2(EUCAST_R), color="black", linestyle="--", lw=1.5, label="R breakpoint")
    ax.axvline(np.log2(0.125), color="orange", linestyle="--", lw=1.5, label="0.125 mg/L")
    ax.axvline(np.log2(0.03125), color="red", linestyle="--", lw=1.5, label="0.03125 (floor)")
    ax.set_xlabel("log2(MIC)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("H3: Paediatric MIC distribution 2007-2016\n(are all values at the panel floor?)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(REPORTS / "paeds_h3_floor.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> paeds_h3_floor.png")


# ---------------------------------------------------------------------------
# H4 — Are paediatric isolates genuinely more susceptible?
# ---------------------------------------------------------------------------

def h4_biology(df: pd.DataFrame, paeds: pd.DataFrame) -> None:
    df = df.copy()
    df["age_simple"] = "Other"
    df.loc[df["Age Group"] == "0 - 17",                          "age_simple"] = "Paediatric"
    df.loc[df["Age Group"].isin(["18 - 30", "31 - 60", "61+"]),  "age_simple"] = "Adult/Elderly"

    flat_years = df[df["Year"].between(2007, 2016)]
    shared = (
        set(flat_years[flat_years["age_simple"] == "Paediatric"]["Country"].unique())
        & set(flat_years[flat_years["age_simple"] == "Adult/Elderly"]["Country"].unique())
    )
    print(f"\n  Countries with both paediatric + adult data in 2007-2016: {len(shared)}")

    controlled = flat_years[flat_years["Country"].isin(shared)]
    comp = controlled.groupby("age_simple").agg(
        n             = ("mic_value", "count"),
        mic90         = ("mic_value", lambda x: x.quantile(0.90)),
        pct_resistant = ("mic_value", lambda x: (x >= EUCAST_R).mean()),
        pct_censored  = ("is_censored", "mean"),
    )
    comp["pct_resistant"] *= 100
    comp["pct_censored"]  *= 100
    print("\n  Controlled comparison (same countries, 2007-2016):")
    print(comp.to_string(float_format="{:.3f}".format))

    # Military proxy
    WOUND_SOURCES = {"Wound", "Abscess", "Skin and Soft Tissue"}
    df["military_proxy"] = (
        df["Source"].isin(WOUND_SOURCES)
        & (df["Gender"] == "Male")
        & (df["Age Group"].isin(["18 - 30", "31 - 60"]))
    )
    mil = df[df["military_proxy"]]
    print(f"\n  Military proxy: {len(mil):,} isolates ({len(mil)/len(df)*100:.2f}% of dataset)")
    print(f"\n  Top 20 contributing countries:")
    ctry = mil["Country"].value_counts().head(20)
    for country, n in ctry.items():
        print(f"    {country:<30} {n:>5,}  ({n/len(mil)*100:.1f}%)")

    print(f"\n  N per year (threshold=50):")
    mil_yr = mil.groupby("Year").size().reset_index(name="n")
    for _, r in mil_yr.iterrows():
        flag = "" if r.n >= 50 else "  <- unreliable"
        print(f"    {int(r.Year)}: n={int(r.n)}{flag}")

    fig, ax = plt.subplots(figsize=(14, 6))
    ctry_all = mil["Country"].value_counts().head(30).reset_index()
    ctry_all.columns = ["Country", "n"]
    ax.barh(ctry_all["Country"][::-1], ctry_all["n"][::-1], color="#d62728", alpha=0.8)
    ax.set_xlabel("Number of isolates", fontsize=11)
    ax.set_title("Military Proxy — Isolates by Country (top 30, all years)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(REPORTS / "military_proxy_countries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> military_proxy_countries.png")


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "raw"
    n = 5

    print(f"[1/{n}] Load data")
    df    = load(data_dir)
    paeds = df[df["Age Group"] == "0 - 17"].copy()
    print(f"  Total rows:       {len(df):,}")
    print(f"  Paediatric rows:  {len(paeds):,}  ({len(paeds)/len(df)*100:.1f}%)")

    print(f"\n[2/{n}] H1 — Sample size per year")
    h1_sample_size(paeds)

    print(f"\n[3/{n}] H2 — Geographic bias")
    h2_geography(paeds)

    print(f"\n[4/{n}] H3 — Censoring floor")
    h3_censoring_floor(paeds)

    print(f"\n[5/{n}] H4 — Biology + military proxy countries")
    h4_biology(df, paeds)

    print(f"\nDone. Charts saved to: {REPORTS}")


if __name__ == "__main__":
    main()
