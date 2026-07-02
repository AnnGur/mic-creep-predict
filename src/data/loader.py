"""ATLAS data loader for MIC creep prediction."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

ATLAS_FILENAME    = "atlas_vivli_2004_2024.csv"
ENTASIS_FILENAME  = "IST-Entasis_Acinetobacter-Surveillance_2016-2021.xlsx"
SIDERO_FILENAME   = "Updated_Shionogi Five year SIDERO-WT Surveillance data(without strain number)_Vivli_220409.xlsx"
VENATORX_FILENAME = "Venatorx surveillance data_2024_06_06.xlsx"

METADATA_COLS = [
    "Isolate Id", "Species", "Country", "Gender", "Age Group",
    "Source", "Year", "Speciality",
]
GENE_COLS = ["KPC", "NDM", "OXA", "VIM", "IMP", "GES"]

# Standard meropenem doubling-dilution series used to detect censored encoding
_MIC_DILUTIONS = [0.03, 0.06, 0.12, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]


def _nearest_dilution(val: float) -> float:
    return min(_MIC_DILUTIONS, key=lambda d: abs(d - val))


def _parse_entasis_mic(val) -> tuple:
    """Parse Entasis numeric MIC encoding.

    Convention observed in the file:
      x.0001  -> ">x"  (censored high) -> impute as x*2 (next doubling)
      x-0.0001 (e.g. 0.0299 = 0.03-0.0001) -> "<=x" (censored low) -> impute as x/2
      exact dilution value -> not censored
    """
    if pd.isna(val):
        return None, None, None
    val = float(val)
    nearest = _nearest_dilution(val)
    diff = val - nearest
    if abs(diff - 0.0001) < 0.00005:        # x.0001 pattern — censored high
        imputed = nearest * 2
        return imputed, np.log2(imputed), True
    elif abs(diff + 0.0001) < 0.00005:      # x - 0.0001 pattern — censored low
        imputed = nearest / 2
        return imputed, np.log2(imputed), True
    else:
        if val <= 0:
            return None, None, None
        return val, np.log2(val), False


class ATLASLoader:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def load(self, species: str, antibiotic: str = "Meropenem") -> pd.DataFrame:
        path = self.data_dir / ATLAS_FILENAME

        # Read only the columns we need to keep memory low on a 1M-row file
        all_cols = pd.read_csv(path, nrows=0).columns.tolist()
        gene_present = [g for g in GENE_COLS if g in all_cols]
        abx_col = antibiotic if antibiotic in all_cols else None
        if abx_col is None:
            raise ValueError(f"Antibiotic column '{antibiotic}' not found in ATLAS data")

        use_cols = METADATA_COLS + [abx_col] + gene_present
        use_cols = [c for c in use_cols if c in all_cols]

        df = pd.read_csv(path, usecols=use_cols, low_memory=False)
        df = df[df["Species"] == species].copy()
        df = df[df[antibiotic].notna()].copy()
        df = df.reset_index(drop=True)
        return df


def load_entasis_abaumannii(
    data_dir: Path,
    train_only: bool = True,
    train_end: int = 2018,
) -> pd.DataFrame:
    """Load Entasis A. baumannii surveillance data and return an ATLAS-compatible DataFrame.

    The Entasis file encodes censored MICs numerically (x.0001 = >x, x-0.0001 = <=x).
    Gene flags (KPC, NDM, etc.) are absent — all set to False.

    train_only: if True (default), return only rows with Year <= train_end to avoid
                contaminating the ATLAS test window (2019-2022).
    """
    path = Path(data_dir) / ENTASIS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Entasis file not found: {path}")

    raw = pd.read_excel(path)

    # --- MIC parsing ---
    parsed = raw["Meropenem"].apply(_parse_entasis_mic)
    raw["mic_value"]   = parsed.apply(lambda t: t[0])
    raw["mic_log2"]    = parsed.apply(lambda t: t[1])
    raw["is_censored"] = parsed.apply(lambda t: bool(t[2]) if t[2] is not None else False)
    raw = raw[raw["mic_value"].notna() & (raw["mic_value"] > 0)].copy()
    raw["is_resistant"] = raw["mic_value"] >= 8

    # --- Column renames to ATLAS schema ---
    raw = raw.rename(columns={"YearCollected": "Year", "BodyLocation": "Source",
                               "FacilityName": "Speciality"})

    # --- Gender: M/F -> Male/Female ---
    raw["Gender"] = raw["Gender"].map({"M": "Male", "F": "Female"})

    # --- Age: numeric -> ATLAS categorical bins ---
    def _bin_age(age):
        if pd.isna(age):
            return "18 - 30"
        a = float(age)
        if a <= 17:   return "0 - 17"
        if a <= 30:   return "18 - 30"
        if a <= 60:   return "31 - 60"
        return "61+"

    raw["Age Group"] = raw["Age"].apply(_bin_age)
    raw["age_group_broad"] = raw["Age Group"].map({
        "0 - 17": "Paediatric (0-17)",
        "18 - 30": "Adult (18-60)",
        "31 - 60": "Adult (18-60)",
        "61+": "Elderly (61+)",
    }).fillna("Adult (18-60)")

    # --- Military proxy (wound + male + 18-60) ---
    wound_kw = {"wound", "abscess", "skin"}
    raw["military_proxy"] = (
        raw["Source"].str.lower().str.contains("|".join(wound_kw), na=False)
        & (raw["Gender"] == "Male")
        & (raw["Age Group"].isin(["18 - 30", "31 - 60"]))
    )

    # --- Carbapenemase genes not collected in Entasis ---
    for gene in GENE_COLS:
        raw[f"{gene}_pos"] = False

    # --- Dataset tag (for debugging) ---
    raw["dataset"] = "entasis"

    if train_only:
        raw = raw[raw["Year"] <= train_end].copy()

    raw = raw.reset_index(drop=True)
    print(f"  Entasis A. baumannii loaded: {len(raw):,} rows "
          f"({raw['Year'].min()}-{raw['Year'].max()}, "
          f"{raw['Country'].nunique()} countries, "
          f"{raw['is_resistant'].mean()*100:.0f}% resistant)")
    return raw


def _shared_supplementary_fields(raw: pd.DataFrame, dataset_tag: str) -> pd.DataFrame:
    """Apply common post-processing for supplementary datasets after MIC parsing."""
    raw["is_resistant"] = raw["mic_value"] >= 8
    raw["military_proxy"] = False
    for gene in GENE_COLS:
        raw[f"{gene}_pos"] = False
    raw["dataset"] = dataset_tag
    return raw


def load_sidero_wt(
    data_dir: Path,
    species: str,
    train_only: bool = True,
    train_end: int = 2018,
) -> pd.DataFrame:
    """Load Shionogi SIDERO-WT surveillance data for the given species.

    species: exact organism name, e.g. 'Klebsiella pneumoniae'
    SIDERO-WT does not collect Age or Gender — demographic features default to
    the adult reference category (gender_male=0, age_paediatric=0, age_elderly=0).
    Panel ceiling: 64 mg/L (cefiderocol study, meropenem goes to >64).
    """
    path = Path(data_dir) / SIDERO_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"SIDERO-WT file not found: {path}")

    raw = pd.read_excel(path)
    raw = raw[raw["Organism Name"].str.contains(species, case=False, na=False)].copy()
    raw = raw[raw["Meropenem"].notna()].copy()

    parsed = raw["Meropenem"].apply(_parse_entasis_mic)
    raw["mic_value"]   = parsed.apply(lambda t: t[0])
    raw["mic_log2"]    = parsed.apply(lambda t: t[1])
    raw["is_censored"] = parsed.apply(lambda t: bool(t[2]) if t[2] is not None else False)
    raw = raw[raw["mic_value"].notna() & (raw["mic_value"] > 0)].copy()

    raw = raw.rename(columns={"Year Collected": "Year", "Body Location": "Source"})

    # No Age/Gender in SIDERO-WT — set to adult reference so OHE columns are 0
    raw["Gender"]         = "Unknown"
    raw["Age Group"]      = "31 - 60"
    raw["age_group_broad"] = "Adult (18-60)"
    raw["Speciality"]     = None   # -> ward_general (reference, no dummy fired)

    raw = _shared_supplementary_fields(raw, "sidero_wt")

    if train_only:
        raw = raw[raw["Year"] <= train_end].copy()

    raw = raw.reset_index(drop=True)
    print(f"  SIDERO-WT {species}: {len(raw):,} rows "
          f"({raw['Year'].min()}-{raw['Year'].max()}, "
          f"{raw['Country'].nunique()} countries, "
          f"{raw['is_resistant'].mean()*100:.0f}% resistant)")
    return raw


def load_venatorx_kpneumoniae(
    data_dir: Path,
    train_only: bool = True,
    train_end: int = 2018,
) -> pd.DataFrame:
    """Load Venatorx K. pneumoniae surveillance data (no A. baumannii in this dataset).

    Has Age, Gender, Facility, BodySite — full demographic columns like ATLAS.
    Panel ceiling: 64 mg/L. MEM_MIC column name for meropenem.
    """
    path = Path(data_dir) / VENATORX_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Venatorx file not found: {path}")

    raw = pd.read_excel(path)
    raw = raw[raw["Organism"].str.contains("Klebsiella pneumoniae", case=False, na=False)].copy()

    # MEM_MIC uses "-", "NG" (not given), "NT" (not tested) for missing
    _non_numeric = {"-", "NG", "NA", "NT", "nd", ""}
    raw = raw[~raw["MEM_MIC"].astype(str).str.strip().isin(_non_numeric)].copy()
    raw["MEM_MIC"] = pd.to_numeric(raw["MEM_MIC"], errors="coerce")
    raw = raw[raw["MEM_MIC"].notna()].copy()

    parsed = raw["MEM_MIC"].apply(_parse_entasis_mic)
    raw["mic_value"]   = parsed.apply(lambda t: t[0])
    raw["mic_log2"]    = parsed.apply(lambda t: t[1])
    raw["is_censored"] = parsed.apply(lambda t: bool(t[2]) if t[2] is not None else False)
    raw = raw[raw["mic_value"].notna() & (raw["mic_value"] > 0)].copy()

    raw = raw.rename(columns={"BodySite": "Source", "Facility": "Speciality"})

    raw["Gender"] = raw["Gender"].map({"M": "Male", "F": "Female"})

    def _bin_age(age):
        try:
            a = float(age)
        except (TypeError, ValueError):
            return "31 - 60"  # non-numeric sentinel ("NG" etc.) -> adult reference
        if a <= 17:   return "0 - 17"
        if a <= 30:   return "18 - 30"
        if a <= 60:   return "31 - 60"
        return "61+"

    raw["Age Group"] = raw["Age"].apply(_bin_age)
    raw["age_group_broad"] = raw["Age Group"].map({
        "0 - 17": "Paediatric (0-17)",
        "18 - 30": "Adult (18-60)",
        "31 - 60": "Adult (18-60)",
        "61+": "Elderly (61+)",
    }).fillna("Adult (18-60)")

    wound_kw = {"wound", "abscess", "skin"}
    raw["military_proxy"] = (
        raw["Source"].str.lower().str.contains("|".join(wound_kw), na=False)
        & (raw["Gender"] == "Male")
        & (raw["Age Group"].isin(["18 - 30", "31 - 60"]))
    )

    raw["is_resistant"] = raw["mic_value"] >= 8
    for gene in GENE_COLS:
        raw[f"{gene}_pos"] = False
    raw["dataset"] = "venatorx"

    if train_only:
        raw = raw[raw["Year"] <= train_end].copy()

    raw = raw.reset_index(drop=True)
    print(f"  Venatorx K. pneumoniae: {len(raw):,} rows "
          f"({raw['Year'].min()}-{raw['Year'].max()}, "
          f"{raw['Country'].nunique()} countries, "
          f"{raw['is_resistant'].mean()*100:.0f}% resistant)")
    return raw
