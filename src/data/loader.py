"""ATLAS data loader for MIC creep prediction."""

import pandas as pd
from pathlib import Path
from typing import Optional

ATLAS_FILENAME = "atlas_vivli_2004_2024.csv"

METADATA_COLS = [
    "Isolate Id", "Species", "Country", "Gender", "Age Group",
    "Source", "Year", "Speciality",
]
GENE_COLS = ["KPC", "NDM", "OXA", "VIM", "IMP", "GES"]


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
