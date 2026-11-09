"""Rebuild the historical stump-model feature table from raw competition data.

The original selected v19 ensemble used a compact legacy feature set for its
shallow boosted-stump component. This script creates that set locally so the
large derived table does not need to be stored in Git.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEGACY_DIR = ROOT / "reproducibility_runs" / "legacy_features"

# `feat2.add` contains deterministic clinical ratios and missingness features.
from feat2 import add


features = pd.read_parquet(LEGACY_DIR / "features.parquet")
features = add(features).drop(
    columns=["admit_year", "days_since_start", "admit_month"], errors="ignore"
)
output = LEGACY_DIR / "final_dataset.csv.gz"
features.to_csv(output, index=False, compression="gzip")
print(f"Wrote {output} with {features.shape[0]:,} rows and {features.shape[1]:,} columns.")
