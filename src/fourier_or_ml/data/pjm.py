"""PJM hourly load loader (Kaggle: robikscube/hourly-energy-consumption).

Treats the dataset as a 12-zone panel. Handles the known quirks:
DST duplicates/gaps, missing hours, and the differing spans per zone.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ZONES = ("AEP", "COMED", "DAYTON", "DEOK", "DOM", "DUQ", "EKPC",
         "FE", "NI", "PJME", "PJMW", "PJM_Load")


def load_zone(raw_dir: str | Path, zone: str) -> pd.Series:
    """Load one zone as a clean hourly series (UTC-naive local time).

    - duplicate timestamps (DST fall-back) are averaged
    - missing hours are reindexed and linearly interpolated (short gaps only)
    """
    path = Path(raw_dir) / f"{zone}_hourly.csv"
    df = pd.read_csv(path, parse_dates=["Datetime"])
    col = [c for c in df.columns if c.endswith("_MW")][0]
    s = df.set_index("Datetime")[col].sort_index()
    s = s.groupby(level=0).mean()  # DST duplicates
    full = pd.date_range(s.index.min(), s.index.max(), freq="h")
    s = s.reindex(full).interpolate(limit=6)  # only bridge short gaps
    s.name = zone
    return s


def load_panel(raw_dir: str | Path, zones: tuple[str, ...] = ZONES) -> dict[str, pd.Series]:
    out = {}
    for z in zones:
        try:
            out[z] = load_zone(raw_dir, z)
        except FileNotFoundError:
            pass
    if not out:
        raise FileNotFoundError(
            f"No PJM csv files found in {raw_dir}. Download with: "
            "kaggle datasets download -d robikscube/hourly-energy-consumption -p data/raw --unzip"
        )
    return out
