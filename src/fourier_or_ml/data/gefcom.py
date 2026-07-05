"""GEFCom2014 load-track loader.

The GEFCom2014 data are distributed as a zip accompanying Hong et al. (2016),
International Journal of Forecasting 32(3) — see the paper's appendix link
(Dropbox) or request from the organizers. Place the extracted load-track task
files under data/raw/gefcom2014/ (files named like L1-train.csv with columns
ZONEID, TIMESTAMP, LOAD, w1..w25 temperature stations).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_gefcom_load(raw_dir: str | Path, task_file: str = "L1-train.csv") -> pd.DataFrame:
    """Load one GEFCom2014-L task file as an hourly frame with load + mean temp.

    Returns a DataFrame indexed by timestamp with columns ['y', 'temp'].
    """
    path = Path(raw_dir) / "gefcom2014" / task_file
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the GEFCom2014 load track per Hong et al. "
            "(2016) and extract it to data/raw/gefcom2014/."
        )
    df = pd.read_csv(path)
    ts_col = "TIMESTAMP" if "TIMESTAMP" in df.columns else df.columns[1]
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.set_index(ts_col).sort_index()
    temp_cols = [c for c in df.columns if c.lower().startswith("w")]
    out = pd.DataFrame({"y": df["LOAD"]})
    if temp_cols:
        out["temp"] = df[temp_cols].mean(axis=1)
    return out.dropna(subset=["y"])
