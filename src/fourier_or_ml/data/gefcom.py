"""GEFCom2014 load-track loader.

Data: Hong et al. (2016), IJF 32(3) — GEFCom2014-L_V2.zip from the official
Dropbox link. Expected layout: data/raw/gefcom2014/L{n}-train.csv with columns
ZONEID, TIMESTAMP, LOAD, w1..w25 (temperature stations).

Timestamp quirk: GEFCom uses unpadded 'mdyyyy h:mm' strings (e.g. '112001
1:00', '4142003 7:00'), which are ambiguous to parse directly. Because the
records are strictly hourly and continuous, the index is generated as an
hourly range anchored at the (unambiguous) first timestamp instead.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def _parse_anchor(ts: str) -> pd.Timestamp:
    """Parse one 'mdyyyy h:mm' timestamp (used only for the first row)."""
    date_str, time_str = ts.strip().split()
    year = int(date_str[-4:])
    md = date_str[:-4]
    if len(md) == 2:
        month, day = int(md[0]), int(md[1])
    elif len(md) == 4:
        month, day = int(md[:2]), int(md[2:])
    else:  # len 3: try m/dd then mm/d
        m1, d1 = int(md[0]), int(md[1:])
        if 1 <= d1 <= 31:
            month, day = m1, d1
        else:
            month, day = int(md[:2]), int(md[2])
    hour, minute = map(int, time_str.split(":"))
    return pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)


def load_gefcom_load(raw_dir: str | Path, task: int = 1) -> pd.DataFrame:
    """Load one GEFCom2014-L task file.

    Returns a DataFrame indexed by hourly timestamps with columns
    ['y', 'temp'] (temp = mean of the 25 stations), rows with missing load
    dropped (the first years of each file are temperature-only history).
    """
    path = Path(raw_dir) / "gefcom2014" / f"L{task}-train.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download GEFCom2014-L_V2.zip (Hong et al. 2016) "
            "and copy the Task */L*-train.csv files into data/raw/gefcom2014/."
        )
    df = pd.read_csv(path)
    anchor = _parse_anchor(str(df["TIMESTAMP"].iloc[0]))
    df.index = pd.date_range(anchor, periods=len(df), freq="h")
    temp_cols = [c for c in df.columns if re.fullmatch(r"w\d+", c)]
    out = pd.DataFrame({"y": df["LOAD"]})
    if temp_cols:
        out["temp"] = df[temp_cols].mean(axis=1)
    return out.dropna(subset=["y"])
