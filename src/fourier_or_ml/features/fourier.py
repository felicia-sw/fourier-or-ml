"""Fourier (harmonic) term construction for multi-seasonal series.

Note on notation: K is the number of sine/cosine harmonic pairs per seasonal
period m (K <= m/2), NOT the period itself. Periods for hourly load are
m = 24 (daily), 168 (weekly), 8766 (annual).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fourier_terms(
    t: np.ndarray,
    period: float,
    k: int,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Build K sine/cosine pairs for one seasonal period.

    Parameters
    ----------
    t : integer time index (0, 1, 2, ...), one step per observation.
    period : seasonal period m in observations (e.g. 24 for daily on hourly data).
    k : number of harmonic pairs; must satisfy 1 <= k <= period / 2.
    prefix : column-name prefix; defaults to ``m<period>``.
    """
    if not 1 <= k <= period / 2:
        raise ValueError(f"k={k} must be in [1, m/2]={period / 2} for period m={period}")
    prefix = prefix or f"m{int(period)}"
    cols = {}
    for j in range(1, k + 1):
        angle = 2.0 * np.pi * j * t / period
        cols[f"{prefix}_sin{j}"] = np.sin(angle)
        cols[f"{prefix}_cos{j}"] = np.cos(angle)
    return pd.DataFrame(cols, index=pd.RangeIndex(len(t)) if not hasattr(t, "index") else None)


def multi_seasonal_fourier(
    n: int,
    orders: dict[float, int],
    t0: int = 0,
) -> pd.DataFrame:
    """Fourier design matrix for several seasonal periods.

    Parameters
    ----------
    n : number of observations.
    orders : mapping period -> K, e.g. {24: 8, 168: 4, 8766: 3}.
    t0 : time index of the first observation (lets train/test share phase).
    """
    t = np.arange(t0, t0 + n)
    parts = [fourier_terms(t, period, k) for period, k in orders.items()]
    out = pd.concat(parts, axis=1)
    out.index = pd.RangeIndex(n)
    return out
