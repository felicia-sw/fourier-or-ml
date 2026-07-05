"""Assemble the full deterministic design matrix shared by all models.

Matched-information-set protocol: all models receive the SAME underlying
calendar information, but each model family selects its own *encoding* of it
via column prefixes:

- DHR (M1):        m*_sin/cos (Fourier), cal_is_holiday, cal_is_weekend, trend
- LightGBM (M2):   cal_* raw calendar integers, trend  (no Fourier)
- Hybrid A (M3):   everything (calendar + Fourier + trend)

The information content is matched; the encoding is a property of the model
family, which is exactly the design choice under study.
"""
from __future__ import annotations

import pandas as pd

from .fourier import multi_seasonal_fourier
from .calendar import calendar_features, trend_index

DEFAULT_ORDERS = {24: 8, 168: 4, 8766: 3}

# canonical prefix groups
FOURIER_PREFIXES = ("m",)
CALENDAR_PREFIXES = ("cal_",)
TREND_PREFIXES = ("trend",)
DHR_PREFIXES = FOURIER_PREFIXES + ("cal_is_holiday", "cal_is_weekend") + TREND_PREFIXES
LGBM_PREFIXES = CALENDAR_PREFIXES + TREND_PREFIXES
ALL_PREFIXES = None  # None = every column


def deterministic_features(
    index: pd.DatetimeIndex,
    t0: int,
    fourier_orders: dict[float, int] | None = None,
    include_trend: bool = True,
    country: str = "US",
) -> pd.DataFrame:
    """Build the full Fourier + calendar + trend matrix for a window.

    `t0` is the global integer offset of the first timestamp — it keeps
    train/test windows on the same Fourier phase and trend line (never reset
    it to 0 per window).
    """
    orders = fourier_orders or DEFAULT_ORDERS
    n = len(index)
    F = multi_seasonal_fourier(n, orders, t0=t0)
    F.index = index
    C = calendar_features(index, country=country).add_prefix("cal_")
    parts = [F, C]
    if include_trend:
        tr = trend_index(n, t0=t0)
        tr.index = index
        parts.append(tr)
    return pd.concat(parts, axis=1)


def select_columns(X: pd.DataFrame, prefixes: tuple[str, ...] | None) -> pd.DataFrame:
    """Select the columns a model family is allowed to see."""
    if prefixes is None:
        return X
    cols = [c for c in X.columns if any(c.startswith(p) for p in prefixes)]
    return X[cols]
