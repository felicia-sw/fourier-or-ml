"""Assemble the deterministic design matrix shared by all models.

Crucial for the matched-information-set protocol: DHR and LightGBM receive
exactly this matrix in Scenario S1; in S2, lags are added inside each model
with the same lag set.
"""
from __future__ import annotations

import pandas as pd

from .fourier import multi_seasonal_fourier
from .calendar import calendar_features, trend_index

DEFAULT_ORDERS = {24: 8, 168: 4, 8766: 3}


def deterministic_features(
    index: pd.DatetimeIndex,
    t0: int,
    fourier_orders: dict[float, int] | None = None,
    include_trend: bool = True,
    country: str = "US",
) -> pd.DataFrame:
    """Build Fourier + calendar + trend features for a window.

    Parameters
    ----------
    index : timestamps of the window (train or future).
    t0 : global integer offset of the first timestamp (keeps train/test on
         the same Fourier phase and trend line — do not reset to 0 per window).
    """
    orders = fourier_orders or DEFAULT_ORDERS
    n = len(index)
    F = multi_seasonal_fourier(n, orders, t0=t0)
    F.index = index
    C = calendar_features(index, country=country)[["is_weekend", "is_holiday"]]
    parts = [F, C]
    if include_trend:
        tr = trend_index(n, t0=t0)
        tr.index = index
        parts.append(tr)
    return pd.concat(parts, axis=1)
