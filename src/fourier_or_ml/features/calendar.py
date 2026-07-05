"""Calendar, holiday, trend, and lag features."""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import holidays as _holidays
except ImportError:  # pragma: no cover
    _holidays = None

DEFAULT_LAGS = (1, 24, 168)
DEFAULT_ROLLS = (24, 168)


def calendar_features(index: pd.DatetimeIndex, country: str = "US") -> pd.DataFrame:
    """Deterministic calendar features from a DatetimeIndex."""
    df = pd.DataFrame(index=index)
    df["hour"] = index.hour
    df["day_of_week"] = index.dayofweek
    df["month"] = index.month
    df["day_of_year"] = index.dayofyear
    df["is_weekend"] = (index.dayofweek >= 5).astype(int)
    if _holidays is not None:
        hol = _holidays.country_holidays(country, years=sorted(set(index.year)))
        df["is_holiday"] = np.fromiter((d in hol for d in index.date), dtype=int, count=len(index))
    else:
        df["is_holiday"] = 0
    return df


def trend_index(n: int, t0: int = 0, scale: float = 1e-4) -> pd.Series:
    """Linear trend index. Scaled so coefficients are well-conditioned.

    Note: tree models cannot extrapolate this — that is a *studied property*
    in this project (see detrending ablations), not an oversight.
    """
    return pd.Series(np.arange(t0, t0 + n) * scale, name="trend")


def lag_features(
    y: pd.Series,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    rolls: tuple[int, ...] = DEFAULT_ROLLS,
) -> pd.DataFrame:
    """Autoregressive features (Scenario S2 only). Uses only past values —
    rolling means are shifted by 1 so no current/future leakage."""
    df = pd.DataFrame(index=y.index)
    for l in lags:
        df[f"lag_{l}"] = y.shift(l)
    for w in rolls:
        df[f"rollmean_{w}"] = y.shift(1).rolling(w).mean()
    return df
