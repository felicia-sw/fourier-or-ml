"""Seasonal naive baseline (M0)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Forecaster


class SeasonalNaive(Forecaster):
    """y_hat(t+h) = y(t + h - m*ceil(h/m)): repeat the last full season."""

    def __init__(self, period: int = 24):
        self.period = period
        self.name = f"snaive_{period}"

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> "SeasonalNaive":
        self._tail = y.dropna().to_numpy()[-self.period:]
        self._last_index = y.index[-1]
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        reps = int(np.ceil(horizon / self.period))
        vals = np.tile(self._tail, reps)[:horizon]
        idx = pd.date_range(self._last_index, periods=horizon + 1, freq="h")[1:]
        return pd.Series(vals, index=idx, name=self.name)
