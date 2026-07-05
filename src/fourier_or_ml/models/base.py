"""Forecaster interface shared by all models.

All models implement fit(y, X) / predict(horizon, X_future) where X carries
the deterministic features (Fourier, calendar, trend, holiday) and — in
Scenario S2 — lag features are handled internally by each model so that all
models see the same information set.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Forecaster(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> "Forecaster":
        ...

    @abstractmethod
    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        """Forecast the next `horizon` steps after the end of the training data."""
        ...
