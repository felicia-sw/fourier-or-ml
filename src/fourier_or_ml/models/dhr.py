"""Dynamic Harmonic Regression (M1).

OLS on Fourier terms + trend + holiday dummies, with an ARMA model on the
residuals for the dynamic part (Hyndman & Athanasopoulos 2021, ch. 10).
Setting error_order=None gives the plain OLS harmonic regression ablation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.arima.model import ARIMA

from .base import Forecaster
from ..features.build import DHR_PREFIXES, select_columns


class DynamicHarmonicRegression(Forecaster):
    def __init__(
        self,
        error_order: tuple[int, int, int] | None = (2, 0, 1),
        prefixes: tuple[str, ...] | None = DHR_PREFIXES,
    ):
        self.error_order = error_order
        self.prefixes = prefixes
        self.name = "dhr" if error_order else "harmonic_ols"

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> "DynamicHarmonicRegression":
        if X is None:
            raise ValueError("DHR requires deterministic features (Fourier/trend/holiday)")
        Xs = select_columns(X, self.prefixes)
        self._cols = list(Xs.columns)
        Xc = add_constant(Xs.to_numpy(dtype=float), has_constant="add")
        self._ols = OLS(y.to_numpy(dtype=float), Xc).fit()
        self._last_index = y.index[-1]
        resid = pd.Series(self._ols.resid)
        self._arma = None
        if self.error_order is not None:
            try:
                self._arma = ARIMA(resid.to_numpy(), order=self.error_order).fit()
            except Exception:
                self._arma = None  # fall back to OLS-only if ARMA fails to converge
        return self

    @property
    def harmonic_coefficients(self) -> pd.Series:
        """Named coefficients — used by the interpretability analysis
        (compared against SHAP-implied profiles)."""
        return pd.Series(self._ols.params[1:], index=self._cols)

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        if X_future is None or len(X_future) < horizon:
            raise ValueError("X_future with deterministic features for the horizon is required")
        Xf = add_constant(
            X_future.iloc[:horizon][self._cols].to_numpy(dtype=float), has_constant="add"
        )
        mean = self._ols.predict(Xf)
        if self._arma is not None:
            mean = mean + np.asarray(self._arma.forecast(horizon))
        idx = pd.date_range(self._last_index, periods=horizon + 1, freq="h")[1:]
        return pd.Series(mean, index=idx, name=self.name)
