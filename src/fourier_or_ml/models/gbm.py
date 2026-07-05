"""LightGBM forecasters (M2, M3) and the DHR-residual hybrid (M4).

Feature encodings under the matched-information-set protocol:
- M2 lgbm:     raw calendar integers + trend (prefixes cal_, trend) — no Fourier
- M3 hybrid_a: calendar + Fourier + trend (prefixes=None -> all columns)
Scenario S1 = deterministic only; S2 adds lags {1,24,168} + shifted rolling
means, mirroring the information carried by DHR's ARMA errors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import lightgbm as lgb

from .base import Forecaster
from .dhr import DynamicHarmonicRegression
from ..features.build import LGBM_PREFIXES, select_columns
from ..features.calendar import lag_features, DEFAULT_LAGS, DEFAULT_ROLLS

_DEFAULT_PARAMS = dict(
    objective="regression",
    num_leaves=63,
    learning_rate=0.05,
    n_estimators=800,
    min_child_samples=50,
    verbosity=-1,
)


class LGBMForecaster(Forecaster):
    def __init__(
        self,
        use_lags: bool = False,
        params: dict | None = None,
        prefixes: tuple[str, ...] | None = LGBM_PREFIXES,
        name: str | None = None,
    ):
        self.use_lags = use_lags
        self.params = {**_DEFAULT_PARAMS, **(params or {})}
        self.prefixes = prefixes
        self.name = name or ("lgbm_s2" if use_lags else "lgbm_s1")

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> "LGBMForecaster":
        if X is None:
            raise ValueError("LGBMForecaster requires deterministic features")
        Xs = select_columns(X, self.prefixes)
        self._det_cols = list(Xs.columns)
        feats = Xs.copy()
        if self.use_lags:
            feats = pd.concat([feats, lag_features(y)], axis=1)
        mask = feats.notna().all(axis=1)
        self._model = lgb.LGBMRegressor(**self.params).fit(feats[mask], y[mask])
        self._feat_cols = list(feats.columns)
        self._y_hist = y.copy()
        self._last_index = y.index[-1]
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        if X_future is None or len(X_future) < horizon:
            raise ValueError("X_future with deterministic features for the horizon is required")
        idx = pd.date_range(self._last_index, periods=horizon + 1, freq="h")[1:]
        Xf = X_future.iloc[:horizon][self._det_cols].copy()
        Xf.index = idx

        if not self.use_lags:
            preds = self._model.predict(Xf)
            return pd.Series(preds, index=idx, name=self.name)

        # recursive multi-step forecast with lag features
        hist = self._y_hist.copy()
        preds = []
        for ts in idx:
            row = Xf.loc[[ts]].copy()
            for l in DEFAULT_LAGS:
                row[f"lag_{l}"] = hist.iloc[-l]
            for w in DEFAULT_ROLLS:
                row[f"rollmean_{w}"] = hist.iloc[-w:].mean()
            yhat = float(self._model.predict(row[self._feat_cols])[0])
            preds.append(yhat)
            hist.loc[ts] = yhat
        return pd.Series(preds, index=idx, name=self.name)

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """TreeSHAP attributions for the interpretability analysis."""
        import shap
        return shap.TreeExplainer(self._model).shap_values(X[self._feat_cols])


class DHRResidualHybrid(Forecaster):
    """Hybrid B (M4): DHR mean forecast + LightGBM trained on DHR residuals."""

    def __init__(self, use_lags: bool = False):
        self.name = "hybrid_b"
        self._dhr = DynamicHarmonicRegression(error_order=None)  # residuals go to LGBM instead
        self._lgbm = LGBMForecaster(use_lags=use_lags, name="hybrid_b_resid")

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> "DHRResidualHybrid":
        self._dhr.fit(y, X)
        fitted = self._dhr._ols.fittedvalues
        resid = pd.Series(y.to_numpy() - fitted, index=y.index)
        self._lgbm.fit(resid, X)
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        base = self._dhr.predict(horizon, X_future)
        corr = self._lgbm.predict(horizon, X_future)
        out = base + corr.to_numpy()
        out.name = self.name
        return out
