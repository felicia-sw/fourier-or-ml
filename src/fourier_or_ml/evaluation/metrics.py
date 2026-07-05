"""Forecast accuracy metrics: MASE (primary), RMSE, sMAPE, pinball loss."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = (np.abs(yt) + np.abs(yp)) / 2
    mask = denom > 0
    return float(np.mean(np.abs(yt - yp)[mask] / denom[mask]) * 100)


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, m: int = 24) -> float:
    """MASE scaled by the in-sample seasonal-naive MAE (period m)."""
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    tr = np.asarray(y_train, float)
    scale = np.mean(np.abs(tr[m:] - tr[:-m]))
    if scale <= 0:
        return np.nan
    return float(np.mean(np.abs(yt - yp)) / scale)


def pinball(y_true: np.ndarray, y_pred_q: np.ndarray, q: float) -> float:
    yt, yq = np.asarray(y_true, float), np.asarray(y_pred_q, float)
    diff = yt - yq
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def score_all(y_true, y_pred, y_train, m: int = 24) -> dict[str, float]:
    return {
        "mase": mase(y_true, y_pred, y_train, m=m),
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
