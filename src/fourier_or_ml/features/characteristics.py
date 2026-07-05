"""Time-series characteristic extraction for the meta-analysis.

Characteristics per evaluation window:
- STL-based seasonal strength per period (Wang, Smith & Hyndman style)
- trend strength
- spectral entropy
- remainder (noise) variance share
- lumpiness / stability
- anomaly density (share of |z| > 3 STL remainders)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import periodogram
from statsmodels.tsa.seasonal import MSTL


def _strength(component_var: float, resid_var: float) -> float:
    """Seasonal/trend strength: max(0, 1 - Var(remainder)/Var(remainder+component))."""
    denom = component_var + resid_var
    if denom <= 0:
        return 0.0
    return max(0.0, 1.0 - resid_var / denom)


def spectral_entropy(y: np.ndarray) -> float:
    """Normalized Shannon entropy of the periodogram. Near 0 = strongly
    deterministic/seasonal, near 1 = noise-like."""
    _, psd = periodogram(y - np.mean(y))
    psd = psd[psd > 0]
    p = psd / psd.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def extract_characteristics(
    y: pd.Series,
    periods: tuple[int, ...] = (24, 168),
    anomaly_z: float = 3.0,
) -> dict[str, float]:
    """Extract the meta-analysis characteristics from one series window.

    Annual period is omitted from STL by default because windows shorter than
    ~2 years cannot estimate it reliably; include 8766 only for long windows.
    """
    yv = y.dropna().to_numpy(dtype=float)
    res = MSTL(yv, periods=list(periods)).fit()
    seasonal = np.column_stack([np.asarray(res.seasonal)[:, i] for i in range(len(periods))]) \
        if np.asarray(res.seasonal).ndim > 1 else np.asarray(res.seasonal).reshape(-1, 1)
    remainder = np.asarray(res.resid)
    trend = np.asarray(res.trend)

    out: dict[str, float] = {}
    rv = float(np.var(remainder))
    for i, m in enumerate(periods):
        out[f"seasonal_strength_{m}"] = _strength(float(np.var(seasonal[:, i])), rv)
    # trend strength computed on deseasonalized series
    out["trend_strength"] = _strength(float(np.var(trend)), rv)
    out["spectral_entropy"] = spectral_entropy(yv)
    out["remainder_variance_share"] = rv / float(np.var(yv)) if np.var(yv) > 0 else 1.0

    # lumpiness & stability over non-overlapping daily tiles
    tile = 24
    ntile = len(yv) // tile
    tiles = yv[: ntile * tile].reshape(ntile, tile)
    out["lumpiness"] = float(np.var(tiles.var(axis=1)))
    out["stability"] = float(np.var(tiles.mean(axis=1)))

    z = (remainder - remainder.mean()) / (remainder.std() + 1e-12)
    out["anomaly_density"] = float(np.mean(np.abs(z) > anomaly_z))
    return out
