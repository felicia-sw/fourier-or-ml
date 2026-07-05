"""Diebold-Mariano test with the Harvey, Leybourne & Newbold (1997)
small-sample correction, for h-step forecasts."""
from __future__ import annotations

import numpy as np
from scipy import stats


def dm_test(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1,
    power: int = 2,
) -> tuple[float, float]:
    """Two-sided DM test on forecast error sequences e1, e2 (same targets).

    Returns (statistic, p_value). Negative statistic => model 1 more accurate.
    Loss differential d_t = |e1|^power - |e2|^power; long-run variance via
    rectangular kernel with h-1 lags (standard for h-step forecasts).
    """
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    dbar = d.mean()
    # autocovariances up to h-1
    gamma = [np.mean((d[k:] - dbar) * (d[:n - k] - dbar)) for k in range(h)]
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var_d <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(var_d)
    # Harvey correction
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_star = dm * correction
    p = 2 * stats.t.sf(np.abs(dm_star), df=n - 1)
    return float(dm_star), float(p)


def dm_from_error_table(
    errors: "pd.DataFrame",
    model_a: str,
    model_b: str,
    horizon: int,
) -> tuple[float, float]:
    """DM test from a long-format error table (origin, model, step, error),
    as produced by rolling_origin_backtest(collect_errors=True).

    Uses the step-`horizon` error across origins for both models. With
    non-overlapping origins (step >= horizon), h=1 in the correction is
    appropriate; for overlapping origins pass the errors to `dm_test` directly
    with the proper h.
    """
    import pandas as pd  # local import to keep scipy-only module light
    sub = errors[errors.step == horizon].pivot(index="origin", columns="model", values="error")
    sub = sub[[model_a, model_b]].dropna()
    return dm_test(sub[model_a].to_numpy(), sub[model_b].to_numpy(), h=1)
