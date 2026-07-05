"""Characteristic-driven meta-analysis (proposal Section 4.7).

Fits a meta-regression of log(MASE_DHR / MASE_LGBM) on window characteristics
and horizon. Positive response => LightGBM better; negative => DHR better.
Mixed effects (zone/dataset) to be added once the panel results exist —
statsmodels MixedLM slot marked below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

RESPONSE = "log_mase_ratio"


def build_meta_table(
    results: pd.DataFrame,
    characteristics: pd.DataFrame,
    model_a: str = "dhr",
    model_b: str = "lgbm_s1",
    keys: tuple[str, ...] = ("origin", "horizon"),
) -> pd.DataFrame:
    """Join backtest results with per-window characteristics.

    `results`: tidy output of rolling_origin_backtest (origin, model, horizon, mase, ...)
    `characteristics`: one row per evaluation window with the extracted characteristics.
    `keys`: identifying columns; on multi-series runs include the series id
    (e.g. ("cell_id", "origin", "horizon")) — origins repeat across series,
    so joining on origin alone would mix windows from different series.
    """
    a = results[results.model == model_a].set_index(list(keys))["mase"]
    b = results[results.model == model_b].set_index(list(keys))["mase"]
    ratio = np.log(a / b).rename(RESPONSE).reset_index()
    join_cols = [k for k in keys if k != "horizon" and k in characteristics.columns]
    return ratio.merge(characteristics, on=join_cols, how="left")


def fit_meta_regression(table: pd.DataFrame, groups: str | None = None):
    """OLS meta-regression; pass groups (e.g. 'zone') for MixedLM random intercepts."""
    feats = [c for c in table.columns if c not in (RESPONSE, "origin", "horizon", "zone", "dataset")]
    formula = f"{RESPONSE} ~ " + " + ".join(feats) + " + C(horizon)"
    if groups and groups in table.columns:
        return smf.mixedlm(formula, table, groups=table[groups]).fit()
    return smf.ols(formula, table).fit()


def decision_frontier_accuracy(table: pd.DataFrame, fitted) -> float:
    """Share of windows where the meta-regression predicts the correct winner."""
    pred = fitted.predict(table)
    return float(np.mean(np.sign(pred) == np.sign(table[RESPONSE])))
