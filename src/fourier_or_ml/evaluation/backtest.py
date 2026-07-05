"""Rolling-origin (expanding window) backtest under matched information sets."""
from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from tqdm import tqdm

from ..features.build import deterministic_features
from .metrics import score_all


def rolling_origin_backtest(
    y: pd.Series,
    model_factories: dict[str, Callable[[], "Forecaster"]],
    horizons: tuple[int, ...] = (1, 24, 168, 720),
    initial_train: int = 24 * 365,
    step: int = 24 * 30,          # retrain monthly
    fourier_orders: dict[float, int] | None = None,
    max_origins: int | None = None,
    country: str = "US",
) -> pd.DataFrame:
    """Expanding-window evaluation.

    At each origin: fit every model on y[:origin] with the SAME deterministic
    feature matrix, forecast max(horizons) steps, score each horizon.
    Returns a tidy DataFrame: origin x model x horizon x metrics.
    """
    y = y.dropna()
    H = max(horizons)
    origins = range(initial_train, len(y) - H, step)
    if max_origins is not None:
        origins = list(origins)[:max_origins]

    rows = []
    for origin in tqdm(list(origins), desc="rolling origin"):
        y_train = y.iloc[:origin]
        y_test = y.iloc[origin : origin + H]
        X_train = deterministic_features(y_train.index, t0=0,
                                         fourier_orders=fourier_orders, country=country)
        X_future = deterministic_features(y_test.index, t0=origin,
                                          fourier_orders=fourier_orders, country=country)
        for name, factory in model_factories.items():
            model = factory().fit(y_train, X_train)
            preds = model.predict(H, X_future)
            for h in horizons:
                scores = score_all(y_test.iloc[:h].to_numpy(), preds.iloc[:h].to_numpy(),
                                   y_train.to_numpy())
                rows.append({"origin": y.index[origin], "model": name, "horizon": h, **scores})
    return pd.DataFrame(rows)
