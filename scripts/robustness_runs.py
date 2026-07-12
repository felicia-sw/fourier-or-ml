#!/usr/bin/env python
"""Robustness runs for the paper: K-order sensitivity + LightGBM detrending ablation.

    python scripts/robustness_runs.py --ksens    # DHR/hybrid_a under 4 Fourier-order sets
    python scripts/robustness_runs.py --detrend  # lgbm_s1 raw vs linearly detrended

Both: 12 PJM zones x 12 monthly origins (final year), horizons {1,24,168,720},
expanding window — same protocol as the main run, subsampled origins.
Checkpointed per zone x variant; rerun the same command to resume.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fourier_or_ml.data.pjm import load_panel, ZONES
from fourier_or_ml.evaluation.backtest import rolling_origin_backtest
from fourier_or_ml.models.base import Forecaster
from fourier_or_ml.models.dhr import DynamicHarmonicRegression
from fourier_or_ml.models.gbm import LGBMForecaster

HORIZONS = (1, 24, 168, 720)
STEP = 720
N_ORIGINS = 12

# K = number of harmonics per period m (K <= m/2); base = main-run setting
K_VARIANTS = {
    "K_low":   {24: 4,  168: 2, 8766: 2},
    "K_base":  {24: 8,  168: 4, 8766: 3},
    "K_high":  {24: 12, 168: 6, 8766: 4},
    "K_xhigh": {24: 12, 168: 8, 8766: 6},  # 24 capped at m/2 = 12
}


class DetrendedLGBM(Forecaster):
    """lgbm_s1 on linearly detrended y; OLS trend extrapolated back at predict.

    Ablation target: tree models cannot extrapolate a trend outside the training
    range of the target — detrending moves that burden to the linear component.
    """

    def __init__(self):
        self.name = "lgbm_s1_detrended"
        self._lgbm = LGBMForecaster(use_lags=False, name=self.name)

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> "DetrendedLGBM":
        t = np.arange(len(y), dtype=float)
        self._coef = np.polyfit(t, y.to_numpy(), 1)
        self._n = len(y)
        resid = pd.Series(y.to_numpy() - np.polyval(self._coef, t), index=y.index)
        self._lgbm.fit(resid, X)
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        base = self._lgbm.predict(horizon, X_future)
        t_fut = np.arange(self._n, self._n + horizon, dtype=float)
        return pd.Series(base.to_numpy() + np.polyval(self._coef, t_fut),
                         index=base.index, name=self.name)


def run_variants(panel: dict, variants: list[tuple[str, dict, dict]], res_path: Path) -> None:
    """variants: list of (variant_label, model_factories, fourier_orders)."""
    done = set()
    if res_path.exists():
        done = set(map(tuple, pd.read_csv(res_path, usecols=["series", "variant"])
                       .drop_duplicates().itertuples(index=False)))
    for zone in ZONES:
        if zone not in panel:
            continue
        y = panel[zone]
        initial = len(y.dropna()) - STEP * N_ORIGINS - max(HORIZONS)
        for label, factories, fourier in variants:
            if (f"pjm_{zone}", label) in done:
                print(f"[skip] {zone}/{label}")
                continue
            t0 = time.time()
            res = rolling_origin_backtest(
                y, factories, horizons=HORIZONS, initial_train=initial,
                step=STEP, fourier_orders=fourier,
            )
            res.insert(0, "series", f"pjm_{zone}")
            res.insert(1, "variant", label)
            res.to_csv(res_path, mode="a", header=not res_path.exists(), index=False)
            print(f"[done] {zone}/{label}: {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ksens", action="store_true")
    ap.add_argument("--detrend", action="store_true")
    ap.add_argument("--out", default="results/full")
    args = ap.parse_args()
    out = Path(args.out)
    panel = load_panel("data/raw")

    if args.ksens:
        variants = [
            (label, {"dhr": lambda: DynamicHarmonicRegression(),
                     "hybrid_a": lambda: LGBMForecaster(use_lags=False, prefixes=None,
                                                        name="hybrid_a")}, fourier)
            for label, fourier in K_VARIANTS.items()
        ]
        run_variants(panel, variants, out / "ksens_results.csv")

    if args.detrend:
        variants = [("detrend_ablation",
                     {"lgbm_s1": lambda: LGBMForecaster(use_lags=False),
                      "lgbm_s1_detrended": lambda: DetrendedLGBM()},
                     K_VARIANTS["K_base"])]
        run_variants(panel, variants, out / "detrend_results.csv")


if __name__ == "__main__":
    main()
