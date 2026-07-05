#!/usr/bin/env python
"""Synthetic-grid experiment: backtest DHR vs LightGBM across sampled grid
cells, attach window characteristics, and fit the meta-regression.

This is the M3/M4 pipeline in miniature. Start small:

    python scripts/run_synthetic_grid.py --cells 20 --max-origins 2

and scale up once timings are known.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from fourier_or_ml.data.synthetic import factorial_grid, generate
from fourier_or_ml.evaluation.backtest import rolling_origin_backtest
from fourier_or_ml.models.dhr import DynamicHarmonicRegression
from fourier_or_ml.models.gbm import LGBMForecaster
from fourier_or_ml.meta.regression import build_meta_table, fit_meta_regression, \
    decision_frontier_accuracy, RESPONSE

FACTORIES = {
    "dhr": lambda: DynamicHarmonicRegression(),
    "lgbm_s1": lambda: LGBMForecaster(use_lags=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=20, help="number of grid cells to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=24 * 365 + 24 * 90)  # 1y train + eval room
    ap.add_argument("--max-origins", type=int, default=2)
    ap.add_argument("--out", default="results/synthetic_grid")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    configs = factorial_grid(replicates=1, n=args.n)
    configs = rng.sample(configs, min(args.cells, len(configs)))

    all_results = []
    for cfg in tqdm(configs, desc="grid cells"):
        y = generate(cfg)["y"]
        res = rolling_origin_backtest(
            y, FACTORIES,
            horizons=(24, 168),
            initial_train=24 * 365,
            step=24 * 30,
            fourier_orders={24: 6, 168: 3},
            max_origins=args.max_origins,
            char_window=24 * 60,
        )
        for k, v in asdict(cfg).items():
            res[k] = v
        res["cell_id"] = cfg.cell_id
        all_results.append(res)

    results = pd.concat(all_results, ignore_index=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "results.csv", index=False)

    # meta-analysis: characteristics -> log MASE ratio
    char_cols = [c for c in results.columns if c.startswith(("seasonal_strength", "trend_strength",
                 "spectral_entropy", "remainder_variance", "lumpiness", "stability",
                 "anomaly_density"))]
    chars = results[results.model == "dhr"].groupby("origin", as_index=False)[char_cols].first() \
        if char_cols else None
    if chars is not None and len(results["origin"].unique()) >= 10:
        table = build_meta_table(results, chars)
        table = table.dropna()
        if len(table) >= 10:
            fitted = fit_meta_regression(table)
            print(fitted.summary())
            acc = decision_frontier_accuracy(table, fitted)
            print(f"\nin-sample decision-frontier accuracy: {acc:.1%}")
            table.to_csv(out / "meta_table.csv", index=False)
    else:
        print("Too few origins for meta-regression — raise --cells / --max-origins.")

    print(results.groupby(["model", "horizon"])["mase"].mean().round(3).unstack())
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
