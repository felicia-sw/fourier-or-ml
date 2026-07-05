#!/usr/bin/env python
"""Synthetic-grid experiment: backtest DHR vs LightGBM across sampled grid
cells, attach window characteristics, and fit the meta-regression.

Supports chunked execution (results are appended), so large runs can be split:

    python scripts/run_synthetic_grid.py --cells 20 --offset 0
    python scripts/run_synthetic_grid.py --cells 20 --offset 20
    python scripts/run_synthetic_grid.py --fit-meta          # analyze accumulated results
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

CHAR_PREFIXES = ("seasonal_strength", "trend_strength", "spectral_entropy",
                 "remainder_variance", "lumpiness", "stability", "anomaly_density")


def fit_meta(results: pd.DataFrame, out: Path) -> None:
    char_cols = [c for c in results.columns if c.startswith(CHAR_PREFIXES)]
    chars = results[results.model == "dhr"].groupby(["cell_id", "origin"], as_index=False)[char_cols].first()
    table = build_meta_table(results, chars, keys=("cell_id", "origin", "horizon")).dropna()
    if len(table) < 20:
        print(f"Only {len(table)} usable rows — run more cells before fitting the meta-regression.")
        return
    fitted = fit_meta_regression(table.drop(columns=["cell_id"]))
    print(fitted.summary())
    acc = decision_frontier_accuracy(table, fitted)
    share_dhr = float((table[RESPONSE] < 0).mean())
    print(f"\nDHR wins {share_dhr:.1%} of windows; in-sample decision-frontier accuracy: {acc:.1%}")
    table.to_csv(out / "meta_table.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=20, help="number of grid cells this run")
    ap.add_argument("--offset", type=int, default=0, help="skip the first N sampled cells")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=24 * 365 + 24 * 90)
    ap.add_argument("--max-origins", type=int, default=2)
    ap.add_argument("--out", default="results/synthetic_grid")
    ap.add_argument("--fit-meta", action="store_true", help="only analyze accumulated results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.csv"

    if args.fit_meta:
        fit_meta(pd.read_csv(results_path), out)
        return

    rng = random.Random(args.seed)
    configs = factorial_grid(replicates=1, n=args.n)
    rng.shuffle(configs)
    configs = configs[args.offset : args.offset + args.cells]

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
        # cfg_ prefix prevents config labels (e.g. anomaly_density='low') from
        # colliding with the measured numeric characteristics of the same name
        for k, v in asdict(cfg).items():
            res[f"cfg_{k}"] = v
        res["cell_id"] = cfg.cell_id
        all_results.append(res)

    results = pd.concat(all_results, ignore_index=True)
    if results_path.exists():
        results = pd.concat([pd.read_csv(results_path), results], ignore_index=True)
        results = results.drop_duplicates(subset=["cell_id", "origin", "model", "horizon"])
    results.to_csv(results_path, index=False)

    print(results.groupby(["model", "horizon"])["mase"].mean().round(3).unstack())
    print(f"\naccumulated: {results.cell_id.nunique()} cells -> {results_path}")


if __name__ == "__main__":
    main()
