#!/usr/bin/env python
"""Run the rolling-origin backtest from a yaml config.

Usage:
    python scripts/run_backtest.py --config configs/smoke.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from fourier_or_ml.data.synthetic import SyntheticConfig, generate
from fourier_or_ml.data.pjm import load_zone
from fourier_or_ml.evaluation.backtest import rolling_origin_backtest
from fourier_or_ml.models.naive import SeasonalNaive
from fourier_or_ml.models.dhr import DynamicHarmonicRegression
from fourier_or_ml.models.gbm import LGBMForecaster, DHRResidualHybrid

MODEL_FACTORIES = {
    "snaive": lambda: SeasonalNaive(24),
    "harmonic_ols": lambda: DynamicHarmonicRegression(error_order=None),
    "dhr": lambda: DynamicHarmonicRegression(),
    "lgbm_s1": lambda: LGBMForecaster(use_lags=False),                       # calendar encoding, no Fourier
    "lgbm_s2": lambda: LGBMForecaster(use_lags=True),
    "hybrid_a": lambda: LGBMForecaster(use_lags=False, prefixes=None, name="hybrid_a"),  # calendar + Fourier
    "hybrid_b": lambda: DHRResidualHybrid(),
}


def load_series(cfg: dict) -> pd.Series:
    src = cfg["data"]["source"]
    if src == "synthetic":
        s = generate(SyntheticConfig(**cfg["data"].get("synthetic", {})))["y"]
        s.name = "synthetic"
        return s
    if src == "pjm":
        return load_zone(cfg["data"].get("raw_dir", "data/raw"), cfg["data"]["zone"])
    raise ValueError(f"unknown source {src}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--origin-offset", type=int, default=0,
                    help="skip the first N origins (for chunked runs; results are appended)")
    ap.add_argument("--max-origins", type=int, default=None, help="override config max_origins")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    y = load_series(cfg)
    factories = {m: MODEL_FACTORIES[m] for m in cfg["models"]}
    bt = cfg.get("backtest", {})
    results = rolling_origin_backtest(
        y,
        factories,
        horizons=tuple(bt.get("horizons", (1, 24, 168))),
        initial_train=bt.get("initial_train", 24 * 365),
        step=bt.get("step", 24 * 30),
        fourier_orders={int(k): v for k, v in cfg.get("fourier_orders", {24: 8, 168: 4}).items()},
        max_origins=args.max_origins if args.max_origins is not None else bt.get("max_origins"),
        origin_offset=args.origin_offset,
        train_window=bt.get("train_window"),
    )

    out = Path(cfg.get("output", "results")) / (Path(args.config).stem + "_results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and args.origin_offset > 0:
        results = pd.concat([pd.read_csv(out, parse_dates=["origin"]), results],
                            ignore_index=True)
        results = results.drop_duplicates(subset=["origin", "model", "horizon"])
    results.to_csv(out, index=False)
    print(results.groupby(["model", "horizon"])["mase"].mean().round(3).unstack())
    print(f"\norigins so far: {results.origin.nunique()}  saved -> {out}")


if __name__ == "__main__":
    main()
