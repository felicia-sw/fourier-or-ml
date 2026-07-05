#!/usr/bin/env python
"""Overnight orchestrator: the full experimental run.

Covers, resumably (completed pieces are skipped on restart):
  1. All 12 PJM zones, full model set, 36 monthly origins over the final
     3 years, expanding window, horizons {1, 24, 168, 720}, raw errors stored
     for Diebold-Mariano testing.
  2. GEFCom2014-L task 1, same protocol (24 monthly origins).
  3. Synthetic factorial grid: all 405 cells x 3 origins with characteristics.

Launch (macOS; caffeinate keeps the machine awake with the lid open):

    caffeinate -i python scripts/run_all.py > results/overnight.log 2>&1

Rough expected runtime on an M4-class laptop: 3-6 hours.
--smoke runs a tiny version of everything first (~2 min) to fail fast.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from fourier_or_ml.data.pjm import load_panel, ZONES
from fourier_or_ml.data.gefcom import load_gefcom_load
from fourier_or_ml.data.synthetic import factorial_grid, generate
from fourier_or_ml.evaluation.backtest import rolling_origin_backtest
from fourier_or_ml.models.naive import SeasonalNaive
from fourier_or_ml.models.dhr import DynamicHarmonicRegression
from fourier_or_ml.models.gbm import LGBMForecaster, DHRResidualHybrid

FULL_MODELS = {
    "snaive": lambda: SeasonalNaive(24),
    "harmonic_ols": lambda: DynamicHarmonicRegression(error_order=None),
    "dhr": lambda: DynamicHarmonicRegression(),
    "lgbm_s1": lambda: LGBMForecaster(use_lags=False),
    "lgbm_s2": lambda: LGBMForecaster(use_lags=True),
    "hybrid_a": lambda: LGBMForecaster(use_lags=False, prefixes=None, name="hybrid_a"),
    "hybrid_b": lambda: DHRResidualHybrid(),
}
GRID_MODELS = {k: FULL_MODELS[k] for k in ("dhr", "lgbm_s1", "hybrid_a")}

HORIZONS = (1, 24, 168, 720)
FOURIER = {24: 8, 168: 4, 8766: 3}
FOURIER_SHORT = {24: 6, 168: 3}  # synthetic series are too short for annual terms
STEP = 720


def run_real_series(y: pd.Series, tag: str, out: Path, n_origins: int, smoke: bool) -> None:
    res_path, err_path = out / f"{tag}_results.csv", out / f"{tag}_errors.parquet"
    if res_path.exists():
        print(f"[skip] {tag} (exists)")
        return
    if smoke:
        n_origins = 1
    initial = len(y.dropna()) - STEP * n_origins - max(HORIZONS)
    t0 = time.time()
    results, errors = rolling_origin_backtest(
        y, FULL_MODELS, horizons=HORIZONS, initial_train=initial, step=STEP,
        fourier_orders=FOURIER, collect_errors=True,
    )
    results.insert(0, "series", tag)
    results.to_csv(res_path, index=False)
    errors.to_parquet(err_path, index=False)
    print(f"[done] {tag}: {results.origin.nunique()} origins in {time.time()-t0:.0f}s")


def run_grid(out: Path, replicates: int, max_origins: int, smoke: bool) -> None:
    res_path = out / "grid_results.csv"
    done_cells = set()
    if res_path.exists():
        done_cells = set(pd.read_csv(res_path, usecols=["cell_id"]).cell_id.unique())
    configs = factorial_grid(replicates=replicates, n=24 * 365 + 24 * 150)
    if smoke:
        configs = configs[:2]
    todo = [c for c in configs if c.cell_id not in done_cells]
    print(f"[grid] {len(todo)} cells to run ({len(done_cells)} already done)")
    buffer = []
    for i, cfg in enumerate(todo):
        y = generate(cfg)["y"]
        res = rolling_origin_backtest(
            y, GRID_MODELS, horizons=(24, 168, 720), initial_train=24 * 365,
            step=STEP, fourier_orders=FOURIER_SHORT,
            max_origins=1 if smoke else max_origins, char_window=24 * 60,
        )
        from dataclasses import asdict
        for k, v in asdict(cfg).items():
            res[f"cfg_{k}"] = v
        res["cell_id"] = cfg.cell_id
        buffer.append(res)
        if len(buffer) >= 10 or i == len(todo) - 1:  # checkpoint every 10 cells
            block = pd.concat(buffer, ignore_index=True)
            block.to_csv(res_path, mode="a", header=not res_path.exists(), index=False)
            buffer = []
            print(f"[grid] {i+1}/{len(todo)} cells", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end test (~2 min)")
    ap.add_argument("--out", default="results/full")
    ap.add_argument("--n-origins", type=int, default=36)
    ap.add_argument("--grid-replicates", type=int, default=1)
    ap.add_argument("--grid-origins", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()

    # 1. PJM panel
    panel = load_panel("data/raw")
    zones = list(ZONES if not args.smoke else ZONES[:1])
    for z in zones:
        if z in panel:
            run_real_series(panel[z], f"pjm_{z}", out, args.n_origins, args.smoke)

    # 2. GEFCom2014-L task 1
    try:
        g = load_gefcom_load("data/raw", task=1)["y"]
        run_real_series(g, "gefcom_L1", out, min(args.n_origins, 24), args.smoke)
    except FileNotFoundError as e:
        print(f"[warn] GEFCom skipped: {e}")

    # 3. synthetic factorial grid
    run_grid(out, args.grid_replicates, args.grid_origins, args.smoke)

    print(f"\nALL DONE in {(time.time()-start)/3600:.1f} h -> {out}")


if __name__ == "__main__":
    main()
