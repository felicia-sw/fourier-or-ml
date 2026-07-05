#!/usr/bin/env python
"""Generate the synthetic factorial grid (or a subset) to parquet files."""
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from fourier_or_ml.data.synthetic import factorial_grid, generate, grid_metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/synthetic")
    ap.add_argument("--replicates", type=int, default=30)
    ap.add_argument("--n", type=int, default=24 * 365 * 2)
    ap.add_argument("--limit", type=int, default=None, help="only first N series (for testing)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    configs = factorial_grid(replicates=args.replicates, n=args.n)
    if args.limit:
        configs = configs[: args.limit]
    grid_metadata(configs).to_csv(out / "grid_metadata.csv", index=False)
    for cfg in tqdm(configs, desc="generating"):
        df = generate(cfg)
        df.to_parquet(out / f"{cfg.cell_id}.parquet")
    print(f"wrote {len(configs)} series to {out}")


if __name__ == "__main__":
    main()
