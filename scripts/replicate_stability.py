#!/usr/bin/env python
"""Meta-regression coefficient stability across synthetic-grid replicates.

Fits the characteristic x horizon meta-regression separately per generator
seed and pooled, and reports coefficient/sign agreement:

    python scripts/replicate_stability.py [--grid results/full/grid_results.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analyze_grid import RESPONSE, grid_meta_table, meta_regression_with_interactions

NUISANCE = r"Intercept|C\(horizon\)\[T"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="results/full/grid_results.csv")
    ap.add_argument("--analysis-dir", default="results/analysis")
    args = ap.parse_args()

    g = pd.read_csv(args.grid)
    seeds = sorted(g.cfg_seed.unique())
    print(f"{g.cell_id.nunique()} cells, seeds {seeds}")

    per_seed = {}
    for seed in seeds:
        t = grid_meta_table(g[g.cfg_seed == seed])
        f = meta_regression_with_interactions(t)
        per_seed[seed] = pd.DataFrame({"coef": f.params, "p": f.pvalues})
        print(f"seed {seed}: {len(t)} windows, LGBM win share {(t[RESPONSE] > 0).mean():.1%}")

    cmp = pd.concat(per_seed, axis=1)
    coefs = cmp.xs("coef", axis=1, level=1)
    sig_any = (cmp.xs("p", axis=1, level=1) < 0.05).any(axis=1)
    keep = sig_any & ~cmp.index.str.match(NUISANCE)
    print("\nterms significant (p<.05) in any seed:")
    print(cmp[keep].round(4).to_string())
    signs = np.sign(coefs[keep])
    print(f"sign agreement: {(signs.nunique(axis=1) == 1).mean():.0%}")
    print("coef correlation between seeds (excl. intercept):")
    print(coefs.drop(index="Intercept").corr().round(3).to_string())

    t_all = grid_meta_table(g)
    f_all = meta_regression_with_interactions(t_all)
    pooled = pd.DataFrame({"coef": f_all.params, "p": f_all.pvalues}).round(4)
    print(f"\npooled fit ({len(t_all)} windows), significant characteristic terms:")
    print(pooled[(pooled.p < .05) & ~pooled.index.str.match(NUISANCE)].to_string())
    print(f"R2 = {f_all.rsquared:.3f}")

    adir = Path(args.analysis_dir)
    cmp[~cmp.index.str.match(NUISANCE)].to_csv(adir / "replicate_stability.csv")
    pooled.to_csv(adir / "grid_meta_coefs_v2_pooled.csv")
    t_all.to_csv(adir / "grid_meta_table_v2_pooled.csv", index=False)


if __name__ == "__main__":
    main()
