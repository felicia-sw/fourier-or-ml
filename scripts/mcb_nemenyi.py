#!/usr/bin/env python
"""MCB / Friedman-Nemenyi analysis across the 12-zone PJM panel.

Per horizon: rank the 7 models by mean MASE within each zone (blocks = zones),
Friedman test, then Nemenyi critical distance CD = q_alpha * sqrt(k(k+1)/(12n)).
Models whose mean-rank interval [r - CD/2, r + CD/2] overlaps the best model's
are not significantly different (MCB reading). Outputs a tidy table and an
MCB-style figure per horizon.

    python scripts/mcb_nemenyi.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, studentized_range

OUT = Path("results/analysis")


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    q = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return q * np.sqrt(k * (k + 1) / (6 * n))


def main():
    files = sorted(glob.glob("results/full/pjm_*_results.csv"))
    panel = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"{len(files)} zones, models: {sorted(panel.model.unique())}")

    rows = []
    for h, sub in panel.groupby("horizon"):
        # mean MASE per zone x model, then rank models within each zone
        zm = sub.groupby(["series", "model"])["mase"].mean().unstack()
        ranks = zm.rank(axis=1)
        k, n = ranks.shape[1], ranks.shape[0]
        stat, p = friedmanchisquare(*[zm[c] for c in zm.columns])
        cd = nemenyi_cd(k, n)
        mean_ranks = ranks.mean().sort_values()
        best = mean_ranks.iloc[0]
        for m, r in mean_ranks.items():
            rows.append({"horizon": h, "model": m, "mean_rank": r,
                         "mean_mase": zm[m].mean(), "cd": cd,
                         "friedman_p": p, "in_best_set": r - best < cd})
        # MCB-style figure
        fig, ax = plt.subplots(figsize=(6, 3.2))
        xs = np.arange(len(mean_ranks))
        ax.errorbar(xs, mean_ranks.values, yerr=cd / 2, fmt="o", capsize=4, color="k")
        ax.axhspan(best - cd / 2, best + cd / 2, alpha=0.15, color="tab:blue")
        ax.set_xticks(xs, mean_ranks.index, rotation=30, ha="right")
        ax.set_ylabel("mean rank (12 zones)")
        ax.set_title(f"MCB, h={h}  (Nemenyi CD={cd:.2f}, Friedman p={p:.1e})")
        fig.tight_layout()
        fig.savefig(OUT / f"mcb_h{h}.png", dpi=200)
        plt.close(fig)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "mcb_nemenyi.csv", index=False)
    for h, sub in res.groupby("horizon"):
        print(f"\nh={h}  Friedman p={sub.friedman_p.iloc[0]:.2e}  CD={sub.cd.iloc[0]:.2f}")
        print(sub[["model", "mean_rank", "mean_mase", "in_best_set"]]
              .to_string(index=False, formatters={"mean_rank": "{:.2f}".format,
                                                  "mean_mase": "{:.3f}".format}))


if __name__ == "__main__":
    main()
