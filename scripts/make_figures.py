#!/usr/bin/env python
"""Publication figures (200+ dpi, serif). Regenerates figs 1-3 with grid v2
and adds the frontier-validation figure (fig5); fig4 comes from
interpretability.py.

    python scripts/make_figures.py

Colors: Okabe-Ito subset, CVD-validated (worst adjacent deltaE 17.9, deutan);
snaive / harmonic_ols are neutral references. Fixed model->hue assignment.
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

OUT = Path("results/analysis")
DPI = 300
plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})

MODEL_COLORS = {  # fixed assignment, never cycled
    "dhr": "#0072B2", "lgbm_s1": "#D55E00", "lgbm_s2": "#E69F00",
    "hybrid_a": "#009E73", "hybrid_b": "#CC79A7",
    "snaive": "#999999", "harmonic_ols": "#666666",
}
MODEL_STYLE = {"snaive": ":", "harmonic_ols": (0, (1, 1))}
HORIZONS = [1, 24, 168, 720]


def fig1_mase_by_horizon(panel: pd.DataFrame) -> None:
    m = panel.groupby(["model", "horizon"])["mase"].mean().unstack()[HORIZONS]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    xs = np.arange(len(HORIZONS))
    order = m[720].sort_values().index  # legend ordered by h=720 value
    for name in order:
        row = m.loc[name]
        ax.plot(xs, row.values, ls=MODEL_STYLE.get(name, "-"), lw=1.4,
                marker="o", ms=3.5, color=MODEL_COLORS[name], label=name)
    ax.set_xticks(xs, [f"h={h}" for h in HORIZONS])
    ax.set_yscale("log")
    ax.set_yticks([0.1, 0.2, 0.5, 1.0, 2.0], ["0.1", "0.2", "0.5", "1.0", "2.0"])
    ax.set_xlim(-0.2, len(HORIZONS) - 0.8)
    ax.set_ylabel("mean MASE (12 PJM zones)")
    ax.legend(frameon=False, fontsize=7, loc="lower right", ncol=2)
    ax.grid(axis="y", lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_mase_by_horizon.png", dpi=DPI)
    plt.close(fig)


def fig2_winrate_heatmap(grid: pd.DataFrame) -> None:
    p = grid.pivot_table("mase", index=["cell_id", "origin", "horizon"],
                         columns="model").reset_index()
    p["lgbm_wins"] = (p.lgbm_s1 < p.dhr).astype(float)
    cfg = grid[["cell_id", "cfg_seasonal_strength", "cfg_snr", "cfg_trend",
                "cfg_anomaly_density", "cfg_nonlinearity"]].drop_duplicates()
    p = p.merge(cfg, on="cell_id")
    factors = [("cfg_trend", "trend"), ("cfg_seasonal_strength", "seasonal strength"),
               ("cfg_snr", "SNR"), ("cfg_anomaly_density", "anomaly density"),
               ("cfg_nonlinearity", "nonlinearity")]
    rows, labels = [], []
    for col, lab in factors:
        w = p.groupby([col, "horizon"])["lgbm_wins"].mean().unstack()
        rows.append(w)
        labels += [f"{lab}: {lv}" for lv in w.index]
    W = pd.concat(rows)
    cmap = LinearSegmentedColormap.from_list(
        "dhr_lgbm", ["#0072B2", "#f2f0ee", "#D55E00"])  # DHR pole - neutral - LGBM pole
    fig, ax = plt.subplots(figsize=(3.6, 0.32 * len(W) + 1.0))
    im = ax.imshow(W.values, cmap=cmap, norm=TwoSlopeNorm(0.5, 0, 1), aspect="auto")
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            ax.text(j, i, f"{W.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="#222222")
    ax.set_xticks(range(W.shape[1]), [f"h={h}" for h in W.columns])
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_title("LightGBM win share vs DHR (grid v2, 810 cells)", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.7, label="win share (0.5 = tied)")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_winrate_heatmap.png", dpi=DPI)
    plt.close(fig)


def fig3_characteristic_space() -> None:
    v1 = pd.read_csv(OUT / "grid_meta_table_v1.csv")
    v2 = pd.read_csv(OUT / "grid_meta_table_v2_pooled.csv")
    pjm = pd.read_csv(OUT / "pjm_meta_table.csv")
    gef = pd.read_csv(OUT / "gefcom_transfer_table.csv")
    dd = lambda d: d.drop_duplicates(subset=[c for c in ("cell_id", "zone", "origin")
                                             if c in d.columns])
    panels = [("trend_strength", "spectral_entropy"),
              ("seasonal_strength_168", "remainder_variance_share")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for ax, (cx, cy) in zip(axes, panels):
        for d, lab, c, m, s in [
            (dd(v1), "grid v1", "#bbbbbb", "o", 4),
            (dd(v2), "grid v2", "#009E73", "o", 4),
            (dd(pjm), "PJM", "#0072B2", "s", 9),
            (dd(gef), "GEFCom", "#D55E00", "^", 12),
        ]:
            ax.scatter(d[cx], d[cy], s=s, c=c, marker=m, alpha=0.45,
                       lw=0, label=lab)
        ax.set_xlabel(cx.replace("_", " "))
        ax.set_ylabel(cy.replace("_", " "))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc="upper center", frameon=False, fontsize=8,
               markerscale=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "fig3_characteristic_space.png", dpi=DPI)
    plt.close(fig)


def fig5_frontier_validation() -> None:
    # numbers from the recorded analyses (findings.md 2026-07-06 / 2026-07-12)
    rows = [  # (label, accuracy, baseline)
        ("synthetic cell CV h=24 (v1)", 0.620, 0.523),
        ("synthetic cell CV h=168 (v1)", 0.689, 0.573),
        ("synthetic cell CV h=720 (v1)", 0.694, 0.541),
        ("synthetic window CV (v1)", 0.594, 0.517),
        ("synthetic window CV (v2 pooled)", 0.652, 0.650),
        ("within-PJM leave-zone-out", 0.652, 0.631),
        ("transfer: grid v1 -> GEFCom", 0.611, 0.611),
        ("transfer: PJM -> GEFCom", 0.597, 0.611),
        ("transfer: PJM+grid v1 -> GEFCom", 0.667, 0.611),
        ("transfer: PJM+grid v2 -> GEFCom", 0.611, 0.611),
    ]
    labels = [r[0] for r in rows][::-1]
    acc = np.array([r[1] for r in rows])[::-1]
    base = np.array([r[2] for r in rows])[::-1]
    ys = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.barh(ys, acc, height=0.6, color="#0072B2", alpha=0.85)
    ax.scatter(base, ys, marker="|", s=180, c="#222222", lw=1.4,
               label="majority-class baseline", zorder=3)
    for y, a in zip(ys, acc):
        ax.text(a - 0.004, y, f"{a:.1%}", va="center", ha="right",
                fontsize=7, color="white")
    ax.set_yticks(ys, labels, fontsize=7.5)
    ax.set_xlim(0.45, 0.78)
    ax.set_xlabel("winner-prediction accuracy")
    ax.axvline(0.70, color="#D55E00", lw=0.8, ls=":")
    ax.text(0.701, len(rows) - 0.4, "70% target", fontsize=7, color="#D55E00")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left",
              bbox_to_anchor=(0.0, 1.0))
    ax.grid(axis="x", lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_frontier_validation.png", dpi=DPI)
    plt.close(fig)


def main():
    panel = pd.concat([pd.read_csv(f) for f in
                       sorted(glob.glob("results/full/pjm_*_results.csv"))])
    fig1_mase_by_horizon(panel)
    grid = pd.read_csv("results/full/grid_results.csv")
    fig2_winrate_heatmap(grid)
    fig3_characteristic_space()
    fig5_frontier_validation()
    print("saved fig1, fig2, fig3, fig5 ->", OUT)


if __name__ == "__main__":
    main()
