#!/usr/bin/env python
"""Grid revalidation: meta-regression + frontier CV + GEFCom transfer.

Reconstructs the v1 analysis pipeline (results/analysis/findings.md) as
committed code so grid v2 can be compared against v1 like-for-like:

    python scripts/analyze_grid.py --grid results/full/grid_results_v1.csv --tag v1
    python scripts/analyze_grid.py --grid results/full/grid_results.csv    --tag v2

Reuses results/analysis/pjm_meta_table.csv and gefcom_transfer_table.csv
(the PJM/GEFCom runs are unchanged between grid versions). Transfer uses
scale-free characteristics only (lumpiness/stability are scale-dependent).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fourier_or_ml.meta.regression import RESPONSE, build_meta_table

ALL_CHARS = [
    "seasonal_strength_24", "seasonal_strength_168", "trend_strength",
    "spectral_entropy", "remainder_variance_share", "lumpiness",
    "stability", "anomaly_density",
]
SCALE_FREE = [c for c in ALL_CHARS if c not in ("lumpiness", "stability")]
HORIZONS = (24, 168, 720)


def grid_meta_table(grid: pd.DataFrame) -> pd.DataFrame:
    chars = (grid[grid.model == "dhr"]
             .groupby(["cell_id", "origin"], as_index=False)[ALL_CHARS].first())
    table = build_meta_table(grid, chars, keys=("cell_id", "origin", "horizon"))
    return table[table.horizon.isin(HORIZONS)].dropna().reset_index(drop=True)


def meta_regression_with_interactions(table: pd.DataFrame):
    terms = " + ".join(f"{c} * C(horizon)" for c in ALL_CHARS)
    return smf.ols(f"{RESPONSE} ~ {terms}", table).fit()


def _clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def _xy(table: pd.DataFrame, chars: list[str], resp: str):
    X = pd.get_dummies(table[chars + ["horizon"]], columns=["horizon"], dtype=float)
    y = (table[resp] > 0).astype(int)  # 1 = LightGBM wins
    return X, y


def window_cv(table: pd.DataFrame, groups: pd.Series, resp: str = RESPONSE,
              chars: list[str] = ALL_CHARS) -> tuple[float, float]:
    X, y = _xy(table, chars, resp)
    accs = [
        _clf().fit(X.iloc[tr], y.iloc[tr]).score(X.iloc[te], y.iloc[te])
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups)
    ]
    return float(np.mean(accs)), float(np.std(accs))


def cell_cv_by_horizon(table: pd.DataFrame, resp: str = RESPONSE,
                       chars: list[str] = ALL_CHARS) -> pd.DataFrame:
    rows = []
    for h, sub in table.groupby("horizon"):
        cells = sub.groupby("cell_id").agg({**{c: "mean" for c in chars}, resp: "mean"})
        X, y = cells[chars], (cells[resp] > 0).astype(int)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        accs = [
            _clf().fit(X.iloc[tr], y.iloc[tr]).score(X.iloc[te], y.iloc[te])
            for tr, te in cv.split(X, y)
        ]
        rows.append({"horizon": h, "acc": np.mean(accs),
                     "baseline": max(y.mean(), 1 - y.mean()), "n_cells": len(cells)})
    return pd.DataFrame(rows)


def transfer(train_tables: list[tuple[pd.DataFrame, str]], test: pd.DataFrame,
             test_resp: str = "log_ratio") -> float:
    parts = []
    for t, resp in train_tables:
        p = t[SCALE_FREE + ["horizon"]].copy()
        p["win"] = (t[resp] > 0).astype(int)
        parts.append(p)
    train = pd.concat(parts, ignore_index=True)
    Xtr = pd.get_dummies(train[SCALE_FREE + ["horizon"]], columns=["horizon"], dtype=float)
    Xte, yte = _xy(test, SCALE_FREE, test_resp)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)
    return float(_clf().fit(Xtr, train["win"]).score(Xte, yte))


def coverage(grid_table: pd.DataFrame, pjm: pd.DataFrame, gef: pd.DataFrame) -> pd.DataFrame:
    def q(df, c):
        return df[c].quantile([0.05, 0.95]).round(3).tolist()
    return pd.DataFrame(
        {"grid": {c: q(grid_table, c) for c in SCALE_FREE},
         "pjm": {c: q(pjm, c) for c in SCALE_FREE},
         "gefcom": {c: q(gef, c) for c in SCALE_FREE}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="results/full/grid_results.csv")
    ap.add_argument("--tag", default="v2", help="suffix for output files")
    ap.add_argument("--analysis-dir", default="results/analysis")
    args = ap.parse_args()

    adir = Path(args.analysis_dir)
    grid = pd.read_csv(args.grid)
    table = grid_meta_table(grid)
    pjm = pd.read_csv(adir / "pjm_meta_table.csv")
    gef = pd.read_csv(adir / "gefcom_transfer_table.csv")
    print(f"grid {args.tag}: {grid.cell_id.nunique()} cells, {len(table)} meta rows; "
          f"PJM {len(pjm)} rows; GEFCom {len(gef)} rows")

    print("\n=== characteristic-space coverage (5-95% quantiles) ===")
    print(coverage(table, pjm, gef).to_string())

    print("\n=== meta-regression with characteristic x horizon interactions ===")
    fitted = meta_regression_with_interactions(table)
    coefs = pd.DataFrame({"coef": fitted.params, "p": fitted.pvalues}).round(4)
    print(coefs[coefs.p < 0.10].to_string())
    print(f"R2 = {fitted.rsquared:.3f}   (positive coef = LightGBM better)")
    coefs.to_csv(adir / f"grid_meta_coefs_{args.tag}.csv")

    print("\n=== frontier CV (logistic, horizons 24/168/720) ===")
    share = float((table[RESPONSE] > 0).mean())
    m, s = window_cv(table, table.cell_id)
    print(f"window-level grouped 5-fold: {m:.1%} +/- {s:.1%} "
          f"(baseline {max(share, 1 - share):.1%})")
    cell = cell_cv_by_horizon(table)
    print(cell.to_string(index=False,
                         formatters={"acc": "{:.1%}".format, "baseline": "{:.1%}".format}))

    print("\n=== transfer -> GEFCom (n=%d, scale-free chars only) ===" % len(gef))
    base = max((gef.log_ratio > 0).mean(), (gef.log_ratio < 0).mean())
    for name, trains in [
        (f"grid-{args.tag} only", [(table, RESPONSE)]),
        ("PJM only", [(pjm, "log_ratio")]),
        (f"PJM + grid-{args.tag} pooled", [(table, RESPONSE), (pjm, "log_ratio")]),
    ]:
        print(f"{name:26s}: {transfer(trains, gef):.1%}   (baseline {base:.1%})")

    table.to_csv(adir / f"grid_meta_table_{args.tag}.csv", index=False)
    print(f"\nsaved: grid_meta_table_{args.tag}.csv, grid_meta_coefs_{args.tag}.csv")


if __name__ == "__main__":
    main()
