#!/usr/bin/env python
"""SHAP vs harmonic-coefficient interpretability comparison (proposal 4.8).

Per PJM zone, on the maximal training window (all but the last 720 h):
- DHR side: harmonic-OLS partial fits reconstruct the deterministic diurnal
  (m=24 terms) and weekly (m=24 + m=168 terms + weekend dummy) profiles.
- LightGBM side: TreeSHAP attributions of lgbm_s1 (last 365 days), grouped by
  clock hour (diurnal) and hour-of-week (weekly), using the hour / day-of-week
  / weekend feature columns.
Both are in load units (MW), centered. Divergence per zone = 1 - Pearson r of
the two profiles; related to the zone-level accuracy gap from the main run.

    python scripts/interpretability.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fourier_or_ml.data.pjm import load_panel, ZONES
from fourier_or_ml.features.build import deterministic_features
from fourier_or_ml.models.dhr import DynamicHarmonicRegression
from fourier_or_ml.models.gbm import LGBMForecaster

FOURIER = {24: 8, 168: 4, 8766: 3}
SHAP_WINDOW = 24 * 365
OUT = Path("results/analysis")
plt.rcParams.update({"font.family": "serif", "font.size": 9})


def dhr_profiles(model: DynamicHarmonicRegression, X: pd.DataFrame):
    coef = model.harmonic_coefficients
    part = lambda cols: X[cols].to_numpy() @ coef[cols].to_numpy()
    daily = part([c for c in X.columns if c.startswith("m24_")])
    weekly = daily + part([c for c in X.columns if c.startswith("m168_")])
    if "cal_is_weekend" in coef.index:
        weekly = weekly + X["cal_is_weekend"].to_numpy() * coef["cal_is_weekend"]
    idx = X.index
    return (pd.Series(daily, index=idx).groupby(idx.hour).mean(),
            pd.Series(weekly, index=idx).groupby(idx.dayofweek * 24 + idx.hour).mean())


def lgbm_profiles(model: LGBMForecaster, X: pd.DataFrame):
    Xs = X.iloc[-SHAP_WINDOW:]
    sv = pd.DataFrame(model.shap_values(Xs), columns=model._feat_cols, index=Xs.index)
    diurnal = sv["cal_hour"].groupby(Xs.index.hour).mean()
    wk_cols = [c for c in ("cal_hour", "cal_day_of_week", "cal_is_weekend") if c in sv]
    weekly = sv[wk_cols].sum(axis=1).groupby(Xs.index.dayofweek * 24 + Xs.index.hour).mean()
    group_share = sv.abs().mean() / sv.abs().mean().sum()
    return diurnal, weekly, group_share


def centered_corr(a: pd.Series, b: pd.Series) -> float:
    return float(np.corrcoef(a - a.mean(), b - b.mean())[0, 1])


def main():
    panel = load_panel("data/raw")
    prof_rows, summ_rows, shares = [], [], {}
    for zone in ZONES:
        if zone not in panel:
            continue
        y = panel[zone].dropna().iloc[:-720]
        X = deterministic_features(y.index, t0=0, fourier_orders=FOURIER)
        dhr = DynamicHarmonicRegression(error_order=None).fit(y, X)
        lgbm = LGBMForecaster(use_lags=False).fit(y, X)
        d_di, d_wk = dhr_profiles(dhr, X)
        l_di, l_wk, share = lgbm_profiles(lgbm, X)
        shares[zone] = share
        r_di, r_wk = centered_corr(d_di, l_di), centered_corr(d_wk, l_wk)
        amp = float(l_wk.std() / d_wk.std())
        summ_rows.append({"zone": zone, "r_diurnal": r_di, "r_weekly": r_wk,
                          "amplitude_ratio": amp})
        for cyc, dp, lp in [("diurnal", d_di, l_di), ("weekly", d_wk, l_wk)]:
            for pos in dp.index:
                prof_rows.append({"zone": zone, "cycle": cyc, "position": pos,
                                  "dhr": dp[pos] - dp.mean(), "lgbm": lp[pos] - lp.mean()})
        print(f"{zone}: r_diurnal={r_di:.3f} r_weekly={r_wk:.3f} amp_ratio={amp:.2f}",
              flush=True)

    summ = pd.DataFrame(summ_rows)
    profiles = pd.DataFrame(prof_rows)
    share_df = pd.DataFrame(shares).T
    share_df.to_csv(OUT / "interp_shap_group_share.csv")
    profiles.to_csv(OUT / "interp_profiles.csv", index=False)

    # relate divergence to the accuracy gap (main-run panel results)
    res = pd.concat([pd.read_csv(f) for f in
                     sorted(glob.glob("results/full/pjm_*_results.csv"))])
    piv = res.pivot_table("mase", index=["series", "origin", "horizon"], columns="model")
    gap = (np.log(piv.dhr / piv.lgbm_s1).groupby(["series", "horizon"]).mean()
           .rename("log_gap").reset_index())
    gap["zone"] = gap.series.str.replace("pjm_", "")
    summ["divergence"] = 1 - summ.r_weekly
    merged = gap.merge(summ, on="zone")
    corr = merged.groupby("horizon").apply(
        lambda g: np.corrcoef(g.divergence, g.log_gap)[0, 1], include_groups=False)
    print("\ncorr(profile divergence, zone log MASE gap) by horizon (n=12):")
    print(corr.round(3).to_string())
    summ.to_csv(OUT / "interp_summary.csv", index=False)
    merged.to_csv(OUT / "interp_divergence_vs_gap.csv", index=False)

    # figure: PJME profile overlays
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
    for ax, cyc in zip(axes, ["diurnal", "weekly"]):
        p = profiles[(profiles.zone == "PJME") & (profiles.cycle == cyc)]
        ax.plot(p.position, p.dhr, label="DHR harmonic reconstruction", lw=1.2)
        ax.plot(p.position, p.lgbm, label="LightGBM SHAP profile", lw=1.2)
        ax.set_xlabel("hour of day" if cyc == "diurnal" else "hour of week")
        ax.set_ylabel("centered contribution (MW)")
        ax.set_title(f"PJME {cyc} profile")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_interpretability_profiles.png", dpi=200)
    print(f"\nsaved: interp_summary.csv, interp_profiles.csv, "
          f"interp_shap_group_share.csv, interp_divergence_vs_gap.csv, fig4")


if __name__ == "__main__":
    main()
