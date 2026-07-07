# HANDOVER.md — continue in Claude Code

Handoff from the Cowork session that set this project up (July 6, 2026).
Read `CLAUDE.md` first (conventions + the current task); this file adds the
full project state, results so far, and the remaining roadmap.

## What this project is

Research code for a Scopus Q1 paper: characteristic-driven comparison of
dynamic harmonic regression (DHR) vs LightGBM vs Fourier-augmented hybrids
for multi-seasonal (hourly) electricity load forecasting. The contribution is
a **decision frontier**: a validated mapping from measurable series
characteristics (STL seasonal strength, spectral entropy, trend strength,
anomaly density, remainder share) to which model family wins. The proposal
(professor-approved) is in the owner's Google Docs: "Research Proposal
(Professor Style) — Characteristic-Driven Comparison of Harmonic Regression
and Gradient Boosting for Multi-Seasonal Electricity Load Forecasting".
Target journals: IJF, ESWA, Energy/Applied Energy, EAAI.

## State: DONE

1. **Framework** (`src/fourier_or_ml/`): 7 models (snaive, harmonic OLS, DHR
   with windowed ARMA errors, LightGBM S1/S2, Hybrid A/B) under a
   matched-information-set protocol enforced by feature-column prefixes;
   rolling-origin backtest (chunkable, resumable, optional raw-error
   collection); MASE/RMSE/sMAPE/pinball; DM test with Harvey correction
   (`dm_from_error_table` works off the stored parquets); characteristic
   extraction (MSTL); meta-regression + frontier. 13 pytest tests pass.
2. **Full real-data runs** (`results/full/`, gitignored): 12 PJM zones x 36
   monthly origins x 7 models x horizons {1,24,168,720}, expanding window,
   errors stored; GEFCom2014 L1 x 24 origins.
3. **Synthetic grid v1**: 405 cells x 3 origins — superseded (see below),
   kept as `grid_results_v1.csv` after the v2 rerun.
4. **Analysis** (`results/analysis/`): findings.md + tables + figures 1-3.
5. **Generator v2** (`data/synthetic.py`): recalibrated so measured
   characteristics overlap real-load 5-95% ranges (sub-annual SNR definition,
   annual weight 5.0, anomaly multiplier fixed x24 -> x5). Committed.

## Key results so far (details in results/analysis/findings.md)

- Horizon regimes on real data (DM-tested, 12 zones): DHR wins h=1 in 11/12
  zones vs lgbm_s1; lgbm_s2 (lags) wins h=1 in 8/12 vs DHR but loses at h=720
  in 2 (recursion drift); hybrid_a is the only model never significantly worse
  than snaive at h>=168 (beats it in 5/12 at 168, 3/12 at 720). h=24 is the
  contested regime (weather-driven; nothing dominates).
- Grid v1 meta-regression (3,645 windows): trend_strength +0.74 (p<.0001,
  interaction grows with horizon), anomaly_density +7.8 (p=.0016); positive =
  LightGBM better. Generative trend TYPE favors DHR while measured STL trend
  strength favors LightGBM (it captures stochastic wandering) — keep this
  measured-vs-generative distinction, it's a paper-level insight.
- Frontier accuracy: synthetic cell-level CV ~69% (h>=168); window-level
  ~59.5%; within-PJM leave-zone-out 65.2% (baseline 63.1%); transfer to
  GEFCom: grid-v1-only failed (36% raw / 61% scale-free = baseline);
  **PJM+grid pooled 66.7% vs 61.1% baseline** — best so far. Diagnosed cause:
  grid v1 didn't cover real-load characteristic space -> generator v2.
- Use ONLY scale-free characteristics for anything cross-dataset
  (lumpiness/stability are scale-dependent — exclude).

## State: NEXT (in order)

1. **Rerun grid with generator v2** — exact commands in CLAUDE.md ("current
   task"). ~3 h; PJM/GEFCom skipped automatically.
2. **Revalidate**: meta-regression w/ characteristic x horizon interactions on
   grid v2; frontier CV; transfer grid-v2 / PJM / pooled -> GEFCom
   (scale-free chars; join on cell_id+origin+horizon). Compare v1 vs v2 in
   findings.md. Decision point: transfer >= ~70% -> frontier is the headline;
   below -> reframe frontier as within-domain tool + honest transfer analysis.
3. **Robustness for the paper**: K-order sensitivity (fourier_orders),
   detrending ablation for LightGBM (trend artifacts), MCB/Nemenyi across the
   12-zone panel, stability of meta-regression coefficients across
   grid replicates (rerun grid with replicates>=2 if compute allows).
4. **Interpretability (proposal §4.8)**: SHAP (LGBMForecaster.shap_values,
   TreeExplainer) aggregated by feature group vs DHR harmonic-coefficient
   reconstruction of diurnal/weekly profiles; quantify divergence, relate to
   the accuracy gap.
5. **Publication figures/tables**: regenerate figs 1-3 with v2 + add frontier
   validation figure; export publication-quality (200+ dpi, serif fonts).

## Division of labor (important)

Claude Code produces **numbers, tables, figures** only. The manuscript prose
is written by the owner in Cowork with a personal writing skill trained on
her professor's revision style — do not draft paper text here. When analysis
is done, summarize numeric findings into results/analysis/findings.md.

## Practical notes

- Env: `.venv`, `pip install -r requirements.txt && pip install -e .`
- `results/` and `data/` are gitignored — never force-add; commit code only.
- Long runs: `caffeinate -i python scripts/run_all.py > results/overnight2.log 2>&1`
  (resumable; grid checkpoints every 10 cells).
- The 7 hard-won conventions in CLAUDE.md ("Conventions & pitfalls") each
  correspond to a real bug we already fixed once. Check them before touching
  features, the generator, or meta-table joins.
