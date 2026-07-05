# CLAUDE.md — fourier-or-ml

Research code for a Scopus Q1 paper: **characteristic-driven comparison of
dynamic harmonic regression (DHR) vs LightGBM vs hybrids for multi-seasonal
electricity load forecasting**. The contribution is NOT "which model wins" —
it is a meta-regression mapping measurable series characteristics (seasonal
strength, spectral entropy, anomaly density, trend strength) to the relative
accuracy gap, yielding a decision frontier. Full proposal lives in the owner's
Google Docs; design summary is in README.md.

## Current state (as of last session)

- Core framework implemented and tested: `pytest -q` → 13 tests, all should pass.
- Datasets in place (gitignored): `data/raw/*_hourly.csv` (12 PJM zones),
  `data/raw/gefcom2014/L*-train.csv` (GEFCom2014 load track).
- Preliminary runs done in a sandbox (small scale): PJME 36 origins with core
  models; 100-cell synthetic grid. Early findings: DHR dominates h=1, hybrid
  wins h≥168; anomaly density (p=0.011) and spectral entropy (p=0.016)
  significantly favor LightGBM; measured trend strength favors LightGBM even
  though generative trend *type* favors DHR (STL trend strength picks up
  stochastic wandering — keep this distinction, it matters for the paper).
- **The full overnight run has NOT been done yet. That is the current task.**

## Task: run the full experiment

```bash
# 0. environment (once)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# 1. verify before burning hours
python -m pytest tests -q          # must be 13 passed
python scripts/run_all.py --smoke  # ~2 min end-to-end sanity check; then:
rm -rf results/full                # remove smoke artifacts before the real run

# 2. the real run (3-6 h on an M4-class machine)
caffeinate -i python scripts/run_all.py > results/overnight.log 2>&1
```

What it does (see `scripts/run_all.py`): all 12 PJM zones × 36 monthly
origins × 7 models × horizons {1,24,168,720}, expanding window, raw errors
saved to parquet for Diebold–Mariano tests; GEFCom2014 L1 same protocol;
405-cell synthetic factorial grid × 3 origins with per-window characteristics.

**Resumability:** completed zone files under `results/full/` are skipped on
restart, and the grid checkpoints every 10 cells. If the run dies, just rerun
the same command. Monitor with `tail -f results/overnight.log`.

After the run finishes: commit `results/full/` is NOT tracked (results/ is
gitignored) — do not force-add it. Commit any code changes only, and push.

## After the run: quick health checks (do these, report results)

```bash
python - <<'EOF'
import pandas as pd, glob
files = sorted(glob.glob("results/full/pjm_*_results.csv"))
print(len(files), "zone result files")   # expect 12
r = pd.concat([pd.read_csv(f) for f in files])
print(r.groupby(["model","horizon"])["mase"].mean().round(3).unstack())
g = pd.read_csv("results/full/grid_results.csv")
print("grid cells:", g.cell_id.nunique())  # expect 405
EOF
```

Sanity expectations: every model beats snaive somewhere; dhr strongest at
h=1; MASE values roughly 0.2–2.5; if lgbm_s2 wins h=1 that's plausible (it
has lag 1). If anything looks wildly off (MASE > 10, NaNs), stop and
investigate before drawing conclusions.

## Repo map (details in README.md)

- `src/fourier_or_ml/models/` — snaive, DHR (OLS + windowed ARMA errors),
  LightGBM S1/S2, hybrids. Feature selection by column prefix enforces the
  matched-information-set protocol — see `features/build.py` docstring.
- `src/fourier_or_ml/evaluation/` — rolling-origin backtest (chunkable),
  MASE/RMSE/sMAPE, DM test (+`dm_from_error_table` for the stored parquets).
- `src/fourier_or_ml/meta/` — meta-regression / decision frontier.
- `scripts/run_synthetic_grid.py` — chunked grid runner + `--fit-meta`.

## Conventions & pitfalls (learned the hard way — do not regress these)

1. **K is the number of Fourier harmonics per period m ∈ {24,168,8766}, K ≤ m/2** —
   never call the period K.
2. **Matched information sets:** DHR gets Fourier encodings, LightGBM (M2)
   gets raw calendar integers, Hybrid A gets both. Never give plain lgbm_s1
   the Fourier columns — that silently turns it into Hybrid A (old bug).
3. **`cfg_` prefixes** on synthetic config labels prevent collision with
   measured characteristics of the same name (old bug: `anomaly_density`).
4. **Join meta-tables on (cell_id, origin, horizon)** — origins repeat across
   series (old bug).
5. Fourier features must be built with the global `t0` offset so train/test
   share phase (tested in `tests/test_fourier.py`).
6. ARMA error fitting uses the last `error_window=4320` residuals only —
   full-history ARMA on hourly data is prohibitively slow.
7. Data files and results/ are gitignored; never commit them.

## After everything: next analysis steps (if asked to continue)

1. DM tests per zone/horizon from `*_errors.parquet` via `dm_from_error_table`;
   MCB/Nemenyi across the 12-zone panel.
2. Meta-regression with interactions (characteristic × horizon) on the full
   grid; out-of-sample frontier validation: train rule on PJM+synthetic,
   test transfer on GEFCom L1. Target: winner-prediction accuracy ≥ 70%.
3. SHAP-vs-harmonic-coefficient interpretability comparison (proposal §4.8).
4. Results/Discussion drafting is handled by the owner with a separate
   writing skill — produce numbers and figures, not manuscript prose.
