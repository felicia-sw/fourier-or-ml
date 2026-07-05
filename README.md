# fourier-or-ml

**A Characteristic-Driven Comparison of Harmonic Regression and Gradient Boosting for Multi-Seasonal Electricity Load Forecasting**

Code for the research project comparing dynamic harmonic regression (DHR), LightGBM, and Fourier-augmented hybrids under matched information sets, with a characteristic-driven meta-analysis that maps measurable series properties (seasonal strength, spectral entropy, noise level, anomaly density) to relative forecast accuracy.

## Research design (short version)

- **Models:** M0 seasonal naive · M1 DHR (Fourier terms + trend + holidays, ARMA errors) · M2 LightGBM · M3 LightGBM + Fourier features (Hybrid A) · M4 DHR + LightGBM on residuals (Hybrid B)
- **Information sets:** S1 deterministic-only (no lags, all models) · S2 autoregressive (same lag set for all models)
- **Data:** PJM 12-zone hourly panel (Kaggle) · GEFCom2014 load track · controlled synthetic generator (factorial grid: seasonal strength × SNR × trend × anomaly density × nonlinearity, ≥30 replicates/cell)
- **Evaluation:** rolling-origin, horizons {1, 24, 168, 720} h; MASE (primary), RMSE, sMAPE; Diebold–Mariano with Harvey correction; MCB across the panel
- **Contribution:** mixed-effects meta-regression of log(MASE_DHR/MASE_LGBM) on series characteristics → empirical decision frontier, transfer-validated on GEFCom2014

Full proposal lives in Google Docs (see project notes).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Data

```bash
# PJM: download from Kaggle (needs kaggle CLI credentials)
kaggle datasets download -d robikscube/hourly-energy-consumption -p data/raw --unzip

# Synthetic grid
python scripts/generate_synthetic.py --out data/processed/synthetic --replicates 30
```

GEFCom2014: request/download per Hong et al. (2016) and place under `data/raw/gefcom2014/`.

## Quickstart

```bash
# smoke test: full pipeline on one synthetic series
python scripts/run_backtest.py --config configs/smoke.yaml

# full backtest (long-running)
python scripts/run_backtest.py --config configs/default.yaml
```

## Repo structure

```
configs/               experiment configs (yaml)
data/raw|processed/    datasets (gitignored)
results/               backtest outputs (gitignored)
scripts/               entry points
src/fourier_or_ml/
  data/                PJM loader, synthetic generator
  features/            fourier terms, calendar/lags, characteristics extraction
  models/              seasonal naive, DHR, LightGBM, hybrids
  evaluation/          metrics, rolling-origin backtest, DM test
  meta/                meta-regression / decision frontier
tests/                 unit tests (pytest)
```

## Roadmap (mirrors proposal work plan)

- [x] M1: repo scaffold, characteristic-extraction pipeline
- [ ] M1: PJM + GEFCom2014 acquisition and cleaning
- [ ] M2: synthetic generator validation; end-to-end baselines on 2 zones
- [ ] M3: full rolling-origin backtest (all zones × scenarios × horizons)
- [ ] M4: meta-regression, decision frontier, transfer validation, DM/MCB
- [ ] M5: SHAP vs harmonic coefficients; robustness (K sensitivity, detrending ablations)
- [ ] M6–7: manuscript + submission package
