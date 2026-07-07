"""Controlled synthetic data generator (the experimental core of the study).

Each series is: trend + sum of harmonic seasonal signals + nonlinear driver
effect + holiday shocks + ARMA(1,1) noise, over a full factorial grid.
Because the ground truth is known, effects of characteristics on the
DHR-vs-LightGBM gap can be identified causally.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product

import numpy as np
import pandas as pd

DAILY, WEEKLY, ANNUAL = 24, 168, 8766

# Factorial levels (proposal Section 4.2).
# GENERATOR v2 — recalibrated 2026-07-06 so measured characteristics of the
# grid overlap the real-load region (PJM/GEFCom 60-day windows: seasonal
# strength ~0.4-0.94, trend strength ~0.4-0.86 — mostly the annual cycle seen
# by short STL windows — remainder share ~0.04-0.25, spectral entropy
# ~0.23-0.49). v1 levels produced far noisier, weaker-seasonal series and the
# decision rule did not transfer (see results/analysis/findings.md).
SEASONAL_STRENGTH = {"low": 1.0, "med": 3.0, "high": 8.0}
# SNR is defined against the SUB-ANNUAL signal (daily+weekly seasonality +
# trend + driver): the annual component is nearly constant inside an
# evaluation window, so calibrating noise against the full-series std (v1)
# made windows far noisier than any real load series.
SNR = {"snr1": 4.0, "snr2": 8.0, "snr3": 16.0, "snr4": 32.0, "snr5": 64.0}
TREND = ("none", "linear", "piecewise")
ANOMALY_DENSITY = {"low": 0.001, "med": 0.005, "high": 0.02}
NONLINEARITY = ("absent", "mild", "strong")

# relative amplitude per seasonal period: annual is weighted up so that short
# STL windows register realistic "trend" strength, as they do on real load
PERIOD_WEIGHTS = {DAILY: 1.0, WEEKLY: 0.7, ANNUAL: 5.0}


@dataclass(frozen=True)
class SyntheticConfig:
    seasonal_strength: str = "med"
    snr: str = "snr3"
    trend: str = "none"
    anomaly_density: str = "med"
    nonlinearity: str = "absent"
    n: int = 24 * 365 * 2  # two years hourly
    seed: int = 0

    @property
    def cell_id(self) -> str:
        return (
            f"ss-{self.seasonal_strength}_snr-{self.snr}_tr-{self.trend}"
            f"_an-{self.anomaly_density}_nl-{self.nonlinearity}_seed-{self.seed}"
        )


def _seasonal_signal(
    t: np.ndarray, rng: np.random.Generator, strength: float
) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic signals with random amplitudes.

    Returns (sub_annual, annual) separately: the noise level is calibrated
    against the sub-annual signal only, because within a 60-day evaluation
    window the annual component is nearly constant (it registers as trend).
    """
    sub = np.zeros_like(t, dtype=float)
    ann = np.zeros_like(t, dtype=float)
    for period, n_harm in ((DAILY, 3), (WEEKLY, 2), (ANNUAL, 2)):
        w = PERIOD_WEIGHTS[period]
        part = np.zeros_like(t, dtype=float)
        for j in range(1, n_harm + 1):
            a, b = rng.normal(0, 1, 2) * w / j
            part += a * np.sin(2 * np.pi * j * t / period) + b * np.cos(2 * np.pi * j * t / period)
        if period == ANNUAL:
            ann += part
        else:
            sub += part
    return strength * sub, strength * ann


def _driver_effect(t: np.ndarray, rng: np.random.Generator, mode: str) -> np.ndarray:
    """Temperature-like smooth driver with optional nonlinear (U-shaped) response,
    mimicking heating/cooling load."""
    if mode == "absent":
        return np.zeros_like(t, dtype=float)
    temp = 10 * np.sin(2 * np.pi * t / ANNUAL) + 4 * np.sin(2 * np.pi * t / DAILY) \
        + rng.normal(0, 1.0, len(t)).cumsum() * 0.01
    if mode == "mild":
        return 0.05 * temp
    # strong: nonlinear U-shape around comfort temperature
    return 0.02 * (temp - 5.0) ** 2 / 5.0


def _arma_noise(n: int, rng: np.random.Generator, phi: float = 0.7, theta: float = 0.3) -> np.ndarray:
    e = rng.normal(0, 1, n + 200)
    x = np.zeros(n + 200)
    for i in range(1, n + 200):
        x[i] = phi * x[i - 1] + e[i] + theta * e[i - 1]
    return x[200:]


def generate(cfg: SyntheticConfig) -> pd.DataFrame:
    """Generate one synthetic hourly series with component ground truth."""
    rng = np.random.default_rng(cfg.seed + hash(cfg.cell_id) % (2**16))
    t = np.arange(cfg.n)

    seasonal_sub, seasonal_ann = _seasonal_signal(t, rng, SEASONAL_STRENGTH[cfg.seasonal_strength])
    seasonal = seasonal_sub + seasonal_ann

    if cfg.trend == "none":
        trend = np.zeros(cfg.n)
    elif cfg.trend == "linear":
        trend = 6.0 * t / cfg.n
    else:  # piecewise
        brk = cfg.n // 2
        trend = np.where(t < brk, 2.0 * t / cfg.n, 2.0 * brk / cfg.n + 10.0 * (t - brk) / cfg.n)

    driver = _driver_effect(t, rng, cfg.nonlinearity)

    signal = seasonal + trend + driver
    # calibrate noise against the sub-annual signal (see SNR comment above)
    signal_sub = seasonal_sub + trend + driver
    noise = _arma_noise(cfg.n, rng)
    noise *= signal_sub.std() / (noise.std() * np.sqrt(SNR[cfg.snr]) + 1e-12)

    # holiday-like shocks: random days with a level shift. The density value is
    # the target share of anomalous *hours* (matches the measured
    # anomaly_density characteristic on real load, median ~0.008); v1's x24
    # day-probability multiplier shocked up to 40% of days and flooded the STL
    # remainder, which is why the v1 grid never reached real-load noise levels.
    shocks = np.zeros(cfg.n)
    n_days = cfg.n // DAILY
    shock_days = rng.random(n_days) < ANOMALY_DENSITY[cfg.anomaly_density] * 5
    for d in np.where(shock_days)[0]:
        shocks[d * DAILY : (d + 1) * DAILY] = rng.normal(-1.2, 0.3) * signal.std()

    idx = pd.date_range("2020-01-01", periods=cfg.n, freq="h")
    return pd.DataFrame(
        {"y": signal + noise + shocks, "seasonal": seasonal, "trend": trend,
         "driver": driver, "noise": noise, "shocks": shocks},
        index=idx,
    )


def factorial_grid(replicates: int = 30, n: int = 24 * 365 * 2) -> list[SyntheticConfig]:
    """Full factorial grid: 3 x 5 x 3 x 3 x 3 = 405 cells x replicates."""
    return [
        SyntheticConfig(ss, snr, tr, an, nl, n=n, seed=r)
        for ss, snr, tr, an, nl, r in product(
            SEASONAL_STRENGTH, SNR, TREND, ANOMALY_DENSITY, NONLINEARITY, range(replicates)
        )
    ]


def grid_metadata(configs: list[SyntheticConfig]) -> pd.DataFrame:
    return pd.DataFrame([asdict(c) | {"cell_id": c.cell_id} for c in configs])
