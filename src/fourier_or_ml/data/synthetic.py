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

# Factorial levels (proposal Section 4.2)
SEASONAL_STRENGTH = {"low": 0.3, "med": 1.0, "high": 3.0}
SNR = {"snr1": 0.5, "snr2": 1.0, "snr3": 2.0, "snr4": 4.0, "snr5": 8.0}
TREND = ("none", "linear", "piecewise")
ANOMALY_DENSITY = {"low": 0.001, "med": 0.005, "high": 0.02}
NONLINEARITY = ("absent", "mild", "strong")


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


def _seasonal_signal(t: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Harmonic signal with random amplitudes over daily/weekly/annual cycles."""
    sig = np.zeros_like(t, dtype=float)
    for period, n_harm in ((DAILY, 3), (WEEKLY, 2), (ANNUAL, 2)):
        for j in range(1, n_harm + 1):
            a, b = rng.normal(0, 1, 2) / j
            sig += a * np.sin(2 * np.pi * j * t / period) + b * np.cos(2 * np.pi * j * t / period)
    return strength * sig


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

    seasonal = _seasonal_signal(t, rng, SEASONAL_STRENGTH[cfg.seasonal_strength])

    if cfg.trend == "none":
        trend = np.zeros(cfg.n)
    elif cfg.trend == "linear":
        trend = 2.0 * t / cfg.n
    else:  # piecewise
        brk = cfg.n // 2
        trend = np.where(t < brk, 1.0 * t / cfg.n, 1.0 * brk / cfg.n + 3.0 * (t - brk) / cfg.n)

    driver = _driver_effect(t, rng, cfg.nonlinearity)

    signal = seasonal + trend + driver
    noise = _arma_noise(cfg.n, rng)
    noise *= signal.std() / (noise.std() * np.sqrt(SNR[cfg.snr]) + 1e-12)

    # holiday-like shocks: random days with a level shift
    shocks = np.zeros(cfg.n)
    n_days = cfg.n // DAILY
    shock_days = rng.random(n_days) < ANOMALY_DENSITY[cfg.anomaly_density] * DAILY
    for d in np.where(shock_days)[0]:
        shocks[d * DAILY : (d + 1) * DAILY] = rng.normal(-2.0, 0.5) * signal.std()

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
