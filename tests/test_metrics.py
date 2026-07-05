import numpy as np

from fourier_or_ml.evaluation.metrics import mase, rmse, smape
from fourier_or_ml.evaluation.dm_test import dm_test


def test_perfect_forecast():
    y = np.random.default_rng(0).normal(size=100)
    assert rmse(y, y) == 0
    assert smape(y + 10, y + 10) == 0


def test_mase_seasonal_naive_is_one():
    """A seasonal-naive forecast evaluated in-sample should have MASE ~ 1."""
    rng = np.random.default_rng(1)
    y = np.tile(np.sin(np.arange(24)), 50) + rng.normal(0, 0.1, 1200)
    train, test = y[:1000], y[1000:1100]
    snaive_pred = y[1000 - 24 : 1100 - 24]
    val = mase(test, snaive_pred, train, m=24)
    assert 0.5 < val < 1.5


def test_dm_detects_better_model():
    rng = np.random.default_rng(2)
    e_good = rng.normal(0, 1, 500)
    e_bad = rng.normal(0, 2, 500)
    stat, p = dm_test(e_good, e_bad, h=1)
    assert stat < 0 and p < 0.01
