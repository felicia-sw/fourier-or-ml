"""Model-level tests, incl. the matched-information-set feature selection."""
import numpy as np
import pandas as pd
import pytest

from fourier_or_ml.data.synthetic import SyntheticConfig, generate
from fourier_or_ml.features.build import deterministic_features
from fourier_or_ml.models.dhr import DynamicHarmonicRegression
from fourier_or_ml.models.gbm import LGBMForecaster


@pytest.fixture(scope="module")
def toy():
    y = generate(SyntheticConfig(n=24 * 120, seed=3, snr="snr5", seasonal_strength="high"))["y"]
    train, test = y.iloc[: 24 * 100], y.iloc[24 * 100 :]
    X_train = deterministic_features(train.index, t0=0, fourier_orders={24: 4, 168: 2})
    X_future = deterministic_features(test.index, t0=len(train), fourier_orders={24: 4, 168: 2})
    return train, test, X_train, X_future


def test_dhr_sees_fourier_not_raw_calendar(toy):
    train, _, X_train, _ = toy
    m = DynamicHarmonicRegression(error_order=None).fit(train, X_train)
    assert any(c.startswith("m24_") for c in m._cols)
    assert "cal_hour" not in m._cols  # raw integers are the LGBM encoding


def test_lgbm_sees_calendar_not_fourier(toy):
    train, _, X_train, _ = toy
    m = LGBMForecaster(use_lags=False).fit(train, X_train)
    assert "cal_hour" in m._det_cols
    assert not any(c.startswith("m24_") for c in m._det_cols)


def test_hybrid_a_sees_both(toy):
    train, _, X_train, _ = toy
    m = LGBMForecaster(use_lags=False, prefixes=None, name="hybrid_a").fit(train, X_train)
    assert "cal_hour" in m._det_cols and any(c.startswith("m24_") for c in m._det_cols)


def test_models_beat_noise_on_easy_series(toy):
    train, test, X_train, X_future = toy
    h = 48
    for model in (DynamicHarmonicRegression(error_order=None), LGBMForecaster()):
        preds = model.fit(train, X_train).predict(h, X_future)
        assert len(preds) == h
        corr = np.corrcoef(test.iloc[:h], preds)[0, 1]
        assert corr > 0.8  # strong seasonal signal should be captured
