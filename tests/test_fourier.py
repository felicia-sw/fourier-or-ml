import numpy as np
import pytest

from fourier_or_ml.features.fourier import fourier_terms, multi_seasonal_fourier


def test_k_bounds():
    t = np.arange(100)
    with pytest.raises(ValueError):
        fourier_terms(t, period=24, k=13)  # K must be <= m/2
    fourier_terms(t, period=24, k=12)  # boundary ok


def test_column_count():
    X = multi_seasonal_fourier(200, {24: 8, 168: 4})
    assert X.shape == (200, 2 * (8 + 4))


def test_phase_continuity():
    """Train (t0=0) and future (t0=n) windows must lie on the same wave."""
    full = multi_seasonal_fourier(48, {24: 3})
    part2 = multi_seasonal_fourier(24, {24: 3}, t0=24)
    np.testing.assert_allclose(full.iloc[24:].to_numpy(), part2.to_numpy(), atol=1e-12)
