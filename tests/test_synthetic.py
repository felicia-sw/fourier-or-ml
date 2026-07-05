import numpy as np

from fourier_or_ml.data.synthetic import SyntheticConfig, generate, factorial_grid


def test_reproducible():
    cfg = SyntheticConfig(n=24 * 30, seed=7)
    a, b = generate(cfg), generate(cfg)
    np.testing.assert_allclose(a["y"], b["y"])


def test_snr_ordering():
    """Higher SNR level -> noise variance smaller relative to signal."""
    lo = generate(SyntheticConfig(snr="snr1", n=24 * 90, seed=1))
    hi = generate(SyntheticConfig(snr="snr5", n=24 * 90, seed=1))
    ratio_lo = lo["noise"].var() / (lo["seasonal"] + lo["trend"] + lo["driver"]).var()
    ratio_hi = hi["noise"].var() / (hi["seasonal"] + hi["trend"] + hi["driver"]).var()
    assert ratio_hi < ratio_lo


def test_grid_size():
    grid = factorial_grid(replicates=2, n=24)
    assert len(grid) == 3 * 5 * 3 * 3 * 3 * 2  # 810
