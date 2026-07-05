"""Characteristic-driven comparison of harmonic regression and gradient boosting
for multi-seasonal electricity load forecasting."""

__version__ = "0.1.0"

# Seasonal periods (hourly data)
PERIOD_DAILY = 24
PERIOD_WEEKLY = 168
PERIOD_ANNUAL = 8766  # 365.25 * 24
PERIODS = (PERIOD_DAILY, PERIOD_WEEKLY, PERIOD_ANNUAL)
