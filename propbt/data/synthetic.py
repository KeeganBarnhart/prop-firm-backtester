"""Deterministic synthetic OHLCV bars for demos and tests (no market data license needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_ohlcv(n: int = 4000, seed: int = 7) -> pd.DataFrame:
    """Deterministic 5m MNQ-ish bars: trending random walk with noise.

    Used for demos and tests so no licensed market data is needed. Timestamps are
    America/New_York starting at a session open, so session-window filters,
    VWAP resets, and opening-range indicators all exercise real transitions.
    Used by the corpus meter (scripts/corpus_import.py) and engine tests.
    """
    rng = np.random.default_rng(seed)
    drift = np.concatenate([
        np.full(n // 2, 0.04),          # up regime
        np.full(n - n // 2, -0.02),     # down regime
    ])
    steps = drift + rng.normal(0.0, 6.0, n)
    close = 18_000 + np.cumsum(steps)
    close = np.maximum(close, 1_000.0)
    spread = np.abs(rng.normal(0.0, 4.0, n)) + 1.0
    high = close + spread
    low = close - spread
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    idx = pd.date_range("2025-01-02 09:30", periods=n, freq="5min", tz="America/New_York")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": rng.integers(100, 5_000, n)},
        index=idx,
    )
