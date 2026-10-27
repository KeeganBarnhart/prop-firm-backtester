"""End-to-end tests for the backtest engine (simple.py).

These tests prove: example strategy JSON -> pydantic Strategy ->
engine run on deterministic synthetic bars -> sane metrics + out-of-sample diagnosis.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from propbt.engines.simple import BacktestResult, run_simple_backtest
from propbt.schema.strategy import Strategy

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "examples" / "example_mnq_ema_cross.json"


def _synthetic_bars(n: int = 4000, seed: int = 7) -> pd.DataFrame:
    """Deterministic 5m MNQ-ish bars: trending random walk with noise."""
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


@pytest.fixture(scope="module")
def strategy() -> Strategy:
    return Strategy.model_validate(json.loads(EXAMPLE.read_text()))


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return _synthetic_bars()


def test_example_strategy_validates(strategy: Strategy):
    assert strategy.name == "MNQ EMA Cross (5m)"


def test_single_run_produces_trades_and_metrics(strategy: Strategy, bars: pd.DataFrame):
    result = run_simple_backtest(strategy, bars)
    assert isinstance(result, BacktestResult)
    assert result.trades, "EMA cross on a trending walk should produce trades"
    assert len(result.equity_curve) > 0
    assert result.final_equity > 0
    d = result.to_dict()
    for key in (
        "trades", "trades_count", "equity_curve", "net_pnl",
        "max_drawdown", "win_rate", "profit_factor", "final_equity",
    ):
        assert key in d, f"key {key} missing from to_dict()"
    assert d["trades_count"] == len(result.trades)
    assert d["final_equity"] == pytest.approx(
        d["starting_equity"] + d["net_pnl"], abs=0.02
    )


def test_run_is_deterministic(strategy: Strategy, bars: pd.DataFrame):
    a = run_simple_backtest(strategy, bars)
    b = run_simple_backtest(strategy, bars)
    assert a.final_equity == b.final_equity
    assert len(a.trades) == len(b.trades)


def test_train_test_split_returns_oos_diagnosis(strategy: Strategy, bars: pd.DataFrame):
    train, test, diag = run_simple_backtest(strategy, bars, train_test_split=0.7)
    assert isinstance(train, BacktestResult)
    assert isinstance(test, BacktestResult)
    # OOS run continues from IS final equity (continuity contract)
    assert test.starting_equity == pytest.approx(train.final_equity)
    assert isinstance(diag, dict)
    assert "warnings" in diag


def test_split_bounds_validated(strategy: Strategy, bars: pd.DataFrame):
    with pytest.raises(ValueError):
        run_simple_backtest(strategy, bars, train_test_split=0.99)


# ── full pipeline: Pine source -> importer -> schema -> engine ──

EXAMPLE_PINE = REPO_ROOT / "examples" / "example_mnq_ema_cross.pine"


def test_pine_to_engine_full_slice(bars: pd.DataFrame):
    """The lifecycle's create->research spine, end to end and off-TV:
    house-dialect Pine -> deterministic importer -> pydantic schema ->
    engine run -> metrics. This is the path every audit and sweep uses."""
    from propbt.importers.pine import import_pine

    result = import_pine(EXAMPLE_PINE.read_text())
    assert result.ok, f"import failed: {[ (e.line, e.message) for e in result.errors ]}"

    imported = Strategy.model_validate(result.strategy_json)
    exits = result.strategy_json["exit"]
    assert exits["stop_loss"]["type"] == "atr"
    assert exits["stop_loss"]["atr_multiplier"] == 2.0
    assert exits["take_profit"] == {"type": "rr", "ratio": 1.5}

    # Inputs become the sweep grid later — all five must be captured.
    inputs = {i["name"] for i in result.pine_inputs}
    assert inputs == {"emaFastLen", "emaSlowLen", "atrLen", "atrMult", "rrRatio"}

    run = run_simple_backtest(imported, bars)
    assert isinstance(run, BacktestResult)
    assert run.trades, "imported strategy should trade on the synthetic walk"
    assert run.to_dict()["trades_count"] == len(run.trades)
