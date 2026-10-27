"""Per-feature engine tests for the dialect-completeness batch:
session windows, new indicators, new condition types, frozen exit variants,
costs, and the ambiguous-fill counter. Style follows test_engine_simple.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from propbt.engines.simple import (
    _compute_indicators,
    _session_of,
    run_simple_backtest,
)
from propbt.schema.strategy import Strategy
from propbt.data.synthetic import synthetic_ohlcv


def _strategy(**overrides) -> Strategy:
    """EMA-cross base strategy; override any top-level schema section."""
    base = {
        "schema_version": "0.1.0",
        "id": "feature-test",
        "name": "Feature Test",
        "market": {
            "instrument": "MNQ",
            "instrument_type": "futures",
            "timeframe": "5m",
            "session_filter": {"type": "always"},
        },
        "indicators": [
            {"id": "ema_fast", "type": "EMA", "params": {"source": "close", "length": 9}},
            {"id": "ema_slow", "type": "EMA", "params": {"source": "close", "length": 21}},
            {"id": "atr_val", "type": "ATR", "params": {"length": 14}},
        ],
        "entry": {
            "long": {"type": "cross_above", "left": "ema_fast", "right": "ema_slow"},
            "short": {"type": "cross_below", "left": "ema_fast", "right": "ema_slow"},
        },
        "exit": {
            "signal_exit": "opposite",
            "stop_loss": {"type": "atr", "atr_length": 14, "atr_multiplier": 2.0},
            "take_profit": {"type": "rr", "ratio": 1.5},
        },
        "risk": {"model": "fixed_contracts", "fixed_contracts": 1, "max_concurrent_positions": 1},
    }
    base.update(overrides)
    return Strategy.model_validate(base)


RTH = {
    "type": "session_window",
    "windows": [{"start": "09:30", "end": "16:00", "tz": "America/New_York"}],
}


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return synthetic_ohlcv()


# ── session windows ──────────────────────────────────────────────────────────


def test_session_mask_bounds(bars):
    s = _strategy(market={
        "instrument": "MNQ", "instrument_type": "futures", "timeframe": "5m",
        "session_filter": RTH,
    })
    sess = _session_of(s, bars)
    local = bars.index.tz_convert("America/New_York")
    inside = sess.in_window.to_numpy()
    minutes = (local.hour * 60 + local.minute).to_numpy()
    assert (minutes[inside] >= 9 * 60 + 30).all()
    assert (minutes[inside] < 16 * 60).all()
    assert sess.session_start.sum() > 1  # multiple sessions in 4000 bars


def test_session_days_filter(bars):
    weekday_only = {
        "type": "session_window",
        "windows": [{"start": "09:30", "end": "16:00", "tz": "America/New_York",
                     "days": [1, 2, 3, 4, 5]}],
    }
    s = _strategy(market={
        "instrument": "MNQ", "instrument_type": "futures", "timeframe": "5m",
        "session_filter": weekday_only,
    })
    sess = _session_of(s, bars)
    local = bars.index.tz_convert("America/New_York")
    assert not sess.in_window[(local.dayofweek + 1) > 5].any()


def test_session_warmup_delays_entry_ok(bars):
    filt = dict(RTH, warmup_bars=6)
    s = _strategy(market={
        "instrument": "MNQ", "instrument_type": "futures", "timeframe": "5m",
        "session_filter": filt,
    })
    sess = _session_of(s, bars)
    # entry_ok is strictly inside in_window and never true on a session's first 6 bars
    assert (sess.entry_ok & ~sess.in_window).sum() == 0
    session_id = sess.session_start.cumsum()
    bar_num = sess.in_window.groupby(session_id).cumsum()
    assert not (sess.entry_ok & (bar_num <= 6)).any()


def test_entries_only_in_session_and_flat_at_end(bars):
    s = _strategy(market={
        "instrument": "MNQ", "instrument_type": "futures", "timeframe": "5m",
        "session_filter": dict(RTH, flat_at_end=True),
    })
    result = run_simple_backtest(s, bars)
    assert result.trades
    ny = "America/New_York"
    for t in result.trades:
        et = pd.Timestamp(t.entry_time).tz_convert(ny)
        m = et.hour * 60 + et.minute
        assert 9 * 60 + 30 <= m < 16 * 60, f"entry outside RTH: {t.entry_time}"
    reasons = {t.exit_reason for t in result.trades}
    assert "session_end" in reasons  # forced flat at least once
    # no position ever survives outside the window: every exit at/before 16:00
    for t in result.trades:
        xt = pd.Timestamp(t.exit_time).tz_convert(ny)
        m = xt.hour * 60 + xt.minute
        assert m <= 16 * 60, f"exit after session close: {t.exit_time}"


# ── indicators ───────────────────────────────────────────────────────────────


def test_vwap_resets_each_day(bars):
    s = _strategy(indicators=[{"id": "vwap_val", "type": "VWAP", "params": {}}])
    out = _compute_indicators(s, bars)
    vwap = out["vwap_val"]
    day = pd.Series(bars.index.tz_convert("America/New_York").date, index=bars.index)
    first_of_day = day.ne(day.shift())
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3
    # On each day's first bar VWAP equals that bar's typical price exactly.
    assert np.allclose(vwap[first_of_day], tp[first_of_day])


def test_supertrend_direction_and_line(bars):
    s = _strategy(indicators=[
        {"id": "st_dir", "type": "SUPERTREND", "params": {"factor": 3.0, "atr_length": 10, "output": "direction"}},
        {"id": "st_line", "type": "SUPERTREND", "params": {"factor": 3.0, "atr_length": 10, "output": "line"}},
    ])
    out = _compute_indicators(s, bars)
    d = out["st_dir"].iloc[20:]
    assert set(np.unique(d)) <= {-1.0, 1.0}
    assert (d == -1.0).any() and (d == 1.0).any(), "both regimes on the two-regime walk"
    # uptrend: line below close; downtrend: line above close
    line = out["st_line"].iloc[20:]
    close = bars["close"].iloc[20:]
    assert (line[d == -1.0] <= close[d == -1.0]).all()
    assert (line[d == 1.0] >= close[d == 1.0]).all()


def test_orb_freezes_after_n_bars(bars):
    s = _strategy(market={
        "instrument": "MNQ", "instrument_type": "futures", "timeframe": "5m",
        "session_filter": RTH,
    }, indicators=[
        {"id": "orb_high", "type": "ORB", "params": {"bars": 6, "side": "high"}},
        {"id": "orb_low", "type": "ORB", "params": {"bars": 6, "side": "low"}},
    ])
    sess = _session_of(s, bars)
    out = _compute_indicators(s, bars, sess)
    session_id = sess.session_start.cumsum()
    bar_num = sess.in_window.groupby(session_id).cumsum()
    # Pick one full session and check freeze semantics.
    sid = session_id[sess.session_start].iloc[1]
    in_this = (session_id == sid) & sess.in_window
    highs = bars["high"][in_this & (bar_num <= 6)]
    frozen = out["orb_high"][in_this & (bar_num > 6)]
    assert not frozen.empty
    assert (frozen == highs.max()).all()
    assert (out["orb_low"][in_this & (bar_num > 6)] == bars["low"][in_this & (bar_num <= 6)].min()).all()


def test_orb_requires_session_filter(bars):
    s = _strategy(indicators=[{"id": "orb_high", "type": "ORB", "params": {"bars": 6}}])
    with pytest.raises(ValueError, match="session_window"):
        run_simple_backtest(s, bars)


def test_htf_ema_no_lookahead(bars):
    s = _strategy(indicators=[
        {"id": "htf_ema", "type": "EMA", "params": {"source": "close", "length": 20, "timeframe": "30m"}},
    ])
    out = _compute_indicators(s, bars)
    htf = out["htf_ema"]
    # Recompute by hand: EMA on 30m closes, stamped at HTF close, ffilled.
    htf_close = bars["close"].resample("30min", closed="right", label="right").last().dropna()
    ema = htf_close.ewm(span=20, adjust=False).mean()
    # A bar strictly inside an HTF bar must carry the PREVIOUS HTF bar's value.
    probe = bars.index[7]  # 10:05 — inside the 10:00→10:30 HTF bar
    expected = ema[ema.index <= probe].iloc[-1]
    assert htf.loc[probe] == pytest.approx(expected)
    completed_before = ema.index[ema.index <= probe]
    assert (bars.index[7] - completed_before[-1]).total_seconds() >= 0


def test_indicator_offset_shifts(bars):
    s = _strategy(indicators=[
        {"id": "hh", "type": "HIGHEST", "params": {"source": "high", "length": 20}},
        {"id": "hh_prev", "type": "HIGHEST", "params": {"source": "high", "length": 20, "offset": 1}},
    ])
    out = _compute_indicators(s, bars)
    assert np.allclose(out["hh_prev"].iloc[25:], out["hh"].shift(1).iloc[25:], equal_nan=True)


# ── condition types ──────────────────────────────────────────────────────────


def test_new_condition_types_run(bars):
    s = _strategy(
        indicators=[
            {"id": "ema_fast", "type": "EMA", "params": {"source": "close", "length": 9}},
            {"id": "ema_slow", "type": "EMA", "params": {"source": "close", "length": 21}},
            {"id": "rsi_val", "type": "RSI", "params": {"source": "close", "length": 14}},
            {"id": "atr_val", "type": "ATR", "params": {"length": 14}},
        ],
        entry={
            "long": {"type": "and", "conditions": [
                {"type": "cross_above", "left": "ema_fast", "right": "ema_slow"},
                {"type": "between", "left": "rsi_val", "lower": 20, "upper": 80},
                {"type": "rising", "left": "ema_slow", "lookback": 3},
            ]},
            "short": {"type": "and", "conditions": [
                {"type": "cross_below", "left": "ema_fast", "right": "ema_slow"},
                {"type": "falling", "left": "ema_slow", "lookback": 3},
            ]},
        },
    )
    result = run_simple_backtest(s, bars)
    assert result.trades, "filtered cross should still trade on the two-regime walk"


# ── exits, costs, caps ───────────────────────────────────────────────────────


def test_indicator_stop_and_range_rr(bars):
    s = _strategy(
        market={
            "instrument": "MNQ", "instrument_type": "futures", "timeframe": "5m",
            "session_filter": dict(RTH, flat_at_end=True),
        },
        indicators=[
            {"id": "orb_high", "type": "ORB", "params": {"bars": 6, "side": "high"}},
            {"id": "orb_low", "type": "ORB", "params": {"bars": 6, "side": "low"}},
        ],
        entry={"long": {"type": "cross_above", "left": "close", "right": "orb_high"},
               "max_entries_per_session": 1},
        exit={
            "signal_exit": "none",
            "stop_loss": {"type": "indicator", "indicator_id": "orb_low"},
            "take_profit": {"type": "range_rr", "ratio": 2.0, "upper": "orb_high", "lower": "orb_low"},
        },
    )
    result = run_simple_backtest(s, bars)
    assert result.trades
    # max one long entry per session
    ny_dates = {}
    for t in result.trades:
        d = pd.Timestamp(t.entry_time).tz_convert("America/New_York").date()
        ny_dates[d] = ny_dates.get(d, 0) + 1
    assert max(ny_dates.values()) == 1


def test_commission_and_slippage_reduce_pnl(bars):
    plain = _strategy()
    costed = _strategy(risk={
        "model": "fixed_contracts", "fixed_contracts": 1, "max_concurrent_positions": 1,
        "commission_per_contract": 0.62, "slippage_points": 0.25,
    })
    a = run_simple_backtest(plain, bars)
    b = run_simple_backtest(costed, bars)
    assert len(a.trades) == len(b.trades)
    # MNQ point value 2.0: each round trip costs 2*0.62 + 2*0.25*2.0 = 2.24
    per_trade = 2 * 0.62 + 2 * 0.25 * 2.0
    assert b.net_pnl == pytest.approx(a.net_pnl - per_trade * len(a.trades), abs=0.5)


def test_ambiguous_fills_counted():
    # One giant bar that spans both stop and target right after entry.
    idx = pd.date_range("2025-01-02 09:30", periods=40, freq="5min", tz="America/New_York")
    close = np.full(40, 100.0)
    close[:20] = np.linspace(90, 100, 20)     # ramp to force a cross
    close[20:] = np.linspace(100, 110, 20)
    df = pd.DataFrame({
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": 1000,
    }, index=idx)
    # entry bar happens somewhere in the ramp; make every later bar huge
    df.loc[df.index[25]:, "high"] = 200.0
    df.loc[df.index[25]:, "low"] = 1.0
    s = _strategy(
        indicators=[
            {"id": "ema_fast", "type": "EMA", "params": {"source": "close", "length": 3}},
            {"id": "ema_slow", "type": "EMA", "params": {"source": "close", "length": 10}},
        ],
        exit={
            "signal_exit": "none",
            # Wide bracket: nothing exits on the small bars; the first giant bar
            # (index 25) spans BOTH levels at once.
            "stop_loss": {"type": "fixed_points", "points": 50.0},
            "take_profit": {"type": "fixed_points", "points": 50.0},
        },
    )
    result = run_simple_backtest(s, df)
    assert result.trades
    assert result.ambiguous_fills >= 1
    ambiguous = [t for t in result.trades if t.ambiguous]
    assert ambiguous and all(t.exit_reason == "stop" for t in ambiguous)
