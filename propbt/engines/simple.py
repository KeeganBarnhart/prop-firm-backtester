"""Real pandas-based backtest engine. Reads strategy schema, executes on OHLCV bars.

Supported subset (strict — anything else raises):
  - Indicators: EMA (incl. higher-timeframe via params.timeframe), SMA, ATR, RSI,
    MACD, BB, VWAP (day-anchored), SUPERTREND, HIGHEST, LOWEST, ORB
    (opening-range high/low). Any indicator accepts params.offset (bars to shift
    back, i.e. Pine's `[n]` history reference).
  - Entry conditions: cross_above/cross_below, gt/lt/eq, between, rising/falling,
    inside_session — composed in and/or/not trees.
  - Session filter: market.session_filter windows (tz-aware, optional ISO-weekday
    `days`, `warmup_bars` entry delay, `flat_at_end` forced close).
  - Exits: stop_loss(atr | fixed_points | fixed_pct | indicator),
    take_profit(rr | range_rr | fixed_points | fixed_pct), signal exit on
    opposite cross.
  - Risk: fixed_fractional, fixed_contracts, fixed_dollar sizing;
    commission_per_contract and slippage_points costs.

Fill model (v1, conservative — documented contract for calibration):
  - Entries fill at the CLOSE of the signal bar (house dialect signals are
    close-based and `process_orders_on_close` semantics).
  - Stop/target levels are FROZEN at entry (ATR/indicator values from the
    entry bar) and never recomputed — matches the house `var float` idiom.
  - If a bar's range spans BOTH the stop and the target, the stop fills first
    (pessimistic) and the trade is flagged `ambiguous`; `ambiguous_fills` on
    the result counts them. A high count means the strategy's P&L depends on
    intrabar path — treat as a robustness warning, not just a stat.
  - `flat_at_end` session closes fill at the OPEN of the first bar outside the
    window (TV's close_all market order on the next bar).
  - Costs: each round trip pays 2 × qty × commission_per_contract, and
    slippage_points is applied adversely on both entry and exit.

This is a faithful bar-by-bar simulator, not a tick engine. Fidelity vs TV is
quantified by scripts/calibrate.py against the stored corpus runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from propbt.schema.strategy import Strategy

# ────────────────────────── result types ──────────────────────────


@dataclass
class Trade:
    side: str            # "long" | "short"
    entry_time: str      # ISO 8601
    entry_price: float
    exit_time: str
    exit_price: float
    qty: float
    pnl: float
    exit_reason: str     # "stop" | "target" | "signal" | "session_end" | "end_of_data"
    ambiguous: bool = False  # exit bar spanned both stop and target (stop-first applied)


@dataclass
class BacktestResult:
    strategy_id: str
    instrument: str
    timeframe: str
    bars: int
    trades: list[Trade]
    equity_curve: list[float]
    equity_times: list[str]
    starting_equity: float
    final_equity: float
    net_pnl: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    ambiguous_fills: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "bars": self.bars,
            "trades": [t.__dict__ for t in self.trades],
            "trades_count": len(self.trades),
            "equity_curve": [round(v, 2) for v in self.equity_curve],
            "equity_times": self.equity_times,
            "starting_equity": round(self.starting_equity, 2),
            "final_equity": round(self.final_equity, 2),
            "net_pnl": round(self.net_pnl, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 3) if self.profit_factor != float("inf") else None,
            "ambiguous_fills": self.ambiguous_fills,
        }


# ────────────────────────── session windows ──────────────────────────


@dataclass
class _Session:
    in_window: pd.Series      # bar is inside a session window
    entry_ok: pd.Series       # in_window minus warm-up bars — entries allowed
    session_start: pd.Series  # first in-window bar of each session
    flat_at_end: bool


def _session_of(strategy: Strategy, bars: pd.DataFrame) -> _Session:
    """Materialize market.session_filter into boolean bar masks. Bars are
    assumed stamped at bar OPEN; a window [start, end) in its own tz."""
    sf = strategy.market.session_filter
    sf = sf.root if hasattr(sf, "root") else sf
    always = pd.Series(True, index=bars.index)
    never = pd.Series(False, index=bars.index)
    sf_type = getattr(sf, "type", "always") if sf is not None else "always"
    if sf is None or sf_type == "always":
        return _Session(always, always, never, False)

    in_window = never.copy()
    for w in sf.windows:
        local = bars.index.tz_convert(str(w.tz))
        minutes = local.hour * 60 + local.minute
        sh, sm = str(w.start).split(":")
        eh, em = str(w.end).split(":")
        start_min, end_min = int(sh) * 60 + int(sm), int(eh) * 60 + int(em)
        if end_min >= start_min:
            hit = (minutes >= start_min) & (minutes < end_min)
        else:  # overnight window (e.g. 18:00-09:00)
            hit = (minutes >= start_min) | (minutes < end_min)
        if w.days:
            days = {int(_unwrap(d)) for d in w.days}
            hit &= pd.Index(local.dayofweek + 1).isin(sorted(days))
        in_window |= pd.Series(hit, index=bars.index)

    session_start = in_window & ~in_window.shift(1, fill_value=False)
    warmup = getattr(sf, "warmup_bars", None)
    max_entry = getattr(sf, "max_entry_bars", None)
    entry_ok = in_window
    if warmup or max_entry:
        session_id = session_start.cumsum()
        bar_in_session = in_window.groupby(session_id).cumsum()
        if warmup:
            entry_ok = entry_ok & (bar_in_session > int(warmup))
        if max_entry:
            entry_ok = entry_ok & (bar_in_session <= int(max_entry))
    return _Session(in_window, entry_ok, session_start, bool(getattr(sf, "flat_at_end", False)))


# ────────────────────────── indicator computation ──────────────────────────


def _compute_indicators(
    strategy: Strategy,
    bars: pd.DataFrame,
    session: "_Session | None" = None,
) -> dict[str, pd.Series]:
    """Returns a dict of indicator_id (possibly dot-suffixed for multi-output
    indicators) → pd.Series. Multi-output examples:
      MACD with id 'macd1' → 'macd1.line', 'macd1.signal', 'macd1.histogram'
      BB with id 'bb1'     → 'bb1.upper', 'bb1.middle', 'bb1.lower'
    Single-output indicators use bare id (e.g. 'ema_fast'). SUPERTREND and ORB
    select their output via params.output / params.side so condition operands
    stay dot-free (schema Operand forbids dots).
    Any indicator honours params.offset — Pine's `[n]` history reference."""
    out: dict[str, pd.Series] = {}
    for ind in strategy.indicators:
        t = ind.type.value if hasattr(ind.type, "value") else ind.type
        p = ind.params or {}
        before = set(out)

        if t == "EMA":
            src = bars[p.get("source", "close")]
            if p.get("timeframe"):
                out[ind.id] = _htf_ema(src, int(p["length"]), str(p["timeframe"]))
            else:
                out[ind.id] = src.ewm(span=int(p["length"]), adjust=False).mean()

        elif t == "SMA":
            src = bars[p.get("source", "close")]
            out[ind.id] = src.rolling(int(p["length"])).mean()

        elif t == "ATR":
            length = int(p["length"])
            high, low, close = bars["high"], bars["low"], bars["close"]
            prev_close = close.shift(1)
            tr = pd.concat(
                [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
                axis=1,
            ).max(axis=1)
            out[ind.id] = tr.ewm(alpha=1 / length, adjust=False).mean()

        elif t == "RSI":
            length = int(p["length"])
            src = bars[p.get("source", "close")]
            delta = src.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)
            avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0.0, float("nan"))
            out[ind.id] = (100 - (100 / (1 + rs))).fillna(50.0)

        elif t == "MACD":
            src = bars[p.get("source", "close")]
            fast = int(p.get("fast", 12))
            slow = int(p.get("slow", 26))
            signal = int(p.get("signal", 9))
            ema_fast = src.ewm(span=fast, adjust=False).mean()
            ema_slow = src.ewm(span=slow, adjust=False).mean()
            line = ema_fast - ema_slow
            sig = line.ewm(span=signal, adjust=False).mean()
            out[f"{ind.id}.line"] = line
            out[f"{ind.id}.signal"] = sig
            out[f"{ind.id}.histogram"] = line - sig

        elif t == "BB":
            src = bars[p.get("source", "close")]
            length = int(p.get("length", 20))
            mult = float(p.get("mult", 2.0))
            mid = src.rolling(length).mean()
            std = src.rolling(length).std(ddof=0)
            out[f"{ind.id}.middle"] = mid
            out[f"{ind.id}.upper"] = mid + mult * std
            out[f"{ind.id}.lower"] = mid - mult * std

        elif t == "HIGHEST":
            src = bars[p.get("source", "high")]
            out[ind.id] = src.rolling(int(p["length"])).max()

        elif t == "LOWEST":
            src = bars[p.get("source", "low")]
            out[ind.id] = src.rolling(int(p["length"])).min()

        elif t == "VWAP":
            # Day-anchored (exchange tz): cumulative typical-price × volume /
            # cumulative volume, reset when the local date changes.
            tz = str(p.get("tz", "America/New_York"))
            tp = (bars["high"] + bars["low"] + bars["close"]) / 3
            day = pd.Series(bars.index.tz_convert(tz).date, index=bars.index)
            grp = day.ne(day.shift()).cumsum()
            pv = (tp * bars["volume"]).groupby(grp).cumsum()
            vv = bars["volume"].groupby(grp).cumsum()
            out[ind.id] = pv / vv.replace(0, float("nan"))

        elif t == "SUPERTREND":
            line, direction = _supertrend(bars, float(p["factor"]), int(p["atr_length"]))
            out[ind.id] = direction if p.get("output", "line") == "direction" else line

        elif t == "ORB":
            if session is None or not session.session_start.any():
                raise ValueError(
                    "ORB indicator requires a session_window filter on market.session_filter"
                )
            hi, lo = _opening_range(bars, session, int(p["bars"]))
            out[ind.id] = lo if p.get("side", "high") == "low" else hi

        else:
            raise NotImplementedError(
                f"simple engine doesn't yet support indicator type {t!r}. "
                f"Supported: EMA, SMA, ATR, RSI, MACD, BB, VWAP, SUPERTREND, "
                f"HIGHEST, LOWEST, ORB."
            )

        offset = int(p.get("offset", 0) or 0)
        if offset:
            for key in set(out) - before:
                out[key] = out[key].shift(offset)
    return out


def _htf_ema(src: pd.Series, length: int, timeframe: str) -> pd.Series:
    """EMA computed on a higher timeframe, projected back with last-completed-bar
    semantics (request.security lookahead_off): the value stamped at each HTF
    bar's END is forward-filled onto the base index, so no base bar ever sees
    an HTF bar that hasn't closed yet."""
    freq = _TF_FREQ.get(timeframe)
    if freq is None:
        raise NotImplementedError(f"HTF EMA timeframe {timeframe!r}; known: {sorted(_TF_FREQ)}")
    htf_close = src.resample(freq, closed="right", label="right").last().dropna()
    ema = htf_close.ewm(span=length, adjust=False).mean()
    return ema.reindex(src.index.union(ema.index)).ffill().reindex(src.index)


_TF_FREQ = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "60min", "2h": "120min", "4h": "240min", "1d": "1D",
}


def _supertrend(bars: pd.DataFrame, factor: float, atr_length: int) -> tuple[pd.Series, pd.Series]:
    """Classic Supertrend. Returns (line, direction) with Pine's convention:
    direction −1 = uptrend (line below price), +1 = downtrend."""
    high, low, close = bars["high"], bars["low"], bars["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / atr_length, adjust=False).mean()
    hl2 = (high + low) / 2
    upper = (hl2 + factor * atr).to_numpy()
    lower = (hl2 - factor * atr).to_numpy()
    c = close.to_numpy()

    n = len(bars)
    fu = upper.copy()  # final bands
    fl = lower.copy()
    direction = np.ones(n)
    line = np.full(n, np.nan)
    for i in range(1, n):
        fu[i] = upper[i] if (upper[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lower[i] if (lower[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if direction[i - 1] < 0:  # uptrend: flip when close breaks the lower band
            direction[i] = 1.0 if c[i] < fl[i] else -1.0
        else:                      # downtrend: flip when close breaks the upper band
            direction[i] = -1.0 if c[i] > fu[i] else 1.0
        line[i] = fl[i] if direction[i] < 0 else fu[i]
    idx = bars.index
    return pd.Series(line, index=idx), pd.Series(direction, index=idx)


def _opening_range(
    bars: pd.DataFrame, session: "_Session", orb_bars: int
) -> tuple[pd.Series, pd.Series]:
    """Opening-range high/low: running extreme of the first `orb_bars` in-session
    bars, frozen for the rest of the session; NaN before a session's range starts."""
    in_w = session.in_window.to_numpy()
    starts = session.session_start.to_numpy()
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    n = len(bars)
    hi = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    count = 0
    cur_hi = np.nan
    cur_lo = np.nan
    for i in range(n):
        if starts[i]:
            count = 0
            cur_hi = np.nan
            cur_lo = np.nan
        if in_w[i] and count < orb_bars:
            count += 1
            cur_hi = high[i] if np.isnan(cur_hi) else max(cur_hi, high[i])
            cur_lo = low[i] if np.isnan(cur_lo) else min(cur_lo, low[i])
        hi[i] = cur_hi
        lo[i] = cur_lo
    idx = bars.index
    return pd.Series(hi, index=idx), pd.Series(lo, index=idx)


# ────────────────────────── condition evaluation ──────────────────────────


def _unwrap(value: Any) -> Any:
    """Drill through RootModel wrappers until we get a primitive."""
    while hasattr(value, "root"):
        value = value.root
    return value


def _operand_series(operand: Any, bars: pd.DataFrame, indicators: dict[str, pd.Series]) -> pd.Series:
    operand = _unwrap(operand)
    if isinstance(operand, (int, float)):
        return pd.Series(float(operand), index=bars.index)
    s = str(operand)
    # Indicator id (possibly dot-suffixed: 'macd1.signal', 'bb1.upper').
    if s in indicators:
        return indicators[s]
    # Bare id for a multi-output indicator → default sub-key.
    if f"{s}.line" in indicators:
        return indicators[f"{s}.line"]    # MACD default = line
    if f"{s}.middle" in indicators:
        return indicators[f"{s}.middle"]  # BB default = middle
    if s in bars.columns:
        return bars[s]
    if s == "hl2":
        return (bars["high"] + bars["low"]) / 2
    if s == "hlc3":
        return (bars["high"] + bars["low"] + bars["close"]) / 3
    if s == "ohlc4":
        return (bars["open"] + bars["high"] + bars["low"] + bars["close"]) / 4
    raise ValueError(f"unknown operand {operand!r}")


def _eval_condition(
    cond: Any,
    bars: pd.DataFrame,
    indicators: dict[str, pd.Series],
    in_session: pd.Series | None = None,
) -> pd.Series:
    cond = cond.root if hasattr(cond, "root") else cond
    t = cond.type.value if hasattr(cond.type, "value") else cond.type
    if t in ("and", "or", "not"):
        sub = [_eval_condition(c, bars, indicators, in_session) for c in cond.conditions]
        if t == "and":
            result = sub[0]
            for s in sub[1:]:
                result = result & s
            return result
        if t == "or":
            result = sub[0]
            for s in sub[1:]:
                result = result | s
            return result
        return ~sub[0]
    if t == "inside_session":
        if in_session is None:
            return pd.Series(True, index=bars.index)
        return in_session
    left = _operand_series(cond.left, bars, indicators)
    if t == "between":
        lower = _operand_series(cond.lower, bars, indicators)
        upper = _operand_series(cond.upper, bars, indicators)
        return (left >= lower) & (left <= upper)
    if t in ("rising", "falling"):
        lookback = int(cond.lookback or 1)
        return left > left.shift(lookback) if t == "rising" else left < left.shift(lookback)
    right = _operand_series(cond.right, bars, indicators)
    if t == "cross_above":
        return (left > right) & (left.shift(1) <= right.shift(1))
    if t == "cross_below":
        return (left < right) & (left.shift(1) >= right.shift(1))
    if t == "gt":
        return left > right
    if t == "lt":
        return left < right
    if t == "eq":
        return left == right
    raise NotImplementedError(f"condition type {t!r} not implemented in simple engine")


# ────────────────────────── exit pricing ──────────────────────────


@dataclass
class _ExitLevels:
    stop: float | None
    target: float | None


def _exit_levels(
    strategy: Strategy,
    side: str,
    entry_price: float,
    atr_at_entry: float | None,
    ind_at_entry: "dict[str, float] | None" = None,
) -> _ExitLevels:
    """Stop/target prices FROZEN at entry — ATR and indicator values are read
    from the entry bar and never recomputed (the house `var float` idiom)."""
    ex = strategy.exit
    stop = None
    target = None

    def _ind(indicator_id: str, what: str) -> float:
        val = (ind_at_entry or {}).get(indicator_id)
        if val is None or not np.isfinite(val):
            raise ValueError(f"{what} references indicator {indicator_id!r} with no value at entry")
        return float(val)

    sl = ex.stop_loss.root if (ex.stop_loss is not None and hasattr(ex.stop_loss, "root")) else ex.stop_loss
    tp = ex.take_profit.root if (ex.take_profit is not None and hasattr(ex.take_profit, "root")) else ex.take_profit

    if sl is not None:
        st = sl.type.value if hasattr(sl.type, "value") else sl.type
        if st == "atr":
            if atr_at_entry is None:
                raise ValueError("ATR stop requires an ATR indicator in strategy.indicators")
            dist = atr_at_entry * float(sl.atr_multiplier)
        elif st == "fixed_points":
            dist = float(sl.points)
        elif st == "fixed_pct":
            dist = entry_price * float(sl.pct) / 100.0
        elif st == "indicator":
            # Stop AT an indicator level (e.g. opening-range boundary), frozen
            # at entry. Distance is derived, sign checked below.
            ind_id = sl.indicator_id
            if side == "short" and getattr(sl, "indicator_id_short", None):
                ind_id = sl.indicator_id_short
            level = _ind(ind_id, "indicator stop")
            dist = (entry_price - level) if side == "long" else (level - entry_price)
            if dist <= 0:
                dist = abs(dist) or entry_price * 0.001  # degenerate: level on wrong side
        elif st == "range_level":
            # Beyond a band boundary by a fraction of its width (buffered ORB stop).
            upper = _ind(sl.upper, "range_level stop")
            lower = _ind(sl.lower, "range_level stop")
            width = abs(upper - lower)
            r = float(getattr(sl, "offset_ratio", None) or 0.0)
            level = (lower - r * width) if side == "long" else (upper + r * width)
            dist = (entry_price - level) if side == "long" else (level - entry_price)
            if dist <= 0:
                dist = abs(dist) or entry_price * 0.001
        else:
            raise NotImplementedError(f"stop_loss type {st!r} not implemented")
        stop = entry_price - dist if side == "long" else entry_price + dist

    if tp is not None:
        tt = tp.type.value if hasattr(tp.type, "value") else tp.type
        if tt == "rr":
            if stop is None:
                raise ValueError("rr take_profit requires a stop_loss to anchor the R distance")
            r_dist = abs(entry_price - stop)
            target_dist = r_dist * float(tp.ratio)
            target = entry_price + target_dist if side == "long" else entry_price - target_dist
        elif tt == "atr":
            if atr_at_entry is None:
                raise ValueError("ATR take_profit requires an ATR indicator in strategy.indicators")
            d = atr_at_entry * float(tp.atr_multiplier)
            target = entry_price + d if side == "long" else entry_price - d
        elif tt == "fixed_points":
            target = entry_price + float(tp.points) if side == "long" else entry_price - float(tp.points)
        elif tt == "fixed_pct":
            d = entry_price * float(tp.pct) / 100.0
            target = entry_price + d if side == "long" else entry_price - d
        elif tt == "range_rr":
            # Target = entry ± (band width at entry) × ratio — the ORB idiom
            # where R is the opening range, not the stop distance.
            width = _ind(tp.upper, "range_rr target") - _ind(tp.lower, "range_rr target")
            d = abs(width) * float(tp.ratio)
            target = entry_price + d if side == "long" else entry_price - d
        else:
            raise NotImplementedError(f"take_profit type {tt!r} not implemented")

    return _ExitLevels(stop=stop, target=target)


# ────────────────────────── sizing ──────────────────────────


def _position_size(strategy: Strategy, equity: float, stop_dist: float, point_value: float) -> float:
    risk = strategy.risk
    model = risk.model.value if hasattr(risk.model, "value") else risk.model
    if model == "fixed_fractional":
        risk_dollar = equity * float(risk.risk_per_trade or 0.01)
        per_unit_risk = stop_dist * point_value
        if per_unit_risk <= 0:
            return 0.0
        return max(1.0, round(risk_dollar / per_unit_risk))
    if model == "fixed_dollar":
        return 1.0  # placeholder, would need contract spec
    if model == "fixed_contracts":
        return float(risk.fixed_contracts or 1)
    return 1.0


# ────────────────────────── instrument metadata ──────────────────────────


def _point_value(instrument: str) -> float:
    from propbt.instruments import point_value

    return point_value(instrument)


# ────────────────────────── main loop ──────────────────────────


def run_simple_backtest(
    strategy: Strategy,
    bars: pd.DataFrame,
    *,
    starting_equity: float = 50_000.0,
    train_test_split: float | None = None,
) -> "BacktestResult | tuple[BacktestResult, BacktestResult, dict]":
    """Run backtest. If `train_test_split` is set (e.g. 0.7), runs in-sample on
    the first portion of bars, out-of-sample on the rest, and returns both
    results plus a robustness diagnosis dict."""
    if train_test_split is not None:
        if not (0.1 <= train_test_split <= 0.95):
            raise ValueError("train_test_split must be between 0.1 and 0.95")
        cut = int(len(bars) * train_test_split)
        train_bars = bars.iloc[:cut]
        test_bars = bars.iloc[cut:]
        train_r = _run_one(strategy, train_bars, starting_equity)
        # Test starts with whatever equity ended train (more realistic continuity)
        test_r = _run_one(strategy, test_bars, train_r.final_equity)
        diag = _diagnose_oos(train_r, test_r)
        return train_r, test_r, diag

    return _run_one(strategy, bars, starting_equity)


def _diagnose_oos(train: "BacktestResult", test: "BacktestResult") -> dict:
    """Flag suspect out-of-sample (OOS) performance vs. in-sample (IS)."""
    warnings: list[str] = []
    is_pf = train.profit_factor if train.profit_factor != float("inf") else 99.0
    oos_pf = test.profit_factor if test.profit_factor != float("inf") else 99.0

    if oos_pf < 1.0 and is_pf >= 1.0:
        warnings.append(
            f"OOS profit factor {oos_pf:.2f} < 1.0 while IS PF was {is_pf:.2f} — "
            f"strategy likely overfit to the training period."
        )
    if test.max_drawdown > train.max_drawdown * 1.5 and train.max_drawdown > 0:
        warnings.append(
            f"OOS max drawdown ${test.max_drawdown:,.0f} is {test.max_drawdown / train.max_drawdown:.1f}× the IS drawdown — "
            f"regime shift or parameter sensitivity."
        )
    if train.win_rate > 0 and test.win_rate < train.win_rate * 0.7:
        warnings.append(
            f"OOS win rate {test.win_rate * 100:.1f}% is well below IS win rate {train.win_rate * 100:.1f}% — "
            f"signal degraded out-of-sample."
        )
    severity = "high" if warnings else "ok"
    if oos_pf < 0.7:
        severity = "high"
    elif warnings and "ok" == severity:
        severity = "warn"

    return {
        "severity": severity,
        "warnings": warnings,
        "is_pf": is_pf if is_pf != 99.0 else None,
        "oos_pf": oos_pf if oos_pf != 99.0 else None,
        "is_dd": train.max_drawdown,
        "oos_dd": test.max_drawdown,
        "is_wr": train.win_rate,
        "oos_wr": test.win_rate,
        "is_net_pnl": train.net_pnl,
        "oos_net_pnl": test.net_pnl,
    }


def _run_one(
    strategy: Strategy,
    bars: pd.DataFrame,
    starting_equity: float,
) -> BacktestResult:
    session = _session_of(strategy, bars)
    indicators = _compute_indicators(strategy, bars, session)

    long_signal = (
        _eval_condition(strategy.entry.root.long, bars, indicators, session.in_window)
        if strategy.entry.root.long is not None
        else pd.Series(False, index=bars.index)
    )
    short_signal = (
        _eval_condition(strategy.entry.root.short, bars, indicators, session.in_window)
        if strategy.entry.root.short is not None
        else pd.Series(False, index=bars.index)
    )
    # Session gate: entries only inside a window (past any warm-up); signals
    # keep firing outside for the opposite-cross exit check.
    long_entry_ok = long_signal & session.entry_ok
    short_entry_ok = short_signal & session.entry_ok

    max_per_session = getattr(strategy.entry.root, "max_entries_per_session", None)
    session_starts = session.session_start

    atr_id = next(
        (ind.id for ind in strategy.indicators
         if (ind.type.value if hasattr(ind.type, "value") else ind.type) == "ATR"),
        None,
    )
    atr_series = indicators.get(atr_id) if atr_id else None

    instrument = strategy.market.instrument
    point_value = _point_value(instrument)
    commission = float(getattr(strategy.risk, "commission_per_contract", None) or 0.0)
    slippage = float(getattr(strategy.risk, "slippage_points", None) or 0.0)

    equity = starting_equity
    peak = equity
    max_dd = 0.0
    equity_curve: list[float] = [equity]
    equity_times: list[str] = [bars.index[0].isoformat()]
    trades: list[Trade] = []
    ambiguous_fills = 0
    entries_this_session = {"long": 0, "short": 0}

    position = None  # dict | None

    for i in range(1, len(bars)):
        ts = bars.index[i]
        bar = bars.iloc[i]
        price = float(bar["close"])
        high = float(bar["high"])
        low = float(bar["low"])

        if bool(session_starts.iloc[i]):
            entries_this_session = {"long": 0, "short": 0}

        # Check exits on existing position using this bar's range.
        if position is not None:
            exit_price = None
            exit_reason = None
            ambiguous = False

            # Session close first: the position must not see the out-of-session
            # bar's range. close_all is a market order — fills at this bar's open.
            if session.flat_at_end and not bool(session.in_window.iloc[i]):
                exit_price = float(bar["open"])
                exit_reason = "session_end"
            elif position["side"] == "long":
                stop_hit = position["stop"] is not None and low <= position["stop"]
                target_hit = position["target"] is not None and high >= position["target"]
                if stop_hit:
                    exit_price = position["stop"]
                    exit_reason = "stop"
                    ambiguous = target_hit  # bar spans both — stop-first, flagged
                elif target_hit:
                    exit_price = position["target"]
                    exit_reason = "target"
            else:  # short
                stop_hit = position["stop"] is not None and high >= position["stop"]
                target_hit = position["target"] is not None and low <= position["target"]
                if stop_hit:
                    exit_price = position["stop"]
                    exit_reason = "stop"
                    ambiguous = target_hit
                elif target_hit:
                    exit_price = position["target"]
                    exit_reason = "target"

            # Signal exit (opposite cross) — checked if no stop/target hit.
            if exit_price is None and strategy.exit.signal_exit == "opposite":
                opp = short_signal.iloc[i] if position["side"] == "long" else long_signal.iloc[i]
                if bool(opp):
                    exit_price = price
                    exit_reason = "signal"

            if exit_price is not None:
                side = position["side"]
                qty = position["qty"]
                entry = position["entry_price"]
                pnl_points = (exit_price - entry) if side == "long" else (entry - exit_price)
                # Costs: adverse slippage on both fills + commission per contract per fill.
                pnl = (pnl_points - 2 * slippage) * qty * point_value - 2 * commission * qty
                equity += pnl
                if ambiguous:
                    ambiguous_fills += 1
                trades.append(Trade(
                    side=side,
                    entry_time=position["entry_time"],
                    entry_price=entry,
                    exit_time=ts.isoformat(),
                    exit_price=exit_price,
                    qty=qty,
                    pnl=round(pnl, 2),
                    exit_reason=exit_reason,
                    ambiguous=ambiguous,
                ))
                # Equity curve only records points when a trade closes — sparse, trade-indexed.
                equity_curve.append(equity)
                equity_times.append(ts.isoformat())
                if equity > peak:
                    peak = equity
                dd = peak - equity
                if dd > max_dd:
                    max_dd = dd
                position = None

        # Open new position if flat and a signal fires.
        if position is None:
            want_long = bool(long_entry_ok.iloc[i])
            want_short = bool(short_entry_ok.iloc[i])
            if max_per_session is not None:
                if want_long and entries_this_session["long"] >= int(max_per_session):
                    want_long = False
                if want_short and entries_this_session["short"] >= int(max_per_session):
                    want_short = False
            if want_long or want_short:
                side = "long" if want_long else "short"
                entry_price = price
                atr_now = float(atr_series.iloc[i]) if atr_series is not None else None
                ind_now = {k: float(v.iloc[i]) for k, v in indicators.items()}
                levels = _exit_levels(strategy, side, entry_price, atr_now, ind_now)
                stop_dist = abs(entry_price - levels.stop) if levels.stop is not None else 0.0
                qty = _position_size(strategy, equity, stop_dist, point_value)
                if qty > 0:
                    entries_this_session[side] += 1
                    position = {
                        "side": side,
                        "entry_time": ts.isoformat(),
                        "entry_price": entry_price,
                        "stop": levels.stop,
                        "target": levels.target,
                        "qty": qty,
                    }

    # Close any open position at end-of-data
    if position is not None:
        side = position["side"]
        entry = position["entry_price"]
        last_price = float(bars["close"].iloc[-1])
        pnl_points = (last_price - entry) if side == "long" else (entry - last_price)
        pnl = (pnl_points - 2 * slippage) * position["qty"] * point_value \
            - 2 * commission * position["qty"]
        equity += pnl
        trades.append(Trade(
            side=side,
            entry_time=position["entry_time"],
            entry_price=entry,
            exit_time=bars.index[-1].isoformat(),
            exit_price=last_price,
            qty=position["qty"],
            pnl=round(pnl, 2),
            exit_reason="end_of_data",
        ))
        equity_curve.append(equity)
        equity_times.append(bars.index[-1].isoformat())
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl <= 0]
    win_rate = (len(wins) / len(trades)) if trades else 0.0
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    return BacktestResult(
        strategy_id=strategy.id,
        instrument=strategy.market.instrument,
        timeframe=strategy.market.timeframe.value if hasattr(strategy.market.timeframe, "value") else strategy.market.timeframe,
        bars=len(bars),
        trades=trades,
        equity_curve=equity_curve,
        equity_times=equity_times,
        starting_equity=starting_equity,
        final_equity=equity,
        net_pnl=equity - starting_equity,
        max_drawdown=max_dd,
        win_rate=win_rate,
        profit_factor=pf,
        ambiguous_fills=ambiguous_fills,
    )
