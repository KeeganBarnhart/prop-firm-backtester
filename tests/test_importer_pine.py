"""Importer idiom tests — one test per house-dialect construct.

Grows as more of the Pine dialect is supported. Each
test feeds a minimal inline Pine source through import_pine and asserts the
schema JSON it emits; corpus-wide acceptance lives in test_corpus_import.py.
"""

from __future__ import annotations

from propbt.importers.pine import import_pine
from propbt.schema.strategy import Strategy

MULTILINE_DECL = '''\
//@version=5
strategy(
    title         = "Multi-line Decl",
    overlay       = true,
    initial_capital = 50000,
    default_qty_type = strategy.fixed,
    default_qty_value = 2,
    pyramiding    = 0,
    commission_type = strategy.commission.cash_per_contract,
    commission_value = 0.62,
    slippage      = 1,
    process_orders_on_close = true)

emaFast = ta.ema(close, 9)
emaSlow = ta.ema(close, 21)
atrVal  = ta.atr(14)

longSignal = ta.crossover(emaFast, emaSlow)

if longSignal
    stopDist  = atrVal * 2.0
    stopPrice = close - stopDist
    tpPrice   = close + stopDist * 1.5
    strategy.entry("long", strategy.long)
    strategy.exit("long-exit", from_entry="long", stop=stopPrice, limit=tpPrice)
'''


def test_multiline_strategy_decl_imports():
    result = import_pine(MULTILINE_DECL)
    assert result.ok, [e.message for e in result.errors]
    assert result.detected_name == "Multi-line Decl"
    Strategy.model_validate(result.strategy_json)


def test_header_qty_becomes_fixed_contracts():
    result = import_pine(MULTILINE_DECL)
    assert result.ok
    risk = result.strategy_json["risk"]
    assert risk["model"] == "fixed_contracts"
    assert risk["fixed_contracts"] == 2


def test_missing_qty_falls_back_to_fixed_fractional():
    src = MULTILINE_DECL.replace("    default_qty_value = 2,\n", "")
    result = import_pine(src)
    assert result.ok
    assert result.strategy_json["risk"]["model"] == "fixed_fractional"


def test_comment_with_parens_does_not_trigger_joining():
    src = MULTILINE_DECL.replace(
        "emaFast = ta.ema(close, 9)",
        "// note (this comment has an open paren\nemaFast = ta.ema(close, 9)",
    )
    result = import_pine(src)
    assert result.ok, [e.message for e in result.errors]


# ── P3: sessions, conditions, frozen levels ──────────────────────────────────

SESSION_STRATEGY = '''\
//@version=6
strategy("Session Test", overlay=true, initial_capital=50000, commission_type=strategy.commission.cash_per_contract, commission_value=0.62, default_qty_value=1)

fastLen  = input.int(9, "Fast")
slowLen  = input.int(21, "Slow")
atrLen   = input.int(14, "ATR")
stopMult = input.float(1.5, "Stop Mult")
tpMult   = input.float(3.0, "TP Mult")
useVwap  = input.bool(true, "Use VWAP")

inSession = not na(time(timeframe.period, "0930-1200:23456", "America/New_York"))

fastEma = ta.ema(close, fastLen)
slowEma = ta.ema(close, slowLen)
atrVal  = ta.atr(atrLen)
vwapVal = ta.vwap
crossUp = ta.crossover(fastEma, slowEma)
crossDn = ta.crossunder(fastEma, slowEma)
vwapBull = close > vwapVal
vwapBear = close < vwapVal

var int sessionBar = 0
if inSession and not inSession[1]
    sessionBar := 0
else if inSession
    sessionBar += 1
warmedUp = sessionBar >= 6

var float longStop = na
var float longTp   = na
var float shortStop = na
var float shortTp   = na

longCond  = barstate.isconfirmed and inSession and warmedUp and crossUp and (not useVwap or vwapBull)
shortCond = barstate.isconfirmed and inSession and warmedUp and crossDn and (not useVwap or vwapBear)

if longCond and strategy.position_size == 0
    longStop := close - atrVal * stopMult
    longTp   := close + atrVal * tpMult
    strategy.entry("Long", strategy.long)

if shortCond and strategy.position_size == 0
    shortStop := close + atrVal * stopMult
    shortTp   := close - atrVal * tpMult
    strategy.entry("Short", strategy.short)

if strategy.position_size > 0
    strategy.exit("LX", from_entry="Long", stop=longStop, limit=longTp)
if strategy.position_size < 0
    strategy.exit("SX", from_entry="Short", stop=shortStop, limit=shortTp)

if not inSession and strategy.position_size != 0
    strategy.close_all(comment="EOD")
'''


def test_session_window_with_days_and_warmup():
    r = import_pine(SESSION_STRATEGY)
    assert r.ok, [e.message for e in r.errors]
    sf = r.strategy_json["market"]["session_filter"]
    assert sf["type"] == "session_window"
    assert sf["flat_at_end"] is True
    assert sf["warmup_bars"] == 6
    w = sf["windows"][0]
    assert (w["start"], w["end"], w["tz"]) == ("09:30", "12:00", "America/New_York")
    assert w["days"] == [1, 2, 3, 4, 5]  # TV 23456 (Mon-Fri) -> ISO
    Strategy.model_validate(r.strategy_json)


def test_bool_input_folds_and_atr_target():
    r = import_pine(SESSION_STRATEGY)
    assert r.ok
    # useVwap=true bakes the vwap comparison into the condition tree
    tree = str(r.strategy_json["entry"]["long"])
    assert "vwap_val" in tree
    ex = r.strategy_json["exit"]
    assert ex["stop_loss"] == {"type": "atr", "atr_length": 14, "atr_multiplier": 1.5}
    assert ex["take_profit"] == {"type": "atr", "atr_length": 14, "atr_multiplier": 3.0}
    # bool input rides along in the sweep grid
    assert {"name": "useVwap", "kind": "bool", "default": True} in r.pine_inputs


def test_bool_input_false_disables_branch():
    src = SESSION_STRATEGY.replace('input.bool(true, "Use VWAP")', 'input.bool(false, "Use VWAP")')
    r = import_pine(src)
    assert r.ok
    assert "vwap_val" not in str(r.strategy_json["entry"]["long"])


def test_eod_tail_window_truncates_session():
    src = '''//@version=6
strategy("EOD Tail", overlay=true, default_qty_value=1)
atrVal = ta.atr(14)
emaF = ta.ema(close, 9)
emaS = ta.ema(close, 21)
inSession = not na(time(timeframe.period, "0930-1600", "America/New_York"))
nearClose = not na(time(timeframe.period, "1555-1600", "America/New_York"))
sig = ta.crossover(emaF, emaS)
if sig and inSession
    stopDist = atrVal * 2.0
    stopPrice = close - stopDist
    tpPrice = close + stopDist * 1.5
    strategy.entry("Long", strategy.long)
    strategy.exit("X", from_entry="Long", stop=stopPrice, limit=tpPrice)
if nearClose and strategy.position_size != 0
    strategy.close_all(comment="EOD")
'''
    r = import_pine(src)
    assert r.ok, [e.message for e in r.errors]
    sf = r.strategy_json["market"]["session_filter"]
    assert sf["windows"][0]["end"] == "15:55"
    assert sf["flat_at_end"] is True


def test_input_session_and_entry_kwargs():
    src = '''//@version=6
strategy("Input Session", overlay=true, default_qty_value=1)
sessionStr = input.session("0930-1200", title="Window")
atrVal = ta.atr(14)
prevHigh = ta.highest(high, 20)[1]
inSession = not na(time(timeframe.period, sessionStr, "America/New_York"))
sessionEnd = inSession[1] and not inSession
if inSession and close > prevHigh and strategy.position_size == 0
    stopPrice := close - 1.5 * atrVal
    tpPrice := close + 2.0 * (close - stopPrice)
    strategy.entry("Long", strategy.long, qty=1, comment="Breakout")
if strategy.position_size > 0
    strategy.exit("X", from_entry="Long", stop=stopPrice, limit=tpPrice)
if sessionEnd
    strategy.close_all(comment="End")
'''
    src = src.replace("stopPrice :=", "stopPrice =").replace("tpPrice :=", "tpPrice =")
    r = import_pine(src)
    assert r.ok, [e.message for e in r.errors]
    sf = r.strategy_json["market"]["session_filter"]
    assert sf["windows"][0] == {"start": "09:30", "end": "12:00", "tz": "America/New_York"}
    assert sf["flat_at_end"] is True
    ex = r.strategy_json["exit"]
    assert ex["stop_loss"] == {"type": "atr", "atr_length": 14, "atr_multiplier": 1.5}
    assert ex["take_profit"] == {"type": "rr", "ratio": 2.0}
    # prevHigh keeps its [1] history reference as an offset param
    hh = [i for i in r.strategy_json["indicators"] if i["type"] == "HIGHEST"][0]
    assert hh["params"]["offset"] == 1


def test_ternary_session_guard_folds():
    src = '''//@version=6
strategy("Ternary", overlay=true, default_qty_value=1)
useSession = input.bool(true, "Filter")
atrVal = ta.atr(14)
emaF = ta.ema(close, 9)
emaS = ta.ema(close, 21)
inSession = useSession ? not na(time(timeframe.period, "0930-1600", "America/New_York")) : true
sig = ta.crossover(emaF, emaS)
if sig and inSession
    stopDist = atrVal * 2.0
    stopPrice = close - stopDist
    tpPrice = close + stopDist * 1.5
    strategy.entry("Long", strategy.long)
    strategy.exit("X", from_entry="Long", stop=stopPrice, limit=tpPrice)
'''
    r = import_pine(src)
    assert r.ok, [e.message for e in r.errors]
    assert r.strategy_json["market"]["session_filter"]["type"] == "session_window"


def test_commission_and_slippage_flow_to_risk():
    r = import_pine(SESSION_STRATEGY)
    assert r.ok
    assert r.strategy_json["risk"]["commission_per_contract"] == 0.62


# ── P4/P5: opening-range family, supertrend, HTF EMA ─────────────────────────

ORB_STRATEGY = '''\
//@version=6
strategy("ORB Test", overlay=true, default_qty_value=1, commission_type=strategy.commission.cash_per_contract, commission_value=0.62)

orbBars     = input.int(6, "ORB Bars")
maxEntryBar = input.int(20, "Max Entry Bar")
rrRatio     = input.float(2.0, "R:R")

inSession = not na(time(timeframe.period, "0930-1600", "America/New_York"))
sessionStart = not inSession[1] and inSession

var int   barsSinceOpen = 0
var float orbHigh       = na
var float orbLow        = na
var bool  tradedLong    = false

if sessionStart
    barsSinceOpen := 1
    orbHigh       := high
    orbLow        := low
    tradedLong    := false
else if inSession
    barsSinceOpen += 1
    if barsSinceOpen <= orbBars
        orbHigh := math.max(orbHigh, high)
        orbLow  := math.min(orbLow,  low)

orbDone = inSession and barsSinceOpen > orbBars and barsSinceOpen <= maxEntryBar

longEntry = orbDone and barstate.isconfirmed and not tradedLong and strategy.position_size == 0 and close > orbHigh

var float stopLong = na
var float targLong = na

if longEntry
    _range     = orbHigh - orbLow
    stopLong  := orbLow
    targLong  := close + _range * rrRatio
    tradedLong := true
    strategy.entry("Long", strategy.long)

if strategy.position_size > 0
    strategy.exit("LX", "Long", stop=stopLong, limit=targLong)

if not inSession and strategy.position_size != 0
    strategy.close_all(comment="EOD")
'''


def test_orb_construction_and_exits():
    r = import_pine(ORB_STRATEGY)
    assert r.ok, [e.message for e in r.errors]
    inds = {i["id"]: i for i in r.strategy_json["indicators"]}
    assert inds["orb_high"]["type"] == "ORB"
    assert inds["orb_high"]["params"] == {"bars": 6, "side": "high"}
    assert inds["orb_low"]["params"] == {"bars": 6, "side": "low"}
    ex = r.strategy_json["exit"]
    assert ex["stop_loss"] == {"type": "indicator", "indicator_id": "orb_low"}
    assert ex["take_profit"] == {"type": "range_rr", "ratio": 2.0,
                                 "upper": "orb_high", "lower": "orb_low"}
    sf = r.strategy_json["market"]["session_filter"]
    assert sf["warmup_bars"] == 6          # entries only after the range forms
    assert sf["max_entry_bars"] == 20      # and only in the first 20 bars
    assert sf["flat_at_end"] is True
    assert r.strategy_json["entry"]["max_entries_per_session"] == 1
    Strategy.model_validate(r.strategy_json)


def test_orb_buffered_stop_range_level():
    src = ORB_STRATEGY.replace(
        "    _range     = orbHigh - orbLow\n    stopLong  := orbLow\n    targLong  := close + _range * rrRatio\n",
        "    _range     = orbHigh - orbLow\n    _stop      = orbLow - 0.25 * _range\n"
        "    _dist      = close - _stop\n    stopLong  := _stop\n    targLong  := close + _dist * rrRatio\n",
    )
    r = import_pine(src)
    assert r.ok, [e.message for e in r.errors]
    ex = r.strategy_json["exit"]
    assert ex["stop_loss"] == {"type": "range_level", "upper": "orb_high",
                               "lower": "orb_low", "offset_ratio": 0.25}
    assert ex["take_profit"] == {"type": "rr", "ratio": 2.0}


def test_supertrend_destructure_and_flip_to_cross():
    src = '''//@version=6
strategy("ST", overlay=true, default_qty_value=1)
atrVal = ta.atr(10)
[st_line, st_dir] = ta.supertrend(3.0, 10)
bull_now  = not na(st_dir) and st_dir < 0
bear_now  = not na(st_dir) and st_dir > 0
bull_flip = bull_now and bear_now[1]
if barstate.isconfirmed
    if bull_flip
        stopP = close - atrVal * 1.0
        tpP   = close + atrVal * 2.5
        strategy.entry("Long", strategy.long)
        strategy.exit("LX", from_entry="Long", stop=stopP, limit=tpP)
'''
    r = import_pine(src)
    assert r.ok, [e.message for e in r.errors]
    tree = r.strategy_json["entry"]["long"]
    assert tree == {"type": "cross_below", "left": "st_dir", "right": 0.0}
    inds = {i["id"]: i for i in r.strategy_json["indicators"]}
    assert inds["st_dir"]["params"]["output"] == "direction"


def test_completion_flag_becomes_warmup():
    src = '''//@version=6
strategy("Completion", overlay=true, default_qty_value=1)
num_bars = input.int(6, "OR Bars")
atrVal = ta.atr(14)
is_rth    = not na(time(timeframe.period, "0930-1600", "America/New_York"))
rth_start = is_rth and not is_rth[1]
var float or_high     = na
var int   bar_count   = 0
var bool  or_complete = false
if rth_start
    or_high     := na
    bar_count   := 0
    or_complete := false
if is_rth and not or_complete
    or_high   := na(or_high) ? high : math.max(or_high, high)
    bar_count += 1
    if bar_count >= num_bars
        or_complete := true
sig = or_complete and barstate.isconfirmed and close > or_high
if sig and strategy.position_size == 0
    stopP = close - atrVal * 1.5
    tpP   = close + atrVal * 3.0
    strategy.entry("Long", strategy.long)
    strategy.exit("LX", from_entry="Long", stop=stopP, limit=tpP)
'''
    r = import_pine(src)
    assert r.ok, [e.message for e in r.errors]
    sf = r.strategy_json["market"]["session_filter"]
    assert sf["warmup_bars"] == 6
    inds = {i["id"]: i for i in r.strategy_json["indicators"]}
    assert inds["or_high"]["params"]["bars"] == 6


def test_htf_ema_via_request_security():
    src = '''//@version=6
strategy("HTF", overlay=true, default_qty_value=1)
htfLen = input.int(20, "HTF Len")
atrVal = ta.atr(14)
emaF = ta.ema(close, 9)
emaS = ta.ema(close, 21)
htfEmaVal = request.security(syminfo.tickerid, "30", ta.ema(close, htfLen), lookahead=barmerge.lookahead_off)
sig = ta.crossover(emaF, emaS) and close > htfEmaVal
if sig
    stopD = atrVal * 2.0
    stopP = close - stopD
    tpP   = close + stopD * 1.5
    strategy.entry("Long", strategy.long)
    strategy.exit("LX", from_entry="Long", stop=stopP, limit=tpP)
'''
    r = import_pine(src)
    assert r.ok, [e.message for e in r.errors]
    inds = {i["id"]: i for i in r.strategy_json["indicators"]}
    assert inds["htf_ema_val"]["type"] == "EMA"
    assert inds["htf_ema_val"]["params"]["timeframe"] == "30m"
