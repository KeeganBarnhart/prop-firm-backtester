"""Instrument registry: contract metadata the engine needs for P&L math.

A wrong point value silently poisons every downstream number (P&L, drawdown,
calibration deltas), so unknown instruments RAISE instead of defaulting.
Extend the table when a new market is onboarded — and add its calibration
family at the same time (scripts/calibrate.py).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    symbol: str
    point_value: float   # currency per 1.0 price point per contract/unit
    tick_size: float


_REGISTRY: dict[str, Instrument] = {
    # CME micro + e-mini index futures
    "MNQ": Instrument("MNQ", point_value=2.0, tick_size=0.25),
    "MES": Instrument("MES", point_value=5.0, tick_size=0.25),
    "NQ": Instrument("NQ", point_value=20.0, tick_size=0.25),
    "ES": Instrument("ES", point_value=50.0, tick_size=0.25),
    # Non-futures entries kept from the engine's original table
    "EUR/USD": Instrument("EUR/USD", point_value=100_000.0, tick_size=0.00001),
    "BTC/USD": Instrument("BTC/USD", point_value=1.0, tick_size=0.01),
}


def get_instrument(symbol: str) -> Instrument:
    inst = _REGISTRY.get(symbol.upper())
    if inst is None:
        raise KeyError(
            f"unknown instrument {symbol!r} — add it to backend/app/core/instruments.py "
            f"(known: {sorted(_REGISTRY)})"
        )
    return inst


def point_value(symbol: str) -> float:
    return get_instrument(symbol).point_value
