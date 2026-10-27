"""Prop-firm rule engine. Loads YAML configs, evaluates pass/fail for backtest results.

Strict subset of rules supported in v0.1:
  - trailing_drawdown.amount (EOD only)
  - profit_target.amount (must_hit_to_pass)

Intraday DD, consistency rules, min trading days, and scaling plans are
deferred to Milestone 4 / PRP-02.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROP_FIRMS_DIR = Path(__file__).resolve().parents[2] / "prop_firms"


@dataclass
class PropFirmRule:
    id: str
    name: str
    account_size: float
    profit_target: float | None
    profit_target_required: bool
    trailing_dd: float | None  # EOD only in v0.1


@dataclass
class PassResult:
    firm_id: str
    passes: bool
    profit_target_hit: bool | None
    dd_under_floor: bool | None
    reasons: list[str]


def load_firms() -> dict[str, PropFirmRule]:
    out: dict[str, PropFirmRule] = {}
    if not PROP_FIRMS_DIR.exists():
        return out
    for p in sorted(PROP_FIRMS_DIR.glob("*.yaml")):
        cfg = yaml.safe_load(p.read_text())
        rules = cfg.get("rules", {})
        td = rules.get("trailing_drawdown") or {}
        pt = rules.get("profit_target") or {}
        firm = PropFirmRule(
            id=cfg["id"],
            name=cfg["name"],
            account_size=float(cfg.get("account_size", 0)),
            profit_target=float(pt["amount"]) if pt.get("amount") is not None else None,
            profit_target_required=bool(pt.get("must_hit_to_pass", False)),
            trailing_dd=float(td["amount"]) if td.get("enabled") and td.get("amount") is not None else None,
        )
        out[firm.id] = firm
    return out


def evaluate(net_pnl: float, max_drawdown: float, firm: PropFirmRule) -> PassResult:
    reasons: list[str] = []

    profit_target_hit: bool | None = None
    if firm.profit_target is not None:
        profit_target_hit = net_pnl >= firm.profit_target
        if firm.profit_target_required and not profit_target_hit:
            reasons.append(f"net P&L ${net_pnl:,.0f} < profit target ${firm.profit_target:,.0f}")

    dd_under_floor: bool | None = None
    if firm.trailing_dd is not None:
        dd_under_floor = max_drawdown <= firm.trailing_dd
        if not dd_under_floor:
            reasons.append(f"max DD ${max_drawdown:,.0f} > trailing DD floor ${firm.trailing_dd:,.0f}")

    passes = not reasons
    return PassResult(
        firm_id=firm.id,
        passes=passes,
        profit_target_hit=profit_target_hit,
        dd_under_floor=dd_under_floor,
        reasons=reasons,
    )
