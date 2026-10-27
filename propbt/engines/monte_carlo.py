"""Monte Carlo simulation — trade-order shuffle / bootstrap resampling.

Takes a per-trade P&L list and resamples it N times to produce a
distribution of possible equity outcomes. Surfaces:
  - Percentile equity bands (p5/p25/p50/p75/p95) — the "fan chart"
  - Terminal equity distribution
  - Max-drawdown distribution + histogram
  - Probabilities: P(profitable), P(passes prop firm gates), P(ruin)

This engine is execution-agnostic — it does not know or care how the trades
were produced. The trade list can come from the backtest engine in this repo
or from any source that yields a list of floats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

import numpy as np

from propbt.propfirm.rules import evaluate as prop_firm_evaluate, load_firms


@dataclass
class MonteCarloResult:
    strategy_id: str
    config: dict[str, Any]
    base_trades: int
    base_net_pnl: float
    base_max_dd: float
    sims: int

    percentile_curves: dict[str, list[float]] = field(default_factory=dict)

    terminal_p5: float = 0.0
    terminal_p25: float = 0.0
    terminal_p50: float = 0.0
    terminal_p75: float = 0.0
    terminal_p95: float = 0.0
    terminal_mean: float = 0.0
    terminal_std: float = 0.0

    dd_p5: float = 0.0
    dd_p50: float = 0.0
    dd_p95: float = 0.0
    dd_mean: float = 0.0
    dd_max: float = 0.0

    prob_profit: float = 0.0
    prob_ruin: float = 0.0
    prop_firm_pass_rates: list[dict] = field(default_factory=list)

    dd_histogram: list[int] = field(default_factory=list)
    dd_histogram_bins: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def run_monte_carlo(
    trades: Iterable[float],
    *,
    strategy_id: str = "",
    sims: int = 1000,
    resample: Literal["shuffle", "bootstrap"] = "shuffle",
    starting_equity: float = 50_000.0,
    ruin_pct: float = 0.20,
    seed: int = 42,
) -> MonteCarloResult:
    if sims < 10 or sims > 100_000:
        raise ValueError(f"sims must be in [10, 100000], got {sims}")
    if resample not in ("shuffle", "bootstrap"):
        raise ValueError(f"resample must be 'shuffle' or 'bootstrap', got {resample!r}")

    trade_pnls = np.asarray(list(trades), dtype=np.float64)
    n_trades = len(trade_pnls)
    if n_trades < 5:
        raise ValueError(
            f"need at least 5 trades to run Monte Carlo; got {n_trades}"
        )

    base_net_pnl = float(trade_pnls.sum())
    base_equity = starting_equity + np.cumsum(trade_pnls)
    base_equity_full = np.concatenate([[starting_equity], base_equity])
    base_running_peak = np.maximum.accumulate(base_equity_full)
    base_max_dd = float((base_running_peak - base_equity_full).max())

    rng = np.random.default_rng(seed)

    if resample == "shuffle":
        idx = np.stack([rng.permutation(n_trades) for _ in range(sims)])
    else:
        idx = rng.integers(0, n_trades, size=(sims, n_trades))
    resampled = trade_pnls[idx]

    cumulative = np.cumsum(resampled, axis=1)
    equity = starting_equity + cumulative
    full_equity = np.concatenate(
        [np.full((sims, 1), starting_equity), equity], axis=1
    )

    pct_levels = [5, 25, 50, 75, 95]
    pct_curves: dict[str, list[float]] = {}
    for p in pct_levels:
        band = np.percentile(full_equity, p, axis=0)
        pct_curves[f"p{p}"] = [float(round(v, 2)) for v in band]

    terminals = full_equity[:, -1]
    t_p5, t_p25, t_p50, t_p75, t_p95 = np.percentile(terminals, [5, 25, 50, 75, 95])

    running_peak = np.maximum.accumulate(full_equity, axis=1)
    drawdowns = running_peak - full_equity
    max_dds = drawdowns.max(axis=1)
    d_p5, d_p50, d_p95 = np.percentile(max_dds, [5, 50, 95])

    prob_profit = float((terminals > starting_equity).mean())
    ruin_threshold = starting_equity * (1.0 - ruin_pct)
    prob_ruin = float((full_equity.min(axis=1) <= ruin_threshold).mean())

    firms = load_firms()
    pass_rates: list[dict] = []
    for fid, firm in firms.items():
        passes = np.array([
            prop_firm_evaluate(
                float(terminals[i] - starting_equity),
                float(max_dds[i]),
                firm,
            ).passes
            for i in range(sims)
        ])
        pass_rates.append({
            "firm_id": fid,
            "firm_name": firm.name,
            "pass_rate": float(passes.mean()),
            "passes": int(passes.sum()),
            "total": sims,
        })

    dd_top = float(np.percentile(max_dds, 99))
    hist, bin_edges = np.histogram(max_dds, bins=20, range=(0, max(dd_top, 1.0)))

    return MonteCarloResult(
        strategy_id=strategy_id,
        config={
            "sims": sims,
            "resample": resample,
            "starting_equity": starting_equity,
            "ruin_pct": ruin_pct,
            "seed": seed,
            "n_trades": int(n_trades),
        },
        base_trades=n_trades,
        base_net_pnl=round(base_net_pnl, 2),
        base_max_dd=round(base_max_dd, 2),
        sims=sims,
        percentile_curves=pct_curves,
        terminal_p5=round(float(t_p5), 2),
        terminal_p25=round(float(t_p25), 2),
        terminal_p50=round(float(t_p50), 2),
        terminal_p75=round(float(t_p75), 2),
        terminal_p95=round(float(t_p95), 2),
        terminal_mean=round(float(terminals.mean()), 2),
        terminal_std=round(float(terminals.std()), 2),
        dd_p5=round(float(d_p5), 2),
        dd_p50=round(float(d_p50), 2),
        dd_p95=round(float(d_p95), 2),
        dd_mean=round(float(max_dds.mean()), 2),
        dd_max=round(float(max_dds.max()), 2),
        prob_profit=round(prob_profit, 4),
        prob_ruin=round(prob_ruin, 4),
        prop_firm_pass_rates=pass_rates,
        dd_histogram=[int(c) for c in hist],
        dd_histogram_bins=[round(float(b), 2) for b in bin_edges],
    )
