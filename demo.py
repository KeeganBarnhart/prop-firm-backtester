"""End-to-end demo: Pine Script strategy -> backtest -> out-of-sample check -> Monte Carlo + prop firm pass rate.

Runs on deterministic synthetic bars, so no market data or API keys are needed.

    python demo.py
"""
from __future__ import annotations

from pathlib import Path

from propbt.data.synthetic import synthetic_ohlcv
from propbt.engines.monte_carlo import run_monte_carlo
from propbt.engines.simple import run_simple_backtest
from propbt.importers.pine import import_pine
from propbt.schema.strategy import Strategy

ROOT = Path(__file__).resolve().parent


def main() -> None:
    # 1. Import a TradingView Pine Script strategy into the strategy schema
    source = (ROOT / "examples" / "example_mnq_ema_cross.pine").read_text()
    imported = import_pine(source)
    if not imported.ok:
        raise SystemExit(f"Import failed: {imported.errors}")
    strategy = Strategy.model_validate(imported.strategy_json)
    print(f"Imported strategy: {imported.detected_name or 'unnamed'}")

    # 2. Backtest with a 70/30 in-sample / out-of-sample split
    bars = synthetic_ohlcv(n=6000)
    train, test, diagnosis = run_simple_backtest(strategy, bars, train_test_split=0.7)
    print(f"\nIn-sample:      {len(train.trades):>4} trades | net P&L ${train.net_pnl:>10,.2f} | "
          f"max DD ${train.max_drawdown:>9,.2f} | win rate {train.win_rate:.0%}")
    print(f"Out-of-sample:  {len(test.trades):>4} trades | net P&L ${test.net_pnl:>10,.2f} | "
          f"max DD ${test.max_drawdown:>9,.2f} | win rate {test.win_rate:.0%}")
    print(f"Robustness check: {diagnosis['severity'].upper()}")
    for w in diagnosis["warnings"]:
        print(f"  - {w}")

    # 3. Monte Carlo on the full trade list, including prop firm pass rates
    full = run_simple_backtest(strategy, bars)
    mc = run_monte_carlo([t.pnl for t in full.trades], sims=2000, strategy_id=strategy.id)
    print(f"\nMonte Carlo ({mc.sims} trade-order shuffles):")
    print(f"  P(profitable)          {mc.prob_profit:.0%}")
    print(f"  P(ruin, -20%)          {mc.prob_ruin:.0%}")
    print(f"  Max drawdown p50 / p95 ${mc.dd_p50:,.0f} / ${mc.dd_p95:,.0f}")
    for firm in mc.prop_firm_pass_rates:
        print(f"  {firm['firm_name']}: pass rate {firm['pass_rate']:.0%}")


if __name__ == "__main__":
    main()
