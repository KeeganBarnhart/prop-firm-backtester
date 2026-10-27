# Prop Firm Backtester

A Python backtesting engine for futures strategies that answers one question traders usually skip: **would this strategy actually survive a prop firm evaluation?**

Most tools stop at a single backtest. This one takes a TradingView Pine Script strategy, backtests it, checks it out-of-sample, then runs thousands of Monte Carlo simulations against real prop firm rules (trailing drawdown, profit target) to estimate a pass rate.

## What it does

1. **Pine Script importer**: parses a TradingView strategy (EMAs, conditions, stops, targets, session windows) into a typed strategy schema.
2. **Backtest engine**: bar-by-bar simulation with slippage, commissions, position sizing, session filters, and multi-timeframe indicators (built on pandas and NumPy).
3. **Out-of-sample check**: splits the data into in-sample and out-of-sample periods and flags signs of overfitting, such as a profit factor that collapses or a drawdown that balloons.
4. **Monte Carlo**: shuffles or bootstraps the trade list thousands of times to estimate the odds of profit, ruin, and drawdown ranges.
5. **Prop firm rules**: evaluates each simulated path against a firm's rules defined in YAML (`prop_firms/`), producing a pass rate.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python demo.py      # Pine Script -> backtest -> OOS check -> Monte Carlo + prop firm pass rate
pytest              # 36 tests
```

The demo runs on deterministic synthetic price data, so no market data subscription or API keys are needed.

Example output:

```
Imported strategy: MNQ EMA Cross (9/21)

In-sample:       110 trades | net P&L $   -371.47 | max DD $   713.88 | win rate 42%
Out-of-sample:    53 trades | net P&L $     66.40 | max DD $   420.73 | win rate 47%
Robustness check: OK

Monte Carlo (2000 trade-order shuffles):
  P(profitable)          0%
  P(ruin, -20%)          0%
  Max drawdown p50 / p95 $858 / $1,224
  Tradeify Select 50K: pass rate 0%
```

A simple EMA cross loses money on this data and has a 0% evaluation pass rate. That's the point: the tool tells you before you pay for a challenge.

## Project layout

```
propbt/
  importers/pine.py      Pine Script -> strategy schema
  schema/strategy.py     Pydantic strategy model
  engines/simple.py      Backtest engine + out-of-sample diagnosis
  engines/monte_carlo.py Monte Carlo simulation + prop firm pass rates
  propfirm/rules.py      Prop firm rule evaluation
  data/synthetic.py      Synthetic OHLCV bars for demos and tests
prop_firms/              Prop firm rule definitions (YAML)
examples/                Example Pine Script strategy + its JSON form
tests/                   pytest suite
```

## Tech

Python 3.12 · pandas · NumPy · Pydantic · PyYAML · pytest

---

Built by [Keegan Barnhart](https://www.linkedin.com/in/keegan-david-barnhart). Not financial advice; simulated results do not guarantee future performance.
