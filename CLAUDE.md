# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Automated LEAPS buy-signal strategy for AAPL (later MSFT, GOOGL), validated by backtesting against historical data with auto-research parameter tuning.

## Strategy specification

The full strategy spec lives in `.claude/skills/buildAaplLeaps/SKILL.md` (invocable as the `buildAaplLeaps` skill). It defines the dual-mode entry signals (Mode A: MACD bottom-divergence + VIX filter; Mode B: trend-pullback continuation), option selection via BSM pricing, tiered exit rules, the parameter search space, and the auto-research protocol. **Read it before touching any strategy logic — it is the living spec that code must follow.**

## Commands

```bash
uv sync                                          # install deps
uv run backtest.py --mode A                      # baseline backtest, training period (2015-2024)
uv run backtest.py --mode B  --start 2025-01-01  # OOS test
uv run backtest.py --mode AB --trades            # combined mode + full trade log + per-mode attribution
uv run backtest.py --refresh                     # re-download price/VIX/earnings data
uv run optimize.py --mode A --n 300 --out best_A.json           # random hyperparameter search
uv run optimize.py --mode A --refine best_A.json --out best_A2.json  # fine-grained search around a result
```

There is no test suite yet (no pytest config). `backtest.py --trades` and manual sanity-checking of trade logs is the current verification method.

## Architecture

- `strategy/data.py` — fetches & caches AAPL price + VIX (parquet) and AAPL earnings dates (CSV, via yfinance `get_earnings_dates`, with a fallback to the CSV cache if the live scrape fails)
- `strategy/signals.py` — Mode A (MACD bullish divergence + VIX filter) and Mode B (MA20/MA50 trend-pullback + MACD convergence + IV rank) entry signals, plus `in_earnings_blackout()`
- `strategy/options.py` — BSM option pricing/delta/strike-search, realized-vol IV proxy
- `strategy/portfolio.py` — `Position`/`Trade` (tagged with `signal_mode` for A/B attribution), `Portfolio.step()` handling DTE exits, tiered profit-taking, FIFO rotation under a NAV deployment cap
- `strategy/metrics.py` — CAGR/drawdown/Sharpe/Calmar/win-rate/composite score, plus `summary_by_mode()` for AB attribution
- `backtest.py` / `optimize.py` — CLI entry points, both take `--mode A|B|AB`

`options.py`, `portfolio.py`, and `metrics.py` were ported near-verbatim from the QQQ LEAPS reference implementation at `~/repo/leaps/leaps/` — check there first when debugging BSM pricing or exit-tier logic, since that codebase has more historical iteration behind it.

The full strategy spec (why each rule exists, parameter search space, auto-research protocol, known limitations) lives in `.claude/skills/buildAaplLeaps/SKILL.md`. **Read it before changing strategy logic — it is the living design doc that the code must follow, and should be updated whenever the code's actual behavior diverges from what it describes.**
