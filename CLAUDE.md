# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Automated LEAPS buy-signal strategy for AAPL (later MSFT, GOOGL), validated by backtesting against historical data with auto-research parameter tuning.

## Strategy specification

The full strategy spec lives in `.claude/skills/buildAaplLeaps/SKILL.md` (invocable as the `buildAaplLeaps` skill). It defines the entry signals (Mode A: MACD bottom-divergence + VIX filter; Mode B: trend-pullback continuation; Mode C: simplified deep-pullback `Price < MA5 < MA20` entry / `Price > MA5 > MA20` + >35% profit FIFO exit, basket-sized position model, a bear-regime DTE extension that's adopted (2.5yr LEAPS when entering during a death cross) plus disabled-by-default stop-loss / death-cross entry filter / DTE-floor loss-rolling — all tested and found not to help (the roll spikes Max DD to 68%), see SKILL.md), option selection via BSM pricing, tiered exit rules, the parameter search space, and the auto-research protocol. **Read it before touching any strategy logic — it is the living spec that code must follow.**

## Commands

```bash
uv sync                                          # install deps
uv run backtest.py --mode A                      # baseline backtest, training period (default --end 2024-12-31; use --end 2020-09-30 to match the current train/OOS split, see SKILL.md)
uv run backtest.py --mode B  --start 2020-10-01  # OOS test — train/OOS split is now a 5.75yr/5.75yr 50/50 (2015-01-01 midpoint 2020-10-01), not the old 10yr/1.5yr split; see SKILL.md "回测区间"
uv run backtest.py --mode AB --trades            # combined mode + full trade log + per-mode attribution
uv run backtest.py --mode C  --trades            # simplified deep-pullback mode (basket sizing: $50k fixed budget / 10 baskets, one basket/week sized by price, FIFO signal exit at >35% profit or 6mo-DTE)
uv run backtest.py --refresh                     # re-download price/VIX/earnings data
uv run optimize.py --mode A --n 300 --out best_A.json           # random hyperparameter search, ranked by (train score - overfit_penalty on train/OOS CAGR gap); A/B/AB only, Mode C not wired in
uv run optimize.py --mode A --refine best_A.json --out best_A2.json  # fine-grained search around a result
```

There is no test suite yet (no pytest config). `backtest.py --trades` and manual sanity-checking of trade logs is the current verification method.

## Architecture

- `strategy/data.py` — fetches & caches AAPL price + VIX (parquet) and AAPL earnings dates (CSV, via yfinance `get_earnings_dates`, with a fallback to the CSV cache if the live scrape fails)
- `strategy/signals.py` — Mode A (MACD bullish divergence + VIX filter), Mode B (MA20/MA50 trend-pullback + MACD convergence + IV rank), and Mode C (`deep_pullback_entry()`: price < MA5 < MA20 entry; `deep_rally_exit()`: price > MA5 > MA20 exit) entry/exit signals, plus `death_cross_regime()`: MA50 < MA200, computed for every mode in `backtest.py::run()` — drives the `dte_days_bear` DTE-extension override (adopted for Mode C, tested and *not* adopted for A/B — see SKILL.md) and Mode C's disabled-by-default `death_cross_filter` entry block, plus `in_earnings_blackout()`
- `strategy/options.py` — BSM option pricing/delta/strike-search, realized-vol IV proxy
- `strategy/portfolio.py` — `Position`/`Trade` (tagged with `signal_mode` for A/B/C attribution), `Portfolio.step()` handling DTE exits for all modes plus two mutually-exclusive profit-exit paths: tiered `_exit_reason()` (A/B) or FIFO signal-gated exit throttled to one sale/week (`fixed_sizing=True`, Mode C); entries likewise branch between NAV%-based FIFO rotation with an 80%-of-NAV cap (A/B) and Mode C's basket sizing — `Portfolio.initial_cash` (fixed at construction) × `max_deploy_pct` ÷ `n_baskets` gives a fixed dollar budget per entry, converted to whole contracts at the day's option cost, throttled to one entry/week (`contracts_per_entry` can still override to a fixed count); `Portfolio._roll()` implements an optional (disabled-by-default) DTE-floor rollover for losing Mode C positions
- `strategy/metrics.py` — CAGR/drawdown/Sharpe/Calmar/win-rate/composite score, plus `summary_by_mode()` for AB attribution
- `backtest.py` — CLI entry point, takes `--mode A|B|AB|C`
- `optimize.py` — CLI entry point, takes `--mode A|B|AB` (Mode C not yet in the search space); every trial is backtested on both train and OOS windows and ranked by `train_score - overfit_penalty(train_cagr, oos_cagr)` (`strategy/metrics.py`), default OOS window is train_end+1 day through the latest cached data — pass `--no-oos-penalty` for the old training-score-only ranking

`options.py`, `portfolio.py`, and `metrics.py` were ported near-verbatim from the QQQ LEAPS reference implementation at `~/repo/leaps/leaps/` — check there first when debugging BSM pricing or exit-tier logic, since that codebase has more historical iteration behind it.

The full strategy spec (why each rule exists, parameter search space, auto-research protocol, known limitations) lives in `.claude/skills/buildAaplLeaps/SKILL.md`. **Read it before changing strategy logic — it is the living design doc that the code must follow, and should be updated whenever the code's actual behavior diverges from what it describes.**
