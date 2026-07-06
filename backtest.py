"""AAPL LEAPS Backtest — dual-mode entry signals.

Mode A: MACD bullish divergence + VIX elevation (bottom-divergence reversal).
Mode B: MA20>MA50 trend + MA20 pullback stabilization + MACD convergence +
        low IV rank (trend-pullback continuation).
Mode AB: either A or B fires (A takes priority on same-day overlap).

See .claude/skills/buildAaplLeaps/SKILL.md for the full strategy spec.

Usage:
    uv run backtest.py --mode A                            # default params, train period
    uv run backtest.py --mode B  --start 2025-01-01        # OOS test
    uv run backtest.py --mode AB --refresh                 # re-download data
    uv run backtest.py --mode A  --params best_A.json      # load params from optimizer
    uv run backtest.py --mode AB --trades                  # print trade log + mode attribution
"""
import argparse
import json
from datetime import date

from strategy.data import load, load_earnings_dates
from strategy.metrics import summary, summary_by_mode, score
from strategy.options import realized_vol
from strategy.portfolio import Portfolio
from strategy.signals import (
    compute_macd,
    bullish_divergence,
    vix_elevated,
    signal_strength,
    pullback_entry,
    iv_rank,
    in_earnings_blackout,
    deep_pullback_entry,
    deep_rally_exit,
    death_cross_regime,
)

DEFAULT_PARAMS: dict = {
    # shared
    "target_delta": 0.60,
    "dte_days": 365,
    "dte_days_bear": None,  # e.g. 545 (~18mo) — DTE used instead of dte_days when an entry (any mode) opens during a death cross (MA50<MA200); None disables
    "lot_pct": 0.05,
    "lot_pct_max": 0.15,
    "min_months_remaining": 6,
    "min_hold_months": 3,
    "earnings_blackout": 7,
    "tier1_months": 4,
    "tier1_profit": 0.50,
    "tier2_months": 6,
    "tier2_profit": 0.30,
    "tier3_months": 9,
    "tier3_profit": 0.10,
    "force_months": 9,
    "tp1_close_pct": 1.0,
    "tp2_close_pct": 1.0,
    # Mode A — bottom-divergence reversal
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_sig": 9,
    "div_lookback": 20,
    "div_min_gap": 5,
    "vix_ma": 20,
    "neg_hist": True,
    # Mode B — trend-pullback continuation
    "pullback_lookback": 5,
    "touch_tolerance": 0.005,
    "hist_converge_days": 3,
    "iv_rank_max": 0.40,
    "ma_short": 5,
    "ma_mid": 20,
    "ma_long": 50,
}

# Mode C — simplified deep-pullback: entry on price < MA5 < MA20. Position
# budget = initial_cash × max_deploy_pct (fixed at $50k for a $100k account,
# never recalculated off NAV growth), split into n_baskets equal dollar
# baskets (default 10 × $5k). At most one entry per week spends one basket's
# worth of cash on as many whole contracts as it buys — cheap contracts (2015)
# fill a basket with more contracts than expensive ones (2025), so sizing
# scales with price without compounding with NAV. Exit FIFO on price > MA5 >
# MA20 with per-position profit > 35% (max 1 sale/week — "sell by basket,
# FIFO"), or forced at 6 months before expiry regardless of P&L. tier1/2/3 and
# force_months from DEFAULT_PARAMS are unused here — Mode C's fixed_sizing
# branch in portfolio.step() replaces the tiered/_exit_reason exit path
# entirely, and lot_pct (the NAV%-based sizing A/B use) is unused too.
DEFAULT_PARAMS_C: dict = {
    **DEFAULT_PARAMS,
    "target_delta": 0.80,
    "dte_days": 365,
    "min_months_remaining": 6,
    "fixed_sizing": True,
    "contracts_per_entry": None,  # None = size from the basket budget below; set an int to force a fixed count
    "n_baskets": 10,
    "entry_cooldown_days": 7,
    "max_deploy_pct": 0.50,   # total budget = 50% of initial capital, split across n_baskets
    "exit_profit_min": 0.35,
    "stop_loss_pct": None,    # e.g. 0.30 = close immediately at -30% P&L; None disables (tested 2026-07-05, not adopted — see SKILL.md)
    "death_cross_filter": False,  # block new entries while MA50 < MA200; tested 2026-07-05, not adopted — see SKILL.md
    "ma_death_mid": 50,
    "ma_death_long": 200,
    "dte_days_bear": 913,     # ~2.5yr — DTE used instead of dte_days when an entry opens during a death cross (MA50<MA200); the longest LEAPS commonly listed. None disables. Tested 2026-07-05, adopted — see SKILL.md
    "dte_roll_losers": False,  # roll a losing position at the DTE floor into a fresh expiry instead of realizing the loss; tested 2026-07-05, not adopted (spikes Max DD to 68%) — see SKILL.md
    "exit_cooldown_days": 7,
    "ma_short": 5,
    "ma_mid": 20,
}

INITIAL_CASH = 100_000.0


def run(
    mode: str = "A",
    params: dict | None = None,
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    data=None,
    earnings_dates=None,
    refresh: bool = False,
    ticker: str = "AAPL",
) -> dict:
    """Run a full backtest for the given mode ("A", "B", or "AB").

    Returns metrics dict including curve, trades, and (for AB) by-mode attribution.
    """
    if mode not in ("A", "B", "AB", "C"):
        raise ValueError(f"mode must be 'A', 'B', 'AB', or 'C', got {mode!r}")
    if params is None:
        params = DEFAULT_PARAMS_C if mode == "C" else DEFAULT_PARAMS
    if data is None:
        data = load(refresh=refresh, ticker=ticker)
    if earnings_dates is None:
        earnings_dates = load_earnings_dates(refresh=refresh, ticker=ticker)

    macd_line, macd_signal, hist = compute_macd(
        data["price"], params["macd_fast"], params["macd_slow"], params["macd_sig"]
    )

    pf = Portfolio(INITIAL_CASH)

    sub = data.loc[start:end]
    warmup_a = params["macd_slow"] + params["div_lookback"] + 5
    warmup_b = params["ma_long"] + max(params["pullback_lookback"], params["hist_converge_days"]) + 2

    use_a = mode in ("A", "AB")
    use_b = mode in ("B", "AB")
    use_c = mode == "C"
    warmup_c = params["ma_mid"]

    for d, row in sub.iterrows():
        global_i = data.index.get_loc(d)
        S = float(row["price"])
        sigma = realized_vol(data["price"].iloc[: global_i + 1])
        blackout = in_earnings_blackout(d, earnings_dates, params["earnings_blackout"])

        sig_a = False
        strength = 0.0
        if use_a and not blackout and global_i >= warmup_a:
            p_win = data["price"].iloc[global_i - params["div_lookback"] : global_i + 1]
            h_win = hist.iloc[global_i - params["div_lookback"] : global_i + 1]
            v_win = data["vix"].iloc[: global_i + 1]
            div = bullish_divergence(
                p_win, h_win, params["div_lookback"], params["div_min_gap"],
                neg_hist=params.get("neg_hist", True),
            )
            sig_a = div and vix_elevated(v_win, params["vix_ma"])
            if sig_a:
                strength = signal_strength(
                    p_win, h_win, v_win,
                    params["div_lookback"], params["div_min_gap"], params["vix_ma"],
                )

        sig_b = False
        if use_b and not blackout and global_i >= warmup_b:
            p_full = data["price"].iloc[: global_i + 1]
            h_full = hist.iloc[: global_i + 1]
            m_full = macd_line.iloc[: global_i + 1]
            pullback = pullback_entry(
                p_full, h_full, m_full,
                params["pullback_lookback"], params["touch_tolerance"], params["hist_converge_days"],
                params["ma_short"], params["ma_mid"], params["ma_long"],
            )
            sig_b = pullback and iv_rank(p_full) < params["iv_rank_max"]

        # Death-cross regime flag (MA50 < MA200): computed unconditionally for
        # every mode, not just C — it drives the dte_days_bear override below
        # (buying a longer-dated LEAPS when opening a position during a bear
        # regime gives a prolonged correction, e.g. 2015-2016, more room to
        # recover before the DTE-proactive-exit forces a sale) as well as
        # Mode C's own entry filter. See death_cross_regime() and SKILL.md.
        in_death_cross = False
        ma_death_long = params.get("ma_death_long", 200)
        if global_i >= ma_death_long - 1:
            p_death_win = data["price"].iloc[global_i - ma_death_long + 1 : global_i + 1]
            in_death_cross = death_cross_regime(
                p_death_win, params.get("ma_death_mid", 50), ma_death_long
            )

        sig_c = False
        sig_c_exit = False
        if use_c:
            blocked_by_death_cross = params.get("death_cross_filter", False) and in_death_cross
            if global_i >= warmup_c:
                p_win = data["price"].iloc[global_i - warmup_c + 1 : global_i + 1]
                if not blackout and not blocked_by_death_cross:
                    sig_c = deep_pullback_entry(p_win, params["ma_short"], params["ma_mid"])
                # Exit signal is not blocked by earnings blackout or death
                # cross — both only gate new entries.
                sig_c_exit = deep_rally_exit(p_win, params["ma_short"], params["ma_mid"])

        # Same-day overlap: Mode A takes priority (rarer, higher-conviction signal)
        signal: str | None = None
        step_params = params
        if sig_a:
            signal = "A"
            lo = params.get("lot_pct", 0.05)
            hi = params.get("lot_pct_max", lo)
            step_params = {**params, "lot_pct": lo + (hi - lo) * strength}
        elif sig_b:
            signal = "B"
        elif sig_c:
            signal = "C"

        # Bear-regime DTE extension applies uniformly across A/B/C — an entry
        # opened during a death cross buys dte_days_bear (e.g. 18mo) instead
        # of the regular dte_days (e.g. 12mo); None (the default for A/B)
        # leaves them on a single fixed DTE, unaffected.
        if signal and in_death_cross and params.get("dte_days_bear") is not None:
            step_params = {**step_params, "dte_days": params["dte_days_bear"]}

        pf.step(d, S, sigma, signal, step_params, exit_signal=sig_c_exit, in_death_cross=in_death_cross)

    m = summary(pf.curve, pf.trades)
    m["curve"] = pf.curve
    m["trades"] = pf.trades
    m["positions"] = pf.positions
    m["score"] = score(pf.curve, pf.trades)
    m["by_mode"] = summary_by_mode(pf.trades)
    return m


def _print_report(m: dict, mode: str, start: str, end: str, ticker: str, show_trades: bool = False) -> None:
    print(f"\n{'─' * 52}")
    print(f"  {ticker} LEAPS Backtest  [mode={mode}]  {start} → {end}")
    print(f"{'─' * 52}")
    print(f"  Final Value   ${m['final_value']:>12,.0f}   (start ${INITIAL_CASH:,.0f})")
    print(f"  CAGR          {m['cagr']:>11.1%}")
    print(f"  Max Drawdown  {m['max_dd']:>11.1%}")
    print(f"  Sharpe        {m['sharpe']:>12.2f}")
    print(f"  Calmar        {m['calmar']:>12.2f}")
    print(f"  Win Rate      {m['win_rate']:>11.1%}")
    print(f"  # Trades      {m['n_trades']:>12d}")
    print(f"  Score         {m['score']:>12.4f}")
    print(f"{'─' * 52}")

    if mode == "AB" and m["by_mode"]:
        print("  Attribution by signal mode:")
        for signal_mode, stats in sorted(m["by_mode"].items()):
            print(
                f"    Mode {signal_mode:<3} {stats['n_trades']:>4} trades   "
                f"win_rate={stats['win_rate']:.1%}   avg_pnl={stats['avg_pnl_pct']:+.1%}"
            )
        print(f"{'─' * 52}")

    if m["trades"]:
        by_reason: dict[str, int] = {}
        for t in m["trades"]:
            by_reason[t.reason] = by_reason.get(t.reason, 0) + 1
        print("  Exit reasons:")
        for reason, cnt in sorted(by_reason.items(), key=lambda x: -x[1]):
            print(f"    {reason:<14} {cnt:>4} trades")
        print(f"{'─' * 52}")

    if show_trades and m["trades"]:
        print(f"\n  {'#':>3}  {'Mode':4}  {'Entry':10}  {'Exit':10}  {'Contracts':>9}  {'Entry$':>8}  {'Exit$':>8}  {'P&L':>7}  {'Cost':>10}  Reason")
        print(f"  {'─'*3}  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*9}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*10}  {'─'*6}")
        for i, t in enumerate(m["trades"], 1):
            sign = "+" if t.pnl_pct >= 0 else ""
            entry = str(t.entry_date)[:10]
            exit_ = str(t.exit_date)[:10]
            contracts = t.shares / 100
            cost = t.entry_premium * t.shares
            print(
                f"  {i:>3}  {t.signal_mode:4}  {entry:10}  {exit_:10}  {contracts:>9.1f}  "
                f"${t.entry_premium:>7.2f}  ${t.exit_premium:>7.2f}  "
                f"{sign}{t.pnl_pct:>6.1%}  ${cost:>9,.0f}  {t.reason}"
            )
        print(f"  {'─'*3}  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*9}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*10}  {'─'*6}")

    if show_trades and m.get("positions"):
        print(f"\n  Open positions on {end}:")
        print(f"  {'Mode':4}  {'Entry':10}  {'Expiry':10}  {'Strike':>8}  {'Entry$':>8}")
        print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}")
        for pos in m["positions"]:
            print(
                f"  {pos.signal_mode:4}  {str(pos.entry_date)[:10]:10}  {str(pos.expiry_date)[:10]:10}  "
                f"${pos.strike:>7.1f}  ${pos.entry_premium:>7.2f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="AAPL LEAPS backtest")
    parser.add_argument("--mode", default="A", choices=["A", "B", "AB", "C"], help="Entry signal mode")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--refresh", action="store_true", help="Re-download price/earnings data")
    parser.add_argument("--params", help="JSON file with parameter overrides")
    parser.add_argument("--ticker", default="AAPL", help="Underlying ticker (default: AAPL)")
    parser.add_argument("--trades", action="store_true", help="Print individual trade log")
    args = parser.parse_args()

    params = dict(DEFAULT_PARAMS_C if args.mode == "C" else DEFAULT_PARAMS)
    if args.params:
        with open(args.params) as f:
            params.update(json.load(f))

    print(f"Loading data for {args.ticker}…")
    data = load(refresh=args.refresh, ticker=args.ticker)
    earnings_dates = load_earnings_dates(refresh=args.refresh, ticker=args.ticker)
    print(f"Data: {data.index[0].date()} → {data.index[-1].date()}  ({len(data)} days)")
    print(f"Earnings dates: {len(earnings_dates)} known reports")

    print(f"Running backtest [mode={args.mode}] {args.start} → {args.end}…")
    m = run(args.mode, params, args.start, args.end, data, earnings_dates, ticker=args.ticker)
    _print_report(m, args.mode, args.start, args.end, args.ticker, show_trades=args.trades)


if __name__ == "__main__":
    main()
