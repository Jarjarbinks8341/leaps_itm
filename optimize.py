"""Random hyperparameter search for the AAPL LEAPS strategy.

Usage:
    uv run optimize.py --mode A                        # 300 trials, default dates
    uv run optimize.py --mode B  --n 500 --workers 4   # parallel with 4 processes
    uv run optimize.py --mode AB --out best_AB.json    # custom output path
    uv run optimize.py --mode A  --refine best_A.json  # fine-grained search around a result

The optimizer samples randomly from PARAM_GRID, enforces first-principles
constraints, runs each backtest on the training period, and ranks by
composite score. Top result is saved to --out for use with backtest.py.

Every trial is also backtested on the OOS window (train_end+1 day through the
latest cached data, unless --oos-start/--oos-end override it) and an
overfit_penalty() is subtracted from the training score based on the
train/OOS CAGR gap — added 2026-07-05 after PARAM_GRID tightening alone
failed to stop the optimizer from finding params that inflate training CAGR
without holding up OOS (see SKILL.md Mode AB overfitting diagnosis).
Pass --no-oos-penalty to fall back to training-score-only ranking.

Note: the full grid (Mode A + Mode B params) is always sampled regardless of
--mode — params unused by the selected mode are simply ignored by run(), so
a single grid/constraint set covers all three modes without branching.
"""
import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta

import pandas as pd

from strategy.data import load, load_earnings_dates
from strategy.metrics import overfit_penalty
from backtest import run, DEFAULT_PARAMS

PARAM_GRID: dict[str, list] = {
    # shared
    "target_delta": [0.50, 0.55, 0.60, 0.65, 0.70],
    "dte_days": [300, 330, 365, 400, 430],
    "lot_pct": [0.03, 0.05, 0.07],
    "lot_pct_max": [0.10, 0.15, 0.20, 0.25],
    "min_months_remaining": [3, 4, 5, 6],
    "min_hold_months": [2, 3, 4],  # 1 removed 2026-07-05 — optimizer kept exploiting it for high-frequency overfitting (see SKILL.md)
    "earnings_blackout": [0, 5, 7, 10],  # 0 = disabled (verify necessity)
    "tier1_months": [3, 4, 5],
    "tier1_profit": [0.30, 0.40, 0.50, 0.60],
    "tier2_months": [5, 6, 7],
    "tier2_profit": [0.20, 0.25, 0.30],
    "tier3_months": [8, 9],
    "tier3_profit": [0.05, 0.10, 0.15],
    "force_months": [9, 10, 12],
    "tp1_close_pct": [0.50, 0.67, 1.00],
    "tp2_close_pct": [0.67, 1.00],
    # Mode A
    "macd_fast": [8, 10, 12, 16],
    "macd_slow": [24, 26, 28, 30],
    "macd_sig": [7, 9, 12],
    "div_lookback": [10, 15, 20, 25],
    "div_min_gap": [3, 5, 7],
    "vix_ma": [10, 20, 30],
    "neg_hist": [True, False],
    # Mode B
    "pullback_lookback": [3, 5, 7],
    "touch_tolerance": [0.0, 0.005, 0.01],
    "hist_converge_days": [2, 3, 4],
    "iv_rank_max": [0.30, 0.40, 0.50, 1.00],  # 1.00 = disabled (verify necessity)
}


def _valid(p: dict) -> bool:
    """First-principles constraints: reject logically impossible configs."""
    return (
        p["macd_fast"] < p["macd_slow"]
        and (p["macd_slow"] - p["macd_fast"]) >= 16
        and p["div_min_gap"] < p["div_lookback"] // 2
        and p["tier1_months"] < p["tier2_months"]
        and p["tier2_months"] < p["tier3_months"]
        and p["tier3_months"] <= p["force_months"]
        and p["lot_pct"] < p["lot_pct_max"]
    )


def _sample() -> dict:
    for _ in range(1000):
        p = {**DEFAULT_PARAMS, **{k: random.choice(v) for k, v in PARAM_GRID.items()}}
        if _valid(p):
            return p
    return dict(DEFAULT_PARAMS)


def _make_neighborhood(base: dict) -> dict[str, list]:
    """Fine-grained grid centred on base params for refinement search."""
    def _floats(v, steps, lo=0.01, hi=1.0):
        return sorted({max(lo, min(hi, round(v + d, 3))) for d in steps})

    def _ints(v, steps, lo=1):
        return sorted({max(lo, v + d) for d in steps})

    spec: dict[str, list] = {
        "target_delta": _floats(base["target_delta"], [-0.10, -0.05, 0, 0.05, 0.10], lo=0.30, hi=0.85),
        "dte_days": _ints(base["dte_days"], [-60, -30, 0, 30, 60], lo=180),
        "lot_pct": _floats(base["lot_pct"], [-0.02, -0.01, 0, 0.01, 0.02], lo=0.01, hi=0.30),
        "lot_pct_max": _floats(base["lot_pct_max"], [-0.05, -0.02, 0, 0.02, 0.05], lo=0.03, hi=0.50),
        "min_months_remaining": _ints(base["min_months_remaining"], [-2, -1, 0, 1, 2], lo=1),
        "min_hold_months": _ints(base["min_hold_months"], [-1, 0, 1, 2], lo=2),
        "earnings_blackout": _ints(base["earnings_blackout"], [-5, -2, 0, 2, 5], lo=0),
        "tier1_months": _ints(base["tier1_months"], [-1, 0, 1], lo=2),
        "tier1_profit": _floats(base["tier1_profit"], [-0.10, -0.05, 0, 0.05, 0.10], lo=0.10, hi=0.80),
        "tier2_months": _ints(base["tier2_months"], [-1, 0, 1, 2], lo=3),
        "tier2_profit": _floats(base["tier2_profit"], [-0.05, 0, 0.05, 0.10], lo=0.10, hi=0.60),
        "tier3_months": _ints(base["tier3_months"], [-1, 0, 1], lo=4),
        "tier3_profit": _floats(base["tier3_profit"], [-0.05, 0, 0.05, 0.10], lo=0.10, hi=0.40),
        "force_months": _ints(base["force_months"], [-2, -1, 0, 1, 2, 4], lo=6),
        "tp1_close_pct": [0.50, 0.67, 1.00],
        "tp2_close_pct": [0.67, 1.00],
        "macd_fast": _ints(base["macd_fast"], [-4, -2, 0, 2, 4], lo=4),
        "macd_slow": _ints(base["macd_slow"], [-4, -2, 0, 2, 4], lo=10),
        "macd_sig": _ints(base["macd_sig"], [-3, -1, 0, 1, 3], lo=3),
        "div_lookback": _ints(base["div_lookback"], [-5, -3, 0, 3, 5, 10], lo=5),
        "div_min_gap": _ints(base["div_min_gap"], [-2, -1, 0, 1, 2], lo=2),
        "vix_ma": _ints(base["vix_ma"], [-10, -5, 0, 5, 10], lo=5),
        "neg_hist": [True, False],
        "pullback_lookback": _ints(base["pullback_lookback"], [-2, -1, 0, 1, 2], lo=2),
        "touch_tolerance": _floats(base["touch_tolerance"], [-0.005, -0.002, 0, 0.002, 0.005], lo=0.0, hi=0.03),
        "hist_converge_days": _ints(base["hist_converge_days"], [-1, 0, 1, 2], lo=1),
        "iv_rank_max": _floats(base["iv_rank_max"], [-0.10, -0.05, 0, 0.05, 0.10], lo=0.10, hi=1.00),
    }
    return spec


def _sample_refine(neighborhood: dict) -> dict:
    for _ in range(2000):
        p = {**DEFAULT_PARAMS, **{k: random.choice(v) for k, v in neighborhood.items()}}
        if _valid(p):
            return p
    return dict(DEFAULT_PARAMS)


def _run_one(args_tuple) -> tuple[float, dict, dict] | None:
    mode, params, start, end, oos_start, oos_end, penalty_tolerance, penalty_weight, ticker = args_tuple
    try:
        m = run(mode, params, start, end, ticker=ticker)
        train_score = m["score"]
        if oos_start is not None:
            m_oos = run(mode, params, oos_start, oos_end, ticker=ticker)
            penalty = overfit_penalty(m["cagr"], m_oos["cagr"], penalty_tolerance, penalty_weight)
            m["oos_cagr"] = m_oos["cagr"]
            m["oos_max_dd"] = m_oos["max_dd"]
            m["oos_n_trades"] = m_oos["n_trades"]
            m["train_score"] = train_score
            m["penalty"] = penalty
            adjusted_score = train_score - penalty
        else:
            adjusted_score = train_score
        return (adjusted_score, params, m)
    except Exception:
        return None


def search(
    mode: str = "A",
    n: int = 300,
    train_start: str = "2015-01-01",
    train_end: str = "2020-09-30",
    workers: int = 1,
    neighborhood: dict | None = None,
    ticker: str = "AAPL",
    oos_start: str | None = None,
    oos_end: str | None = None,
    penalty_tolerance: float = 0.10,
    penalty_weight: float = 1.0,
    apply_oos_penalty: bool = True,
) -> list[tuple[float, dict, dict]]:
    """Run random search, return results sorted best-first.

    Each trial's ranking score is (training score - overfit_penalty), where
    the penalty is computed by also backtesting the same params on the OOS
    window (default: the day after train_end through the latest cached data)
    and comparing train vs OOS CAGR. Set apply_oos_penalty=False to rank by
    training score alone (the old behavior).
    """
    data = load(ticker=ticker)
    load_earnings_dates(ticker=ticker)

    resolved_oos_start = None
    resolved_oos_end = None
    if apply_oos_penalty:
        resolved_oos_start = oos_start or (pd.Timestamp(train_end) + timedelta(days=1)).strftime("%Y-%m-%d")
        resolved_oos_end = oos_end or data.index[-1].strftime("%Y-%m-%d")

    sampler = (lambda: _sample_refine(neighborhood)) if neighborhood else _sample
    jobs = [
        (mode, sampler(), train_start, train_end, resolved_oos_start, resolved_oos_end,
         penalty_tolerance, penalty_weight, ticker)
        for _ in range(n)
    ]

    results: list[tuple[float, dict, dict]] = []

    if workers <= 1:
        for i, job in enumerate(jobs, 1):
            result = _run_one(job)
            if result:
                results.append(result)
            if i % 20 == 0:
                best = max(r[0] for r in results) if results else 0
                print(f"  [{i}/{n}] best score so far: {best:.4f}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_run_one, job): i for i, job in enumerate(jobs, 1)}
            done = 0
            for fut in as_completed(futures):
                done += 1
                result = fut.result()
                if result:
                    results.append(result)
                if done % 20 == 0:
                    best = max(r[0] for r in results) if results else 0
                    print(f"  [{done}/{n}] best score so far: {best:.4f}")

    results.sort(key=lambda x: x[0], reverse=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="AAPL LEAPS strategy parameter optimizer")
    parser.add_argument("--mode", default="A", choices=["A", "B", "AB"], help="Entry signal mode")
    parser.add_argument("--n", type=int, default=300, help="Number of random trials")
    parser.add_argument("--train-start", default="2015-01-01")
    parser.add_argument("--train-end", default="2020-09-30")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--out", default="best_params.json", help="Output path for best params")
    parser.add_argument("--refine", help="Refine search around params from this JSON file")
    parser.add_argument("--ticker", default="AAPL", help="Underlying ticker (default: AAPL)")
    parser.add_argument("--oos-start", help="OOS window start (default: day after --train-end)")
    parser.add_argument("--oos-end", help="OOS window end (default: latest cached data)")
    parser.add_argument("--penalty-tolerance", type=float, default=0.10, help="Train/OOS CAGR gap allowed before penalty kicks in")
    parser.add_argument("--penalty-weight", type=float, default=1.0, help="Multiplier on the overfit penalty")
    parser.add_argument("--no-oos-penalty", action="store_true", help="Rank by training score alone (old behavior)")
    args = parser.parse_args()

    neighborhood = None
    if args.refine:
        with open(args.refine) as f:
            base = json.load(f)
        neighborhood = _make_neighborhood(base)
        print(f"Refining around {args.refine}")

    print(f"Starting {'refine' if neighborhood else 'random'} search: {args.n} trials, {args.workers} worker(s)")
    print(f"Ticker: {args.ticker}  |  Mode: {args.mode}  |  Training period: {args.train_start} → {args.train_end}")
    if not args.no_oos_penalty:
        print(f"OOS penalty: ON  |  tolerance={args.penalty_tolerance:.0%}  weight={args.penalty_weight}\n")
    else:
        print("OOS penalty: OFF (--no-oos-penalty)\n")

    results = search(
        args.mode, args.n, args.train_start, args.train_end, args.workers, neighborhood, args.ticker,
        oos_start=args.oos_start, oos_end=args.oos_end,
        penalty_tolerance=args.penalty_tolerance, penalty_weight=args.penalty_weight,
        apply_oos_penalty=not args.no_oos_penalty,
    )

    if not results:
        print("No valid results. Check data and constraints.")
        return

    print(f"\n{'─' * 90}")
    print(f"  Top 5 results (training period {args.train_start} → {args.train_end})")
    print(f"{'─' * 90}")
    for rank, (s, p, m) in enumerate(results[:5], 1):
        line = (
            f"  #{rank}  score={s:.4f}  CAGR={m['cagr']:.1%}  "
            f"MaxDD={m['max_dd']:.1%}  Sharpe={m['sharpe']:.2f}  "
            f"Trades={m['n_trades']}"
        )
        if "oos_cagr" in m:
            line += f"  |  OOS CAGR={m['oos_cagr']:.1%}  OOS MaxDD={m['oos_max_dd']:.1%}  penalty={m['penalty']:.4f}"
        print(line)
    print(f"{'─' * 90}")

    best_score, best_params, best_m = results[0]
    print(f"\nBest params (score {best_score:.4f}):")
    print(json.dumps(best_params, indent=2))

    with open(args.out, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nSaved to {args.out}")
    print(f"Run: uv run backtest.py --mode {args.mode} --ticker {args.ticker} --params {args.out}")


if __name__ == "__main__":
    main()
