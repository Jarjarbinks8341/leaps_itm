"""Performance metrics for backtesting evaluation."""
import math
from datetime import date

from strategy.portfolio import Trade


def cagr(curve: list[tuple[date, float]]) -> float:
    if len(curve) < 2:
        return 0.0
    dates, vals = zip(*curve)
    years = (dates[-1] - dates[0]).days / 365.25
    if years <= 0 or vals[0] <= 0:
        return 0.0
    return (vals[-1] / vals[0]) ** (1.0 / years) - 1.0


def max_drawdown(curve: list[tuple[date, float]]) -> float:
    _, vals = zip(*curve)
    peak = vals[0]
    worst = 0.0
    for v in vals:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > worst:
            worst = dd
    return worst


def sharpe(curve: list[tuple[date, float]], rf: float = 0.045) -> float:
    if len(curve) < 2:
        return 0.0
    vals = [v for _, v in curve]
    rets = [(vals[i] - vals[i - 1]) / vals[i - 1] for i in range(1, len(vals))]
    if not rets:
        return 0.0
    n = len(rets)
    mean_r = sum(rets) / n
    variance = sum((r - mean_r) ** 2 for r in rets) / n
    std_r = math.sqrt(variance) if variance > 0 else 0.0
    if std_r == 0:
        return 0.0
    daily_rf = rf / 252
    return (mean_r - daily_rf) / std_r * math.sqrt(252)


def calmar(curve: list[tuple[date, float]]) -> float:
    c = cagr(curve)
    dd = max_drawdown(curve)
    return c / dd if dd > 0 else 0.0


def win_rate(trades: list[Trade]) -> float:
    if not trades:
        return 0.0
    return sum(1 for t in trades if t.pnl_pct > 0) / len(trades)


def summary(curve: list[tuple[date, float]], trades: list[Trade]) -> dict:
    return {
        "final_value": curve[-1][1] if curve else 0.0,
        "cagr": cagr(curve),
        "max_dd": max_drawdown(curve),
        "sharpe": sharpe(curve),
        "calmar": calmar(curve),
        "win_rate": win_rate(trades),
        "n_trades": len(trades),
    }


def score(
    curve: list[tuple[date, float]],
    trades: list[Trade],
    weights: dict | None = None,
) -> float:
    """Composite score for parameter optimization.

    Normalizes each metric to a comparable scale before weighting.
    Higher is always better (max_dd is negated).
    Weights: cagr=0.40, neg_dd=0.30, sharpe=0.20, calmar=0.10
    """
    if weights is None:
        weights = {"cagr": 0.40, "neg_dd": 0.30, "sharpe": 0.20, "calmar": 0.10}

    c = cagr(curve)
    dd = max_drawdown(curve)
    s = sharpe(curve)
    cal = calmar(curve)

    return (
        weights.get("cagr", 0) * c
        + weights.get("neg_dd", 0) * (-dd)
        + weights.get("sharpe", 0) * s / 5.0      # Sharpe ~0–5 → 0–1
        + weights.get("calmar", 0) * cal / 10.0   # Calmar ~0–10 → 0–1
    )


def summary_by_mode(trades: list[Trade]) -> dict[str, dict]:
    """Break out win_rate / avg pnl / count by signal_mode (for mode=AB attribution)."""
    by_mode: dict[str, list[Trade]] = {}
    for t in trades:
        by_mode.setdefault(t.signal_mode or "?", []).append(t)

    out: dict[str, dict] = {}
    for mode, mode_trades in by_mode.items():
        n = len(mode_trades)
        out[mode] = {
            "n_trades": n,
            "win_rate": sum(1 for t in mode_trades if t.pnl_pct > 0) / n if n else 0.0,
            "avg_pnl_pct": sum(t.pnl_pct for t in mode_trades) / n if n else 0.0,
        }
    return out
