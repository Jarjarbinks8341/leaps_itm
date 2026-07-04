"""BSM option pricing, delta, and realized volatility helpers."""
import math

import pandas as pd

R = 0.045   # risk-free rate
Q = 0.005   # AAPL dividend yield (approx.)


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def call_delta(S: float, K: float, T: float, sigma: float, r: float = R, q: float = Q) -> float:
    """Black-Scholes delta for a European call."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return math.exp(-q * T) * _ncdf(d1)


def call_price(S: float, K: float, T: float, sigma: float, r: float = R, q: float = Q) -> float:
    """Black-Scholes price for a European call."""
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)


def strike_for_delta(
    S: float, T: float, sigma: float, target: float = 0.6, r: float = R, q: float = Q
) -> float:
    """Binary-search for the strike K that produces the target delta."""
    lo, hi = S * 0.30, S * 1.10
    for _ in range(64):
        mid = (lo + hi) / 2.0
        if call_delta(S, mid, T, sigma, r, q) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def realized_vol(prices: pd.Series, window: int = 30) -> float:
    """Annualized realized volatility using the last `window` daily returns.

    Used as IV proxy in BSM. Falls back to a 20% floor so BSM never degenerates,
    floored at 5% minimum.
    """
    ret = prices.pct_change().dropna()
    w = min(window, len(ret))
    if w < 2:
        return 0.20
    vol = float(ret.iloc[-w:].std() * math.sqrt(252))
    return max(vol, 0.05)
