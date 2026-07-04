"""Entry signals for AAPL LEAPS.

Mode A (bottom-divergence reversal): MACD bullish divergence + VIX elevation.
Ported unchanged from the QQQ strategy — see SKILL.md "Mode A".

Mode B (trend-pullback continuation): MA20>MA50 trend gate + pullback-to-MA20
stabilization + MA5 turning up + MACD momentum converging + low IV rank.
New signal, see SKILL.md "Mode B".

Shared: earnings blackout (blocks new entries only, both modes).
"""
import math

import pandas as pd


def compute_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    sig: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (macd_line, signal_line, histogram)."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    signal = line.ewm(span=sig, adjust=False).mean()
    hist = line - signal
    return line, signal, hist


# ── Mode A: bottom-divergence reversal ──────────────────────────────────────


def bullish_divergence(
    prices: pd.Series,
    hist: pd.Series,
    lookback: int = 20,
    min_gap: int = 5,
    neg_hist: bool = True,
) -> bool:
    """Detect MACD bullish divergence at the last bar.

    Algorithm: split the lookback window into two halves separated by min_gap.
    - Prior low  = minimum price in [i-lookback, i-min_gap]
    - Recent low = minimum price in [i-min_gap, i]
    Divergence = price_recent < price_prior  AND  hist_recent > hist_prior
    If neg_hist=True (default): also require both histogram values negative.

    No look-ahead: only uses data up to and including the current bar.
    """
    n = len(prices)
    if n < lookback + 1 or min_gap >= lookback:
        return False

    p = prices.values
    h = hist.values

    prior_p = p[n - lookback : n - min_gap]
    recent_p = p[n - min_gap : n]

    if len(prior_p) == 0 or len(recent_p) == 0:
        return False

    prior_local = prior_p.argmin()
    recent_local = recent_p.argmin()

    price_prior = prior_p[prior_local]
    price_recent = recent_p[recent_local]

    hist_prior = h[n - lookback + prior_local] if n - lookback + prior_local < n else h[-lookback]
    hist_recent = h[n - min_gap + recent_local]

    divergence = price_recent < price_prior and hist_recent > hist_prior
    if neg_hist:
        return divergence and hist_prior < 0 and hist_recent < 0
    return divergence


def vix_elevated(vix: pd.Series, ma_window: int = 20) -> bool:
    """Return True if current VIX is above its own moving average."""
    if len(vix) < ma_window + 1:
        return False
    ma = float(vix.rolling(ma_window).mean().iloc[-1])
    return float(vix.iloc[-1]) > ma


def signal_strength(
    prices: pd.Series,
    hist: pd.Series,
    vix: pd.Series,
    lookback: int,
    min_gap: int,
    vix_ma_window: int,
) -> float:
    """Compute Mode A signal conviction [0.0, 1.0] when a divergence has fired.

    Two components (equal weight): VIX excess above its MA, and divergence
    magnitude (price drop % + MACD histogram recovery %).
    """
    vix_cur = float(vix.iloc[-1])
    vix_ma = float(vix.rolling(vix_ma_window).mean().iloc[-1])
    vix_excess = max(0.0, (vix_cur - vix_ma) / vix_ma)
    vix_score = min(1.0, vix_excess / 0.60)

    n = len(prices)
    p = prices.values
    h = hist.values

    prior_p = p[n - lookback : n - min_gap]
    recent_p = p[n - min_gap : n]
    prior_local = prior_p.argmin()
    recent_local = recent_p.argmin()

    price_prior = prior_p[prior_local]
    price_recent = recent_p[recent_local]
    hist_prior = h[n - lookback + prior_local]
    hist_recent = h[n - min_gap : n][recent_local]

    price_drop_pct = max(0.0, (price_prior - price_recent) / price_prior)
    hist_denom = max(abs(hist_prior), 1e-6)
    hist_recovery = max(0.0, (hist_recent - hist_prior) / hist_denom)

    div_score = min(1.0, price_drop_pct * 8 + hist_recovery * 0.15)

    return (vix_score + div_score) / 2.0


# ── Mode B: trend-pullback continuation ─────────────────────────────────────


def iv_rank(prices: pd.Series, vol_window: int = 30, lookback: int = 252) -> float:
    """Percentile rank [0.0, 1.0] of current realized vol vs trailing `lookback`-day
    distribution of realized vol. Low rank = options relatively cheap.

    Returns 0.5 (neutral) when there isn't enough history to form the distribution.
    """
    rets = prices.pct_change().dropna()
    if len(rets) < vol_window + lookback:
        return 0.5

    rolling_vol = (rets.rolling(vol_window).std() * math.sqrt(252)).dropna()
    if len(rolling_vol) < lookback:
        return 0.5

    window = rolling_vol.iloc[-lookback:]
    current = window.iloc[-1]
    return float((window < current).sum() / len(window))


def _macd_converging_or_crossed(
    macd_line: pd.Series,
    hist: pd.Series,
    converge_days: int = 3,
    cross_lookback: int = 3,
) -> bool:
    """True if histogram is negative but has risen for `converge_days` straight
    bars (converging toward a golden cross), or the histogram flipped from
    negative to positive within the last `cross_lookback` bars while the MACD
    line itself was already above the zero axis (a "golden cross above zero").
    """
    h = hist.values
    m = macd_line.values
    n = len(h)
    if n < converge_days + 1:
        return False

    recent = h[-(converge_days + 1) :]
    if recent[-1] < 0 and all(recent[i] < recent[i + 1] for i in range(len(recent) - 1)):
        return True

    lookback = min(cross_lookback, n - 1)
    for i in range(1, lookback + 1):
        idx = n - i
        if h[idx - 1] <= 0 < h[idx] and m[idx] > 0:
            return True
    return False


def pullback_entry(
    prices: pd.Series,
    hist: pd.Series,
    macd_line: pd.Series,
    pullback_lookback: int = 5,
    touch_tolerance: float = 0.005,
    hist_converge_days: int = 3,
    ma_short: int = 5,
    ma_mid: int = 20,
    ma_long: int = 50,
) -> bool:
    """Mode B entry: uptrend pullback to MA20 that's stabilizing, with MA5
    turning up and MACD momentum converging. See SKILL.md Mode B, conditions 1-4.
    (Condition 5, IV rank filter, and condition 6, earnings blackout, are
    checked separately by the caller.)
    """
    n = len(prices)
    min_len = ma_long + max(pullback_lookback, hist_converge_days) + 2
    if n < min_len:
        return False

    ma5 = prices.rolling(ma_short).mean()
    ma20 = prices.rolling(ma_mid).mean()
    ma50 = prices.rolling(ma_long).mean()

    # 1. Trend confirmation: only buy pullbacks within an established uptrend
    if not (ma20.iloc[-1] > ma50.iloc[-1]):
        return False

    # 2. Pullback to MA20 and stabilization: touched/dipped below MA20 recently,
    #    but today's close has recovered back to/above it
    recent_low = prices.iloc[-pullback_lookback:].min()
    ma20_now = float(ma20.iloc[-1])
    touched = recent_low <= ma20_now * (1 + touch_tolerance)
    stabilized = float(prices.iloc[-1]) >= ma20_now
    if not (touched and stabilized):
        return False

    # 3. Short-term momentum turning up
    if not (ma5.iloc[-1] > ma5.iloc[-2] and float(prices.iloc[-1]) > float(ma5.iloc[-1])):
        return False

    # 4. MACD momentum converging toward (or freshly past) a golden cross
    if not _macd_converging_or_crossed(macd_line, hist, hist_converge_days):
        return False

    return True


# ── Shared ───────────────────────────────────────────────────────────────────


def in_earnings_blackout(
    current_date: pd.Timestamp,
    earnings_dates: pd.DatetimeIndex,
    blackout_days: int = 7,
) -> bool:
    """True if `current_date` falls within `blackout_days` before the next
    upcoming earnings report. Blocks new entries only — does not affect exits.
    """
    if blackout_days <= 0 or len(earnings_dates) == 0:
        return False
    future = earnings_dates[earnings_dates >= current_date]
    if len(future) == 0:
        return False
    days_until = (future.min() - current_date).days
    return 0 <= days_until <= blackout_days
