"""Fetch and cache AAPL + VIX historical daily data, plus AAPL earnings dates."""
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent.parent


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"backtest_cache_{ticker.replace('-', '_')}.parquet"


def _earnings_cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"earnings_dates_{ticker.replace('-', '_')}.csv"


def load(start: str = "2014-01-01", end: str | None = None, refresh: bool = False, ticker: str = "AAPL") -> pd.DataFrame:
    """Return DataFrame with columns [price, vix], indexed by date.

    Fetches from Yahoo Finance on first call (or when refresh=True),
    then reads from a local parquet cache on subsequent calls.
    Start at 2014 to give warm-up room for 2015 training start.
    """
    cache = _cache_path(ticker)
    if cache.exists() and not refresh:
        df = pd.read_parquet(cache)
        if not df.empty:
            return df

    px = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)["Close"]

    px = px.squeeze()
    vix = vix.squeeze()

    df = pd.DataFrame({"price": px, "vix": vix}).dropna()
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "date"

    df.to_parquet(cache)
    return df


def load_earnings_dates(refresh: bool = False, ticker: str = "AAPL") -> pd.DatetimeIndex:
    """Return sorted DatetimeIndex of historical + upcoming earnings report dates.

    Fetches via yfinance's earnings-calendar scrape (covers back to ~2002 for AAPL),
    caches to CSV. Falls back to the CSV cache if the live fetch fails (the scrape
    endpoint is occasionally flaky/rate-limited) so a backtest never silently runs
    with zero earnings-blackout coverage.
    """
    cache = _earnings_cache_path(ticker)

    if not refresh and cache.exists():
        cached = pd.read_csv(cache, parse_dates=["date"])
        return pd.DatetimeIndex(cached["date"]).sort_values()

    try:
        t = yf.Ticker(ticker)
        ed = t.get_earnings_dates(limit=80)
        dates = pd.DatetimeIndex(ed.index).tz_localize(None).normalize().unique().sort_values()
        pd.DataFrame({"date": dates}).to_csv(cache, index=False)
        return dates
    except Exception:
        if cache.exists():
            cached = pd.read_csv(cache, parse_dates=["date"])
            return pd.DatetimeIndex(cached["date"]).sort_values()
        raise
