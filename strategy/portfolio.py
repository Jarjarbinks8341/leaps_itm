"""Position management: open/close LEAPS, tiered exit, FIFO rotation.

Ported from the QQQ strategy, extended with `signal_mode` tagging on
Position/Trade so mode=AB backtests can attribute performance back to
Mode A (bottom-divergence) vs Mode B (trend-pullback) entries.
"""
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import NamedTuple

from strategy.options import call_price, strike_for_delta


class Trade(NamedTuple):
    entry_date: date
    exit_date: date
    entry_premium: float
    exit_premium: float
    pnl_pct: float
    reason: str
    shares: float = 0.0
    signal_mode: str = ""


@dataclass
class Position:
    entry_date: date
    expiry_date: date
    strike: float
    entry_premium: float   # BSM price per share at entry
    shares: float          # whole number: contracts * 100
    signal_mode: str = ""  # "A" or "B" — which entry mode opened this position
    used_tiers: list = field(default_factory=list)  # tiers already partially closed

    @property
    def cost(self) -> float:
        return self.entry_premium * self.shares

    @property
    def contracts(self) -> int:
        return int(self.shares // 100)

    def months_held(self, d: date) -> float:
        return (d - self.entry_date).days / 30.44

    def months_to_expiry(self, d: date) -> float:
        return max((self.expiry_date - d).days, 0) / 30.44

    def tte_years(self, d: date) -> float:
        return max((self.expiry_date - d).days, 0) / 365.0

    def current_premium(self, d: date, S: float, sigma: float) -> float:
        return call_price(S, self.strike, self.tte_years(d), sigma)

    def current_value(self, d: date, S: float, sigma: float) -> float:
        return self.current_premium(d, S, sigma) * self.shares

    def pnl_pct(self, d: date, S: float, sigma: float) -> float:
        cp = self.current_premium(d, S, sigma)
        return (cp - self.entry_premium) / self.entry_premium


class Portfolio:
    def __init__(self, cash: float, max_deploy_pct: float = 0.80):
        self.cash = cash
        self.max_deploy_pct = max_deploy_pct  # max fraction of NAV in options
        self.positions: list[Position] = []
        self.trades: list[Trade] = []
        self.curve: list[tuple[date, float]] = []

    # ── valuation ─────────────────────────────────────────────────────────────

    def nav(self, d: date, S: float, sigma: float) -> float:
        return self.cash + sum(p.current_value(d, S, sigma) for p in self.positions)

    def option_value(self, d: date, S: float, sigma: float) -> float:
        return sum(p.current_value(d, S, sigma) for p in self.positions)

    # ── open / close ───────────────────────────────────────────────────────────

    def _open(self, d: date, S: float, sigma: float, params: dict, signal_mode: str = "") -> bool:
        lot = int(params.get("lot_size", 1))
        T = params["dte_days"] / 365.0
        K = strike_for_delta(S, T, sigma, params["target_delta"])
        premium = call_price(S, K, T, sigma)
        if premium <= 0:
            return False
        cost = lot * 100 * premium
        if self.cash < cost:
            return False
        self.cash -= cost
        expiry = d + timedelta(days=params["dte_days"])
        self.positions.append(Position(d, expiry, K, premium, lot * 100, signal_mode))
        return True

    def _close(self, pos: Position, d: date, S: float, sigma: float, reason: str):
        exit_premium = pos.current_premium(d, S, sigma)
        proceeds = exit_premium * pos.shares
        pnl = (exit_premium - pos.entry_premium) / pos.entry_premium
        self.cash += proceeds
        self.positions.remove(pos)
        self.trades.append(Trade(pos.entry_date, d, pos.entry_premium, exit_premium, pnl, reason, pos.shares, pos.signal_mode))

    def _partial_close(self, pos: Position, d: date, S: float, sigma: float, reason: str, close_pct: float):
        contracts_to_close = max(1, round(pos.contracts * close_pct))
        if contracts_to_close >= pos.contracts:
            self._close(pos, d, S, sigma, reason)
            return
        shares_to_close = contracts_to_close * 100
        exit_premium = pos.current_premium(d, S, sigma)
        pnl = (exit_premium - pos.entry_premium) / pos.entry_premium
        self.cash += exit_premium * shares_to_close
        pos.shares -= shares_to_close
        pos.used_tiers = [*pos.used_tiers, reason]
        self.trades.append(Trade(pos.entry_date, d, pos.entry_premium, exit_premium, pnl, f"{reason}_partial", shares_to_close, pos.signal_mode))

    # ── daily step ─────────────────────────────────────────────────────────────

    def step(self, d: date, S: float, sigma: float, signal: str | None, params: dict) -> float:
        """Process one trading day. Returns end-of-day NAV.

        `signal` is None (no entry today) or a string signal_mode ("A"/"B")
        naming which mode fired. If both fired on the same day, the caller
        resolves priority before calling step() (Mode A wins — see backtest.py).
        """
        max_dep = params.get("max_deploy_pct", self.max_deploy_pct)
        min_rem = params.get("min_months_remaining", 6)

        # 1. DTE exit: proactively sell when < min_months_remaining to expiry
        for pos in list(self.positions):
            if pos.months_to_expiry(d) < min_rem:
                self._close(pos, d, S, sigma, "dte")

        # 2. Tiered profit / force exits
        for pos in list(self.positions):
            months = pos.months_held(d)
            pnl = pos.pnl_pct(d, S, sigma)
            reason = _exit_reason(months, pnl, params, pos.used_tiers)
            if reason:
                close_pct = params.get(f"{reason}_close_pct", 1.0) if reason.startswith("tp") else 1.0
                if close_pct < 1.0:
                    self._partial_close(pos, d, S, sigma, reason, close_pct)
                else:
                    self._close(pos, d, S, sigma, reason)

        # 3. Record NAV
        current_nav = self.nav(d, S, sigma)
        self.curve.append((d, current_nav))

        # 4. Entry: compute whole contracts from NAV × lot_pct, FIFO if over cap
        if signal:
            T = params["dte_days"] / 365.0
            K = strike_for_delta(S, T, sigma, params["target_delta"])
            premium = call_price(S, K, T, sigma)
            lot_pct = params.get("lot_pct", 0.05)
            contracts = max(1, int(current_nav * lot_pct / (premium * 100)))
            lot_cost = contracts * 100 * premium

            # FIFO-rotate oldest until there's room under the NAV cap
            while (self.option_value(d, S, sigma) + lot_cost > current_nav * max_dep
                   and self.positions):
                oldest = min(self.positions, key=lambda p: p.entry_date)
                self._close(oldest, d, S, sigma, "fifo")

            if self.option_value(d, S, sigma) + lot_cost <= current_nav * max_dep:
                self._open(d, S, sigma, {**params, "lot_size": contracts}, signal_mode=signal)

        return self.nav(d, S, sigma)


def _exit_reason(months: float, pnl: float, p: dict, used_tiers: list | None = None) -> str | None:
    """Return exit reason string or None if position should be held.

    used_tiers: tiers already partially closed on this position — skipped here.
    """
    used = used_tiers or []
    if months > p["force_months"]:
        return "force"
    if months < p.get("min_hold_months", 0):
        return None
    if "tp1" not in used and months <= p["tier1_months"] and pnl >= p["tier1_profit"]:
        return "tp1"
    if "tp2" not in used and months <= p["tier2_months"] and pnl >= p["tier2_profit"]:
        return "tp2"
    if "tp3" not in used and months <= p["tier3_months"] and pnl >= p["tier3_profit"]:
        return "tp3"
    return None
