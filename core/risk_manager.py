"""
Risk management: position sizing and account-level circuit breakers.

This is deliberately separate from the strategies -- strategies decide WHEN
to enter and WHERE the stop/target go; this module decides HOW MUCH to risk
and whether trading should be halted.

CHANGE (Oct 2026): the max-drawdown halt is now tracked PER SYMBOL rather
than account-wide. The original account-wide version meant a single manual
reset reset every symbol's drawdown baseline simultaneously, regardless of
where each symbol's own P&L happened to sit at that moment - a pair reset
right after a losing stretch got a flattering "fresh start" baseline that
made ordinary mean-reversion look like an edge, while a pair reset at a
different point in its own cycle didn't get the same benefit. Tracking per
symbol means a reset (manual or automatic) only ever affects the symbol it
actually applies to.

The halt also now clears itself automatically after `drawdown_cooloff_days`
rather than requiring a manual restart, and every halt/reset is recorded in
`risk_events` so reset timing can always be checked against performance
after the fact, instead of being invisible.

Daily-loss and max-concurrent-trades limits remain account-wide - those are
genuinely account-level concerns (protecting the whole account in one day,
limiting total simultaneous exposure), not per-symbol ones.
"""
from dataclasses import dataclass


@dataclass
class RiskConfig:
    account_balance: float = 10_000.0
    risk_per_trade_pct: float = 1.0       # % of balance risked per trade
    max_daily_loss_pct: float = 3.0       # halt new trades for the day if hit (account-wide)
    max_drawdown_pct: float = 10.0        # halt a SYMBOL if its own drawdown hits this
    max_concurrent_trades: int = 3        # account-wide
    pip_value_per_lot: float = 10.0       # approx USD per pip per standard lot (varies by pair)
    drawdown_cooloff_days: float = 3.0    # auto-resume a halted symbol after this many days


class RiskManager:
    def __init__(self, config: RiskConfig):
        self.cfg = config
        self.balance = config.account_balance
        self.daily_start_balance = config.account_balance
        self.current_date = None
        self.current_datetime = None
        self.open_trades = 0

        # per-symbol drawdown tracking, decoupled from the shared account balance
        self._symbol_cum_pnl = {}       # symbol -> running sum of that symbol's own realized P&L
        self._symbol_peak = {}          # symbol -> peak of that symbol's own cumulative P&L
        self._symbol_halted_since = {}  # symbol -> datetime halt started, or None if not halted

        self.risk_events = []           # list of dicts, for writing to a log / CSV

    def new_bar(self, bar_datetime):
        """Call once per bar with the bar's full datetime (not just date) -
        the cooloff timer needs real elapsed time, not just a changed date."""
        bar_date = bar_datetime.date() if hasattr(bar_datetime, "date") else bar_datetime
        if self.current_date is None or bar_date != self.current_date:
            self.current_date = bar_date
            self.daily_start_balance = self.balance
        self.current_datetime = bar_datetime
        self._check_auto_reset()

    def _check_auto_reset(self):
        if self.current_datetime is None:
            return
        for symbol, halted_since in list(self._symbol_halted_since.items()):
            if halted_since is None:
                continue
            elapsed_days = (self.current_datetime - halted_since).total_seconds() / 86400
            if elapsed_days >= self.cfg.drawdown_cooloff_days:
                self._symbol_halted_since[symbol] = None
                # resume from the current level - do NOT restore the old peak,
                # that would just re-create the same "forgiven drawdown" problem
                self._symbol_peak[symbol] = self._symbol_cum_pnl.get(symbol, 0.0)
                self.risk_events.append({
                    "timestamp": self.current_datetime, "symbol": symbol, "event": "auto_reset",
                    "note": f"auto-resumed after {self.cfg.drawdown_cooloff_days}-day cooloff",
                })

    def can_open_trade(self, symbol: str) -> tuple[bool, str]:
        if self._symbol_halted_since.get(symbol) is not None:
            halted_since = self._symbol_halted_since[symbol]
            remaining = self.cfg.drawdown_cooloff_days
            if self.current_datetime is not None:
                elapsed_days = (self.current_datetime - halted_since).total_seconds() / 86400
                remaining = max(0.0, self.cfg.drawdown_cooloff_days - elapsed_days)
            return False, f"{symbol} halted on max drawdown - auto-resume in {remaining:.1f}d"

        if self.open_trades >= self.cfg.max_concurrent_trades:
            return False, "max concurrent trades reached"

        daily_loss_pct = (self.daily_start_balance - self.balance) / self.daily_start_balance * 100
        if daily_loss_pct >= self.cfg.max_daily_loss_pct:
            return False, "daily loss limit hit"

        cum = self._symbol_cum_pnl.get(symbol, 0.0)
        peak = self._symbol_peak.get(symbol, 0.0)
        drawdown_pct = (peak - cum) / self.cfg.account_balance * 100
        if drawdown_pct >= self.cfg.max_drawdown_pct:
            self._symbol_halted_since[symbol] = self.current_datetime
            self.risk_events.append({
                "timestamp": self.current_datetime, "symbol": symbol, "event": "halted",
                "note": f"drawdown {drawdown_pct:.1f}% (symbol-specific)",
            })
            return False, f"{symbol} max drawdown breached ({drawdown_pct:.1f}%) - auto-reset in {self.cfg.drawdown_cooloff_days}d"

        return True, ""

    def position_size_lots(self, stop_loss_pips: float) -> float:
        """Risk-based position size. Returns lots (0 if stop is invalid)."""
        if stop_loss_pips <= 0:
            return 0.0
        risk_amount = self.balance * (self.cfg.risk_per_trade_pct / 100)
        pips_value_needed = risk_amount / stop_loss_pips
        lots = pips_value_needed / self.cfg.pip_value_per_lot
        return round(max(lots, 0.0), 2)

    def on_trade_open(self):
        self.open_trades += 1

    def on_trade_close(self, pnl: float, symbol: str):
        self.balance += pnl
        self.open_trades = max(0, self.open_trades - 1)
        self._symbol_cum_pnl[symbol] = self._symbol_cum_pnl.get(symbol, 0.0) + pnl
        self._symbol_peak[symbol] = max(self._symbol_peak.get(symbol, 0.0), self._symbol_cum_pnl[symbol])

    def reset_halt(self, symbol: str):
        """Manual override to resume trading a specific symbol early,
        before its cooloff period elapses. Still logged."""
        if self._symbol_halted_since.get(symbol) is not None:
            self._symbol_halted_since[symbol] = None
            self._symbol_peak[symbol] = self._symbol_cum_pnl.get(symbol, 0.0)
            self.risk_events.append({
                "timestamp": self.current_datetime, "symbol": symbol, "event": "manual_reset",
                "note": "manually reset before cooloff elapsed",
            })
