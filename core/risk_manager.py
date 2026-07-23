"""
Risk management: position sizing and account-level circuit breakers.

This is deliberately separate from the strategies -- strategies decide WHEN
to enter and WHERE the stop/target go; this module decides HOW MUCH to risk
and whether trading should be halted.
"""
from dataclasses import dataclass


@dataclass
class RiskConfig:
    account_balance: float = 10_000.0
    risk_per_trade_pct: float = 1.0       # % of balance risked per trade
    max_daily_loss_pct: float = 3.0       # halt new trades for the day if hit
    max_drawdown_pct: float = 10.0        # halt ALL trading if hit (needs manual reset)
    max_concurrent_trades: int = 3
    pip_value_per_lot: float = 10.0       # approx USD per pip per standard lot (varies by pair)


class RiskManager:
    def __init__(self, config: RiskConfig):
        self.cfg = config
        self.balance = config.account_balance
        self.peak_balance = config.account_balance
        self.daily_start_balance = config.account_balance
        self.current_date = None
        self.open_trades = 0
        self.trading_halted = False  # tripped by max drawdown, needs manual reset
        self.halt_reason = None

    def new_bar(self, bar_date):
        """Call once per bar/day to roll the daily-loss counter."""
        if self.current_date is None or bar_date != self.current_date:
            self.current_date = bar_date
            self.daily_start_balance = self.balance

    def can_open_trade(self) -> tuple[bool, str]:
        if self.trading_halted:
            return False, self.halt_reason
        if self.open_trades >= self.cfg.max_concurrent_trades:
            return False, "max concurrent trades reached"

        daily_loss_pct = (self.daily_start_balance - self.balance) / self.daily_start_balance * 100
        if daily_loss_pct >= self.cfg.max_daily_loss_pct:
            return False, "daily loss limit hit"

        drawdown_pct = (self.peak_balance - self.balance) / self.peak_balance * 100
        if drawdown_pct >= self.cfg.max_drawdown_pct:
            self.trading_halted = True
            self.halt_reason = f"max drawdown breached ({drawdown_pct:.1f}%) - manual reset required"
            return False, self.halt_reason

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

    def on_trade_close(self, pnl: float):
        self.balance += pnl
        self.peak_balance = max(self.peak_balance, self.balance)
        self.open_trades = max(0, self.open_trades - 1)

    def reset_halt(self):
        """Manual override to resume trading after a max-drawdown halt."""
        self.trading_halted = False
        self.halt_reason = None
