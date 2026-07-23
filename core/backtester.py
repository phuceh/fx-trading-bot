"""
Simple event-driven backtester.

Walks forward bar by bar. On each bar:
  1. Check open trades for stop-loss / take-profit hits (using that bar's
     high/low - conservative: if both SL and TP are within the same bar's
     range we assume SL was hit first, since we can't know intrabar order
     from OHLC alone).
  2. If a new signal fired on the previous bar, open a trade sized by the
     RiskManager.

This is intentionally simple (one position per signal, no partial closes,
no slippage/spread modeling) -- it's a starting point for validating
strategy logic, not a broker-grade simulator. See README for what to add
before trusting results with real money.
"""
from dataclasses import dataclass, field
import pandas as pd
from core.risk_manager import RiskManager


@dataclass
class Trade:
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    lots: float
    exit_time: pd.Timestamp = None
    exit_price: float = None
    pnl: float = None
    exit_reason: str = None


def _pips(price_diff: float, pair: str) -> float:
    # JPY pairs use 2 decimal pips; everything else here (no JPY pairs in
    # your list) uses 4 decimal pips. Kept general in case you add JPY pairs.
    pip_size = 0.01 if "JPY" in pair else 0.0001
    return price_diff / pip_size


def run_backtest(df: pd.DataFrame, risk_mgr: RiskManager, pair: str,
                  spread_pips: float = 1.0) -> dict:
    """
    df must already contain 'long_signal' and 'short_signal' boolean columns
    plus a way to compute trade params - pass a `build_trade_params_fn`
    closure instead if strategies differ; here we expect the caller to have
    pre-computed stop/target per bar (see run_backtest_with_builder below).
    """
    raise NotImplementedError("Use run_backtest_with_builder")


def run_backtest_with_builder(df: pd.DataFrame, build_trade_params_fn, risk_mgr: RiskManager,
                               pair: str, spread_pips: float = 1.0) -> dict:
    trades: list[Trade] = []
    open_trade: Trade = None
    equity_curve = []
    pip_size = 0.01 if "JPY" in pair else 0.0001
    spread_price = spread_pips * pip_size

    for i in range(len(df) - 1):
        row = df.iloc[i]
        risk_mgr.new_bar(row.name.date())

        # --- manage open trade ---
        if open_trade is not None:
            bar = df.iloc[i]
            hit_sl = hit_tp = False
            if open_trade.direction == "long":
                hit_sl = bar["low"] <= open_trade.stop_loss
                hit_tp = bar["high"] >= open_trade.take_profit
            else:
                hit_sl = bar["high"] >= open_trade.stop_loss
                hit_tp = bar["low"] <= open_trade.take_profit

            if hit_sl or hit_tp:
                # conservative: if both hit in the same bar, assume SL first
                exit_price = open_trade.stop_loss if hit_sl else open_trade.take_profit
                reason = "stop_loss" if hit_sl else "take_profit"
                price_diff = (exit_price - open_trade.entry_price) if open_trade.direction == "long" \
                    else (open_trade.entry_price - exit_price)
                pips = _pips(price_diff, pair)
                pnl = pips * open_trade.lots * risk_mgr.cfg.pip_value_per_lot

                open_trade.exit_time = bar.name
                open_trade.exit_price = exit_price
                open_trade.pnl = pnl
                open_trade.exit_reason = reason
                risk_mgr.on_trade_close(pnl)
                trades.append(open_trade)
                open_trade = None

        # --- open new trade on signal ---
        if open_trade is None:
            direction = None
            if row.get("long_signal"):
                direction = "long"
            elif row.get("short_signal"):
                direction = "short"

            if direction:
                can_open, reason = risk_mgr.can_open_trade()
                if can_open:
                    params = build_trade_params_fn(df, i, direction)
                    if params:
                        entry_price = params["entry_price"] + (spread_price if direction == "long" else 0)
                        stop_distance_pips = _pips(params["stop_distance"], pair)
                        lots = risk_mgr.position_size_lots(stop_distance_pips)
                        if lots > 0:
                            open_trade = Trade(
                                direction=direction,
                                entry_time=df.index[i + 1],
                                entry_price=entry_price,
                                stop_loss=params["stop_loss"],
                                take_profit=params["take_profit"],
                                lots=lots,
                            )
                            risk_mgr.on_trade_open()

        equity_curve.append({"time": row.name, "balance": risk_mgr.balance})

    return {
        "trades": trades,
        "equity_curve": pd.DataFrame(equity_curve).set_index("time") if equity_curve else pd.DataFrame(),
        "final_balance": risk_mgr.balance,
        "trading_halted": risk_mgr.trading_halted,
        "halt_reason": risk_mgr.halt_reason,
    }
