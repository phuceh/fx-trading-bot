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
    initial_stop_distance: float = None  # risk in price terms at entry, used for breakeven/trailing triggers
    peak_price: float = None             # best price reached so far (for trailing)
    breakeven_triggered: bool = False
    entry_bar_index: int = None          # bar index at entry, for time-based exits
    exit_time: pd.Timestamp = None
    exit_price: float = None
    pnl: float = None
    exit_reason: str = None


@dataclass
class TrailingConfig:
    """Optional dynamic exit management, applied on top of the strategy's
    initial stop/target. All distances are expressed as a multiple of the
    trade's OWN initial risk (entry-to-stop distance), not ATR or pips -
    keeps this independent of instrument/timeframe.

    breakeven_at_rr: once the trade is up by this multiple of initial risk,
        move the stop to entry price (locks in "can't lose" but not yet
        trailing further). None disables this.
    trail_activate_rr: once the trade is up by this multiple of initial
        risk, start trailing the stop behind the peak price. None disables
        trailing entirely (stop stays fixed, or at breakeven if triggered).
    trail_distance_rr: how far behind the peak price to trail the stop,
        in multiples of initial risk. Only relevant if trail_activate_rr
        is set.
    remove_fixed_tp: if True, the strategy's original take-profit is
        ignored once trailing has activated - the trade only exits via
        stop (fixed, breakeven, or trailing) or end of data. Lets winners
        run further than the fixed target, at the cost of giving back
        some profit before the trail catches it.
    """
    breakeven_at_rr: float = None
    trail_activate_rr: float = None
    trail_distance_rr: float = None
    remove_fixed_tp: bool = False
    max_hold_bars: int = None  # time-based exit: close at market if still open after N bars


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
                               pair: str, spread_pips: float = 1.0,
                               trailing: "TrailingConfig" = None) -> dict:
    trades: list[Trade] = []
    open_trade: Trade = None
    equity_curve = []
    pip_size = 0.01 if "JPY" in pair else 0.0001
    spread_price = spread_pips * pip_size

    idx = df.index
    dates = idx.date
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    long_sig = df["long_signal"].to_numpy() if "long_signal" in df.columns else None
    short_sig = df["short_signal"].to_numpy() if "short_signal" in df.columns else None
    n = len(df)

    for i in range(n - 1):
        risk_mgr.new_bar(dates[i])

        # --- manage open trade ---
        if open_trade is not None:
            # update peak price and apply breakeven/trailing BEFORE checking
            # for exits this bar, using this bar's high/low as the extremes
            # reached (favorable-side first, so a trailing stop can still
            # catch a reversal within the same bar - conservative-ish).
            if trailing is not None:
                if open_trade.direction == "long":
                    open_trade.peak_price = max(open_trade.peak_price, highs[i])
                    favorable_move = open_trade.peak_price - open_trade.entry_price
                else:
                    open_trade.peak_price = min(open_trade.peak_price, lows[i])
                    favorable_move = open_trade.entry_price - open_trade.peak_price

                risk_unit = open_trade.initial_stop_distance
                favorable_rr = favorable_move / risk_unit if risk_unit > 0 else 0

                if (trailing.breakeven_at_rr is not None and not open_trade.breakeven_triggered
                        and favorable_rr >= trailing.breakeven_at_rr):
                    new_stop = open_trade.entry_price
                    if open_trade.direction == "long":
                        open_trade.stop_loss = max(open_trade.stop_loss, new_stop)
                    else:
                        open_trade.stop_loss = min(open_trade.stop_loss, new_stop)
                    open_trade.breakeven_triggered = True

                if (trailing.trail_activate_rr is not None
                        and favorable_rr >= trailing.trail_activate_rr):
                    trail_dist = trailing.trail_distance_rr * risk_unit
                    if open_trade.direction == "long":
                        new_stop = open_trade.peak_price - trail_dist
                        open_trade.stop_loss = max(open_trade.stop_loss, new_stop)
                    else:
                        new_stop = open_trade.peak_price + trail_dist
                        open_trade.stop_loss = min(open_trade.stop_loss, new_stop)

            hit_sl = hit_tp = hit_time = False
            if open_trade.direction == "long":
                hit_sl = lows[i] <= open_trade.stop_loss
                if not (trailing is not None and trailing.remove_fixed_tp):
                    hit_tp = highs[i] >= open_trade.take_profit
            else:
                hit_sl = highs[i] >= open_trade.stop_loss
                if not (trailing is not None and trailing.remove_fixed_tp):
                    hit_tp = lows[i] <= open_trade.take_profit

            if (not hit_sl and not hit_tp and trailing is not None
                    and trailing.max_hold_bars is not None
                    and (i - open_trade.entry_bar_index) >= trailing.max_hold_bars):
                hit_time = True

            if hit_sl or hit_tp or hit_time:
                if hit_time:
                    exit_price = opens[i]  # exit at this bar's open (conservative: not the best price)
                    reason = "time_exit"
                else:
                    # conservative: if both hit in the same bar, assume SL first
                    exit_price = open_trade.stop_loss if hit_sl else open_trade.take_profit
                    reason = "stop_loss" if hit_sl else "take_profit"
                    if hit_sl and open_trade.breakeven_triggered and exit_price == open_trade.entry_price:
                        reason = "breakeven_stop"
                price_diff = (exit_price - open_trade.entry_price) if open_trade.direction == "long" \
                    else (open_trade.entry_price - exit_price)
                pips = _pips(price_diff, pair)
                pnl = pips * open_trade.lots * risk_mgr.cfg.pip_value_per_lot

                open_trade.exit_time = idx[i]
                open_trade.exit_price = exit_price
                open_trade.pnl = pnl
                open_trade.exit_reason = reason
                risk_mgr.on_trade_close(pnl)
                trades.append(open_trade)
                open_trade = None

        # --- open new trade on signal ---
        if open_trade is None:
            direction = None
            if long_sig is not None and long_sig[i]:
                direction = "long"
            elif short_sig is not None and short_sig[i]:
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
                                entry_time=idx[i + 1],
                                entry_price=entry_price,
                                stop_loss=params["stop_loss"],
                                take_profit=params["take_profit"],
                                lots=lots,
                                initial_stop_distance=abs(entry_price - params["stop_loss"]),
                                peak_price=entry_price,
                                entry_bar_index=i + 1,
                            )
                            risk_mgr.on_trade_open()

        equity_curve.append((idx[i], risk_mgr.balance))

    equity_df = pd.DataFrame(equity_curve, columns=["time", "balance"]).set_index("time") \
        if equity_curve else pd.DataFrame()

    return {
        "trades": trades,
        "equity_curve": equity_df,
        "final_balance": risk_mgr.balance,
        "trading_halted": risk_mgr.trading_halted,
        "halt_reason": risk_mgr.halt_reason,
    }
