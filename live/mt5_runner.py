"""
MT5 live/demo runner for the Fade the Streak + RSI strategy.

Windows only (or Wine) - requires the MT5 terminal installed, logged in,
and the `MetaTrader5` Python package (`pip install MetaTrader5`).

Usage:
    1. Copy config.example.ini to config.ini and fill in your details.
    2. Run: python live/mt5_runner.py
    3. Leave dry_run = true in config.ini until you've watched it run and
       are confident it's behaving correctly. It will log every decision
       without placing any orders while dry_run is true.
    4. Set dry_run = false to actually place demo trades.

This reuses the exact same strategy logic (core/, strategies/) already
validated in backtesting - the signal-generation code here is identical to
what ran against historical data, not a reimplementation.
"""
import configparser
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategies import strategy_fade  # noqa: E402

STATE_FILE = os.path.join(os.path.dirname(__file__), "live_state.json")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "runner.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("mt5_runner")

TRADE_LOG_PATH = os.path.join(LOG_DIR, "trades.csv")
DECISION_LOG_PATH = os.path.join(LOG_DIR, "decisions.csv")


def load_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.ini")
    if not os.path.exists(cfg_path):
        raise SystemExit(
            "config.ini not found. Copy config.example.ini to config.ini "
            "and fill in your MT5 login details first."
        )
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)
    return cfg


MAGIC_NUMBER = 20260724


def log_trade_open(state, ticket, symbol, direction, lots, entry_price, stop_loss, take_profit, bar_time):
    state.setdefault("open_positions", {})
    state["open_positions"][str(ticket)] = {
        "symbol": symbol, "direction": direction, "lots": lots,
        "entry_price": entry_price, "stop_loss": stop_loss, "take_profit": take_profit,
        "bar_time": bar_time, "open_time": datetime.now(timezone.utc).isoformat(),
    }


def check_closed_positions(mt5, state):
    """Called once per poll cycle. Compares the positions we're tracking
    (opened by this bot, keyed by ticket) against what MT5 currently
    reports open. Anything we were tracking that's no longer open has
    closed - pull the actual closing deal from MT5's history for the real
    profit/loss (not an estimate) and log it to trades.csv."""
    open_positions = state.get("open_positions", {})
    if not open_positions:
        return

    still_open_tickets = {str(p.ticket) for p in (mt5.positions_get(magic=MAGIC_NUMBER) or [])}

    for ticket_str, info in list(open_positions.items()):
        if ticket_str in still_open_tickets:
            continue  # still open, nothing to do

        ticket = int(ticket_str)
        deals = mt5.history_deals_get(position=ticket)
        profit = commission = swap = 0.0
        exit_price = None
        exit_time = None
        if deals:
            for d in deals:
                if d.entry == 1:  # DEAL_ENTRY_OUT - a closing deal
                    profit += d.profit
                    commission += d.commission
                    swap += d.swap
                    exit_price = d.price
                    exit_time = pd.to_datetime(d.time, unit="s").isoformat()

        net_pnl = profit + commission + swap
        row = {
            "symbol": info["symbol"], "direction": info["direction"], "lots": info["lots"],
            "bar_time": info["bar_time"], "open_time": info["open_time"],
            "entry_price": info["entry_price"], "stop_loss": info["stop_loss"],
            "take_profit": info["take_profit"], "exit_price": exit_price, "exit_time": exit_time,
            "raw_profit_before_fees": profit, "commission": commission, "swap": swap,
            "profit_after_fees": net_pnl,
        }
        log.info(f"TRADE CLOSED: {info['symbol']} {info['direction']} {info['lots']} lots -- "
                 f"net P&L: {net_pnl:.2f}")
        append_csv_row(TRADE_LOG_PATH, row)
        del open_positions[ticket_str]

    state["open_positions"] = open_positions


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_bar_time": {}, "daily_start_balance": None, "daily_date": None,
             "peak_balance": None, "trading_halted": False, "halt_reason": None,
             "open_positions": {}}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def append_csv_row(path, row: dict):
    df = pd.DataFrame([row])
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


def fetch_bars(mt5, symbol: str, n_bars: int = 300) -> pd.DataFrame:
    tf = mt5.TIMEFRAME_H1
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n_bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No data returned for {symbol}. Is it in Market Watch?")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time").rename(columns={"tick_volume": "volume"})
    return df[["open", "high", "low", "close", "volume"]]


def compute_lot_size(mt5, symbol: str, account_balance: float, risk_pct: float,
                      stop_distance_price: float) -> float:
    """Position size using the SYMBOL'S ACTUAL tick value/size from MT5,
    more accurate than the backtester's flat pip-value assumption."""
    info = mt5.symbol_info(symbol)
    if info is None:
        return 0.0
    risk_amount = account_balance * (risk_pct / 100)
    ticks_at_risk = stop_distance_price / info.trade_tick_size if info.trade_tick_size else 0
    value_per_lot_at_risk = ticks_at_risk * info.trade_tick_value
    if value_per_lot_at_risk <= 0:
        return 0.0
    lots = risk_amount / value_per_lot_at_risk
    lots = max(info.volume_min, min(info.volume_max, round(lots / info.volume_step) * info.volume_step))
    return round(lots, 2)


def check_risk_limits(state, balance: float, cfg) -> tuple:
    today = datetime.now(timezone.utc).date().isoformat()
    if state["daily_date"] != today:
        state["daily_date"] = today
        state["daily_start_balance"] = balance
    if state["peak_balance"] is None:
        state["peak_balance"] = balance
    state["peak_balance"] = max(state["peak_balance"], balance)

    if state.get("trading_halted"):
        return False, state.get("halt_reason", "halted")

    daily_loss_pct = (state["daily_start_balance"] - balance) / state["daily_start_balance"] * 100
    if daily_loss_pct >= float(cfg["risk"]["max_daily_loss_pct"]):
        return False, f"daily loss limit hit ({daily_loss_pct:.1f}%)"

    drawdown_pct = (state["peak_balance"] - balance) / state["peak_balance"] * 100
    if drawdown_pct >= float(cfg["risk"]["max_drawdown_pct"]):
        state["trading_halted"] = True
        state["halt_reason"] = f"max drawdown breached ({drawdown_pct:.1f}%) - manual reset required"
        return False, state["halt_reason"]

    return True, ""


def process_symbol(mt5, symbol: str, cfg, state, dry_run: bool):
    df = fetch_bars(mt5, symbol)
    last_closed_bar_time = df.index[-2]  # -1 is the still-forming current bar

    key = symbol
    last_seen = state["last_bar_time"].get(key)
    if last_seen == str(last_closed_bar_time):
        return  # already processed this bar

    signals = strategy_fade.generate_signals(
        df,
        streak_length=int(cfg["strategy"]["streak_length"]),
        rsi_filter=cfg["strategy"].getboolean("rsi_filter"),
        rsi_threshold=float(cfg["strategy"]["rsi_threshold"]),
    )

    # -2 is the last CLOSED bar (the signal bar); the entry, if any, would
    # be at the open of bar -1 (the currently-forming bar) - matching the
    # backtester's "enter at next bar's open" convention.
    idx = len(signals) - 2
    row = signals.iloc[idx]
    direction = None
    if row.get("long_signal"):
        direction = "long"
    elif row.get("short_signal"):
        direction = "short"

    decision_row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "bar_time": str(last_closed_bar_time),
        "signal": direction or "none",
        "action": None,
        "lots": None,
        "stop_loss": None,
        "take_profit": None,
        "mt5_result": None,
    }

    if direction:
        account = mt5.account_info()
        balance = account.balance if account else 0.0
        can_trade, reason = check_risk_limits(state, balance, cfg)

        existing_positions = mt5.positions_get(symbol=symbol)
        already_open = existing_positions is not None and len(existing_positions) > 0
        open_count = len(mt5.positions_get() or [])
        max_concurrent = int(cfg["risk"]["max_concurrent_trades"])

        if not can_trade:
            decision_row["action"] = f"skipped: {reason}"
        elif already_open:
            decision_row["action"] = "skipped: position already open on this symbol"
        elif open_count >= max_concurrent:
            decision_row["action"] = "skipped: max concurrent trades reached"
        else:
            params = strategy_fade.build_trade_params(
                signals, idx, direction,
                stop_atr_mult=float(cfg["strategy"]["stop_atr_mult"]),
                rr_ratio=float(cfg["strategy"]["rr_ratio"]),
            )
            if params is None:
                decision_row["action"] = "skipped: invalid trade params (e.g. zero ATR)"
            else:
                lots = compute_lot_size(
                    mt5, symbol, balance, float(cfg["risk"]["risk_per_trade_pct"]),
                    params["stop_distance"],
                )
                if lots <= 0:
                    decision_row["action"] = "skipped: computed lot size was zero"
                else:
                    decision_row["action"] = f"{'[DRY RUN] would open' if dry_run else 'OPENING'} " \
                                              f"{direction} {lots} lots, SL={params['stop_loss']:.5f} " \
                                              f"TP={params['take_profit']:.5f}"
                    decision_row["lots"] = lots
                    decision_row["stop_loss"] = params["stop_loss"]
                    decision_row["take_profit"] = params["take_profit"]

                    if not dry_run:
                        order_type = mt5.ORDER_TYPE_BUY if direction == "long" else mt5.ORDER_TYPE_SELL
                        tick = mt5.symbol_info_tick(symbol)
                        price = tick.ask if direction == "long" else tick.bid
                        request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": symbol,
                            "volume": lots,
                            "type": order_type,
                            "price": price,
                            "sl": params["stop_loss"],
                            "tp": params["take_profit"],
                            "deviation": 20,
                            "magic": MAGIC_NUMBER,
                            "comment": "fade_streak_rsi_bot",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_IOC,
                        }
                        result = mt5.order_send(request)
                        decision_row["mt5_result"] = str(result.retcode) if result else "FAILED"
                        log.info(f"Order sent for {symbol}: retcode={decision_row.get('mt5_result')}")

                        if result and result.retcode == 10009:  # TRADE_RETCODE_DONE
                            # find the newly opened position to start tracking it for close-detection
                            new_positions = mt5.positions_get(symbol=symbol, magic=MAGIC_NUMBER) or []
                            matched = [p for p in new_positions if str(p.ticket) not in state.get("open_positions", {})]
                            if matched:
                                pos = matched[0]
                                log_trade_open(state, pos.ticket, symbol, direction, lots,
                                                price, params["stop_loss"], params["take_profit"],
                                                str(last_closed_bar_time))
                            else:
                                log.warning(f"Order for {symbol} confirmed but couldn't locate the "
                                            f"new position to track it - close P&L won't be logged "
                                            f"for this trade. Check MT5 History manually.")
    else:
        decision_row["action"] = "no signal"

    log.info(f"{symbol} @ {last_closed_bar_time}: {decision_row['action']}")
    append_csv_row(DECISION_LOG_PATH, decision_row)
    state["last_bar_time"][key] = str(last_closed_bar_time)


def main():
    cfg = load_config()
    dry_run = cfg["run"].getboolean("dry_run")
    poll_seconds = int(cfg["run"]["poll_seconds"])
    pairs = [p.strip() for p in cfg["strategy"]["pairs"].split(",")]

    if dry_run:
        log.warning("Running in DRY RUN mode - no real orders will be placed.")
    else:
        log.warning("!!! LIVE MODE - this WILL place real orders on your account (even if it's a demo) !!!")

    import MetaTrader5 as mt5

    login = cfg["mt5"]["login"].strip()
    password = cfg["mt5"]["password"]
    server = cfg["mt5"]["server"].strip()

    if not mt5.initialize(login=int(login), password=password, server=server):
        raise SystemExit(f"MT5 initialize() failed: {mt5.last_error()}")

    account = mt5.account_info()
    log.info(f"Connected to MT5. Account: {account.login}, balance: {account.balance} {account.currency}")

    state = load_state()

    try:
        while True:
            for symbol in pairs:
                try:
                    process_symbol(mt5, symbol, cfg, state, dry_run)
                except Exception as e:
                    log.error(f"Error processing {symbol}: {e}")

            if not dry_run:
                try:
                    check_closed_positions(mt5, state)
                except Exception as e:
                    log.error(f"Error checking closed positions: {e}")

            save_state(state)
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        log.info("Stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
