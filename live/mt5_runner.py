"""
Live/demo MT5 runner for the Fade the Streak + RSI strategy.

Polls MT5 every `poll_seconds`, checks for a newly-closed H1 candle on each
configured symbol, evaluates the strategy, and places/manages trades
through MT5 (or just logs what it WOULD do, if dry_run=true).

Incorporates fixes made across this project's live-shakedown period:
  - fill-mode fallback (tries IOC, then FOK, then RETURN - brokers vary)
  - human-readable retcode messages in the log
  - robust close detection (won't log a blank row if a position merely
    looks "not open" transiently - only finalizes on a real closing deal)
  - per-symbol drawdown tracking with automatic cooloff-based reset, and
    a risk_events.csv log of every halt/reset (Oct 2026 change - see
    core/risk_manager.py for the full reasoning)
"""
import configparser
import csv
import json
import logging
import os
import time
from datetime import datetime, timezone

import pandas as pd

from core.risk_manager import RiskManager, RiskConfig
from strategies import strategy_fade

MAGIC_NUMBER = 20260724
STATE_PATH = os.path.join(os.path.dirname(__file__), "live_state.json")
DECISION_LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "decisions.csv")
TRADE_LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "trades.csv")
RISK_EVENT_LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "risk_events.csv")
RUNNER_LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "runner.log")

os.makedirs(os.path.join(os.path.dirname(__file__), "logs"), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(RUNNER_LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def send_order_with_fallback_filling(mt5, base_request: dict):
    """Different brokers/symbols support different order-filling modes, and
    there's no reliable way to know which without asking - so try the
    common ones in order and use whichever the broker actually accepts.
    Logs which one worked so you know for next time."""
    fill_modes = [
        ("IOC", mt5.ORDER_FILLING_IOC),
        ("FOK", mt5.ORDER_FILLING_FOK),
        ("RETURN", mt5.ORDER_FILLING_RETURN),
    ]
    last_result = None
    for name, mode in fill_modes:
        request = {**base_request, "type_filling": mode}
        result = mt5.order_send(request)
        if result and result.retcode == 10009:
            log.info(f"Order filled using '{name}' filling mode.")
            return result
        last_result = result
        if result and result.retcode != 10030:  # 10030 = INVALID_FILL specifically
            break  # a different error - retrying with another fill mode won't help
    return last_result


def load_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.ini")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing {cfg_path} - copy config.example.ini and fill in your details.")
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)
    return cfg


def log_trade_open(state, ticket, symbol, direction, lots, entry_price, stop_loss, take_profit, bar_time):
    state["open_positions"][str(ticket)] = {
        "symbol": symbol, "direction": direction, "lots": lots,
        "entry_price": entry_price, "stop_loss": stop_loss, "take_profit": take_profit,
        "bar_time": str(bar_time), "open_time": datetime.now(timezone.utc).isoformat(),
    }


def check_closed_positions(mt5, state, risk_mgr: RiskManager):
    open_positions = state["open_positions"]
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

        if exit_price is None:
            # Not in the open-positions list, but no closing deal found
            # either - most likely a transient hiccup in the positions_get
            # query, not a real close. Keep tracking it and check again
            # next poll, rather than logging a blank/zero row.
            log.warning(f"{info['symbol']} ticket {ticket} isn't in MT5's open positions, but no "
                        f"closing deal was found in history either - will re-check next poll "
                        f"rather than assume it closed.")
            continue

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
        risk_mgr.on_trade_close(net_pnl, info["symbol"])
        del open_positions[ticket_str]


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"last_bar_time": {}, "open_positions": {}}


def save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


def append_csv_row(path, row: dict):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def flush_risk_events(risk_mgr: RiskManager, last_flushed_count: int) -> int:
    """Append any new risk events (halts/resets) to risk_events.csv since
    the last flush. Returns the new total count flushed."""
    events = risk_mgr.risk_events
    for event in events[last_flushed_count:]:
        append_csv_row(RISK_EVENT_LOG_PATH, {
            "timestamp": event["timestamp"], "symbol": event["symbol"],
            "event": event["event"], "note": event["note"],
        })
        log.warning(f"RISK EVENT: {event['symbol']} {event['event']} - {event['note']}")
    return len(events)


def fetch_bars(mt5, symbol: str, n_bars: int = 300) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, n_bars)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").rename(columns={"tick_volume": "volume"})
    return df[["open", "high", "low", "close", "volume"]]


def compute_lot_size(mt5, symbol: str, account_balance: float, risk_pct: float,
                      stop_distance_price: float) -> float:
    info = mt5.symbol_info(symbol)
    pip_size = 0.01 if "JPY" in symbol else 0.0001
    stop_pips = stop_distance_price / pip_size
    tick_value = info.trade_tick_value
    tick_size = info.trade_tick_size
    pip_value_per_lot = (pip_size / tick_size) * tick_value if tick_size else 10.0
    risk_amount = account_balance * (risk_pct / 100)
    lots = risk_amount / (stop_pips * pip_value_per_lot) if stop_pips > 0 else 0
    lots = max(info.volume_min, min(info.volume_max, round(lots / info.volume_step) * info.volume_step))
    return round(lots, 2)


def check_risk_limits(state, balance: float, cfg) -> tuple:
    return True, ""  # superseded by risk_mgr.can_open_trade(symbol) per-symbol check


def process_symbol(mt5, symbol: str, cfg, state, risk_mgr: RiskManager, dry_run: bool):
    df = fetch_bars(mt5, symbol, n_bars=300)
    last_closed_bar_time = df.index[-2]
    last_seen = state["last_bar_time"].get(symbol)
    if last_seen == str(last_closed_bar_time):
        return  # already processed this bar

    risk_mgr.new_bar(last_closed_bar_time)

    sig = strategy_fade.generate_signals(
        df,
        streak_length=int(cfg["strategy"]["streak_length"]),
        rsi_filter=cfg["strategy"].getboolean("rsi_filter"),
        rsi_threshold=float(cfg["strategy"]["rsi_threshold"]),
    )
    i = len(df) - 2  # index of the last fully-closed bar
    row = sig.iloc[i]

    decision_row = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "symbol": symbol,
        "bar_time": str(last_closed_bar_time), "action": "no signal",
        "reason": "no signal", "mt5_result": None, "lots": None, "sl": None, "tp": None,
    }

    direction = "long" if row["long_signal"] else ("short" if row["short_signal"] else None)

    if direction:
        params = strategy_fade.build_trade_params(
            sig, i, direction,
            stop_atr_mult=float(cfg["strategy"]["stop_atr_mult"]),
            rr_ratio=float(cfg["strategy"]["rr_ratio"]),
        )
        if params is None:
            decision_row["action"] = "skipped: invalid ATR/params"
        else:
            existing = mt5.positions_get(symbol=symbol)
            if existing:
                decision_row["action"] = "skipped: position already open on this symbol"
            else:
                can_open, reason = risk_mgr.can_open_trade(symbol)
                if not can_open:
                    decision_row["action"] = f"skipped: {reason}"
                else:
                    account_info = mt5.account_info()
                    lots = compute_lot_size(mt5, symbol, account_info.balance,
                                             float(cfg["risk"]["risk_per_trade_pct"]),
                                             params["stop_distance"])
                    decision_row["lots"] = lots
                    decision_row["sl"] = params["stop_loss"]
                    decision_row["tp"] = params["take_profit"]

                    if dry_run:
                        decision_row["action"] = f"[DRY RUN] would open {direction} {lots} lots"
                        log.info(f"{symbol} @ {last_closed_bar_time}: [DRY RUN] {direction} "
                                 f"{lots} lots, SL={params['stop_loss']:.5f} TP={params['take_profit']:.5f}")
                    else:
                        order_type = mt5.ORDER_TYPE_BUY if direction == "long" else mt5.ORDER_TYPE_SELL
                        tick = mt5.symbol_info_tick(symbol)
                        price = tick.ask if direction == "long" else tick.bid
                        base_request = {
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
                        }
                        result = send_order_with_fallback_filling(mt5, base_request)
                        decision_row["mt5_result"] = str(result.retcode) if result else "FAILED"
                        log.info(f"Order sent for {symbol}: retcode={decision_row.get('mt5_result')}")

                        RETCODE_MEANINGS = {
                            10009: "Done - order placed successfully",
                            10027: "AutoTrading is DISABLED in the MT5 terminal - click the "
                                   "'Algo Trading' button in the MT5 toolbar to enable it, then "
                                   "no action needed here, it'll pick up the next signal",
                            10018: "Market is closed for this symbol right now",
                            10019: "Not enough money in the account for this trade size",
                            10004: "Requote - price moved before the order could fill",
                            10006: "Order rejected by the broker/server",
                            10013: "Invalid request - check symbol/volume/price are all valid",
                            10030: "None of the order-filling modes tried (IOC/FOK/RETURN) were "
                                   "accepted by this broker for this symbol - unusual, worth "
                                   "checking the symbol's trading specification in MT5",
                        }
                        if result and result.retcode != 10009:
                            meaning = RETCODE_MEANINGS.get(result.retcode, "Unknown error - check MT5's Journal tab for details")
                            log.error(f"ORDER FAILED for {symbol}: retcode {result.retcode} - {meaning}")

                        if result and result.retcode == 10009:
                            decision_row["action"] = f"OPENING {direction} {lots} lots, SL={params['stop_loss']:.5f} TP={params['take_profit']:.5f}"
                            log.info(f"{symbol} @ {last_closed_bar_time}: OPENING {direction} {lots} lots, "
                                     f"SL={params['stop_loss']:.5f} TP={params['take_profit']:.5f}")
                            new_positions = mt5.positions_get(symbol=symbol, magic=MAGIC_NUMBER) or []
                            if new_positions:
                                pos = new_positions[-1]
                                log_trade_open(state, pos.ticket, symbol, direction, lots,
                                               pos.price_open, params["stop_loss"], params["take_profit"],
                                               last_closed_bar_time)
                                risk_mgr.on_trade_open()
                        else:
                            decision_row["action"] = f"FAILED to open: retcode={decision_row['mt5_result']}"
    else:
        decision_row["action"] = "no signal"

    append_csv_row(DECISION_LOG_PATH, decision_row)
    state["last_bar_time"][symbol] = str(last_closed_bar_time)


def main():
    cfg = load_config()
    import MetaTrader5 as mt5

    if not mt5.initialize(login=int(cfg["mt5"]["login"]), password=cfg["mt5"]["password"], server=cfg["mt5"]["server"]):
        log.error(f"MT5 initialize failed: {mt5.last_error()}")
        return

    dry_run = cfg["run"].getboolean("dry_run")
    if dry_run:
        log.warning("Running in DRY RUN mode - no real orders will be placed.")
    else:
        log.warning("!!! LIVE MODE - this WILL place real orders on your account (even if it's a demo) !!!")

    pairs = [p.strip() for p in cfg["strategy"]["pairs"].split(",")]
    state = load_state()

    account_info = mt5.account_info()
    risk_mgr = RiskManager(RiskConfig(
        account_balance=account_info.balance,
        risk_per_trade_pct=float(cfg["risk"]["risk_per_trade_pct"]),
        max_daily_loss_pct=float(cfg["risk"]["max_daily_loss_pct"]),
        max_drawdown_pct=float(cfg["risk"]["max_drawdown_pct"]),
        max_concurrent_trades=int(cfg["risk"]["max_concurrent_trades"]),
        drawdown_cooloff_days=float(cfg["risk"].get("drawdown_cooloff_days", "3.0")),
    ))
    last_flushed_risk_events = 0

    poll_seconds = int(cfg["run"]["poll_seconds"])
    log.info(f"Starting runner for {pairs}, polling every {poll_seconds}s, "
             f"drawdown cooloff={risk_mgr.cfg.drawdown_cooloff_days} days")

    try:
        while True:
            check_closed_positions(mt5, state, risk_mgr)
            for symbol in pairs:
                try:
                    process_symbol(mt5, symbol, cfg, state, risk_mgr, dry_run)
                except Exception:
                    log.exception(f"Error processing {symbol}")
            last_flushed_risk_events = flush_risk_events(risk_mgr, last_flushed_risk_events)
            save_state(state)
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        log.info("Stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
