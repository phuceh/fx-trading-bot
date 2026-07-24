"""
Mock integration test for live/mt5_runner.py - simulates the MetaTrader5
module's API using our own historical CSV data, so we can exercise the
runner's decision logic (bar detection, signal generation, lot sizing,
risk limits) without needing an actual Windows/MT5 install.

This does NOT test real MT5 connectivity, order execution, or broker-
specific behavior (fill modes, symbol suffixes, etc.) - only that the
runner's own logic behaves sensibly given realistic inputs.
"""
import sys
import os
import types
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data_loader import load_from_csv


class FakeAccountInfo:
    def __init__(self, balance=10000.0, login=12345, currency="USD"):
        self.balance = balance
        self.login = login
        self.currency = currency


class FakeSymbolInfo:
    def __init__(self):
        self.trade_tick_value = 1.0
        self.trade_tick_size = 0.00001
        self.volume_min = 0.01
        self.volume_max = 100.0
        self.volume_step = 0.01


class FakeTick:
    def __init__(self, bid, ask):
        self.bid = bid
        self.ask = ask


class FakeMT5:
    """Simulates enough of the MetaTrader5 module's API to exercise
    live/mt5_runner.py's logic using a real historical CSV as the data
    source, advancing bar-by-bar like a live feed would."""

    TIMEFRAME_H1 = "H1"
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    TRADE_ACTION_DEAL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC = 1

    def __init__(self, full_df: pd.DataFrame, current_index: int, balance=10000.0):
        self.full_df = full_df
        self.current_index = current_index  # simulates "now" - only bars up to here are visible
        self.balance = balance
        self.orders_placed = []

    def copy_rates_from_pos(self, symbol, timeframe, start, n_bars):
        end = self.current_index + 1
        start_idx = max(0, end - n_bars)
        window = self.full_df.iloc[start_idx:end]
        rates = []
        for ts, row in window.iterrows():
            rates.append((int(ts.timestamp()), row["open"], row["high"], row["low"],
                          row["close"], int(row.get("volume", 0)), 0, 0))
        dtype = [("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
                 ("close", "f8"), ("tick_volume", "i8"), ("spread", "i4"), ("real_volume", "i8")]
        return np.array(rates, dtype=dtype)

    def account_info(self):
        return FakeAccountInfo(balance=self.balance)

    def symbol_info(self, symbol):
        return FakeSymbolInfo()

    def symbol_info_tick(self, symbol):
        price = self.full_df.iloc[self.current_index]["close"]
        return FakeTick(bid=price - 0.0001, ask=price + 0.0001)

    def positions_get(self, symbol=None):
        return []

    def order_send(self, request):
        self.orders_placed.append(request)
        return types.SimpleNamespace(retcode=10009)  # TRADE_RETCODE_DONE


def run_test():
    import configparser
    cfg = configparser.ConfigParser()
    cfg.read_dict({
        "strategy": {"streak_length": "4", "rsi_filter": "true", "rsi_threshold": "50",
                     "stop_atr_mult": "0.75", "rr_ratio": "2.5"},
        "risk": {"risk_per_trade_pct": "1.0", "max_daily_loss_pct": "3.0",
                 "max_drawdown_pct": "10.0", "max_concurrent_trades": "3"},
    })

    import live.mt5_runner as runner_module

    df = load_from_csv("data/EURUSD_H1.csv")
    n = len(df)
    test_indices = range(200, min(400, n))  # simulate 200 consecutive bar closes

    state = {"last_bar_time": {}, "daily_start_balance": None, "daily_date": None,
             "peak_balance": None, "trading_halted": False, "halt_reason": None}

    signals_found = 0
    errors = 0
    for i in test_indices:
        fake_mt5 = FakeMT5(df, current_index=i, balance=10000.0)
        try:
            runner_module.process_symbol(fake_mt5, "EURUSD", cfg, state, dry_run=True)
            if os.path.exists(runner_module.DECISION_LOG_PATH):
                last_row = pd.read_csv(runner_module.DECISION_LOG_PATH).iloc[-1]
                if "would open" in str(last_row.get("action", "")):
                    signals_found += 1
        except Exception as e:
            errors += 1
            print(f"ERROR at index {i}: {e}")

    print(f"\nProcessed {len(list(test_indices))} simulated bar closes.")
    print(f"Signals detected: {signals_found}")
    print(f"Errors: {errors}")

    if os.path.exists(runner_module.DECISION_LOG_PATH):
        os.remove(runner_module.DECISION_LOG_PATH)
    if os.path.exists(runner_module.STATE_FILE):
        os.remove(runner_module.STATE_FILE)

    if errors == 0:
        print("\nPASS: runner logic executed without errors across 200 simulated bar closes.")
    else:
        print("\nFAIL: see errors above.")


if __name__ == "__main__":
    run_test()
