"""
Tests check_closed_positions() specifically: simulates a position that was
open, then disappears (as if it hit its stop or target), and confirms the
close gets detected and logged with the correct P&L pulled from the
(mocked) MT5 deal history.
"""
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import live.mt5_runner as runner_module


class FakePosition:
    def __init__(self, ticket, magic=20260724):
        self.ticket = ticket
        self.magic = magic


class FakeDeal:
    def __init__(self, entry, profit, commission, swap, price, time):
        self.entry = entry  # 0 = IN, 1 = OUT
        self.profit = profit
        self.commission = commission
        self.swap = swap
        self.price = price
        self.time = time


class FakeMT5ForClose:
    def __init__(self, open_tickets, deals_by_position):
        self.open_tickets = open_tickets
        self.deals_by_position = deals_by_position

    def positions_get(self, symbol=None, magic=None):
        return [FakePosition(t) for t in self.open_tickets]

    def history_deals_get(self, position=None):
        return self.deals_by_position.get(position, [])


def run_test():
    # simulate: position 111 was opened by our bot, is now closed (hit TP)
    state = {
        "open_positions": {
            "111": {"symbol": "EURUSD", "direction": "long", "lots": 1.0,
                     "entry_price": 1.1000, "stop_loss": 1.0950, "take_profit": 1.1080,
                     "bar_time": "2026-07-27 10:00:00", "open_time": "2026-07-27T10:00:00"},
            "222": {"symbol": "GBPUSD", "direction": "short", "lots": 0.5,
                     "entry_price": 1.3300, "stop_loss": 1.3350, "take_profit": 1.3200,
                     "bar_time": "2026-07-27 10:00:00", "open_time": "2026-07-27T10:00:00"},
        }
    }

    # position 111 has closed (no longer in MT5's open list), 222 is still open
    fake_mt5 = FakeMT5ForClose(
        open_tickets=[222],
        deals_by_position={
            111: [
                FakeDeal(entry=0, profit=0, commission=-2.0, swap=0, price=1.1000, time=1753600000),
                FakeDeal(entry=1, profit=80.0, commission=-2.0, swap=-0.5, price=1.1080, time=1753610000),
            ]
        }
    )

    trade_log = runner_module.TRADE_LOG_PATH
    if os.path.exists(trade_log):
        os.remove(trade_log)

    runner_module.check_closed_positions(fake_mt5, state)

    assert "111" not in state["open_positions"], "closed position should be removed from tracking"
    assert "222" in state["open_positions"], "still-open position should remain tracked"

    assert os.path.exists(trade_log), "trades.csv should have been written"
    logged = pd.read_csv(trade_log)
    assert len(logged) == 1, f"expected 1 logged trade, got {len(logged)}"
    row = logged.iloc[0]
    expected_net = 80.0 + -2.0 + -0.5  # raw_profit_before_fees + commission + swap
    assert abs(row["profit_after_fees"] - expected_net) < 0.01, f"net P&L mismatch: {row['profit_after_fees']} vs {expected_net}"
    assert abs(row["raw_profit_before_fees"] - 80.0) < 0.01, "raw profit should be pre-fees"
    assert row["symbol"] == "EURUSD"
    assert abs(row["exit_price"] - 1.1080) < 1e-9

    print("Logged trade row:")
    print(row.to_dict())
    os.remove(trade_log)
    print("\nPASS: close detection correctly identified the closed position, "
          "computed net P&L from gross profit + commission + swap, logged it, "
          "and left the still-open position untouched.")


def run_transient_hiccup_test():
    """The bug that actually happened: position 333 isn't in MT5's open
    list (maybe a transient query hiccup), but there's also no closing
    deal for it yet. Should NOT log a blank row and abandon it - should
    keep tracking it for the next poll."""
    state = {
        "open_positions": {
            "333": {"symbol": "GBPUSD", "direction": "long", "lots": 0.5,
                     "entry_price": 1.3400, "stop_loss": 1.3350, "take_profit": 1.3460,
                     "bar_time": "2026-08-03 16:00:00", "open_time": "2026-08-03T16:00:14"},
        }
    }

    fake_mt5 = FakeMT5ForClose(open_tickets=[], deals_by_position={})  # nothing open, no deals found either

    trade_log = runner_module.TRADE_LOG_PATH
    if os.path.exists(trade_log):
        os.remove(trade_log)

    runner_module.check_closed_positions(fake_mt5, state)

    assert "333" in state["open_positions"], "should still be tracked - not confirmed closed yet"
    assert not os.path.exists(trade_log) or len(pd.read_csv(trade_log)) == 0, \
        "should NOT have logged a blank row"

    print("PASS: transient hiccup (missing from open list, no closing deal found) "
          "correctly kept tracking the position instead of logging a blank row.")


if __name__ == "__main__":
    run_test()
    run_transient_hiccup_test()
