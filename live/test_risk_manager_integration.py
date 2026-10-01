"""
Mock test: confirms the per-symbol halt + auto-reset works correctly when
driven through the actual RiskManager API used by mt5_runner.py.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from datetime import datetime, timedelta, timezone
from core.risk_manager import RiskManager, RiskConfig


def run_test():
    rm = RiskManager(RiskConfig(account_balance=10000.0, max_drawdown_pct=10.0, drawdown_cooloff_days=3.0))

    t0 = datetime(2026, 9, 1, tzinfo=timezone.utc)

    # EURUSD: steady losses spread across many days, should trip its own
    # drawdown halt (spread out so the unrelated daily-loss limit doesn't
    # trigger first)
    rm.new_bar(t0)
    for i in range(12):
        day = t0 + timedelta(days=i)
        rm.new_bar(day)
        rm.on_trade_close(-100.0, "EURUSD")  # 12 x -100 = -1200 = 12% of 10000

    can_open, reason = rm.can_open_trade("EURUSD")
    assert not can_open, "EURUSD should be halted after exceeding 10% drawdown"
    print(f"PASS: EURUSD correctly halted - {reason}")

    # GBPUSD: meanwhile, completely unaffected - this is the actual bug we're fixing
    can_open_gbp, _ = rm.can_open_trade("GBPUSD")
    assert can_open_gbp, "GBPUSD should NOT be affected by EURUSD's halt"
    print("PASS: GBPUSD unaffected by EURUSD's halt (per-symbol isolation working)")

    # advance time past the cooloff period
    rm.new_bar(t0 + timedelta(days=16))
    can_open_after, _ = rm.can_open_trade("EURUSD")
    assert can_open_after, "EURUSD should auto-resume after the cooloff period"
    print("PASS: EURUSD auto-resumed after cooloff period elapsed")

    # risk events were logged
    events = rm.risk_events
    assert any(e["event"] == "halted" and e["symbol"] == "EURUSD" for e in events)
    assert any(e["event"] == "auto_reset" and e["symbol"] == "EURUSD" for e in events)
    print(f"PASS: {len(events)} risk events logged correctly: {[e['event'] for e in events]}")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    run_test()
