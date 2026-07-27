"""Computes summary stats from a backtest result and (optionally) plots the
equity curve."""
import pandas as pd
import numpy as np


def summarize(result: dict, starting_balance: float) -> dict:
    trades = result["trades"]
    n = len(trades)
    if n == 0:
        return {"num_trades": 0, "message": "No trades were generated - "
                "check the signal conditions aren't too restrictive for this data range."}

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / n * 100

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    equity = result["equity_curve"]["balance"] if not result["equity_curve"].empty else pd.Series([starting_balance])
    running_max = equity.cummax()
    drawdown = (running_max - equity) / running_max * 100
    max_drawdown_pct = drawdown.max()

    total_return_pct = (result["final_balance"] - starting_balance) / starting_balance * 100

    return {
        "num_trades": n,
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf (no losing trades)",
        "avg_win": round(np.mean(wins), 2) if wins else 0,
        "avg_loss": round(np.mean(losses), 2) if losses else 0,
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "starting_balance": starting_balance,
        "final_balance": round(result["final_balance"], 2),
        "total_return_pct": round(total_return_pct, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "trading_halted": result["trading_halted"],
        "halt_reason": result["halt_reason"],
    }


def trades_to_dataframe(result: dict) -> pd.DataFrame:
    trades = result["trades"]
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([{
        "entry_time": t.entry_time, "direction": t.direction, "entry_price": t.entry_price,
        "stop_loss": t.stop_loss, "take_profit": t.take_profit, "lots": t.lots,
        "exit_time": t.exit_time, "exit_price": t.exit_price, "pnl": t.pnl,
        "exit_reason": t.exit_reason,
    } for t in trades])


def print_summary(name: str, stats: dict):
    print(f"\n{'=' * 50}\n{name}\n{'=' * 50}")
    if stats.get("num_trades", 0) == 0:
        print(stats.get("message", "No trades."))
        return
    for k, v in stats.items():
        print(f"{k:20s}: {v}")
