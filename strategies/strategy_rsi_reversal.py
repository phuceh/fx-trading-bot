"""
Strategy F - "RSI-Filtered Reversal"

Built directly from the statistical pattern search: the prior candle's
direction has a small mean-reversion tendency, and filtering by RSI(14)
strengthens it. Note the signal is ASYMMETRIC - long side (fade a down
candle when RSI<50) is considerably stronger (z=8.25 in-sample) than the
short side (fade an up candle when RSI>50, z=2.53) - expect most of any
edge to come from the long side.

Rule:
  - Long: previous H1 candle closed down, AND RSI(14) < rsi_threshold
  - Short: previous H1 candle closed up, AND RSI(14) > (100 - rsi_threshold)
  - Stop loss: ATR-based
  - Take profit: fixed reward:risk from the stop distance
"""
import pandas as pd
import numpy as np
from core import indicators as ind


def precompute_base(df: pd.DataFrame, rsi_period: int = 14) -> pd.DataFrame:
    df = df.copy()
    df["atr"] = ind.atr(df, 14)
    df["rsi"] = ind.rsi(df["close"], rsi_period)
    df["direction"] = df["close"] - df["open"]
    return df


def apply_thresholds(base_df: pd.DataFrame, rsi_threshold: float = 50) -> pd.DataFrame:
    df = base_df.copy()
    prev_down = df["direction"].shift(1) < 0
    prev_up = df["direction"].shift(1) > 0

    df["long_signal"] = prev_down & (df["rsi"] < rsi_threshold)
    df["short_signal"] = prev_up & (df["rsi"] > (100 - rsi_threshold))
    return df


def generate_signals(df: pd.DataFrame, rsi_threshold: float = 50) -> pd.DataFrame:
    base = precompute_base(df)
    return apply_thresholds(base, rsi_threshold)


def build_trade_params(df: pd.DataFrame, i: int, direction: str,
                        stop_atr_mult: float = 1.5, rr_ratio: float = 1.5):
    row = df.iloc[i]
    entry_price = df.iloc[i + 1]["open"] if i + 1 < len(df) else None
    if entry_price is None:
        return None

    atr = row["atr"]
    if np.isnan(atr) or atr <= 0:
        return None

    stop_distance = stop_atr_mult * atr
    if direction == "long":
        stop_loss = entry_price - stop_distance
        take_profit = entry_price + rr_ratio * stop_distance
    else:
        stop_loss = entry_price + stop_distance
        take_profit = entry_price - rr_ratio * stop_distance

    return {
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "stop_distance": stop_distance,
        "direction": direction,
    }
