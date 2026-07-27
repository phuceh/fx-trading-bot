"""
Strategy E - "Volatility Breakout" (Donchian-channel style)

Not from your course material and not the statistical mean-reversion
finding either - this is a classic systematic trend-following technique
(the basis of the "Turtle Trading" system and similar CTA/managed-futures
approaches): trade breakouts of the recent price range, on the premise
that a range breakout is more likely to continue than immediately revert.

This is mechanically different from everything else tested in this
project: not a candlestick pattern, not an EMA cross, not RSI, not a
streak/mean-reversion effect. It's a genuinely different hypothesis.

Rule:
  - Long: close breaks above the highest high of the prior N bars
  - Short: close breaks below the lowest low of the prior N bars
  - Optional trend filter: only take the breakout if it agrees with a
    longer-term EMA's direction (filters breakouts against the bigger
    trend, a common addition to reduce false breakouts in choppy markets)
  - Stop loss: ATR-based
  - Take profit: fixed reward:risk from the stop distance
"""
import pandas as pd
import numpy as np
from core import indicators as ind


def precompute_base(df: pd.DataFrame, breakout_period: int = 20, trend_ma_period: int = 100) -> pd.DataFrame:
    df = df.copy()
    df["atr"] = ind.atr(df, 14)
    # shift(1) so the breakout level excludes the current (not-yet-closed
    # in a live sense) bar - a look-ahead-safe rolling channel
    df["rolling_high"] = df["high"].rolling(breakout_period).max().shift(1)
    df["rolling_low"] = df["low"].rolling(breakout_period).min().shift(1)
    df["trend_ema"] = ind.ema(df["close"], trend_ma_period)
    return df


def apply_thresholds(base_df: pd.DataFrame, use_trend_filter: bool = True) -> pd.DataFrame:
    df = base_df.copy()
    breaks_up = df["close"] > df["rolling_high"]
    breaks_down = df["close"] < df["rolling_low"]

    if use_trend_filter:
        trend_up = df["close"] > df["trend_ema"]
        df["long_signal"] = breaks_up & trend_up
        df["short_signal"] = breaks_down & (~trend_up)
    else:
        df["long_signal"] = breaks_up
        df["short_signal"] = breaks_down

    return df


def generate_signals(df: pd.DataFrame, breakout_period: int = 20, trend_ma_period: int = 100,
                      use_trend_filter: bool = True) -> pd.DataFrame:
    base = precompute_base(df, breakout_period, trend_ma_period)
    return apply_thresholds(base, use_trend_filter)


def build_trade_params(df: pd.DataFrame, i: int, direction: str,
                        stop_atr_mult: float = 2.0, rr_ratio: float = 2.0):
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
