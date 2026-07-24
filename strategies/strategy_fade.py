"""
Strategy D - "Fade the Streak"

Not from your course material - this one comes directly from the
statistical pattern we found in the actual price data: on the H1
timeframe, after a run of N consecutive same-direction candles, the next
candle is slightly more likely to reverse than continue (a small,
consistent mean-reversion effect, confirmed across all 3 pairs with
proper confidence intervals).

Rule:
  - After `streak_length` consecutive up candles closing, enter SHORT at
    the next bar's open (fading the up-move).
  - After `streak_length` consecutive down candles, enter LONG (fading
    the down-move).
  - Stop loss: `stop_atr_mult` x ATR(14) from entry.
  - Take profit: `rr_ratio` x stop distance.

This is deliberately simple - the statistical edge is small (52-55% vs
50%), so it doesn't warrant an elaborate rule set. Complexity here would
just be more knobs to overfit.
"""
import pandas as pd
import numpy as np
from core import indicators as ind


def precompute_base(df_h1: pd.DataFrame) -> pd.DataFrame:
    """Expensive-ish part (still cheap here, but kept in the same
    precompute/apply_thresholds pattern as the other strategies for
    consistency and sweep speed)."""
    df = df_h1.copy()
    df["atr"] = ind.atr(df, 14)

    # direction: 1 = up candle, 0 = down candle. True dojis (close==open)
    # break the streak (treated as neither up nor down).
    is_up = df["close"] > df["open"]
    is_down = df["close"] < df["open"]
    direction = pd.Series(np.where(is_up, 1, np.where(is_down, 0, -1)), index=df.index)

    # streak length: consecutive bars with the same direction, ending at
    # (and including) the current bar. Dojis (-1) reset the streak.
    changed = (direction != direction.shift(1)) | (direction == -1)
    group_id = changed.cumsum()
    streak_len = direction.groupby(group_id).cumcount() + 1
    streak_len = streak_len.where(direction != -1, 0)

    df["direction"] = direction
    df["streak_len"] = streak_len
    return df


def apply_thresholds(base_df: pd.DataFrame, streak_length: int = 3, allowed_hours=None,
                      rsi_filter: bool = False, rsi_threshold: float = 50) -> pd.DataFrame:
    """Cheap part - safe to call many times in a sweep.

    allowed_hours: optional iterable of hours (0-23, in the data's own
    timestamp convention - EST/no-DST for this project's HistData source)
    to restrict entries to. None = no restriction (any hour).
    rsi_filter: if True, additionally require RSI(14) < rsi_threshold for
    longs (fading a down-streak) and RSI(14) > (100-rsi_threshold) for
    shorts - combining the streak-based signal with the separate RSI-based
    statistical finding from the pattern search.
    """
    df = base_df.copy()
    streak_here = df["streak_len"] >= streak_length

    # long_signal = fade a down-streak (direction==0 means the streak that
    # just ended was down candles)
    long_sig = streak_here & (df["direction"] == 0)
    short_sig = streak_here & (df["direction"] == 1)

    if rsi_filter:
        if "rsi" not in df.columns:
            df["rsi"] = ind.rsi(df["close"], 14)
        long_sig = long_sig & (df["rsi"] < rsi_threshold)
        short_sig = short_sig & (df["rsi"] > (100 - rsi_threshold))

    df["long_signal"] = long_sig
    df["short_signal"] = short_sig

    if allowed_hours is not None:
        hour_mask = df.index.to_series().dt.hour.isin(allowed_hours).to_numpy()
        df["long_signal"] = df["long_signal"] & hour_mask
        df["short_signal"] = df["short_signal"] & hour_mask

    return df


def generate_signals(df_h1: pd.DataFrame, streak_length: int = 3, allowed_hours=None,
                      rsi_filter: bool = False, rsi_threshold: float = 50) -> pd.DataFrame:
    base = precompute_base(df_h1)
    return apply_thresholds(base, streak_length, allowed_hours, rsi_filter, rsi_threshold)


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
