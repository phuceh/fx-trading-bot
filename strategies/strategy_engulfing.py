"""
Strategy A - "Engulfing / 200-day trend" (from your 'extra notes')

Rules:
  Long:
    - close > 200-period MA (trend filter)
    - RSI(14) > 50 (momentum confirmation)
    - Bullish engulfing candle closes
    -> Enter on the NEXT bar open, after the engulfing candle closes
  Short: mirrored (close < 200 MA, RSI < 50, bearish engulfing)

  Stop loss  : 2x the range (high-low) of the entry/signal candle, placed
               below (long) / above (short) the entry price
  Take profit: 2:1 reward:risk based on that stop distance

Notes / assumptions made explicit:
  - "200 day line" is implemented as a 200-period SMA on whatever timeframe
    you run this on (not literally calendar days unless you run it on D1).
  - "RSI middle line" = 50.
  - Stop is anchored to the *signal candle's* high/low, not the entry bar.
"""
import pandas as pd
from core import indicators as ind


def generate_signals(df: pd.DataFrame, ma_period: int = 200, rsi_period: int = 14) -> pd.DataFrame:
    df = df.copy()
    df["ma200"] = ind.sma(df["close"], ma_period)
    df["rsi"] = ind.rsi(df["close"], rsi_period)
    df["bull_engulf"] = ind.bullish_engulfing(df)
    df["bear_engulf"] = ind.bearish_engulfing(df)

    long_signal = (df["close"] > df["ma200"]) & (df["rsi"] > 50) & df["bull_engulf"]
    short_signal = (df["close"] < df["ma200"]) & (df["rsi"] < 50) & df["bear_engulf"]

    df["long_signal"] = long_signal
    df["short_signal"] = short_signal
    return df


def build_trade_params(df: pd.DataFrame, i: int, direction: str):
    """Given a signal at bar i (the engulfing candle), returns entry/stop/target
    for a trade entered at bar i+1's open."""
    signal_candle = df.iloc[i]
    entry_price = df.iloc[i + 1]["open"] if i + 1 < len(df) else None
    if entry_price is None:
        return None

    candle_range = signal_candle["high"] - signal_candle["low"]
    if candle_range <= 0:
        return None

    stop_distance = 2 * candle_range

    if direction == "long":
        stop_loss = entry_price - stop_distance
        take_profit = entry_price + 2 * stop_distance  # 2:1 reward:risk
    else:
        stop_loss = entry_price + stop_distance
        take_profit = entry_price - 2 * stop_distance

    return {
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "stop_distance": stop_distance,
        "direction": direction,
    }
