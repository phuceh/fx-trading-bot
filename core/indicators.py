"""
Technical indicators and pattern detection used by both strategies.

All functions take/return pandas Series or DataFrames indexed by datetime,
with OHLC columns named: open, high, low, close
"""
import pandas as pd
import numpy as np


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


# ---------------------------------------------------------------------------
# Candlestick patterns
# ---------------------------------------------------------------------------

def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """True on the bar that closes an engulfing pattern (prior red, this green,
    this candle's body fully engulfs the prior candle's body)."""
    prev_open, prev_close = df["open"].shift(1), df["close"].shift(1)
    o, c = df["open"], df["close"]
    prev_bearish = prev_close < prev_open
    this_bullish = c > o
    engulfs = (c >= prev_open) & (o <= prev_close)
    return prev_bearish & this_bullish & engulfs


def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    prev_open, prev_close = df["open"].shift(1), df["close"].shift(1)
    o, c = df["open"], df["close"]
    prev_bullish = prev_close > prev_open
    this_bearish = c < o
    engulfs = (o >= prev_close) & (c <= prev_open)
    return prev_bullish & this_bearish & engulfs


def pin_bar(df: pd.DataFrame, wick_ratio: float = 2.0, body_ratio_max: float = 0.35):
    """Returns (bullish_pin, bearish_pin) boolean Series.

    Bullish pin bar: long lower wick (>= wick_ratio * body), small body,
    close in the upper part of the range.
    Bearish pin bar: long upper wick, small body, close in the lower part.
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    upper_wick = h - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - l

    small_body = (body / rng) <= body_ratio_max
    bullish_pin = small_body & (lower_wick >= wick_ratio * body) & ((c - l) / rng >= 0.6)
    bearish_pin = small_body & (upper_wick >= wick_ratio * body) & ((h - c) / rng >= 0.6)
    return bullish_pin.fillna(False), bearish_pin.fillna(False)


def bullish_bar(df: pd.DataFrame) -> pd.Series:
    return df["close"] > df["open"]


def bearish_bar(df: pd.DataFrame) -> pd.Series:
    return df["close"] < df["open"]


# ---------------------------------------------------------------------------
# Swing points / support-resistance / trendlines (algorithmic approximation)
# ---------------------------------------------------------------------------

def swing_highs_lows(df: pd.DataFrame, left: int = 3, right: int = 3):
    """A bar is a swing high/low if it's the max/min within [left, right] bars
    on either side. Returns (swing_high_bool, swing_low_bool)."""
    high, low = df["high"], df["low"]
    n = len(df)
    is_high = np.zeros(n, dtype=bool)
    is_low = np.zeros(n, dtype=bool)
    h, l = high.values, low.values
    for i in range(left, n - right):
        window_h = h[i - left:i + right + 1]
        window_l = l[i - left:i + right + 1]
        if h[i] == window_h.max() and (window_h == h[i]).sum() == 1:
            is_high[i] = True
        if l[i] == window_l.min() and (window_l == l[i]).sum() == 1:
            is_low[i] = True
    return pd.Series(is_high, index=df.index), pd.Series(is_low, index=df.index)


def nearest_support_resistance(df: pd.DataFrame, swing_low: pd.Series, swing_high: pd.Series,
                                lookback: int = 100):
    """For each bar, returns the most recent swing-low level (support) and
    swing-high level (resistance) within `lookback` bars prior to it."""
    support = pd.Series(np.nan, index=df.index)
    resistance = pd.Series(np.nan, index=df.index)
    low_levels = df["low"].where(swing_low)
    high_levels = df["high"].where(swing_high)
    support = low_levels.rolling(lookback, min_periods=1).apply(
        lambda x: x[~np.isnan(x)][-1] if (~np.isnan(x)).any() else np.nan, raw=True
    ).shift(1)
    resistance = high_levels.rolling(lookback, min_periods=1).apply(
        lambda x: x[~np.isnan(x)][-1] if (~np.isnan(x)).any() else np.nan, raw=True
    ).shift(1)
    return support, resistance


def trendline_from_swings(df: pd.DataFrame, swing_low: pd.Series, swing_high: pd.Series,
                           lookback: int = 100):
    """Very simplified trendline approximation.

    Uptrend line: connects the two most recent swing lows within `lookback`
    bars and projects forward -> returns the projected trendline value at
    each bar (NaN where not enough swing points exist).
    Downtrend line: connects the two most recent swing highs, same idea.

    This is a pragmatic stand-in for manual trendline drawing -- it will not
    match a human's discretionary trendline exactly, but gives the bot an
    objective, backtestable "hits trendline" condition.
    """
    n = len(df)
    up_line = np.full(n, np.nan)
    down_line = np.full(n, np.nan)

    low_idx = np.where(swing_low.values)[0]
    high_idx = np.where(swing_high.values)[0]
    lows = df["low"].values
    highs = df["high"].values

    for i in range(n):
        recent_lows = low_idx[(low_idx < i) & (low_idx >= i - lookback)]
        if len(recent_lows) >= 2:
            x1, x2 = recent_lows[-2], recent_lows[-1]
            y1, y2 = lows[x1], lows[x2]
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
                up_line[i] = y2 + slope * (i - x2)

        recent_highs = high_idx[(high_idx < i) & (high_idx >= i - lookback)]
        if len(recent_highs) >= 2:
            x1, x2 = recent_highs[-2], recent_highs[-1]
            y1, y2 = highs[x1], highs[x2]
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
                down_line[i] = y2 + slope * (i - x2)

    return pd.Series(up_line, index=df.index), pd.Series(down_line, index=df.index)


def pct_retrace(df: pd.DataFrame, swing_low: pd.Series, swing_high: pd.Series, lookback: int = 50):
    """Estimate the retracement % of the most recent swing move, at each bar.

    For an up-move (last swing low -> last swing high before current bar),
    retrace % = (swing_high - current_close) / (swing_high - swing_low).
    For a down-move (last swing high -> last swing low), mirrored.
    Returns (retrace_from_up_move, retrace_from_down_move), each 0-1ish (can exceed 1).
    """
    n = len(df)
    close = df["close"].values
    low_idx = np.where(swing_low.values)[0]
    high_idx = np.where(swing_high.values)[0]
    lows = df["low"].values
    highs = df["high"].values

    retrace_up = np.full(n, np.nan)   # retrace of an up-move (for long setups)
    retrace_down = np.full(n, np.nan)  # retrace of a down-move (for short setups)

    for i in range(n):
        recent_lows = low_idx[(low_idx < i) & (low_idx >= i - lookback)]
        recent_highs = high_idx[(high_idx < i) & (high_idx >= i - lookback)]
        if len(recent_lows) >= 1 and len(recent_highs) >= 1:
            last_low_i = recent_lows[-1]
            last_high_i = recent_highs[-1]
            if last_low_i < last_high_i:
                # up move from low -> high, now retracing down
                move = highs[last_high_i] - lows[last_low_i]
                if move > 0:
                    retrace_up[i] = (highs[last_high_i] - close[i]) / move
            elif last_high_i < last_low_i:
                # down move from high -> low, now retracing up
                move = highs[last_high_i] - lows[last_low_i]
                if move > 0:
                    retrace_down[i] = (close[i] - lows[last_low_i]) / move

    return pd.Series(retrace_up, index=df.index), pd.Series(retrace_down, index=df.index)
