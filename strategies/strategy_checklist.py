"""
Strategy B - "Checklist / Confluence" (from your checklist image)

Long checklist:
  1. Daily trend: 8 EMA above 21 EMA
  2. Execution TF (5/15min) trend: 50 EMA above 200 EMA
  3. 1H trend: 50 EMA above 200 EMA
  4. Price retraced towards/below at least one moving average
  5. Retrace >= 50% of the previous move
  6. Hits trendline
  7. Hits support
  8. Bullish bar (preferably pin-bar)
  9. Risk:Reward >= 1:1

Short checklist is the mirror image.

Implementation notes / assumptions (flagged explicitly, these are judgment
calls needed to make a discretionary checklist backtestable):
  - Higher-timeframe EMAs (daily, 1H) are computed on their own timeframe
    and forward-filled onto the execution timeframe (i.e. "as of the last
    completed daily/1H bar").
  - "Hits trendline" / "hits support/resistance": price is considered to
    have "hit" a level if the bar's low/high comes within `level_tolerance`
    (in price terms, e.g. via ATR-based buffer) of the algorithmic
    trendline/S-R value from core.indicators.
  - "Price retraced towards/below at least one moving average" is
    approximated by requiring price to be within `ma_proximity_atr` ATRs of
    the *execution-timeframe* 50 or 200 EMA.
  - Stop loss: placed just beyond the nearest swing point (support for
    longs, resistance for shorts) plus a small ATR buffer.
  - Take profit: the next opposing swing level (resistance for longs,
    support for shorts). Trade is only taken if this produces RR >= 1:1,
    per the checklist's explicit rule.
"""
import pandas as pd
import numpy as np
from core import indicators as ind


def _htf_trend(df_htf: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    ema_fast = ind.ema(df_htf["close"], fast)
    ema_slow = ind.ema(df_htf["close"], slow)
    return (ema_fast > ema_slow)  # True = up, False = down


def generate_signals(
    df_exec: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_daily: pd.DataFrame,
    swing_left: int = 3,
    swing_right: int = 3,
    swing_lookback: int = 100,
    retrace_min: float = 0.5,
    ma_proximity_atr: float = 0.5,
    level_tolerance_atr: float = 0.3,
    min_rr: float = 1.0,
) -> pd.DataFrame:
    df = df_exec.copy()

    # --- higher timeframe trend, forward-filled onto execution timeframe ---
    daily_trend_up = _htf_trend(df_daily, 8, 21)
    h1_trend_up = _htf_trend(df_1h, 50, 200)

    daily_trend_up = daily_trend_up.reindex(df.index, method="ffill")
    h1_trend_up = h1_trend_up.reindex(df.index, method="ffill")

    # --- execution timeframe trend + indicators ---
    df["ema50"] = ind.ema(df["close"], 50)
    df["ema200"] = ind.ema(df["close"], 200)
    df["atr"] = ind.atr(df, 14)
    exec_trend_up = df["ema50"] > df["ema200"]

    # --- swings / structure ---
    swing_high, swing_low = ind.swing_highs_lows(df, swing_left, swing_right)
    support, resistance = ind.nearest_support_resistance(df, swing_low, swing_high, swing_lookback)
    up_trendline, down_trendline = ind.trendline_from_swings(df, swing_low, swing_high, swing_lookback)
    retrace_up, retrace_down = ind.pct_retrace(df, swing_low, swing_high, swing_lookback)

    # --- candle patterns ---
    bull_pin, bear_pin = ind.pin_bar(df)
    bull_bar = ind.bullish_bar(df)
    bear_bar = ind.bearish_bar(df)
    bullish_signal_bar = bull_pin | bull_bar
    bearish_signal_bar = bear_pin | bear_bar

    # --- "retraced towards/below a MA" proximity check ---
    near_ma_long = (
        (df["close"] - df["ema50"]).abs() <= ma_proximity_atr * df["atr"]
    ) | (
        (df["close"] - df["ema200"]).abs() <= ma_proximity_atr * df["atr"]
    )
    near_ma_short = near_ma_long  # same proximity test, direction irrelevant

    # --- "hits" trendline / support / resistance ---
    tol = level_tolerance_atr * df["atr"]
    hits_up_trendline = (up_trendline.notna()) & ((df["low"] - up_trendline).abs() <= tol)
    hits_down_trendline = (down_trendline.notna()) & ((df["high"] - down_trendline).abs() <= tol)
    hits_support = (support.notna()) & ((df["low"] - support).abs() <= tol)
    hits_resistance = (resistance.notna()) & ((df["high"] - resistance).abs() <= tol)

    long_condition = (
        daily_trend_up.fillna(False)
        & h1_trend_up.fillna(False)
        & exec_trend_up
        & near_ma_long
        & (retrace_up >= retrace_min).fillna(False)
        & (hits_up_trendline | hits_support)
        & bullish_signal_bar
    )

    short_condition = (
        (~daily_trend_up).fillna(False)
        & (~h1_trend_up).fillna(False)
        & (~exec_trend_up)
        & near_ma_short
        & (retrace_down >= retrace_min).fillna(False)
        & (hits_down_trendline | hits_resistance)
        & bearish_signal_bar
    )

    df["support"] = support
    df["resistance"] = resistance
    df["long_signal"] = long_condition
    df["short_signal"] = short_condition
    return df


def build_trade_params(df: pd.DataFrame, i: int, direction: str, min_rr: float = 1.0):
    """Entry at next bar's open. Stop beyond nearest swing structure + ATR
    buffer. Target at the next opposing structure level. Trade rejected if
    RR < min_rr (per checklist rule #9)."""
    if i + 1 >= len(df):
        return None

    row = df.iloc[i]
    entry_price = df.iloc[i + 1]["open"]
    atr_buf = row["atr"] if not np.isnan(row["atr"]) else 0.0

    if direction == "long":
        if np.isnan(row["support"]) or np.isnan(row["resistance"]):
            return None
        stop_loss = row["support"] - 0.5 * atr_buf
        take_profit = row["resistance"]
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
    else:
        if np.isnan(row["support"]) or np.isnan(row["resistance"]):
            return None
        stop_loss = row["resistance"] + 0.5 * atr_buf
        take_profit = row["support"]
        risk = stop_loss - entry_price
        reward = entry_price - take_profit

    if risk <= 0 or reward <= 0:
        return None
    rr = reward / risk
    if rr < min_rr:
        return None

    return {
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "stop_distance": risk,
        "direction": direction,
        "rr": rr,
    }
