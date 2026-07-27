"""
Strategy C - "Backbone" (from 02_a_Trading_Strategy_1_-_Backbone_Strategy.docx)

A mid-term, 1-hour trend-following strategy. Simpler than the checklist
strategy - only two timeframes (Daily trend filter + Hourly execution),
and per the doc, fires infrequently by design.

Long checklist:
  1. Daily trend: 8 EMA above 21 EMA
  2. Hourly trend: 50 EMA above 200 EMA (Hourly IS the execution timeframe)
  3. Price retraced towards the moving averages (50 or 200 EMA)
  4. RSI(6) below 20 (oversold) -- doc-specified levels, not a guess
  5. Retrace >= 50% of the previous move
  6. Hourly confirmation candle: bullish bar (pin bar / engulfing / inside
     bar are noted as stronger confluences for this)
  7. Hits a support/resistance level or trendline
  8. Risk:Reward > 1.5 (doc-specified threshold - stricter than the
     checklist strategy's >= 1.0)

Short checklist is the mirror image (RSI > 80, bearish bar, etc).

Implementation notes / assumptions (same categories as the checklist
strategy, since the underlying structural concepts are identical):
  - Trendline / support-resistance: same algorithmic swing-point
    approximation as strategy_checklist.py.
  - "Retraced towards a MA" = within `ma_proximity_atr` ATRs of the 50 or
    200 EMA (same approach as the checklist strategy).
  - No explicit pin-bar-location rule was specified in the source doc
    beyond "retraced towards the moving averages" - that's the pullback
    condition used here; a separate stricter positioning rule wasn't
    given, so none is added.
  - Stop loss: beyond nearest opposing structure + ATR buffer. Take
    profit: next opposing structure level (previous high/low), matching
    the doc's "target previous lows/highs" note. Trade only taken if this
    produces RR > 1.5, per the doc's explicit rule.
"""
import pandas as pd
import numpy as np
from core import indicators as ind


def _htf_trend(df_htf: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    ema_fast = ind.ema(df_htf["close"], fast)
    ema_slow = ind.ema(df_htf["close"], slow)
    return ema_fast > ema_slow


def precompute_base(
    df_h1: pd.DataFrame,
    df_daily: pd.DataFrame,
    swing_left: int = 3,
    swing_right: int = 3,
    swing_lookback: int = 100,
    rsi_period: int = 6,
) -> pd.DataFrame:
    """Expensive part - run once per pair, reuse across a parameter sweep
    via apply_thresholds()."""
    df = df_h1.copy()

    daily_trend_up = _htf_trend(df_daily, 8, 21).reindex(df.index, method="ffill")

    df["ema50"] = ind.ema(df["close"], 50)
    df["ema200"] = ind.ema(df["close"], 200)
    df["rsi"] = ind.rsi(df["close"], rsi_period)
    df["atr"] = ind.atr(df, 14)
    df["exec_trend_up"] = df["ema50"] > df["ema200"]
    df["daily_trend_up"] = daily_trend_up

    swing_high, swing_low = ind.swing_highs_lows(df, swing_left, swing_right)
    support, resistance = ind.nearest_support_resistance(df, swing_low, swing_high, swing_lookback)
    up_trendline, down_trendline = ind.trendline_from_swings(df, swing_low, swing_high, swing_lookback)
    retrace_up, retrace_down = ind.pct_retrace(df, swing_low, swing_high, swing_lookback)

    bull_pin, bear_pin = ind.pin_bar(df)
    bull_engulf = ind.bullish_engulfing(df)
    bear_engulf = ind.bearish_engulfing(df)
    inside = ind.inside_bar(df)

    # core requirement: ANY directional bar (bullish/bearish close)
    df["bullish_signal_bar"] = ind.bullish_bar(df)
    df["bearish_signal_bar"] = ind.bearish_bar(df)
    # optional/scored: a NAMED stronger pattern specifically (pin bar,
    # engulfing, or inside bar) - separate from just "closed bullish/bearish"
    df["strong_candle_long"] = bull_pin | bull_engulf | inside
    df["strong_candle_short"] = bear_pin | bear_engulf | inside

    df["support"] = support
    df["resistance"] = resistance
    df["up_trendline"] = up_trendline
    df["down_trendline"] = down_trendline
    df["retrace_up"] = retrace_up
    df["retrace_down"] = retrace_down
    return df


def apply_thresholds_scored(
    base_df: pd.DataFrame,
    retrace_min: float = 0.5,
    ma_proximity_atr: float = 0.5,
    level_tolerance_atr: float = 0.3,
    rsi_oversold: float = 20,
    rsi_overbought: float = 80,
    min_optional_hits: int = 1,
) -> pd.DataFrame:
    """Scored variant matching the doc's explicit bold/plain split:
    core (mandatory) = Daily trend, Hourly trend, retraced-to-MA, RSI
    extreme, confirmation bar (enforced separately at trade-build time via
    bullish/bearish_signal_bar), and RR>1.5 (enforced at trade-build time).
    optional (need >= min_optional_hits of 3) = >=50% retrace, specific
    candle type (pin/engulfing/inside - vs a generic directional bar, which
    is already required as core), hits a trendline/support/resistance level.
    """
    df = base_df.copy()

    near_ma = (
        (df["close"] - df["ema50"]).abs() <= ma_proximity_atr * df["atr"]
    ) | (
        (df["close"] - df["ema200"]).abs() <= ma_proximity_atr * df["atr"]
    )

    tol = level_tolerance_atr * df["atr"]
    hits_up_trendline = (df["up_trendline"].notna()) & ((df["low"] - df["up_trendline"]).abs() <= tol)
    hits_down_trendline = (df["down_trendline"].notna()) & ((df["high"] - df["down_trendline"]).abs() <= tol)
    hits_support = (df["support"].notna()) & ((df["low"] - df["support"]).abs() <= tol)
    hits_resistance = (df["resistance"].notna()) & ((df["high"] - df["resistance"]).abs() <= tol)

    retrace_up_ok = (df["retrace_up"] >= retrace_min).fillna(False)
    retrace_down_ok = (df["retrace_down"] >= retrace_min).fillna(False)
    hits_level_long = hits_up_trendline | hits_support
    hits_level_short = hits_down_trendline | hits_resistance

    # optional criteria: retrace>=min, "stronger" candle type (pin/engulf/
    # inside, tracked separately from the generic directional bar), hits a level
    strong_candle_long = df["strong_candle_long"] if "strong_candle_long" in df.columns else pd.Series(False, index=df.index)
    strong_candle_short = df["strong_candle_short"] if "strong_candle_short" in df.columns else pd.Series(False, index=df.index)

    long_optional_score = retrace_up_ok.astype(int) + strong_candle_long.astype(int) + hits_level_long.astype(int)
    short_optional_score = retrace_down_ok.astype(int) + strong_candle_short.astype(int) + hits_level_short.astype(int)

    core_long = (
        df["daily_trend_up"].fillna(False) & df["exec_trend_up"]
        & near_ma & (df["rsi"] <= rsi_oversold)
    )
    core_short = (
        (~df["daily_trend_up"]).fillna(False) & (~df["exec_trend_up"])
        & near_ma & (df["rsi"] >= rsi_overbought)
    )

    df["long_signal"] = core_long & (long_optional_score >= min_optional_hits) & df["bullish_signal_bar"]
    df["short_signal"] = core_short & (short_optional_score >= min_optional_hits) & df["bearish_signal_bar"]
    return df


def apply_thresholds(
    base_df: pd.DataFrame,
    retrace_min: float = 0.5,
    ma_proximity_atr: float = 0.5,
    level_tolerance_atr: float = 0.3,
    rsi_oversold: float = 20,
    rsi_overbought: float = 80,
    min_rr: float = 1.5,
) -> pd.DataFrame:
    """Cheap part - safe to call many times in a parameter sweep."""
    df = base_df.copy()

    near_ma = (
        (df["close"] - df["ema50"]).abs() <= ma_proximity_atr * df["atr"]
    ) | (
        (df["close"] - df["ema200"]).abs() <= ma_proximity_atr * df["atr"]
    )

    tol = level_tolerance_atr * df["atr"]
    hits_up_trendline = (df["up_trendline"].notna()) & ((df["low"] - df["up_trendline"]).abs() <= tol)
    hits_down_trendline = (df["down_trendline"].notna()) & ((df["high"] - df["down_trendline"]).abs() <= tol)
    hits_support = (df["support"].notna()) & ((df["low"] - df["support"]).abs() <= tol)
    hits_resistance = (df["resistance"].notna()) & ((df["high"] - df["resistance"]).abs() <= tol)

    df["long_signal"] = (
        df["daily_trend_up"].fillna(False)
        & df["exec_trend_up"]
        & near_ma
        & (df["retrace_up"] >= retrace_min).fillna(False)
        & (df["rsi"] <= rsi_oversold)
        & (hits_up_trendline | hits_support)
        & df["bullish_signal_bar"]
    )

    df["short_signal"] = (
        (~df["daily_trend_up"]).fillna(False)
        & (~df["exec_trend_up"])
        & near_ma
        & (df["retrace_down"] >= retrace_min).fillna(False)
        & (df["rsi"] >= rsi_overbought)
        & (hits_down_trendline | hits_resistance)
        & df["bearish_signal_bar"]
    )
    return df


def generate_signals(df_h1, df_daily, swing_left=3, swing_right=3, swing_lookback=100,
                      rsi_period=6, retrace_min=0.5, ma_proximity_atr=0.5,
                      level_tolerance_atr=0.3, rsi_oversold=20, rsi_overbought=80,
                      min_rr=1.5):
    """Convenience wrapper. For a sweep, call precompute_base() once and
    apply_thresholds() per combo instead."""
    base = precompute_base(df_h1, df_daily, swing_left, swing_right, swing_lookback, rsi_period)
    return apply_thresholds(base, retrace_min, ma_proximity_atr, level_tolerance_atr,
                             rsi_oversold, rsi_overbought, min_rr)


def build_trade_params(df: pd.DataFrame, i: int, direction: str, min_rr: float = 1.5):
    """Entry at next bar's open. Stop beyond nearest swing structure + ATR
    buffer. Target at the next opposing structure level (previous
    high/low, per the doc). Trade rejected if RR <= min_rr."""
    if i + 1 >= len(df):
        return None

    row = df.iloc[i]
    entry_price = df.iloc[i + 1]["open"]
    atr_buf = row["atr"] if not np.isnan(row["atr"]) else 0.0

    if np.isnan(row["support"]) or np.isnan(row["resistance"]):
        return None

    if direction == "long":
        stop_loss = row["support"] - 0.5 * atr_buf
        take_profit = row["resistance"]
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
    else:
        stop_loss = row["resistance"] + 0.5 * atr_buf
        take_profit = row["support"]
        risk = stop_loss - entry_price
        reward = entry_price - take_profit

    if risk <= 0 or reward <= 0:
        return None
    rr = reward / risk
    if rr <= min_rr:
        return None

    return {
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "stop_distance": risk,
        "direction": direction,
        "rr": rr,
    }
