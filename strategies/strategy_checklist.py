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


def precompute_base(
    df_exec: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_daily: pd.DataFrame,
    swing_left: int = 3,
    swing_right: int = 3,
    swing_lookback: int = 100,
) -> pd.DataFrame:
    """The expensive part: higher-timeframe trend, swing points, trendlines,
    support/resistance, retrace %, candle patterns. None of this depends on
    the threshold parameters (retrace_min, min_rr, level_tolerance_atr,
    ma_proximity_atr) - compute it ONCE per pair and reuse across a
    parameter sweep via apply_thresholds() instead of recomputing per combo.
    """
    df = df_exec.copy()

    daily_trend_up = _htf_trend(df_daily, 8, 21).reindex(df.index, method="ffill")
    h1_trend_up = _htf_trend(df_1h, 50, 200).reindex(df.index, method="ffill")

    df["ema50"] = ind.ema(df["close"], 50)
    df["ema200"] = ind.ema(df["close"], 200)
    df["atr"] = ind.atr(df, 14)
    df["exec_trend_up"] = df["ema50"] > df["ema200"]
    df["daily_trend_up"] = daily_trend_up
    df["h1_trend_up"] = h1_trend_up

    swing_high, swing_low = ind.swing_highs_lows(df, swing_left, swing_right)
    support, resistance = ind.nearest_support_resistance(df, swing_low, swing_high, swing_lookback)
    up_trendline, down_trendline = ind.trendline_from_swings(df, swing_low, swing_high, swing_lookback)
    retrace_up, retrace_down = ind.pct_retrace(df, swing_low, swing_high, swing_lookback)

    bull_pin, bear_pin = ind.pin_bar(df)
    df["bullish_signal_bar"] = bull_pin | ind.bullish_bar(df)
    df["bearish_signal_bar"] = bear_pin | ind.bearish_bar(df)

    df["support"] = support
    df["resistance"] = resistance
    df["up_trendline"] = up_trendline
    df["down_trendline"] = down_trendline
    df["retrace_up"] = retrace_up
    df["retrace_down"] = retrace_down
    return df


def apply_thresholds(
    base_df: pd.DataFrame,
    retrace_min: float = 0.5,
    ma_proximity_atr: float = 0.5,
    level_tolerance_atr: float = 0.3,
    min_rr: float = 1.0,  # noqa: unused here, kept for signature symmetry with build_trade_params
) -> pd.DataFrame:
    """The cheap part: applies threshold comparisons to the precomputed base.
    Safe to call many times (e.g. in a parameter sweep) without recomputing
    swing points / trendlines / retrace each time."""
    df = base_df.copy()

    near_ma_long = (
        (df["close"] - df["ema50"]).abs() <= ma_proximity_atr * df["atr"]
    ) | (
        (df["close"] - df["ema200"]).abs() <= ma_proximity_atr * df["atr"]
    )
    near_ma_short = near_ma_long

    tol = level_tolerance_atr * df["atr"]
    hits_up_trendline = (df["up_trendline"].notna()) & ((df["low"] - df["up_trendline"]).abs() <= tol)
    hits_down_trendline = (df["down_trendline"].notna()) & ((df["high"] - df["down_trendline"]).abs() <= tol)
    hits_support = (df["support"].notna()) & ((df["low"] - df["support"]).abs() <= tol)
    hits_resistance = (df["resistance"].notna()) & ((df["high"] - df["resistance"]).abs() <= tol)

    df["long_signal"] = (
        df["daily_trend_up"].fillna(False)
        & df["h1_trend_up"].fillna(False)
        & df["exec_trend_up"]
        & near_ma_long
        & (df["retrace_up"] >= retrace_min).fillna(False)
        & (hits_up_trendline | hits_support)
        & df["bullish_signal_bar"]
    )

    df["short_signal"] = (
        (~df["daily_trend_up"]).fillna(False)
        & (~df["h1_trend_up"]).fillna(False)
        & (~df["exec_trend_up"])
        & near_ma_short
        & (df["retrace_down"] >= retrace_min).fillna(False)
        & (hits_down_trendline | hits_resistance)
        & df["bearish_signal_bar"]
    )
    return df


def apply_thresholds_scored(
    base_df: pd.DataFrame,
    retrace_min: float = 0.5,
    ma_proximity_atr: float = 0.5,
    level_tolerance_atr: float = 0.3,
    min_rr: float = 1.0,
    min_optional_hits: int = 2,
) -> pd.DataFrame:
    """Scored variant: core criteria (all 3 trend-direction checks + the
    confirmation candle + RR, enforced separately at trade-build time) are
    mandatory. The remaining criteria - retraced-to-MA, >=50% retrace, hits
    trendline, hits support/resistance - are OPTIONAL: at least
    `min_optional_hits` of these 4 must be true, rather than requiring all
    of them. This matches the "as many confirmations as possible, not
    necessarily all" philosophy described in the sibling Backbone doc,
    applied here by inference since this doc doesn't spell out its own
    bold/core split explicitly.
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
    hits_level_long = (hits_up_trendline | hits_support)
    hits_level_short = (hits_down_trendline | hits_resistance)

    # optional score: near_ma, retrace>=min, hits a level (trendline OR S/R
    # counts as ONE optional criterion here, matching this doc's implicit
    # grouping) -- plus retrace direction-specific. 3 optional criteria total.
    long_optional_score = near_ma.astype(int) + retrace_up_ok.astype(int) + hits_level_long.astype(int)
    short_optional_score = near_ma.astype(int) + retrace_down_ok.astype(int) + hits_level_short.astype(int)

    core_long = df["daily_trend_up"].fillna(False) & df["h1_trend_up"].fillna(False) & df["exec_trend_up"]
    core_short = (~df["daily_trend_up"]).fillna(False) & (~df["h1_trend_up"]).fillna(False) & (~df["exec_trend_up"])

    df["long_signal"] = core_long & (long_optional_score >= min_optional_hits) & df["bullish_signal_bar"]
    df["short_signal"] = core_short & (short_optional_score >= min_optional_hits) & df["bearish_signal_bar"]
    return df


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
    """Convenience wrapper: precompute + apply thresholds in one call. For a
    parameter sweep, call precompute_base() once and apply_thresholds() per
    combo instead - much faster."""
    base = precompute_base(df_exec, df_1h, df_daily, swing_left, swing_right, swing_lookback)
    return apply_thresholds(base, retrace_min, ma_proximity_atr, level_tolerance_atr, min_rr)


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
