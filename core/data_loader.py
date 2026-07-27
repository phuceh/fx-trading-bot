"""
Data loading layer.

Two ways to get data:

1. `load_from_mt5(...)` - pulls history directly from a running MT5 terminal
   via the official `MetaTrader5` Python package. This ONLY works on Windows
   (or Wine) with the MT5 terminal installed and logged in -- the package
   wraps the terminal's own DLL, it does not work over a network connection
   or on Linux/macOS natively.

2. `load_from_csv(...)` - loads OHLC data from a CSV file. Use this for
   backtesting on any OS, e.g. after exporting history from MT5
   (Tools > History Center > export) or from another data vendor.
   Expected columns (case-insensitive): time/date, open, high, low, close,
   [volume]
"""
import pandas as pd

TIMEFRAME_MAP_MT5 = {
    "M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}

MAJOR_PAIRS = [
    "AUDCAD", "AUDCHF", "AUDNZD", "AUDUSD", "CADCHF", "EURAUD", "EURGBP",
    "EURUSD", "GBPCHF", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF",
]


def load_from_mt5(symbol: str, timeframe: str, n_bars: int = 5000):
    """Pull `n_bars` most recent candles for `symbol`/`timeframe` from a
    running, logged-in MT5 terminal. Must be run on Windows with the
    MetaTrader5 package installed (`pip install MetaTrader5`).
    """
    import MetaTrader5 as mt5  # local import: only required on Windows

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

    tf_const = getattr(mt5, TIMEFRAME_MAP_MT5[timeframe])
    rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, n_bars)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No data returned for {symbol} {timeframe}. "
                            f"Check the symbol is visible in Market Watch.")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    df = df.rename(columns={"tick_volume": "volume"})
    return df[["open", "high", "low", "close", "volume"]]


def load_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    time_col = next((c for c in ("time", "date", "datetime") if c in df.columns), None)
    if time_col is None:
        raise ValueError("CSV must have a time/date/datetime column")

    df["time"] = pd.to_datetime(df[time_col])
    df = df.set_index("time").sort_index()

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    if "volume" not in df.columns:
        df["volume"] = 0

    return df[["open", "high", "low", "close", "volume"]]


def generate_synthetic(n_bars: int = 3000, seed: int = 42, start_price: float = 1.1000) -> pd.DataFrame:
    """Generates a synthetic random-walk OHLC series purely for smoke-testing
    the pipeline (indicators, strategy logic, backtester) end-to-end when no
    real data is available. NOT for evaluating strategy performance -- only
    for confirming the code runs without errors.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    dt_index = pd.date_range("2023-01-01", periods=n_bars, freq="15min")

    returns = rng.normal(0, 0.0006, n_bars)
    # inject a bit of trend/momentum so EMA crossovers actually occur
    trend = np.sin(np.linspace(0, 12, n_bars)) * 0.0004
    close = start_price * np.exp(np.cumsum(returns + trend))

    high = close * (1 + np.abs(rng.normal(0, 0.0003, n_bars)))
    low = close * (1 - np.abs(rng.normal(0, 0.0003, n_bars)))
    open_ = np.roll(close, 1)
    open_[0] = start_price

    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                        "volume": rng.integers(50, 500, n_bars)}, index=dt_index)
    # ensure high/low actually bound open/close
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df
