# FX Bot — Backtesting Engine (Phase 1)

Two strategies from your course notes, coded up and backtestable, plus a
risk management layer. **This is Phase 1: backtesting only.** No live/demo
execution yet — see "Next steps" below.

## Structure

```
fx_bot/
  core/
    indicators.py     # EMA, RSI, ATR, engulfing/pin-bar detection, swing highs/lows,
                       # algorithmic trendline & support/resistance
    data_loader.py     # MT5 pull (Windows only) + CSV loader + synthetic data generator
    risk_manager.py    # position sizing, daily-loss & max-drawdown circuit breakers
    backtester.py       # event-driven backtest loop
    report.py          # win rate, profit factor, drawdown, etc.
  strategies/
    strategy_engulfing.py   # Strategy A: your "extra notes" (200MA/RSI/engulfing)
    strategy_checklist.py   # Strategy B: your checklist (multi-timeframe confluence)
  run_backtest.py       # CLI entry point
```

## Setup

```bash
pip install pandas numpy
```

To pull data directly from MT5, you additionally need (Windows only, with
the MT5 terminal installed and logged into your broker):

```bash
pip install MetaTrader5
```

The `MetaTrader5` Python package wraps the terminal's own DLL — it only
works on Windows (or Wine) with the terminal open, not from this Linux
sandbox. That's why I validated the code here using synthetic data instead.

## Getting real data

Two options:

1. **Export from MT5**: In the terminal, `View > History Center`, select
   the pair/timeframe, export to CSV. Or use `core/data_loader.load_from_mt5()`
   from a script run on your Windows machine.
2. Any other OHLC CSV source, as long as columns include
   `time/date, open, high, low, close`.

## Running a backtest

Smoke test (synthetic data, just confirms the code runs — **do not use
this to judge strategy quality**, it's random data):

```bash
python run_backtest.py --strategy engulfing --pair EURUSD --synthetic
python run_backtest.py --strategy checklist --pair EURUSD --synthetic
```

With real data:

```bash
# Strategy A only needs one timeframe
python run_backtest.py --strategy engulfing --pair EURUSD --csv-exec data/EURUSD_M15.csv

# Strategy B needs three timeframes (checklist requires Daily + 1H + execution TF)
python run_backtest.py --strategy checklist --pair GBPUSD \
    --csv-exec data/GBPUSD_M15.csv --csv-1h data/GBPUSD_H1.csv --csv-daily data/GBPUSD_D1.csv
```

Trade log gets written to `output/<strategy>_<pair>_trades.csv`.

## Currency pairs

Locked to your 13-pair list in `core/data_loader.MAJOR_PAIRS`. Note: your
list includes NZD/USD but no JPY pairs, even though you listed JPY as one
of the 7 majors — worth double-checking that was intentional, since JPY
pairs are usually a big part of most majors-only universes.

## Judgment calls I had to make (please review these)

Both strategies came from screenshots/notes rather than fully unambiguous
spec, so I made explicit, documented assumptions where the original was
discretionary. These live as comments at the top of each strategy file,
but the big ones:

- **Strategy A**: "200 day line" = 200-period SMA on whatever timeframe you
  run it on (not literally calendar days, unless you run this on Daily
  bars). RSI midline = 50.
- **Strategy B**: trendlines and support/resistance are auto-detected from
  swing highs/lows (connecting the two most recent swing points), not drawn
  by eye — per what you asked for. This will disagree with a human's
  discretionary trendline sometimes. "Hits" a level = comes within
  0.3×ATR of it. "Retraced towards a MA" = within 0.5×ATR of the 50 or 200
  EMA.
- Stop-loss/take-profit logic differs between the two strategies exactly as
  in your two source documents (A: 2× candle range, fixed 2:1 RR. B:
  stop beyond nearest structure, target at next opposing structure level,
  only taken if RR ≥ 1:1).

**Please sanity-check these against your course material** — if any are
wrong, they're single, clearly-commented spots to fix in each strategy
file.

## What this backtester does NOT model yet (important before trusting results)

- Spread is a flat pip estimate, not real variable spread
- No slippage on stop/target fills
- Assumes stop-loss wins ties when both SL and TP are hit within the same
  bar (conservative, but not necessarily what would happen live)
- No news/session filters
- `pip_value_per_lot` in `RiskConfig` is a flat approximation (10 USD/pip
  for a standard lot) — actual pip value varies by pair and your account
  currency; worth plugging in real values per pair before sizing real risk

## Next steps (Phases 2–4, not built yet)

1. **Validate against real historical data** for all 13 pairs, both
   strategies, and compare (that's what you asked to do first)
2. Tighten the backtester (spread/slippage modeling) once a strategy looks
   promising
3. Paper trade on an MT5 demo account (I can build the live connection
   layer using `MetaTrader5` — runs on your Windows machine)
4. Only then, small-size live with a kill switch
