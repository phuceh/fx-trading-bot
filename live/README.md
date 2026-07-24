# Live/Demo Runner

Runs the locked, validated strategy (Fade the Streak + RSI filter) against
a live MT5 terminal, on a demo account by default. Windows only (or Wine)
- see the main README for why.

## Locked strategy configuration

This is the ONE combination that survived proper 2025-tune / 2026-validate
testing across this whole project:
- Entry: after 4 consecutive same-direction H1 candles, fade the streak,
  filtered by RSI(14) < 50 (longs) / > 50 (shorts)
- Stop loss: 0.75 x ATR(14)
- Take profit: 2.5x the stop distance (2.5:1 reward:risk)
- Risk: 1% of account balance per trade (flat - NOT Kelly-sized; the
  Kelly-sized version of this exact combo failed validation, see project
  history)

**Do not change these parameters based on how the demo run looks partway
through.** That's exactly the kind of after-the-fact tweaking this whole
project was built to avoid. If you want to test something different,
that's a new backtest + validation cycle first, not a live edit.

## Setup

1. Install the MT5 terminal, log into your demo account (or open one:
   MT5 > File > Open an Account > search "MetaQuotes-Demo" for a generic
   practice server, or your broker's own demo server for more realistic
   spreads/execution)
2. `pip install MetaTrader5`
3. `cp live/config.example.ini live/config.ini`
4. Fill in `live/config.ini` with your login number, password, and server
   name (exactly as shown in MT5). **This file is gitignored - it never
   gets committed or sent anywhere.**
5. Check the `pairs` list matches your broker's exact symbol names (some
   brokers suffix these, e.g. `EURUSD.a` - check your MT5 Market Watch)

## Running it

```bash
python live/mt5_runner.py
```

**Leave `dry_run = true` in config.ini for the first while.** It will log
every decision (signal detected, position sized, what it would do) without
placing any real orders - even demo orders. Watch `live/logs/decisions.csv`
and `live/logs/runner.log` to confirm it's behaving sensibly before you
let it actually trade.

When ready, set `dry_run = false` to have it place real orders on your
demo account.

## Recommended protocol (decide this BEFORE starting, not after seeing results)

- Run for at least 8-12 weeks given how infrequently this fires
- Decide your success/failure bar now: e.g. "profit factor >= 1.0 across
  the period" - and stick to whatever you decide, whether the outcome is
  good or bad
- Don't restart with different parameters partway through because early
  results look rough
- Review the full `live/logs/decisions.csv` and `live/logs/trades.csv`
  at the end, not daily - daily monitoring just invites the temptation to
  intervene

## What this does NOT do

- No trailing stops, no session filters, no position pyramiding - the
  strategy is exactly what was validated, nothing added
- No automatic parameter re-tuning - if you want to update parameters,
  that requires a fresh backtest/validation cycle, not a live edit
- Basic circuit breakers only (daily loss %, max drawdown %, max
  concurrent trades) - this is not a substitute for you monitoring the
  account

## Safety notes

- Never commit `live/config.ini` (contains your password) - it's
  gitignored already, but double check before any `git add -A`
- The magic number (20260724) tags orders placed by this bot, so you can
  distinguish them from manual trades if you place any yourself
- If MT5 disconnects or the terminal closes, the script will error out -
  it does not automatically reconnect. Consider running it under a
  process supervisor (e.g. `nssm` on Windows) if you want it to survive
  restarts, and check `live/live_state.json` isn't stale before restarting
