# Deployment guide

## Where GitHub fits

- **GitHub = code storage + CI.** Push commits here, and `.github/workflows/backtest.yml`
  auto-runs a synthetic smoke test on every push (catches broken code before
  you deploy it). It does **not** run the bot live — see below for why.
- Real historical CSVs and live trade logs are gitignored by default
  (`data/*.csv`, `output/*.csv`) — data-vendor exports often have licensing
  restrictions on redistribution, and trade logs can contain account-specific
  info you may not want in a public repo. If your repo is private and you
  want them versioned anyway, remove those lines from `.gitignore`.

## Why GitHub Actions can't run the live bot

The `MetaTrader5` Python package talks to a locally running, logged-in MT5
**terminal application** (it's a wrapper around the terminal's own DLL, not
a network API). GitHub Actions runners:
- are ephemeral (destroyed after each job — no persistent logged-in terminal)
- don't support the kind of always-on GUI session MT5 needs
- would require your broker login stored as a secret, executing on
  infrastructure you don't control — riskier than necessary for something
  that needs to run continuously and reliably

So: backtesting → GitHub Actions (Linux, no MT5 needed) is fine.
Live/demo execution → needs a persistent machine with MT5 installed.

## Setting up live/demo execution

1. **Get a Windows VPS.** Options, roughly cheapest to most robust:
   - A general-purpose Windows VM (Azure, AWS EC2 Windows, etc.)
   - A VPS from a provider that specifically markets to MT4/5 EA users —
     these are tuned for low latency to broker servers and often bundle
     MT5 preinstalled
   Either way you want low latency to your broker's server and >99% uptime.

2. **On the VPS:**
   ```powershell
   # Install MT5 terminal, log into your (demo, initially) account
   # Install Python 3.11+
   pip install -r requirements.txt
   pip install MetaTrader5
   git clone <your-repo-url>
   ```

3. **Keep the MT5 terminal running and logged in** — this is the actual
   "always on" requirement. The Python bot connects to it via
   `core/data_loader.load_from_mt5()` (extend this into a live polling loop
   for Phase 3 — not built yet, this repo is backtesting only so far).

4. **Deploying updates**: `git pull` on the VPS after pushing changes,
   restart the bot process. A simple approach to start: run it as a
   scheduled Windows Task or under `nssm` (Windows service wrapper) so it
   restarts if the machine reboots.

5. **Before going live with real money**: run on the demo account for a
   meaningful stretch first (weeks, not days), with the exact same risk
   config you'd use live, and compare live-forward results to the backtest
   — the gap between the two tells you a lot about how much to trust the
   backtest.

## Quick command reference

```bash
# Local dev / backtesting (works anywhere - Linux, Mac, Windows)
git clone <your-repo-url>
cd fx_bot
pip install -r requirements.txt
python run_backtest.py --strategy engulfing --pair EURUSD --synthetic

# On the Windows VPS, once Phase 3 (live connector) exists
git pull
python live_runner.py   # not built yet
```
