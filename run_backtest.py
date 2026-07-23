"""
Run a backtest for either strategy against one currency pair.

Examples
--------
Using synthetic data (smoke test - no real data needed):
    python run_backtest.py --strategy engulfing --pair EURUSD --synthetic

Using CSVs you've exported from MT5 (Strategy A only needs one timeframe):
    python run_backtest.py --strategy engulfing --pair EURUSD \\
        --csv-exec data/EURUSD_M15.csv

Strategy B needs three timeframes:
    python run_backtest.py --strategy checklist --pair EURUSD \\
        --csv-exec data/EURUSD_M15.csv \\
        --csv-1h data/EURUSD_H1.csv \\
        --csv-daily data/EURUSD_D1.csv
"""
import argparse
import pandas as pd

from core.data_loader import load_from_csv, generate_synthetic, MAJOR_PAIRS
from core.risk_manager import RiskManager, RiskConfig
from core.backtester import run_backtest_with_builder
from core import report
from strategies import strategy_engulfing, strategy_checklist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", choices=["engulfing", "checklist"], required=True)
    p.add_argument("--pair", required=True, choices=MAJOR_PAIRS)
    p.add_argument("--synthetic", action="store_true", help="use synthetic data (smoke test only)")
    p.add_argument("--csv-exec", help="CSV path for execution timeframe (e.g. M15)")
    p.add_argument("--csv-1h", help="CSV path for 1H timeframe (checklist strategy only)")
    p.add_argument("--csv-daily", help="CSV path for Daily timeframe (checklist strategy only)")
    p.add_argument("--balance", type=float, default=10_000.0)
    p.add_argument("--risk-pct", type=float, default=1.0)
    args = p.parse_args()

    risk_cfg = RiskConfig(account_balance=args.balance, risk_per_trade_pct=args.risk_pct)

    if args.strategy == "engulfing":
        if args.synthetic:
            df_exec = generate_synthetic(seed=hash(args.pair) % 1000)
        else:
            if not args.csv_exec:
                raise SystemExit("--csv-exec is required unless --synthetic is used")
            df_exec = load_from_csv(args.csv_exec)

        signals = strategy_engulfing.generate_signals(df_exec)
        risk_mgr = RiskManager(risk_cfg)
        result = run_backtest_with_builder(
            signals, strategy_engulfing.build_trade_params, risk_mgr, args.pair
        )

    else:  # checklist
        if args.synthetic:
            df_exec = generate_synthetic(seed=hash(args.pair) % 1000, n_bars=3000)
            df_1h = generate_synthetic(seed=hash(args.pair + "1h") % 1000, n_bars=1500)
            df_1h.index = pd.date_range("2023-01-01", periods=1500, freq="1h")
            df_daily = generate_synthetic(seed=hash(args.pair + "d") % 1000, n_bars=400)
            df_daily.index = pd.date_range("2022-01-01", periods=400, freq="1D")
        else:
            if not (args.csv_exec and args.csv_1h and args.csv_daily):
                raise SystemExit("--csv-exec, --csv-1h and --csv-daily are all required "
                                  "unless --synthetic is used")
            df_exec = load_from_csv(args.csv_exec)
            df_1h = load_from_csv(args.csv_1h)
            df_daily = load_from_csv(args.csv_daily)

        signals = strategy_checklist.generate_signals(df_exec, df_1h, df_daily)
        risk_mgr = RiskManager(risk_cfg)
        result = run_backtest_with_builder(
            signals, strategy_checklist.build_trade_params, risk_mgr, args.pair
        )

    stats = report.summarize(result, args.balance)
    report.print_summary(f"{args.strategy} | {args.pair}", stats)

    trades_df = report.trades_to_dataframe(result)
    if not trades_df.empty:
        out_path = f"output/{args.strategy}_{args.pair}_trades.csv"
        trades_df.to_csv(out_path, index=False)
        print(f"\nTrade log written to {out_path}")


if __name__ == "__main__":
    main()
