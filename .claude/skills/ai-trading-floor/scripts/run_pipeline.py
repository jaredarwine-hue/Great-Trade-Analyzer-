#!/usr/bin/env python3
"""One-command pipeline for the AI Trading Floor toolkit: fetch -> backtest -> dashboard.

This is the "run a simple test" engine the tutorial points beginners at. With no
arguments it fetches ~3 years of daily AAPL, runs a 20/50 SMA crossover, and refreshes
the single aggregate ``reports/dashboard.html`` — everything into CWD-relative folders
(./data, ./results, ./reports). Nothing reads or writes the repository.

``dashboard.html`` is the report to open; it renders every visual (candles, trade
markers, equity curve, stat cards) from ``results/*.json``.

It reuses the kit's own scripts (fetch_data, backtest, dashboard) by calling their
``main()`` functions in-process, so there is ONE source of truth for each step.

Usage:
    python run_pipeline.py
    python run_pipeline.py --ticker MSFT --strategy sma_crossover --fast 10 --slow 30
    python run_pipeline.py --ticker AAPL --strategy rsi_reversion --buy-below 25 --sell-above 60
    python run_pipeline.py --ticker AAPL --strategy breakout --lookback 20 --start 2021-01-01 --end 2024-01-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Import sibling scripts so we call each step's main() directly (single source of truth).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402
import dashboard as dashboard_mod  # noqa: E402
import fetch_data  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch data, run a backtest, and build a report in one command.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ticker", default="AAPL", help="Single ticker to run.")
    parser.add_argument(
        "--strategy", default="sma_crossover", choices=sorted(bt.STRATEGIES),
        help="Which built-in strategy to run.",
    )
    parser.add_argument("--start", help="Start date YYYY-MM-DD (use with --end).")
    parser.add_argument("--end", help="End date YYYY-MM-DD (use with --start).")
    parser.add_argument("--period", default="3y", help="Lookback if no start/end given.")
    parser.add_argument("--capital", type=float, default=10000.0, help="Starting capital.")
    # Strategy params (mirror backtest.py; all optional -> fall back to strategy defaults).
    parser.add_argument("--fast", type=int, help="sma_crossover: fast SMA period.")
    parser.add_argument("--slow", type=int, help="sma_crossover: slow SMA period.")
    parser.add_argument("--rsi-period", type=int, dest="rsi_period", help="rsi_reversion: RSI lookback.")
    parser.add_argument("--buy-below", type=float, dest="buy_below", help="rsi_reversion: buy threshold.")
    parser.add_argument("--sell-above", type=float, dest="sell_above", help="rsi_reversion: sell threshold.")
    parser.add_argument("--lookback", type=int, help="breakout: prior-high lookback in bars.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ticker = args.ticker.strip().upper()

    data_path = Path("data") / f"{ticker}.parquet"
    results_path = Path("results") / f"{ticker}_{args.strategy}.json"
    dashboard_path = Path("reports") / "dashboard.html"

    # --- Step 1: fetch -------------------------------------------------------
    print("=" * 70)
    print(f"STEP 1/3  Fetching {ticker} daily bars into ./data ...")
    print("=" * 70)
    fetch_argv = ["--tickers", ticker, "--interval", "1d", "--outdir", "data"]
    if args.start and args.end:
        fetch_argv += ["--start", args.start, "--end", args.end]
    else:
        fetch_argv += ["--period", args.period]
    if fetch_data.main(fetch_argv) != 0:
        return 1

    # --- Step 2: backtest ----------------------------------------------------
    print("=" * 70)
    print(f"STEP 2/3  Backtesting {args.strategy} ...")
    print("=" * 70)
    bt_argv = [
        "--data", str(data_path), "--strategy", args.strategy,
        "--capital", str(args.capital), "--out", str(results_path),
    ]
    for flag, value in [
        ("--fast", args.fast), ("--slow", args.slow), ("--rsi-period", args.rsi_period),
        ("--buy-below", args.buy_below), ("--sell-above", args.sell_above),
        ("--lookback", args.lookback),
    ]:
        if value is not None:
            bt_argv += [flag, str(value)]
    if bt.main(bt_argv) != 0:
        return 1

    # --- Step 3: report (single aggregate dashboard only) --------------------
    print("=" * 70)
    print("STEP 3/3  Refreshing dashboard.html into ./reports ...")
    print("=" * 70)
    # Everything lives in ONE file: the aggregate dashboard, built from results/*.json.
    dashboard_mod.main(["--results-dir", "results", "--out", str(dashboard_path)])

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Data      : {data_path}")
    print(f"  Results   : {results_path}")
    print(f"  Dashboard : {dashboard_path}  (ALL strategies — open this one; works offline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
