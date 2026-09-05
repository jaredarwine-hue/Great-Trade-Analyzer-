#!/usr/bin/env python3
"""Fetch OHLCV bars from yfinance into the kit's parquet schema (self-contained).

This is the free, no-API-key data path for the AI Trading Floor toolkit. It downloads
bars from Yahoo Finance and normalizes them to the schema in DATA_CONTRACT.md:
columns [Date, Open, High, Low, Close, Volume], plain RangeIndex, timezone-naive
datetime64[ms] (Eastern Time for intraday), OHLCV float64. Intraday timestamps are
converted to naive Eastern Time.

Everything writes to a CWD-RELATIVE folder (default ./data). Nothing in this script
reads from or writes to any repository — it works in any directory you run it from.

Usage examples:
    # Three years of daily AAPL (defaults: --tickers AAPL, --interval 1d, into ./data)
    python fetch_data.py --period 3y

    # Multiple tickers, explicit date range
    python fetch_data.py --tickers AAPL,MSFT --start 2021-01-01 --end 2024-01-01

    # 15-minute intraday (Yahoo only serves ~60 days of intraday history)
    python fetch_data.py --tickers AAPL --period 60d --interval 15m

Output layout (relative to your current directory):
    ./data/<TICKER>.parquet                       (daily)
    ./data/intraday/<TICKER>_<interval>.parquet   (intraday)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCHEMA_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
INTRADAY_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch yfinance bars into the kit's parquet schema (writes to ./data).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tickers",
        default="AAPL",
        help="Comma-separated ticker symbols.",
    )
    parser.add_argument("--start", help="Start date YYYY-MM-DD (use with --end).")
    parser.add_argument("--end", help="End date YYYY-MM-DD (use with --start).")
    parser.add_argument(
        "--period",
        help="Lookback window instead of start/end, e.g. 3y, 6mo, 60d.",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="Bar interval. '1d' = daily; '15m' / '1h' etc. = intraday.",
    )
    parser.add_argument(
        "--outdir",
        default="data",
        help="Output directory (relative to your current directory).",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> list[str]:
    """Validate CLI inputs, return the cleaned ticker list, or fail fast."""
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        sys.exit("ERROR: --tickers produced an empty list. Pass e.g. --tickers AAPL,MSFT")

    has_range = bool(args.start) and bool(args.end)
    has_partial_range = bool(args.start) != bool(args.end)
    if has_partial_range:
        sys.exit("ERROR: --start and --end must be used together.")
    if not has_range and not args.period:
        sys.exit("ERROR: provide either --period (e.g. 3y) or both --start and --end.")
    return tickers


def normalize_to_schema(raw: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Coerce a yfinance frame into the exact kit schema (see DATA_CONTRACT.md)."""
    df = raw.copy()

    # yfinance can return a MultiIndex on columns for single-ticker pulls; flatten it.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Date is the index on a yfinance download -> make it a plain column.
    # Newer yfinance (1.x) leaves the DatetimeIndex unnamed; older versions named
    # it "Date" (daily) or "Datetime" (intraday). Name it so reset_index yields a
    # recognizable column instead of a generic "index".
    if df.index.name not in ("Date", "Datetime"):
        df.index.name = "Date"
    df = df.reset_index()
    date_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={date_col: "Date"})

    # Drop everything the schema doesn't want (Adj Close, Dividends, Stock Splits, ...).
    df = df.rename(columns=str.title)  # normalize "open"->"Open" etc. if lowercased
    missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"yfinance frame missing expected columns: {missing}")
    df = df[SCHEMA_COLUMNS]

    # Timezone handling. Intraday must end up naive Eastern Time; daily is tz-naive.
    df["Date"] = pd.to_datetime(df["Date"])
    if getattr(df["Date"].dt, "tz", None) is not None:
        if interval in INTRADAY_INTERVALS:
            df["Date"] = df["Date"].dt.tz_convert("America/New_York").dt.tz_localize(None)
        else:
            df["Date"] = df["Date"].dt.tz_localize(None)
    df["Date"] = df["Date"].astype("datetime64[ms]")

    for col in PRICE_COLUMNS:
        df[col] = df[col].astype("float64")

    df = df.dropna(subset=PRICE_COLUMNS).reset_index(drop=True)
    return df


def download_one(ticker: str, args: argparse.Namespace) -> pd.DataFrame:
    import yfinance as yf  # imported lazily so --help works without the dep

    kwargs = {"interval": args.interval, "auto_adjust": False, "progress": False}
    if args.period:
        kwargs["period"] = args.period
    else:
        kwargs["start"] = args.start
        kwargs["end"] = args.end
    return yf.download(ticker, **kwargs)


def output_path(ticker: str, interval: str, outdir: Path) -> Path:
    if interval in INTRADAY_INTERVALS:
        return outdir / "intraday" / f"{ticker}_{interval}.parquet"
    return outdir / f"{ticker}.parquet"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tickers = validate_args(args)
    outdir = Path(args.outdir)

    written = 0
    for ticker in tickers:
        print(f"Fetching {ticker} ({args.interval}) ...")
        raw = download_one(ticker, args)
        if raw is None or raw.empty:
            print(f"  WARNING: no data returned for {ticker}. Skipping.")
            continue

        df = normalize_to_schema(raw, args.interval)
        if df.empty:
            print(f"  WARNING: {ticker} had no usable rows after cleaning. Skipping.")
            continue

        out = output_path(ticker, args.interval, outdir)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        written += 1

        start, end = df["Date"].min(), df["Date"].max()
        print(f"  Saved {len(df)} rows to {out}")
        print(f"  Date range: {start} -> {end}")
        print(f"  Columns: {list(df.columns)}")
        print("  Head:")
        print(df.head().to_string(index=False))
        print()

    if written == 0:
        print("No files written. Check your tickers / date range / interval.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
