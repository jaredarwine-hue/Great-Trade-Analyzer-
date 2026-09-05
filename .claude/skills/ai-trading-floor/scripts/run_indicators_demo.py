#!/usr/bin/env python3
"""Demo: compute a handful of indicators on a parquet and prove they're aligned.

Run it to SEE the contract in action — load an engine-schema parquet, compute
~6 indicators, print the last 5 rows as a small table, and assert every output
is index-aligned (same length + same index) as the input. If any assert fails,
the indicator broke the alignment contract and the script exits non-zero.

Usage:
    python scripts/run_indicators_demo.py --parquet data/AAPL.parquet

If --parquet is omitted, the script looks for a sensible default (./data/AAPL.parquet in
your current directory — fetch it first with `scripts/fetch_data.py`) so a quick smoke
test needs no arguments.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Make the sibling indicators module importable whether you run this from the
# repo root or from inside scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import indicators as ind  # noqa: E402

# Default to ./data/AAPL.parquet in the user's current directory (cwd-relative).
DEFAULT_PARQUET = Path("data") / "AAPL.parquet"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute demo indicators on a parquet and verify alignment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--parquet",
        default=str(DEFAULT_PARQUET),
        help="Path to an engine-schema OHLCV parquet.",
    )
    return parser.parse_args(argv)


def load_ohlcv(path: Path) -> pd.DataFrame:
    """Load + sanity-check an engine-schema parquet (see DATA_CONTRACT.md)."""
    if not path.exists():
        sys.exit(f"ERROR: parquet not found: {path}")
    df = pd.read_parquet(path)
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: parquet missing required columns {missing}; got {list(df.columns)}")
    return df


def compute_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a small set of indicators and assemble an aligned display frame."""
    close, high, low = df["Close"], df["High"], df["Low"]

    sma20 = ind.sma(close, 20)
    ema20 = ind.ema(close, 20)
    rsi14 = ind.rsi(close, 14)
    atr14 = ind.atr(high, low, close, 14)
    _, bb_upper, bb_lower = ind.bollinger_bands(close, 20, 2.0)

    # Every computed Series must align to the input index — assert it loudly.
    outputs = {
        "SMA20": sma20,
        "EMA20": ema20,
        "RSI14": rsi14,
        "ATR14": atr14,
        "BB_upper": bb_upper,
        "BB_lower": bb_lower,
    }
    for name, series in outputs.items():
        assert len(series) == len(df), f"{name}: length {len(series)} != input {len(df)}"
        assert series.index.equals(df.index), f"{name}: index not aligned to input"

    table = pd.DataFrame({"Close": close, **outputs})
    table.insert(0, "Date", df["Date"].values)
    return table


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    path = Path(args.parquet)
    df = load_ohlcv(path)

    print(f"Loaded {len(df)} bars from {path}")
    print(f"Date range: {df['Date'].iloc[0]} -> {df['Date'].iloc[-1]}\n")

    table = compute_table(df)

    print("Last 5 rows (Close, SMA20, EMA20, RSI14, ATR14, BB upper/lower):")
    with pd.option_context("display.float_format", lambda v: f"{v:,.4f}"):
        print(table.tail(5).to_string(index=False))

    print("\nAll alignment asserts passed (every output is index-aligned to the input).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
