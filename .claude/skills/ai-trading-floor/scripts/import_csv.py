#!/usr/bin/env python3
"""Import price bars from a CSV/Excel file into the kit's parquet schema.

This is the NO-NETWORK data path for the AI Trading Floor toolkit. Use it when
fetch_data.py can't reach Yahoo Finance — a locked-down network, a datacenter IP
that Yahoo rate-limits, an offline machine — or when your prices come from a
broker export, a paid provider, or a spreadsheet you already have.

You do not need to clean the file first. This script figures out the common
column namings and normalizes everything to DATA_CONTRACT.md: columns
[Date, Open, High, Low, Close, Volume], plain RangeIndex, timezone-naive
datetime64[ms], OHLCV float64.

Where to get a CSV without any coding:
  * Yahoo Finance: open the ticker's "Historical Data" tab in a browser and click
    "Download" — that file imports as-is.
  * Most brokers (Fidelity, Schwab, IBKR, Robinhood) export price history to CSV.
  * Stooq, Nasdaq, and investing.com all offer CSV downloads.

Everything writes to a CWD-RELATIVE folder (default ./data), same as fetch_data.py.

Usage examples:
    # Ticker inferred from the filename (AAPL.csv -> data/AAPL.parquet)
    python import_csv.py AAPL.csv

    # Explicit ticker, and a file whose columns are named oddly
    python import_csv.py ~/Downloads/export.csv --ticker MSFT

    # Several files at once
    python import_csv.py downloads/*.csv

    # 15-minute intraday bars (timestamps are treated as Eastern Time)
    python import_csv.py AAPL_15m.csv --interval 15m

    # Check what would happen without writing anything
    python import_csv.py AAPL.csv --dry-run
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

SCHEMA_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
INTRADAY_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}

# Column aliases seen in the wild, lowercased and stripped of non-letters.
# Order matters within each list: earlier names win when a file has several.
ALIASES: dict[str, list[str]] = {
    "Date": ["date", "datetime", "time", "timestamp", "tradedate", "day", "period"],
    "Open": ["open", "openingprice", "o", "firstprice"],
    "High": ["high", "highprice", "h", "max"],
    "Low": ["low", "lowprice", "l", "min"],
    "Close": ["close", "closingprice", "c", "last", "lastprice", "price", "settle"],
    "Volume": ["volume", "vol", "v", "quantity", "shares", "totalvolume"],
}


def _key(name: object) -> str:
    """Normalize a column label for alias matching: 'Adj. Close ' -> 'adjclose'."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_columns(df: pd.DataFrame) -> tuple[dict[str, str], list[str]]:
    """Map each schema column to a real column in df. Returns (mapping, problems)."""
    available = {_key(c): c for c in df.columns}
    mapping: dict[str, str] = {}
    problems: list[str] = []

    for target, names in ALIASES.items():
        for alias in names:
            # "Close" must not silently match "Adj Close" — that's a different series,
            # and picking it changes every backtest result. Require an exact alias hit.
            if alias in available:
                mapping[target] = available[alias]
                break
        else:
            if target == "Volume":
                continue  # synthesized below; genuinely optional
            problems.append(
                f"  no column found for {target!r} — looked for any of: {', '.join(names)}"
            )

    return mapping, problems


def normalize_to_schema(raw: pd.DataFrame, interval: str, source: Path) -> pd.DataFrame:
    """Coerce an arbitrary price table into the DATA_CONTRACT.md schema."""
    mapping, problems = find_columns(raw)
    if problems:
        raise ValueError(
            f"{source.name}: could not match the required columns.\n"
            + "\n".join(problems)
            + f"\n  columns present: {list(raw.columns)}"
        )

    df = pd.DataFrame({target: raw[src] for target, src in mapping.items()})

    if "Volume" not in df.columns:
        print(f"  note: no volume column found — filling Volume with 0.0")
        df["Volume"] = 0.0

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", format="mixed")
    if df["Date"].isna().all():
        raise ValueError(f"{source.name}: no parseable dates in the date column.")

    # Timezone rule (DATA_CONTRACT.md): stored naive. Intraday means naive ET, so a
    # tz-aware source is converted to New York first; daily bars just lose the tz.
    if getattr(df["Date"].dt, "tz", None) is not None:
        if interval in INTRADAY_INTERVALS:
            df["Date"] = df["Date"].dt.tz_convert("America/New_York")
        df["Date"] = df["Date"].dt.tz_localize(None)

    df["Date"] = df["Date"].astype("datetime64[ms]")

    for col in PRICE_COLUMNS:
        # Strip thousands separators and currency symbols before coercing. Test for
        # "not numeric" rather than "is object": pandas 3.0 reads text columns as the
        # new 'str' dtype, so an `== object` check silently skips the cleaning and
        # every priced-with-a-$ row becomes NaN.
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(str).str.replace(r"[,$€£\s]", "", regex=True)
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    before = len(df)
    df = df.dropna(subset=["Date"] + PRICE_COLUMNS)
    df = df.sort_values("Date").drop_duplicates(subset="Date", keep="last")
    df = df[SCHEMA_COLUMNS].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  note: dropped {dropped} row(s) with unparseable dates or prices")

    if df.empty:
        raise ValueError(f"{source.name}: no usable rows left after cleaning.")

    return df


def check_contract(df: pd.DataFrame) -> list[str]:
    """Re-assert DATA_CONTRACT.md so a bad import fails here, not mid-backtest."""
    issues: list[str] = []
    if list(df.columns) != SCHEMA_COLUMNS:
        issues.append(f"column order is {list(df.columns)}, expected {SCHEMA_COLUMNS}")
    if not str(df["Date"].dtype).startswith("datetime64"):
        issues.append(f"Date dtype is {df['Date'].dtype}, expected datetime64")
    if getattr(df["Date"].dt, "tz", None) is not None:
        issues.append("Date is timezone-aware, expected naive")
    for col in PRICE_COLUMNS:
        if str(df[col].dtype) != "float64":
            issues.append(f"{col} dtype is {df[col].dtype}, expected float64")
    if not df.index.equals(pd.RangeIndex(len(df))):
        issues.append("index is not a plain RangeIndex")
    bad_high = (df["High"] < df[["Open", "Close"]].max(axis=1)).sum()
    bad_low = (df["Low"] > df[["Open", "Close"]].min(axis=1)).sum()
    if bad_high:
        issues.append(f"{bad_high} row(s) where High is below Open/Close")
    if bad_low:
        issues.append(f"{bad_low} row(s) where Low is above Open/Close")
    return issues


def read_any(path: Path) -> pd.DataFrame:
    """Read CSV, TSV, or Excel based on the file's suffix."""
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def ticker_from(path: Path, override: str | None) -> str:
    """Ticker comes from --ticker, else the filename ('aapl_daily.csv' -> AAPL)."""
    if override:
        return override.upper()
    stem = path.stem
    # Trim common decorations: AAPL_daily, AAPL-2024, AAPL (1)
    stem = re.split(r"[_\-. (]", stem)[0]
    return re.sub(r"[^A-Za-z0-9.^=-]", "", stem).upper() or "UNKNOWN"


def output_path(ticker: str, interval: str, outdir: Path) -> Path:
    if interval in INTRADAY_INTERVALS:
        return outdir / "intraday" / f"{ticker}_{interval}.parquet"
    return outdir / f"{ticker}.parquet"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import CSV/Excel price bars into the kit's parquet schema.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="+", help="CSV/TSV/Excel file(s). Globs ok.")
    parser.add_argument(
        "--ticker",
        help="Ticker symbol. Defaults to the filename. Only valid with ONE input file.",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="Bar interval. '1d' = daily; '15m' / '1h' etc. = intraday (naive ET).",
    )
    parser.add_argument(
        "--outdir", default="data", help="Output directory (relative to your CWD)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate, but write nothing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    files: list[Path] = []
    for raw in args.paths:
        matches = sorted(Path().glob(raw)) if any(c in raw for c in "*?[") else [Path(raw)]
        files.extend(matches)

    missing = [f for f in files if not f.is_file()]
    if missing:
        print("Files not found:", ", ".join(str(m) for m in missing), file=sys.stderr)
        return 1
    if not files:
        print("No input files matched.", file=sys.stderr)
        return 1
    if args.ticker and len(files) > 1:
        print("--ticker only works with a single input file.", file=sys.stderr)
        return 1

    outdir = Path(args.outdir)
    written = 0

    for path in files:
        ticker = ticker_from(path, args.ticker)
        print(f"Importing {path} as {ticker} ({args.interval}) ...")
        try:
            df = normalize_to_schema(read_any(path), args.interval, path)
        except Exception as exc:  # noqa: BLE001 — surface the reason plainly
            print(f"  FAILED: {exc}", file=sys.stderr)
            continue

        issues = check_contract(df)
        if issues:
            print("  FAILED contract check:", file=sys.stderr)
            for issue in issues:
                print(f"    - {issue}", file=sys.stderr)
            continue

        span = f"{df['Date'].min()} -> {df['Date'].max()}"
        if args.dry_run:
            print(f"  OK (dry run): {len(df)} rows, {span}")
            continue

        out = output_path(ticker, args.interval, outdir)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"  Wrote {out}  ({len(df)} rows, {span})")
        written += 1

    if not args.dry_run and written:
        print(f"\n{written} file(s) imported. Next:")
        print(f"  python scripts/backtest.py --data {outdir}/<TICKER>.parquet --strategy sma_crossover")
        print("  python scripts/dashboard.py")

    return 0 if (written or args.dry_run) else 1


if __name__ == "__main__":
    raise SystemExit(main())
