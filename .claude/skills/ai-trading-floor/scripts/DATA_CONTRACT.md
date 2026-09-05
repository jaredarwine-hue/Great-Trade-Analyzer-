# Data Contract

Every price file the AI Trading Floor toolkit reads MUST match this schema exactly. If
you port data from ANY provider (yfinance, Polygon, Alpaca, Tiingo, IBKR, a CSV export,
etc.), your job is to produce parquet files in this shape. Match it, and every script in
this kit — `backtest.py`, `render_chart.py`, `report.py` — "just works."

All files live under a `./data` folder in your current working directory. Nothing here
depends on any repository.

## The Schema (non-negotiable)

| Column   | dtype            | Notes |
| -------- | ---------------- | ----- |
| `Date`   | `datetime64[ms]` | **Timezone-naive.** Daily = the trading date at 00:00. Intraday = the bar's open time in **Eastern Time** (naive — no `+00:00`, no `tz`). |
| `Open`   | `float64`        | |
| `High`   | `float64`        | |
| `Low`    | `float64`        | |
| `Close`  | `float64`        | |
| `Volume` | `float64`        | Yes, float — not int. Cast it. |

Plus two structural rules:

1. **Index is a plain `RangeIndex`** (0, 1, 2, …). The `Date` is a regular COLUMN, NOT the
   index. Do not `set_index("Date")`. If you read with a datetime index, `reset_index()` it.
2. **Columns appear in exactly this order:** `Date, Open, High, Low, Close, Volume`. No
   `Adj Close`, no `Dividends`, no `Stock Splits`, no ticker column.

### Example daily row

```
2021-05-13, 124.58, 126.15, 124.26, 124.97, 105850339.0
```

## File naming + location (all relative to your current directory)

- **Daily bars:** `./data/<TICKER>.parquet` — e.g. `./data/AAPL.parquet`
- **Intraday 15-min bars:** `./data/intraday/<TICKER>_15min.parquet` — e.g.
  `./data/intraday/AAPL_15min.parquet`

Ticker symbols are uppercase.

## The timezone rule (most common porting bug)

Intraday `Date` values are **naive Eastern Time**. The regular-hours session is
`09:30`–`16:00` ET, so the first bar of a day reads `2022-05-09 09:30:00` with no tz suffix.

- If your provider returns UTC timestamps, convert to `America/New_York` THEN strip the
  tz (`dt.tz_convert("America/New_York").dt.tz_localize(None)`).
- If your provider returns tz-aware ET timestamps, just strip the tz
  (`dt.tz_localize(None)`).
- Never leave a `+00:00`/UTC offset on the data — downstream code treats the wall-clock
  numbers as ET directly. A UTC-labeled file shifts the whole session 4–5 hours and
  silently breaks every intraday strategy.

## Quick verification snippet

After writing a file, prove it matches the contract:

```python
import pandas as pd

df = pd.read_parquet("data/AAPL.parquet")
assert list(df.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
assert str(df["Date"].dtype).startswith("datetime64")
assert df["Date"].dt.tz is None                       # naive
assert all(str(df[c].dtype) == "float64" for c in ["Open", "High", "Low", "Close", "Volume"])
assert df.index.equals(pd.RangeIndex(len(df)))        # plain integer index
print(df.head())
print(f"{len(df)} rows, {df['Date'].min()} -> {df['Date'].max()}")
```

For intraday files, also confirm the session looks like ET (the 09:30 bar should carry
the day's heaviest opening volume):

```python
di = pd.read_parquet("data/intraday/AAPL_15min.parquet")
assert di["Date"].dt.tz is None
print(di["Date"].head(2).tolist())   # expect e.g. 09:30:00, 09:45:00 (ET, naive)
```

## Default porter checklist (for any new provider)

1. Put the API key in a `.env` file in your working directory (keep it out of git). Load
   it with `python-dotenv` or `os.environ`.
2. Study that provider's bar-download API (pagination limits, adjusted vs unadjusted,
   timezone of timestamps). **Do not mix adjusted and unadjusted prices in one file.**
3. Pull a SMALL sample first (one ticker, ~1 month) and run the verification snippet
   above before pulling the full set.
4. Only after the sample passes, port the rest.

The bundled `fetch_data.py` already implements all of this for the free yfinance provider
— read it as the reference implementation for your own porter.
