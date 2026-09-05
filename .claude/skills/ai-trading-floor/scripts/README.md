# How indicators work in this kit

A starter library for technical indicators, written to be *read*. The goal isn't
just to give you `rsi()` — it's to teach the PATTERN so you can build your own and
trust the numbers. Everything lives in `indicators.py` (pandas + numpy only, no
project imports), and the formulas follow the standard conventions used by
production backtest engines, so your study numbers reproduce the kit's own
`scripts/backtest.py`.

---

## 0. The toolkit at a glance (everything here is self-contained & CWD-relative)

Every script reads/writes folders in the **current working directory** (`./data`,
`./results`, `./reports`) — nothing is hardcoded to a machine, so the skill works
unchanged in any fresh project. Deps: `../requirements.txt`.

| Script | Does |
| --- | --- |
| `fetch_data.py` | Yahoo Finance bars → `./data/<TICKER>.parquet` (version-robust to yfinance's unnamed-index change) |
| `backtest.py` | Run one strategy → `./results/<...>.json`. `simulate()` takes an optional `stop` Series (intrabar ATR stops: exit `min(stop,Open)` on `Low<=stop`, `assert stop<entry`, inverted stops skipped, ambiguous bars counted). |
| `indicators.py` | The indicator library (this doc) |
| `render_chart.py` | Candlestick PNG (MAs, trade markers, `--cutoff`) — see `../CHART_RENDERER.md` |
| `report.py` | Interactive offline Plotly HTML for ONE strategy |
| `dashboard.py` | **The headline output.** Aggregates ALL `results/*.json` + `portfolio_*.json` → one offline `reports/dashboard.html`: asset-class→ticker→strategy tree, Portfolios with IS⇄OOS toggle + correlation heatmap, Sharpe+Calmar cards, strategy-specific overlays, ★-save (localStorage), price embedded once per ticker, portfolios persisted in `results/.portfolio_cache/`. |
| `run_pipeline.py` | fetch → backtest → refresh dashboard (one command) |
| `run_universe.py` | Batch a universe × strategies → IS `results/`, OOS `results/oos/`, grid `results/grid/` |
| `run_rotation.py` | Cross-sectional monthly momentum **rotation** (separate runner — `simulate()` can't rank/rotate) |
| `portfolio.py` | Combine N strategies → correlation matrix + 1-day-lagged inverse-vol/equal-dollar blend + low-corr subset search + deflated Sharpe + IS/OOS. Every `portfolio_<name>.json` now also EMBEDS a `portfolio_walkforward` block for the chosen legs (`--no-walkforward` to skip). |
| `portfolio_walkforward.py` | Portfolio-LEVEL walk-forward: rebuilds portfolio.py's lagged blend of the chosen legs, slices the ONE combined return series into walkforward.py's anchored windows → per-window OOS Sharpe + `wf_efficiency = mean(OOS)/full_is_sharpe`. Imports WINDOWS/EFFICIENCY_IS_EPS from walkforward.py and the combine math from portfolio.py (single source of truth — no redeclaration). Auto-embedded by portfolio.py; also a standalone CLI. |
| `walkforward.py` | Anchored multi-window walk-forward of a SINGLE strategy → per-window OOS Sharpe + efficiency |
| `robustness.py` | Parameter-plateau scoring from the grid (plateau vs spiky) |

**Built-in strategies** (`STRATEGIES` registry in `backtest.py`, all long-only): `sma_crossover`,
`rsi_reversion`, `breakout`, `bollinger_meanrev` (SMA200+RSI filters, ATR + time stop),
`donchian_trend` (55/20 + 2·ATR), `vol_gate_trend` (EMA50/200 + annualized-vol gate + 3·ATR —
an honest GATE, not vol sizing, since the engine is all-in), `dual_momentum`. Add your own via
the "HOW TO ADD YOUR OWN STRATEGY" block at the top of `backtest.py`.

**Honesty rules** live in `../CONVENTIONS.md` — including §8 "Avoiding Overfitting" (walk-forward,
plateau robustness, deflated Sharpe, trade-count/CI, select-by-robustness-not-IS-max).

---

## 1. The one contract every indicator follows

> **in:** a price `Series` (e.g. `close`) **or** an OHLCV `DataFrame`
> **out:** a `pd.Series` (or tuple of Series) **aligned to the input's index**

"Aligned" is the load-bearing word. The output has the **same length** and the
**same index labels** as the input, so this just works:

```python
import pandas as pd
import indicators as ind

df = pd.read_parquet("data/AAPL.parquet")          # [Date, Open, High, Low, Close, Volume]
df["rsi14"] = ind.rsi(df["Close"], 14)             # same length -> drops in cleanly
df["sma20"] = ind.sma(df["Close"], 20)
```

Two flavors of input, depending on what the math needs:

- **Price-Series in** — `sma`, `ema`, `rsi`, `macd`, `obv`, `rolling_high/low`,
  `swing_pivot_*`. These take just the column(s) they use.
- **OHLCV-DataFrame in** — `atr`, `bollinger_bands`, `keltner_channels`,
  `williams_r`, `stochastic`, `vwap_session`, `anchored_vwap`,
  `prior_period_high/low`, `adx`. These need High/Low/Close (and Volume and/or a
  `Date` column for the VWAP family), so they take the relevant Series or the
  whole frame.

Functions that return MULTIPLE lines hand back a **tuple of aligned Series**:

```python
macd_line, signal_line, histogram = ind.macd(df["Close"])
mid, upper, lower               = ind.bollinger_bands(df["Close"])
percent_k, percent_d            = ind.stochastic(df["High"], df["Low"], df["Close"])
```

One function returns **discrete structural points**, not a per-bar Series —
`swing_pivot_lows` / `swing_pivot_highs` return `(positions, levels)` lists,
because a pivot is an event at specific bars, not a value on every bar. Map a
position back to a timestamp with `df.index[pos]` or `df["Date"].iloc[pos]`.

---

## 2. Simple vs Wilder smoothing (the thing beginners trip on)

Most "averages" in technical analysis are one of two families. Getting this wrong
is the #1 reason your RSI doesn't match your charting platform.

| Family | Code | Weighting | Used by |
|---|---|---|---|
| **Simple (SMA)** | `series.rolling(period).mean()` | every bar in the window equal | SMA, Bollinger mid, this kit's `atr` |
| **EMA (span)** | `series.ewm(span=period, adjust=False).mean()`, `alpha = 2/(period+1)` | recent bars more | EMA, MACD, Keltner mid |
| **Wilder (RMA)** | `series.ewm(alpha=1/period, ...).mean()` | recent bars more, but *slower* than span-EMA | RSI, `atr_wilder`, ADX |

Wilder smoothing is just an EMA with a different `alpha` (`1/period` instead of
`2/(period+1)`). It's slower-reacting — that's why a 14-period Wilder RSI looks
calmer than a 14-period EMA would. This kit factors it into one shared helper,
`wilder_smooth()`, so RSI / `atr_wilder` / ADX all smooth identically.

**The ATR fork worth knowing:** textbook ATR is Wilder-smoothed, but the project's
engine computes ATR as a *simple* rolling mean of True Range. This kit's `atr()`
matches the engine (simple mean) so your study numbers reproduce the backtester;
`atr_wilder()` gives you the textbook version if you'd rather match a chart. Same
True Range underneath — only the smoothing differs. When in doubt, match the
engine.

---

## 3. Warmup and NaN handling

An indicator can't produce a value before it has enough history. This kit keeps
those early bars as **`NaN`** rather than dropping them or faking a number — that's
what preserves alignment, and it's honest about "we don't know yet."

- `sma(close, 20)` -> first **19** bars are `NaN` (the window isn't full).
- `rsi(close, 14)` -> first **14** bars are `NaN`.
- `atr(.., 14)` -> first **14** bars `NaN` (True Range is `NaN` on bar 0, then the
  rolling mean needs 14 values).
- `ema(close, 20)` -> no `NaN` after bar 0 (the recursion seeds immediately), but
  treat the first ~20 bars as "still warming up."
- `adx(.., 14)` -> roughly the first **2×period** bars `NaN` (DX is smoothed twice).

Two deliberate choices to notice:

1. **NaN, not a filler.** The engine's `rsi` ends with `.fillna(50.0)`; this kit
   leaves warmup as `NaN`. A teaching library should never let you accidentally
   trade on a "50" that's really "unknown." Decide your own fill at the call site.
2. **Divide-by-zero is masked, not forced.** When a range is flat (e.g.
   `high == low` for a Stochastic window) the bar becomes `NaN`, never `inf`. Look
   for `.where(rng > 0)` in the code — that's the guard.

Always check `len(df) >= period` before reading the tail of a freshly computed
indicator on a short slice, or you'll get an all-NaN column and wonder why.

---

## 4. The session-VWAP reset rule

`vwap_session(df)` is the one indicator with a non-obvious rule: it **resets at
each calendar day** (09:30 ET session open). VWAP is "the volume-weighted average
price *so far today*", so yesterday's flow must not anchor today's line. In code
that's a `groupby(date).cumsum()` instead of a plain `cumsum()`:

```python
cum_pv  = (typical_price * volume).groupby(date).cumsum()   # resets each day
cum_vol = volume.groupby(date).cumsum()
vwap    = cum_pv / cum_vol
```

Because the reset is the whole point, `vwap_session` **requires intraday data**
(>= 2 bars per calendar date) and raises a clear error on daily data — on daily
bars session VWAP would just equal each bar's typical price, which tells you
nothing. This matches the engine's `intraday_vwap_proper`, which rejects daily
input for the same reason.

Need a line that anchors to ONE event and runs forward (an earnings gap, a swing
low) instead of resetting daily? That's `anchored_vwap(df, anchor_date)` — same
cumulative math, but it starts at your chosen date and never resets (bars before
the anchor are `NaN`).

---

## 5. No future leakage (why some functions `shift(1)`)

A research indicator must only use data available **at or before** the bar it's
computed for. Two patterns in this kit encode that:

- **`rolling_high` / `rolling_low` INCLUDE the current bar** — use them for "is
  today a 20-bar high?" reads where seeing today is correct.
- **`prior_period_high` / `prior_period_low` EXCLUDE the current bar** via
  `shift(1)` before rolling. They answer "what level must *today's* bar clear to
  break out?" — and a breakout level that included today's own high would be
  trivially true. The `shift(1)` is the no-leakage guard; the engine's
  `prior_period_*` indicators do exactly this.

When you write your own, ask: *does this bar's value depend on data it couldn't
have known yet?* If yes, `shift` it.

---

## 6. Recipe: add your own indicator

Follow the existing functions as templates. Six steps:

1. **Pick your input shape.** Just `close`? Take `close: pd.Series`. Need
   High/Low/Close or Volume? Take those Series (or the whole `df`). Match the
   nearest existing function's signature so the kit stays consistent.

2. **Write the signature with type hints + defaults.**
   ```python
   def my_indicator(close: pd.Series, period: int = 20) -> pd.Series:
   ```

3. **Write the docstring FIRST.** State (a) the formula, (b) what it's used for,
   (c) the warmup/NaN behavior. If you can't write the formula plainly, you don't
   understand it well enough to implement it.

4. **Vectorize with pandas; reuse the helpers.** Reach for `.rolling()`,
   `.ewm()`, `.shift()`, `.diff()`, `.cumsum()`. Smoothing? Call `ema` or
   `wilder_smooth`. True range? Call `_true_range`. Don't re-derive.

5. **Preserve alignment + guard the edges.** Return a Series on the SAME index as
   the input. Never raise on short input — let the warmup region be `NaN`
   (rolling/ewm do this for free). Mask divide-by-zero with `.where(denom > 0)`.
   For event-style outputs (pivots), return positions+levels lists and return
   them empty when the window can't fit (see `_swing_pivots`).

6. **Prove it.** Add it to `run_indicators_demo.py`'s `compute_table` and confirm
   the alignment asserts still pass:
   ```bash
   source venv/bin/activate
   python scripts/run_indicators_demo.py --parquet data/AAPL.parquet
   ```

A minimal worked example — distance from the close to its own SMA, in percent:

```python
def dist_from_sma_pct(close: pd.Series, period: int = 50) -> pd.Series:
    """Percent the close sits above (+) or below (-) its `period`-bar SMA.

    Formula: 100 * (close - SMA(close, period)) / SMA(close, period).
    Used to gauge how stretched price is from its mean.
    Warmup/NaN: first `period - 1` bars are NaN (SMA isn't full yet).
    """
    ma = sma(close, period)            # reuse the kit's SMA -> aligned, warmup-safe
    return 100.0 * (close - ma) / ma   # division by NaN stays NaN -> alignment intact
```

---

## 7. Quick reference

| Function | Input | Output |
|---|---|---|
| `sma(close, period)` | Series | Series |
| `ema(close, period)` | Series | Series |
| `wilder_smooth(series, period)` | Series | Series (shared helper) |
| `rsi(close, period=14)` | Series | Series [0,100] |
| `macd(close, 12, 26, 9)` | Series | (line, signal, hist) |
| `atr(high, low, close, 14)` | Series×3 | Series (simple-mean, engine match) |
| `atr_wilder(high, low, close, 14)` | Series×3 | Series (textbook Wilder) |
| `bollinger_bands(close, 20, 2.0)` | Series | (mid, upper, lower) |
| `keltner_channels(high, low, close, ...)` | Series×3 | (mid, upper, lower) |
| `williams_r(high, low, close, 14)` | Series×3 | Series [-100,0] |
| `stochastic(high, low, close, 14, 3)` | Series×3 | (%K, %D) |
| `obv(close, volume)` | Series×2 | Series |
| `vwap_session(df)` | OHLCV df | Series (resets daily; intraday only) |
| `anchored_vwap(df, anchor_date)` | OHLCV df | Series (NaN before anchor) |
| `rolling_high/low(series, period)` | Series | Series (incl. current bar) |
| `prior_period_high/low(df, window)` | OHLCV df | Series (excl. current bar) |
| `swing_pivot_lows/highs(series, lookback=3)` | Series | (positions, levels) lists |
| `adx(high, low, close, 14)` | Series×3 | Series [0,100] (trend strength) |

Run `python scripts/run_indicators_demo.py` for a live, alignment-checked example.
