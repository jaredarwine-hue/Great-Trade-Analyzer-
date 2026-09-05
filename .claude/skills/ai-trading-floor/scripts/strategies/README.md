# Strategies — one file per strategy, auto-registered

Every trading strategy lives in **its own file in this folder**. Drop a `<name>.py` here and
it is **registered automatically** as strategy `<name>` — no edit to the engine, no central
list, no CLI wiring. This is the one place strategies are added.

```
scripts/
  backtest.py            <- the ENGINE (executes positions; do NOT add strategies here)
  indicators.py          <- indicator math (sma, ema, rsi, atr, vwap, prior highs/lows, …)
  strategies/
    __init__.py          <- the loader (auto-discovers every <name>.py; don't edit)
    _spec.py             <- the contract: StrategySpec + Param (don't edit)
    sma_crossover.py     <- one strategy
    anchored_vwap_trend.py
    short_breakdown.py   <- a SHORT-side example
    <your_new_one>.py    <- just add a file
```

## The contract (what a strategy is)

A strategy is **one function** that takes the price `DataFrame` and returns a **per-bar
position Series**:

- `+1` = be **long**, `-1` = be **short**, `0` = **flat**.
- The side is decided **entirely by the strategy's own conditions** — there is **no engine or
  CLI direction switch**, and there are **no separate "short" copies** of a strategy.
- A plain **boolean** Series still works: it's the long-only case (`True` → `+1`).
- **Do not shift it yourself** — the engine applies the single central `shift(1)` (no
  look-ahead) and fills at the next bar's open.
- Optionally return `(position, stop)` where `stop` is a per-bar stop-PRICE Series (long stop
  below entry, short stop above — the engine asserts the correct side).

Each file also exports a `SPEC` so the loader can register it and the CLI can expose its flags.

## Add a strategy (the whole recipe)

Create `scripts/strategies/my_strategy.py`:

```python
import numpy as np, pandas as pd
import indicators as ind
from ._spec import Param, StrategySpec

def my_strategy(df, fast=20, slow=50):
    f, s = ind.sma(df["Close"], int(fast)), ind.sma(df["Close"], int(slow))
    pos = np.zeros(len(df), dtype=int)
    pos[(f > s).to_numpy()] = 1     # long when the fast SMA is above the slow
    pos[(f < s).to_numpy()] = -1    # short when it's below   (drop this line for long-only)
    return pd.Series(pos, index=df.index)

SPEC = StrategySpec(
    fn=my_strategy,
    defaults={"fast": 20, "slow": 50},
    params=[
        Param("--fast", "fast", int, "my_strategy: fast SMA period."),
        Param("--slow", "slow", int, "my_strategy: slow SMA period."),
    ],
)
```

Then it just works:

```bash
python scripts/backtest.py --data data/SPY.parquet --strategy my_strategy --fast 10 --slow 40
```

`run_pipeline.py`, `run_universe.py`, and the dashboard all see it automatically — the
strategy name is the filename. (If two strategies legitimately share a flag, e.g. both use
`--lookback`, declare the same `Param` in each; the loader de-dupes by flag.)

## Turning a source (video / article / screenshot / trade log) into a strategy

This is the standard workflow when someone says *"here's a YouTube video — make this a
strategy"*:

1. **Study the source and extract the exact rules.** Write them down as conditions, grounded
   in the source — entry, exit, and which side: *"go LONG when X; go SHORT when Y; otherwise
   flat."* Every rule must trace to something the source actually says (don't invent rules).
   (On the agent floor this is the **Strategy Analyst's** spec; see `../../agent-strategy-analyst.md`.)
2. **Map each condition to an indicator** in `indicators.py` (add a new indicator there only
   if none fits). Decide the parameters and their sensible defaults.
3. **Write one new file** `scripts/strategies/<name>.py` following the recipe above — emit
   `+1` on the long conditions, `-1` on the short conditions, `0` otherwise. (On the agent
   floor this is the **Data Engineer's** job; see `../../agent-data-engineer.md`.)
4. **Backtest it** (`run_pipeline.py --strategy <name> ...`), then read the honest numbers and,
   for anything you'll rely on, walk-forward / out-of-sample it (`walkforward.py`,
   `robustness.py`) per `../../CONVENTIONS.md`. A simple template is a starting point, not an edge.

## Rules that keep it clean

- **One strategy = one file here.** Never add strategy logic to `backtest.py` (that's the engine).
- **No look-ahead:** return the raw position; let the engine shift it. Breakout-style levels
  must use `prior_period_high/low` (they exclude the current bar).
- **Direction is in the conditions**, never a flag.
- Keep the file self-contained: `import indicators as ind`, `from ._spec import StrategySpec, Param`.
