# Backtest Conventions — the rules that keep results honest

These are the bug-prevention rules every backtest in the AI Trading Floor kit must
follow. They are the difference between a number you can trust and a number that's lying
to you. The bundled engine (`scripts/backtest.py`) already enforces all of them; if you
or an agent writes new strategy code, hold it to this same bar.

Everything here is generic and self-contained — it applies to any parquet in `./data`
(kit schema: `Date, Open, High, Low, Close, Volume`, naive datetimes, see
`scripts/DATA_CONTRACT.md`) run through `scripts/`. No external engine required.

---

## 1. No look-ahead (shift your signals)

A signal computed on a given bar may only use information available **at or before that
bar**. The classic leak: deciding to be long *today* using *today's* close, then pretending
you traded at today's open. You couldn't have known today's close at the open.

- **Rule:** turn your strategy into a boolean "should I be long?" Series, then `shift(1)`
  it before trading. A signal seen at the close of bar `i` is acted on at bar `i+1`.
- The bundled engine does this **once, centrally** (`signal.shift(1)` in
  `scripts/backtest.py::simulate`) so no individual strategy can forget it.
- Watch indicators that "see" the current bar: an SMA/RSI/ATR computed *including* bar `i`
  and used to decide bar `i`'s action is a leak. `scripts/indicators.py::prior_period_high`
  already excludes the current bar (`shift(1)` inside) — use that kind of helper for
  breakout levels.

## 2. Realistic entry fills (next-bar-open for retroactive signals)

You cannot fill at a price the market never offered you, and you cannot fill before you
knew the signal.

- **Retroactively discovered signals** (a moving-average cross, an RSI threshold, "the bar
  closed above X") are only known *after* that bar closes → fill at the **next bar's open**.
  This is what the bundled engine does.
- **Pre-computed levels** known *before* the bar (yesterday's high, an opening-range high, a
  support level you'd place a limit order at) can fill same-bar at that level — a trader
  really could have rested the order there in advance.
- For a breakout, never fill below the open: `entry = max(signal_price, bar Open)`. On a
  gap-up day the real fill is the (worse) open, not the level.

## 3. Daily-aggregated Sharpe (never per-trade)

Sharpe must be computed on **daily** portfolio returns, then annualized by `sqrt(252)`.

- **Wrong:** `trade_returns.mean() / trade_returns.std() * sqrt(252)`. With N trades per
  day this inflates Sharpe by roughly `sqrt(N)` — a pure artifact.
- **Right:** aggregate the equity curve to one value per calendar day, take day-over-day
  percent changes, then `mean/std * sqrt(252)`. The bundled engine reports this as
  `sharpe_daily` (`scripts/backtest.py::compute_stats`).

## 4. Stops on the losing side (assert it)

A stop must sit on the side that *loses* money, or every "stop-out" records as a win and
your win rate becomes fiction.

- **Rule:** always `assert stop < entry` for longs and `assert stop > entry` for shorts.
- An inverted stop (e.g. stop placed at a level *above* a long entry) turns a 40%-real
  strategy into a fake 95% win rate. If you ever see >90% WR with mostly stop exits, suspect
  this first.

## 5. Same-bar fill realism (the intraday killer)

On coarse bars (15-min, hourly), if a trade's **entry and target sit close together**, both
can fall inside the *same* bar — and you cannot prove the dip happened before the bounce.

- **Rule:** for any intraday strategy, count how many winning trades have entry and exit on
  the same bar. If that's **> 20% of wins, the result is unreliable** — flag it, rerun
  excluding same-bar wins, or move to finer (1-min) bars.
- This silently kills mean-reversion intraday backtests (dip-buys, gap fills, VWAP reversion).
  Daily bars don't have this problem; the bundled templates run on daily data.

## 6. Flag ambiguous intrabar stop/target

When **both** a stop and a target fall within one bar's high-low range, you can't know
which was hit first without finer data.

- **Rule:** don't silently pick one with `if/elif`. Count these ambiguous bars and report
  them. If they're more than a few percent of trades, the result isn't trustworthy at that
  bar resolution. The conservative default is to assume the *worse* outcome.

## 7. Intraday timestamps are naive Eastern Time

Intraday parquet `Date` values are naive **Eastern Time** (see `scripts/DATA_CONTRACT.md`),
not UTC. The regular-hours session is `09:30`–`16:00` ET.

- **Rule:** filter regular hours with `time(9, 30)`–`time(16, 0)`. Never apply a UTC offset
  like `time(13, 30)` — that shifts the whole session 4-5 hours and breaks every
  time-of-day rule. (Daily bars carry the date at `00:00` and don't need this.)

---

## 8. Avoiding overfitting (does this edge actually generalize?)

Rules 1–7 stop a backtest from *cheating*. These stop a backtest from *fooling you* — a
clean, leak-free result can still be pure curve-fit. This section decides whether a number
is real. It matters most for portfolios and any time you SELECT a winner from many candidates.

- **A high Sharpe on few trades is noise.** Sharpe has a confidence interval that scales with
  `1/sqrt(n)`; on a 1-year window (~250 days) it's roughly ±1.3 wide. A single-name "Sharpe
  > 2" on 1–5 trades is a buy-and-hold artifact, not an edge. **Require ≥ ~30 trades** and
  report the CI, never the bare point estimate. (Lived example: a "Sharpe 2.95" that was one
  trade riding a stock up for a year.)
- **Maximizing in-sample Sharpe SELECTS FOR overfit.** The combo that tops the in-sample
  metric is reliably the worst out-of-sample. **Pick the final strategy/portfolio by
  robustness, not by best in-sample Sharpe.** (Lived example: the IS-max combo went IS 2.03 →
  OOS 0.94; a lower IS 1.80 combo held OOS 1.45.)
- **One out-of-sample window only tests one regime.** A single holdout (e.g. just the last
  year) can flatter or punish a strategy by luck of the regime. Use **walk-forward**
  (`scripts/walkforward.py`): anchored/expanding windows across several years. Report per-window
  OOS Sharpe and **efficiency = mean(OOS Sharpe) / IS Sharpe** — you want ~0.6, positive in
  most windows. **OOS *wildly beating* IS (efficiency ≫ 1) is a regime artifact, not
  robustness** — the mirror image of overfitting.
- **Parameter robustness — plateau, not spike.** A setting only counts if its *neighbors* in
  the param grid also work. Score it (`scripts/robustness.py`: median-neighbor / best-neighbor
  Sharpe); a lone spike that collapses when you nudge a parameter ±1 was fit to that value.
- **Deflate for the search (selection bias).** When you pick the best of N tried combos, the
  winner's Sharpe is biased upward. Compute a **deflated Sharpe** from `combos_tried`; a tidy
  > 2 out of hundreds of attempts is usually luck. (Lived example: a 1.35 IS combo deflated to
  ~0 after accounting for the 337-combo search.)
- **Concentration & correlation stability (portfolios).** Flag if one ticker or a handful of
  days drives most of the return. For a blend, recompute leg correlations on 2–3 subperiods —
  diversification that only existed once won't hold forward.
- **Survivorship.** Yahoo gives currently-listed names only; a hand-picked universe of today's
  survivors flatters any stock-picking backtest. State this caveat on any cross-sectional claim.

The honest deliverable is the OUT-OF-SAMPLE / walk-forward number stated as a *range* with its
caveats — never an in-sample or single-window figure presented as a live expectation.

---

## Quick self-audit before trusting any new backtest

- [ ] Signals shifted by 1 bar (no look-ahead)?
- [ ] Retroactive entries filled at next-bar open (or `max(level, Open)` for breakouts)?
- [ ] Sharpe computed on daily returns, not per-trade?
- [ ] `stop < entry` (long) / `stop > entry` (short) asserted?
- [ ] Intraday: same-bar wins < 20%, ambiguous intrabar bars reported?
- [ ] Intraday: regular-hours filter uses ET constants (9:30–16:00)?
- [ ] A rerunnable `.py` script exists so the result can be reproduced?
- [ ] Enough trades (≥ ~30) and Sharpe reported with its confidence interval, not bare?
- [ ] Selected by robustness (not max in-sample Sharpe), and walk-forward / OOS confirms it?
- [ ] If picked from a search, Sharpe deflated for `combos_tried`?

If any box is unchecked, the numbers aren't trustworthy yet.
