# Orchestrator (Team-Lead Playbook)

> **This is NOT a teammate to spawn.** The team lead (main thread) plays the Orchestrator
> role itself — this file is its playbook for combining strategies into a portfolio. Do not
> create an "orchestrator" agent in the `ai-trading-floor` team; just follow this directly,
> driving the bundled `portfolio.py` / `walkforward.py` / `robustness.py` scripts.

## Role
The Orchestrator sits one level above individual strategies on the `ai-trading-floor` team.
Once two or more strategies have been backtested AND audited, the team lead decides how to
split capital between them to build a smoother combined portfolio — and tests whether the
combination actually beats the best single strategy. It's the "portfolio" step. You only
reach it after you have multiple working backtests; for a first idea you don't need it.

## What You Work With
- **Per-strategy results:** the `./results/<ticker>_<strategy>.json` files the Data Engineer
  produced. Each has a per-bar `equity_curve` and `stats` — those are your inputs.
- **The backtest engine:** `scripts/backtest.py` (to (re)run any sub-strategy you need).
- **The rules:** `CONVENTIONS.md` (the 1-day-lag rule below is the portfolio-specific one).
- Write your combined results as a small JSON/markdown in the user's working dir (e.g.
  `./results/portfolio_<name>.json`) so they're reproducible.

## The 1-Day Lag Rule (the critical portfolio concept)
On day T, the allocation weights must come from day **T-1** data (or earlier). The
orchestrator decides how to split capital BEFORE day T's returns are realized — so any
weight that uses day-T data to set day-T allocation is look-ahead bias.

```
weight_A(T), weight_B(T)  = f(data through T-1)        # decided before T's returns exist
portfolio_return(T)       = weight_A(T)*ret_A(T) + weight_B(T)*ret_B(T)
```

Every allocation rule must `shift(1)` its signal. The sub-strategy returns themselves are
already computed (from each strategy's equity curve), so you only combine them — you don't
re-decide their internal trades.

## How to Combine Strategies
1. Load each sub-strategy's `equity_curve` from its results JSON; convert to daily returns
   (`pct_change`), aligned on date.
2. Define an allocation rule that returns daily `(weight_A, weight_B, …)` summing to ≤ 1,
   computed from data through T-1 (shift it). Examples: fixed split (e.g. 60/40),
   regime-based (more to the trend strategy when a trend filter is on), or signal-strength.
3. Combine: `portfolio_ret(T) = Σ weight_i(T) * ret_i(T)`. Compound into an equity curve.
4. Compute the same stats the engine uses — total return, CAGR, **daily-aggregated Sharpe**,
   max drawdown — and the time-in-each-allocation distribution.
5. Compare against the best single sub-strategy. If the combo doesn't beat it on
   risk-adjusted terms (Sharpe / drawdown), say so plainly — diversification isn't free.

## Things That Bite
- **Look-ahead via same-day weights** — the #1 mistake. Always lag.
- **Equal-dollar vs risk-parity** — a high-volatility strategy dominates an equal-dollar
  blend; inverse-vol weighting can permanently starve it. State which you used and why.
- **Static vs dynamic splits** — a fixed split leaves capital idle when one strategy is in
  cash; a dynamic split keeps it working but adds turnover. Trade-off, not a free lunch.
- **Small-sample combos** — if the "win" comes from a handful of days, it's noise. Check.

## Playbook (the team lead follows this directly — it is NOT a spawn prompt)

```
As the team lead acting as the Orchestrator on team ai-trading-floor: after two or more
strategies are backtested AND audited, you decide how to split capital between them and test
whether the combination beats the best single strategy. This is the portfolio step — do it
yourself, do not spawn a teammate for it.

YOUR INPUTS: the ./results/<ticker>_<strategy>.json files from the Data Engineer — each has
a per-bar equity_curve and stats. You combine those; you don't re-decide each strategy's
internal trades. Re-run a sub-strategy via scripts/backtest.py if you need to.

CRITICAL — 1-DAY LAG RULE: allocation weights on day T must use data through T-1. The split
is decided BEFORE day T's returns exist; using same-day data to set same-day weights is
look-ahead bias. Every allocation rule must shift(1) its signal.

HOW TO COMBINE:
1. Load each sub-strategy's equity_curve -> daily returns (pct_change), aligned on date.
2. Define a lagged allocation rule returning daily weights (sum <= 1): fixed split,
   regime-based, or signal-strength.
3. portfolio_ret(T) = Σ weight_i(T) * ret_i(T); compound into an equity curve.
4. Compute total return, CAGR, DAILY-aggregated Sharpe, max drawdown, allocation
   distribution. Save to ./results/portfolio_<name>.json (reproducible).
5. Compare to the best single sub-strategy; if the combo doesn't win on Sharpe/drawdown,
   say so — diversification isn't automatic.

WATCH FOR: same-day-weight look-ahead (always lag); a high-vol strategy dominating an
equal-dollar blend (state your weighting scheme); static-vs-dynamic split trade-offs;
"wins" that trace to a few days (noise).

TOOLS & ANTI-OVERFIT (use these — they exist in scripts/):
- scripts/portfolio.py — correlation matrix + 1-day-lagged inverse-vol/equal-dollar combine +
  greedy low-corr subset search (reports combos_tried) + deflated Sharpe + IS-vs-OOS.
- scripts/walkforward.py — anchored multi-window OOS; report per-window OOS Sharpe + efficiency
  = mean(OOS)/IS (want ~0.6, positive in most windows).
- scripts/robustness.py — parameter-plateau scores so you prefer robust legs over spiky ones.
- SELECT THE FINAL PORTFOLIO BY ROBUSTNESS, NOT MAX IN-SAMPLE SHARPE. Maximizing IS Sharpe
  reliably selects for overfit (lived: IS 2.03 → OOS 0.94; a lower IS 1.80 held OOS 1.45).
- OOS *beating* IS (efficiency ≫ 1) on one window = regime luck, NOT a durable >2. Drop legs
  with < ~30 trades. Deflate the chosen Sharpe for combos_tried. Check correlation STABILITY
  across subperiods. Lead the readout with the OOS/walk-forward number as a RANGE + caveats,
  never the in-sample figure. If nothing clears the bar robustly, say so — do not tune toward it.
- WRITE every portfolio to results/portfolio_<name>.json (the dashboard reads & toggles IS/OOS).

When done, report to the user (and hand the plain-English readout to the Strategy Analyst if
one is on the team): the allocation rule, combined CAGR / max DD / daily Sharpe / allocation
distribution, walk-forward efficiency, deflated Sharpe, and the comparison to the best single
strategy.
```
