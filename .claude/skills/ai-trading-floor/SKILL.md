---
name: running-ai-trading-floor
description: "Set up and run a beginner-friendly AI trading floor — a bundled Python toolkit plus an optional team of AI quant agents that fetch data, backtest, audit, and chart trading-strategy ideas for you. Use when the user invokes /ai-trading-floor, says 'ai trading floor', 'set up my own AI quant team', 'I want to test a trading strategy as a beginner', 'beginner trading research', 'help me research a trading idea', 'run a backtest for me', 'test my strategy idea', or asks for a guided tutorial to connect data and run their first backtest. Invoke with the arg 'tutorial' (/ai-trading-floor tutorial) for the step-by-step first-time onboarding walkthrough; invoke bare or with a strategy idea to fetch data, run the bundled backtest pipeline, and (optionally) spin up the agent floor for deeper research."
---

# Running an AI Trading Floor

Your "AI trading floor" turns a plain-English trading idea into tested numbers you can
trust. It has two modes:

1. **The bundled toolkit (primary).** A handful of self-contained Python scripts that
   fetch market data, run a backtest, and build a chart + interactive report — no team
   needed. This is how you answer most questions, fast.
2. **The agent floor (advanced).** A small team of specialized AI agents (designer,
   coder, auditor, chart-builder) for deeper, multi-step research once an idea is worth it.

Everything is local and historical — no live trading, no money at risk. All outputs land
in plain folders in the user's current directory (`./data`, `./results`, `./reports`).

## Argument Routing — READ THIS FIRST

- **`/ai-trading-floor tutorial`** → first-time setup walkthrough. READ `./tutorial.md`
  and follow it step-by-step, ONE action at a time, confirming before advancing. Do NOT
  inline the tutorial here.
- **`/ai-trading-floor` (bare) or `/ai-trading-floor <a strategy idea>`** → use the
  toolkit below. If the user has never set up (no `./data`, no venv), gently suggest
  `/ai-trading-floor tutorial` first.

## The Bundled Toolkit (primary path)

Scripts in `./scripts/`. They all read/write CWD-relative folders and depend only on
pandas, numpy, matplotlib, pyarrow, yfinance, plotly (`./requirements.txt`). Activate the
user's venv first, then:

| Script | What it does |
| --- | --- |
| `scripts/fetch_data.py` | Download free Yahoo Finance bars → `./data/<TICKER>.parquet` |
| `scripts/backtest.py` | Run a built-in strategy on a parquet → `./results/<...>.json`. `simulate()` takes an optional `stop` Series (ATR stops). |
| `scripts/render_chart.py` | Dark-theme candlestick PNG (+ trade markers, MAs, cutoff) |
| `scripts/report.py` | Interactive offline Plotly HTML for ONE strategy |
| `scripts/dashboard.py` | **The headline output.** Aggregates EVERY `results/*.json` + `portfolio_*.json` into ONE offline `reports/dashboard.html` — collapsible asset-class→ticker→strategy tree, Portfolios pinned on top with IS⇄OOS toggle + correlation heatmap, Sharpe+Calmar cards, strategy-specific overlays. Point the user at this one file. |
| `scripts/run_pipeline.py` | **One command** that chains fetch → backtest → refreshes `dashboard.html` (the one file to open) |
| `scripts/run_universe.py` | Batch-backtest a whole universe × strategies → IS `results/` + OOS `results/oos/` (+ a param grid in `results/grid/`) |
| `scripts/run_rotation.py` | Cross-sectional monthly momentum **rotation** runner (a SEPARATE runner — `simulate()` is single-instrument all-in and can't do ranking) |
| `scripts/portfolio.py` | Combine N strategies → correlation matrix + 1-day-lagged inverse-vol/equal-dollar blend + IS/OOS; greedy low-corr subset search with deflated Sharpe. Each `portfolio_<name>.json` now also EMBEDS a `portfolio_walkforward` block (the chosen legs' combined OOS Sharpe + efficiency); pass `--no-walkforward` to skip it. |
| `scripts/portfolio_walkforward.py` | Portfolio-LEVEL walk-forward: rebuilds portfolio.py's lagged blend of the chosen legs, slices the ONE combined return series into walkforward.py's anchored windows → per-window OOS Sharpe + `wf_efficiency = mean(OOS)/full_is_sharpe`. Imports WINDOWS/EFFICIENCY_IS_EPS from walkforward.py + the combine math from portfolio.py (no redeclaration). Auto-embedded by portfolio.py; also a standalone CLI. |
| `scripts/walkforward.py` | Anchored walk-forward of a SINGLE strategy (multiple OOS windows) → per-window OOS Sharpe + efficiency |
| `scripts/robustness.py` | Parameter-plateau scoring from the grid (plateau vs spiky = robust vs overfit) |

**Engine capabilities & limits (know these before designing):** `simulate()` executes a
SIGNED position — `+1` long, `-1` short, `0` flat — that the strategy's own conditions emit (a
plain boolean is the long-only case; there is NO direction flag — the side lives in the
conditions). It is all-in on current equity with an OPTIONAL ATR `stop` Series (side-aware:
long stop below entry, short stop above). It has NO native position sizing (so true
volatility-targeting can't be expressed — use an honest "vol_gate" filter and LABEL it as such,
don't call it vol-targeting) and NO cross-sectional ranking (rotation / relative-strength
strategies need their own runner like `run_rotation.py`).

The fastest answer to "test idea X on ticker Y" is `run_pipeline.py`:

```bash
# Defaults that just work (AAPL, 20/50 SMA crossover, ~3 years):
python scripts/run_pipeline.py

# Map a user's idea to flags:
python scripts/run_pipeline.py --ticker MSFT --strategy sma_crossover --fast 10 --slow 30
python scripts/run_pipeline.py --ticker AAPL --strategy rsi_reversion --buy-below 25 --sell-above 60
python scripts/run_pipeline.py --ticker NVDA --strategy breakout --lookback 20 --start 2021-01-01 --end 2024-01-01
```

Built-in strategies (no look-ahead): **sma_crossover** (`--fast --slow`), **rsi_reversion**
(`--rsi-period --buy-below --sell-above`), **breakout** (`--lookback`), plus the short-side
example **short_breakdown** (`--lookback`, emits `-1`). A strategy goes long OR short purely
through its own conditions (return `+1`/`-1`/`0`) — no direction flag. Indicator math lives in
`scripts/indicators.py`.

**Adding a strategy (this is HOW new strategies get made — keep it clean):** every strategy
lives in **its own file in `scripts/strategies/`** and **auto-registers** — the filename IS the
strategy name, and nothing else needs editing (the engine, the CLI flags, `run_pipeline`,
`run_universe`, and the dashboard all pick it up). To add one — including turning a **YouTube
video / article / screenshot / trade log** into a strategy — the flow is: (1) study the source
and write its exact rules as conditions ("go long when X, short when Y, else flat"); (2) map each
to an indicator in `scripts/indicators.py` (add one there only if none fits); (3) drop a new
`scripts/strategies/<name>.py` that emits a `+1/-1/0` position; (4) backtest it, then
walk-forward / OOS anything you'll trust (per `CONVENTIONS.md`). The full recipe + the
from-a-source workflow is in **`scripts/strategies/README.md`** (and the "HOW TO ADD" block atop
`scripts/backtest.py`). NEVER put strategy logic in `backtest.py` — that file is the engine.

### Mapping a plain-English idea to the toolkit

1. Pick the closest built-in strategy (trend cross → `sma_crossover`; "buy the dip,
   oversold" → `rsi_reversion`; "buy new highs / momentum" → `breakout`).
2. Translate the idea's numbers into flags (which averages, which RSI levels, which
   lookback, which ticker, which date range).
3. Run `run_pipeline.py`, then tell the user to OPEN `./reports/dashboard.html` in their
   browser — that ONE file holds every strategy you've ever run plus any portfolios, and
   refreshes itself after each run (the **user** opens it — never run `open`/`xdg-open`
   yourself).
4. Read the results plainly: trades, win rate, total return, CAGR, daily Sharpe, max
   drawdown. Set honest expectations — a simple template is a starting point, not a
   finished edge.

If the closest built-in doesn't capture the idea, that's the cue to use the agent floor.

## The Agent Floor (advanced mode)

For ideas the built-ins can't express, or deeper studies (parameter sweeps, multi-ticker
runs, careful bias audits), spin up a team. The team name is always **`ai-trading-floor`**.

> **Prerequisite:** the agent floor needs `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS="1"` in
> `.claude/settings.json` (and Bash/Task permissions). The tutorial's **Step ⓪** offers to set
> this up with the user's consent; if a team won't spawn, that flag is the first thing to check
> (it may require restarting the Claude Code session to take effect).

Five spawnable roles — for a first deeper idea you usually only need the first three:

| Agent | Plain-English job | Role file |
| --- | --- | --- |
| **Data Engineer** | Writes the strategy code and runs the backtest. The "doer." | `./agent-data-engineer.md` |
| **Strategy Analyst** | Designs the exact rules. Writes specs, never code. | `./agent-strategy-analyst.md` |
| **Auditor** | Checks results for look-ahead bias and bad fills. Quality gate. | `./agent-auditor.md` |
| **Chart Builder** | Renders price + trade charts so you can see it. | `./agent-chart-builder.md` |
| **Optimizer** | Makes slow backtests fast without changing results. | `./agent-optimizer.md` |

> **The team lead IS the Orchestrator — don't spawn one.** Combining audited strategies into
> a portfolio is the team lead's (main thread's) OWN job, done directly with the bundled
> `portfolio.py` / `walkforward.py` / `robustness.py` scripts. `./agent-orchestrator-analyst.md`
> is the team lead's **playbook** for that step, NOT a teammate to spawn — spawning a separate
> "orchestrator" member just duplicates the role the team lead already plays.

To assemble it:

1. Check `~/.claude/teams/ai-trading-floor/config.json` — if idle members of the right
   role already exist, SendMessage them the task instead of respawning.
2. Create the team (only if missing):
   `TeamCreate: team_name="ai-trading-floor", description="Beginner AI quant research floor"`
3. Spawn the agents you need with the `Agent` tool, `team_name="ai-trading-floor"`,
   `run_in_background=true`, `mode="bypassPermissions"`, in parallel. **For each agent,
   paste the ENTIRE Spawn Prompt from its `./agent-<role>.md` file, then add the task on
   top** — skipping it drops the bug-prevention rules and the agent's results can't be
   trusted.
4. Dispatch via SendMessage: Strategy Analyst writes the spec → Data Engineer codes & runs
   it with the bundled scripts → Auditor reviews → Analyst explains the numbers plainly.
   When it's time to combine audited strategies into a portfolio, the **team lead does that
   itself** (the orchestrator role) with `portfolio.py` — it does NOT spawn a teammate for it.

**Coordination rules learned the hard way:**
- **Disk beats inbox.** The SendMessage channel can silently drop a message. Any deliverable
  another agent depends on (a spec, results, a portfolio) MUST be written to a file on disk
  (e.g. `reports/STRATEGY_SPECS.md`, `results/*.json`) AND the path messaged. Never let a
  handoff exist only in chat.
- **Audit gate before combining.** The Data Engineer's batch is NOT trusted until the Auditor
  PASSes the strategy code + results. The team lead (acting as orchestrator) must not build a
  portfolio on unaudited (or stale/prototype) results. Verify the on-disk implementation matches
  the spec before running a batch — prototype-vs-spec drift wasted whole batches here.
- **Files churn during reruns.** A rerun rewrites `results/`; combos that reference deleted
  files are stale. Re-run the combiner on current files before quoting any number.

## Key Conventions (the rules that keep backtests honest)

Every agent and every new strategy must follow `./CONVENTIONS.md`. The essentials:

- **No look-ahead** — shift signals by one bar; a signal seen at a bar's close is acted on
  next bar. (The bundled engine does this centrally.)
- **Realistic fills** — retroactively discovered signals fill at the next bar's open; for
  breakouts never fill below the open (`max(level, Open)`).
- **Daily-aggregated Sharpe**, not per-trade (per-trade is inflated by ~sqrt(trades/day)).
- **Stops on the losing side** — assert `stop < entry` for longs, `stop > entry` for shorts.
- **Same-bar entry+target on intraday bars is the killer** — if >20% of wins are same-bar,
  flag the result as unreliable.
- **Intraday timestamps are naive Eastern Time** (9:30–16:00 ET), not UTC.

**Avoiding overfitting (the discipline that decides whether a result is real)** — see the
"Avoiding Overfitting" section of `./CONVENTIONS.md`. Essentials:
- **A high Sharpe on few trades is noise.** Single-name "Sharpe > 2" with <~30 trades is a
  buy-and-hold artifact; report the trade count + a confidence interval, never the bare number.
- **Maximizing in-sample Sharpe SELECTS FOR overfit.** Pick the final strategy/portfolio by
  *robustness*, not by best in-sample metric.
- **One out-of-sample window only tests one regime.** Use walk-forward (multiple windows);
  OOS *wildly beating* IS is a regime artifact, not durable edge (efficiency should be ~0.6).
- **Parameter robustness:** a setting only counts if its neighbors also work (broad plateau,
  not a lone spike).
- **Deflate for search:** when you pick the best of N combos, haircut the Sharpe for the
  selection (deflated Sharpe); a tidy >2 from a big search is usually luck.

## Important Rules for Running the Floor

- The orchestrator (main thread) NEVER writes code — all implementation goes to agents (or
  to the bundled scripts you invoke directly).
- **The team lead IS the orchestrator** — it plays the portfolio role itself (combining
  audited strategies via `portfolio.py` / `walkforward.py`), and never spawns a separate
  "Orchestrator Analyst" teammate. See `./agent-orchestrator-analyst.md` (the playbook).
- The Strategy Analyst NEVER writes code — only specs, analysis, and plain-English readouts.
- The Auditor reviews EVERY intraday strategy before its results are trusted.
- Every strategy MUST be a rerunnable `.py` (use/extend the bundled scripts), not throwaway
  inline code.
- Each agent does ONLY its assigned task — no refactoring, no "while I'm here" extras.
- Never shut down team members without the user's explicit say-so.

## Where Things Live (all relative to the user's working directory)

- **Toolkit:** `./scripts/` (fetch_data, backtest [the engine], render_chart, report,
  **dashboard**, run_pipeline, run_universe, run_rotation, portfolio, portfolio_walkforward,
  walkforward, robustness, indicators, README)
- **Strategies:** `./scripts/strategies/` — one file per strategy, auto-registered (filename =
  strategy name). This is the ONLY place strategies are added; see `scripts/strategies/README.md`.
- **Data:** `./data/<TICKER>.parquet` (daily), `./data/intraday/<TICKER>_<interval>.parquet`
- **Backtest results:** `./results/<ticker>_<strategy>.json` (in-sample), `./results/oos/` (out-of-sample),
  `./results/grid/` (param sweeps), `./results/portfolio_*.json` (combined portfolios)
- **The dashboard:** `./reports/dashboard.html` (the one file to open; portfolios persist via
  `./results/.portfolio_cache/` so they survive mid-rerun regens)
- **Charts + reports:** `./reports/`
- **Data schema:** `./scripts/DATA_CONTRACT.md`
- **Backtest conventions:** `./CONVENTIONS.md`
- **Chart standard:** `./CHART_RENDERER.md`
- **Dependencies:** `./requirements.txt`
