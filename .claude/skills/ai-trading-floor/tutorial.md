# AI Trading Floor — First-Time Setup Tutorial

You are guiding a total beginner — someone who may not know Python — through setting up
their AI trading floor for the very first time. Be warm, patient, and encouraging. Work
as a narrated CHECKLIST: show the whole list up front, then do ONE step at a time,
explaining WHAT you just did and WHY it matters in plain language, and CONFIRM with the
user before moving to the next step. The user never has to write code — you run everything
for them.

**Make it click-to-start.** Whenever you ask the beginner to decide something (how to begin,
which data source, which sample strategy), offer it as **clickable multiple-choice options**
(the AskUserQuestion tool) with a recommended default first — not an open-ended question. A
beginner should be able to get all the way to their first result by just picking options. If
the venv/deps and `./data` don't exist yet, do the whole Step ① setup for them automatically.

Two important conventions:
- **`<skill>`** below means this skill's folder. Resolve it to the real absolute path of
  the directory containing this file (e.g. `.../.claude/skills/ai-trading-floor`) and use
  that when you run the bundled scripts.
- **Everything else is CWD-relative.** You'll create a workspace folder in the user's
  current directory and run from there, so data and reports land in `./data`, `./results`,
  `./reports` — never anywhere global.

---

## Show the checklist first

Open by telling the user, warmly and plainly, what's about to happen:

> Welcome! Let's get your personal AI trading floor running. A quick one-time setup
> question, then three steps:
>
> ☐ ⓪ **Permissions** — let me run the toolkit for you without asking every time (optional).
> ☐ ① **Set up your toolbox** — install Python's number-crunching libraries (one time).
> ☐ ② **Connect market data** — download real historical prices to your computer.
> ☐ ③ **Run your first test** — backtest a simple strategy and open the report.
>
> No coding from you — I'll do each step and explain it as we go. Nothing here touches
> real money; it's all historical data and simulation. Ready? Let's start with permissions.

Wait for their go-ahead, then do Step ⓪.

---

## Step ⓪ — Permissions (optional, but makes the tutorial smooth)

By default Claude Code asks the user to approve every command (each `python`, `pip`, `ls`).
For a beginner that's a wall of confusing prompts. Offer to pre-approve the toolkit's
permissions ONCE — but only with the user's explicit yes. **Never write settings silently.**

**1. Make sure `.claude/` exists AND the skill lives in `.claude/skills/`.** First ensure the
working dir has a `.claude/` before writing anything: `mkdir -p .claude`. Then make sure the
skill itself is installed under `.claude/skills/`. Depending on how the user obtained the kit,
the `ai-trading-floor/` folder (the one holding this `tutorial.md`, `SKILL.md`, `scripts/`,
`requirements.txt`, etc.) is often sitting **loose in the working directory** instead of under
`.claude/skills/`. If you find it there, MOVE it so `<skill>` and the `CLAUDE.md` paths line up:

```bash
# Only if ./ai-trading-floor exists at the top level and ./.claude/skills/ai-trading-floor does not:
mkdir -p .claude/skills && mv ai-trading-floor .claude/skills/
```

After this, `<skill>` resolves to `./.claude/skills/ai-trading-floor`, which matches the paths
the project `CLAUDE.md` references. (If the skill is instead installed at the USER level —
`~/.claude/skills/ai-trading-floor` — leave it there; just keep a project-local `.claude/` for
this folder's own `settings.json` / `CLAUDE.md`, and resolve `<skill>` to that user-level path.)
Then read `./.claude/settings.json` (and `./.claude/settings.local.json`); if `permissions.allow`
already covers Bash + the file tools, say "you're already set" and skip to Step ①.

**2. Ask — as a clickable choice (AskUserQuestion).** Plainly, e.g.:
> "Want me to enable the recommended permissions so I can run the toolkit for you without
> asking each time? This lets me run shell commands (Python, pip), read/write files in this
> folder, and (optionally) spin up the AI agent floor later. You can remove them anytime by
> editing `.claude/settings.json`."
> Options: **Yes, enable them (recommended)** · **No, ask me each time**

**3a. If YES** — create or MERGE this block into `./.claude/settings.json` (union the
`permissions.allow` list with anything already there; set the env flag; preserve other keys —
do NOT clobber an existing file). Then show the user exactly what you added.
```jsonc
{
  "env": { "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1" },
  "permissions": {
    "allow": ["Bash(*)", "Write", "Edit", "Read", "Glob", "Grep",
              "WebSearch", "WebFetch", "Task(*)", "Skill(*)"]
  }
}
```
Be honest in one line about what it grants: `"Bash(*)"` lets me run shell commands without a
prompt each time, and `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS` turns on the optional multi-agent
"agent floor" (an experimental feature) for later deep research. It's all local; nothing is sent
anywhere. (If they'd rather keep it personal/uncommitted, write to `.claude/settings.local.json`
instead — same shape.)

**3b. If NO** — that's fine; continue and just approve each prompt as it appears. Don't nag.

**4. Offer a project guide (CLAUDE.md).** If there is **no** `./CLAUDE.md` and no
`./.claude/CLAUDE.md` already, offer (clickable yes/no) to drop in a trading-floor project
guide so future sessions follow the honesty/anti-overfit rules automatically. On yes, copy
`<skill>/templates/CLAUDE.md` to `./CLAUDE.md`. **If a CLAUDE.md already exists, do NOT
overwrite it** — just mention they can paste the template's rules in if they want. (Why it
matters: CLAUDE.md is always-loaded, so it keeps "report honest OOS numbers, never tune toward
a target" in force even outside this tutorial.)

Mark ⓪ done (☑) and move to Step ①. (Note: the env-var change may only take effect on the
next Claude Code session — mention that if they plan to use the agent floor.)

---

## Step ① — Set up your toolbox (environment bootstrap)

Goal: a working Python 3 + a private "toolbox" (virtual environment) with the kit's
libraries installed, in a fresh workspace folder in the user's current directory.

**1. Check Python 3 exists.**
```bash
python3 --version
```
- If it prints a version (3.9+), say: "Great — Python is installed." Continue.
- **If it's missing**, STOP and give OS-specific guidance, then ask them to re-run:
  - macOS: "Install with Homebrew: `brew install python3` — or download from python.org."
  - Windows: "Download the installer from python.org and check 'Add Python to PATH'."
  - Linux (Debian/Ubuntu): "`sudo apt install python3 python3-venv python3-pip`."
  Don't proceed until `python3 --version` works.

**2. Create a workspace folder + a virtual environment inside it.**
```bash
mkdir -p my-trading-lab && python3 -m venv my-trading-lab/venv
```
Narrate it: "I made a folder called `my-trading-lab` to keep your work tidy, and a
`venv` inside it — that's a private toolbox so these libraries don't clash with anything
else on your computer."

**3. Install the kit's libraries into that toolbox.**
```bash
my-trading-lab/venv/bin/python -m pip install --quiet -r <skill>/requirements.txt
```
Narrate WHY each piece matters, briefly: "I just installed pandas + numpy (for crunching
price tables), matplotlib + plotly (for the charts and the interactive report), pyarrow
(to read the data files fast), and yfinance (to download free market data). That's your
whole toolbox — a one-time setup."

**4. Confirm it worked.**
```bash
my-trading-lab/venv/bin/python -c "import pandas, numpy, matplotlib, plotly, yfinance; print('Toolbox ready.')"
```
Report a friendly pass. From here on, run the kit's Python as
`my-trading-lab/venv/bin/python` from inside the user's current directory (so outputs land
in `./data`, `./results`, `./reports`). Tell the user step ① is done (☑) and confirm
before step ②.

---

## Step ② — Connect your market data

Goal: get real price history onto the machine in the kit's format. **Ask the user which
source they want** before doing anything:

> Two ways to get data:
> **A (recommended): Free Yahoo Finance** — no account, no API key, great for daily data.
> **B: Your own provider** (Polygon, Alpaca, Tiingo, etc.) — if you already pay for data
> and want full intraday history. A bit more setup.

### Option A — free Yahoo Finance (default)

Fetch a well-known ticker (start with one — it proves the pipe works):
```bash
my-trading-lab/venv/bin/python <skill>/scripts/fetch_data.py --tickers AAPL --period 3y
```
Then narrate the printed output: "That downloaded ~3 years of Apple's daily prices and
saved them to `./data/AAPL.parquet`. Each row is one trading day — the open, high, low,
close, and how many shares traded. This is exactly the format your tools read." Confirm
the file exists and move on:
```bash
ls -lh data/AAPL.parquet
```
Tell them they can add any ticker later by re-running with a different `--tickers`. Mark
② done (☑), confirm, go to step ③.

### Option B — bring your own provider (adapter flow)

1. Ask the user to put their API key in a `.env` file in their working directory (keep it
   out of git). Tell them the exact line, e.g. `POLYGON_API_KEY=...`. Do NOT print/echo it.
2. Read `<skill>/scripts/DATA_CONTRACT.md` together — that's the exact target shape every
   data file must have (`Date, Open, High, Low, Close, Volume`; naive Eastern Time for
   intraday; plain row index; float prices).
3. Study that provider's bar-download API (pagination, adjusted vs unadjusted prices,
   timestamp timezone). For a real port, hand this to a **Data Engineer** on the floor
   (see `SKILL.md`) and have it write a small porter modeled on `scripts/fetch_data.py`
   that writes the SAME schema into `./data`.
4. **Verify a small sample FIRST** — pull ONE ticker, ~1 month, and run the verification
   snippet from `DATA_CONTRACT.md` to confirm columns, dtypes, naive timestamps, and the
   plain index all match. Only port the rest once the sample passes. Then continue to ③.

---

## Step ③ — Run your first test

Goal: prove the whole pipeline works and show the user a real result + interactive report.

Run the one-command pipeline on the data you just fetched:
```bash
my-trading-lab/venv/bin/python <skill>/scripts/run_pipeline.py --ticker AAPL --strategy sma_crossover
```
Narrate what it does: "This runs three steps for you — it makes sure the data's there,
backtests a simple '20/50 moving-average crossover' (buy when the short-term average rises
above the long-term one, sell when it crosses back below), and builds you a report."

Then:
1. Read the printed results table plainly: number of trades, win rate, total return, CAGR,
   daily Sharpe, max drawdown. Explain each in one friendly line.
2. Tell the user to **open the dashboard in their browser**: "I saved an interactive
   dashboard at `./reports/dashboard.html` — double-click it (it works fully offline). It
   holds every strategy you run (and any portfolios), with the price candles + buy/sell
   markers, the account-value curve, and Sharpe/Calmar stat cards. Click any strategy on the
   left to see it." The **user** opens it; you never run `open`/`xdg-open` yourself. Every
   future run drops into this SAME one file.
3. Set honest expectations: "This is a deliberately simple strategy — its only job is to
   prove your data and tools work end-to-end. Now the fun part: describe any idea and we'll
   test it."

Mark ③ done (☑) — all three boxes checked.

---

## Now test your OWN idea (the loop)

Tell the user they can now describe a strategy idea in plain English, and show how it maps
to the toolkit. Built-in strategies:

- **Trend following** ("buy when the fast average crosses above the slow one"):
  `--strategy sma_crossover --fast 20 --slow 50`
- **Buy the dip / oversold** ("buy when RSI is low, sell when it recovers"):
  `--strategy rsi_reversion --buy-below 30 --sell-above 55`
- **Breakout / new highs** ("buy when price breaks above its recent high"):
  `--strategy breakout --lookback 20`

Take their idea, pick the closest one, translate their numbers into flags, and run:
```bash
my-trading-lab/venv/bin/python <skill>/scripts/run_pipeline.py --ticker <THEIR_TICKER> --strategy <NAME> <FLAGS>
```
Then point them back to the SAME `./reports/dashboard.html` — the new strategy just appears
in it (grouped by asset class; click to see its chart). Everything stays in `./data`,
`./results`, `./reports` in their current folder. No new file to hunt for each time.

If their idea doesn't fit any built-in, tell them that's the moment to bring in the full
AI floor — switch to the bare-`/ai-trading-floor` playbook in `SKILL.md` and spin up the
agent team for deeper, custom research. Want to try an idea now?
