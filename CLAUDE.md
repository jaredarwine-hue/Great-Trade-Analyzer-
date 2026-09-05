# AI Trading Floor — Project Guide

This project researches trading-strategy ideas with the AI Trading Floor toolkit
(`.claude/skills/ai-trading-floor/`). It's a **local, historical backtesting sandbox** — no
live trading, no real money, no broker orders. Keep this guide general; the detailed rules and
the specialist work live in the skill's docs and agents (don't duplicate them here).

## How to work
- Use the bundled toolkit in `.claude/skills/ai-trading-floor/scripts/` — don't build a new
  engine; extend `backtest.py` for new strategies (it handles no-look-ahead centrally).
- Run Python via the project venv; all outputs are CWD-relative (`./data`, `./results`,
  `./reports`); never hardcode paths.
- The one file to open is `./reports/dashboard.html` — tell the **user** to open it; don't run
  `open`/`xdg-open` yourself.

## Report results honestly (the principle that always applies)
A backtest's job is to tell the truth, not to look good. Lead with the out-of-sample result,
never present an in-sample or single-window number as "the result," and if there's no real
edge, say so plainly — don't tune toward a target. *How* to test that rigorously (trade counts,
walk-forward, deflated Sharpe, look-ahead, fills) is defined once in `CONVENTIONS.md` and is the
job of the specialist agents (Auditor, Orchestrator) — let them do that work; don't re-derive it.

## The agent floor (optional, advanced)
For deep, multi-step research, spin up the `ai-trading-floor` team — see `SKILL.md` (needs
`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS="1"`). Never shut down team members without the user's
explicit say-so.

## Where the detail lives
- Backtest honesty + overfitting rules → `.claude/skills/ai-trading-floor/CONVENTIONS.md`
- Toolkit, agent roles & coordination → `.claude/skills/ai-trading-floor/SKILL.md`
