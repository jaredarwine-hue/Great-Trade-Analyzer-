#!/usr/bin/env python3
"""Portfolio-level walk-forward — the field portfolio.py doesn't emit natively.

``walkforward.py`` walk-forwards SINGLE strategies. ``portfolio.py`` emits a
deflated Sharpe for the SEARCH but no portfolio-level walk-forward. This helper
closes that gap: it rebuilds the SAME 1-day-lagged blend of the chosen legs that
``portfolio.py`` builds (by importing portfolio.py's own combine functions, so the
blend math is byte-for-byte identical to the engine), then slices that ONE combined
daily-return series into the same anchored windows ``walkforward.py`` uses and
reports per-window OOS Sharpe + efficiency.

CRITICAL definition (as promised to the Strategy Analyst):
    wf_efficiency = mean(OOS window Sharpe) / full_is_sharpe
computed on the COMBINED PORTFOLIO daily returns — NOT an average of per-leg
efficiencies. We build the combined series once over full history, then slice.

Efficiency epsilon guard mirrors ``walkforward.py``: if the full-IS combined Sharpe
is below ``EFFICIENCY_IS_EPS`` the ratio is unstable/misleading, so we report it as
null and let ``mean_oos_sharpe`` speak. Honesty caveat (auditor): efficiency is a
RATIO, not a quality multiple — read it together with full_is_sharpe and
mean_oos_sharpe, never alone.

SINGLE SOURCE OF TRUTH (no redeclaration — conventions can't drift):
  * combine math   -> portfolio.load_many / align_returns / build_weights / combine_returns
  * the windows    -> walkforward.WINDOWS
  * the eps guard  -> walkforward.EFFICIENCY_IS_EPS
  * the Sharpe fn  -> backtest._annualized_sharpe (mean/std(ddof=1) * sqrt(252))

GUARD-SEMANTICS NOTE (for the auditor): this helper uses walkforward.py's EXACT guard
``full_is_sharpe > EFFICIENCY_IS_EPS`` (signed, strict, no abs) — a true no-drift single
source of truth. The Orchestrator's original root ``portfolio_walkforward.py`` used
``abs(full_is_sharpe) >= EFFICIENCY_IS_EPS`` instead; per the Auditor's confirmation the
two differ ONLY when full_is_sharpe is negative (or exactly == eps), and the portfolio's
full_is_sharpe is POSITIVE — so this reproduces the Orchestrator's numbers exactly for
Task #6 while matching walkforward.py. For the negative-IS edge case, walkforward.py's
``null`` is the more honest answer anyway (a negative IS edge shouldn't yield a
"meaningful" efficiency ratio). Team-lead decision, 2026-05-25.

No look-ahead: combine_returns uses portfolio.py's ALREADY-lagged weights
(``weights.shift(1)``), and slicing a precomputed combined series into date windows
adds no leak (a day's combined return only ever used data <= that day).

Usage:
    python portfolio_walkforward.py --scheme inverse_vol --vol-window 20 \
        results/A.json results/B.json results/C.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Sibling-import (like walkforward.py): make the skill's scripts/ dir importable so
# `import portfolio` / `walkforward` / `backtest` resolve regardless of the CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402  (the Sharpe convention — never re-derive)
import portfolio as pf  # noqa: E402  (single source of truth for the combine math)
import walkforward as wf  # noqa: E402  (WINDOWS + EFFICIENCY_IS_EPS — never redeclare)


def portfolio_walkforward(paths: list[str], scheme: str, vol_window: int) -> dict:
    """Build the lagged blend of ``paths`` over full history, then slice into windows.

    Returns the block embedded as top-level ``"portfolio_walkforward"`` in the
    portfolio JSON. ``paths`` must be the legs' FULL-HISTORY result files (not the
    OOS-only ones) so the anchored windows have train history to expand over.
    """
    strategies = pf.load_many(paths, strict=True)
    rets = pf.align_returns(strategies)
    weights = pf.build_weights(rets, scheme, vol_window)
    combined = pf.combine_returns(rets, weights)  # ONE combined daily-return series

    full_is_sharpe = bt._annualized_sharpe(combined)

    win_rows: list[dict] = []
    oos_sharpes: list[float] = []
    n_positive = 0
    for train_end, test_start, test_end in wf.WINDOWS:
        sl = combined[(combined.index >= pd.Timestamp(test_start))
                      & (combined.index <= pd.Timestamp(test_end))]
        oos_days = int(len(sl))
        oos_sharpe = bt._annualized_sharpe(sl) if oos_days > 1 else 0.0
        if oos_days > 1:
            oos_sharpes.append(oos_sharpe)
            if oos_sharpe > 0:
                n_positive += 1
        win_rows.append({
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "oos_sharpe": round(oos_sharpe, 3),
            "oos_days": oos_days,
        })

    mean_oos = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
    # Eps guard: walkforward.py's EXACT signed test (see the GUARD-SEMANTICS NOTE in the
    # module docstring) — true single source of truth, reproduces the Orchestrator's
    # positive-IS numbers for Task #6 and gives the more honest null on a negative IS edge.
    if full_is_sharpe > wf.EFFICIENCY_IS_EPS:
        wf_eff = round(mean_oos / full_is_sharpe, 3)
    else:
        wf_eff = None  # IS edge too small (or negative) for the ratio to mean anything

    return {
        "definition": ("wf_efficiency = mean(OOS window Sharpe) / full_is_sharpe, on the "
                       "COMBINED portfolio daily returns (not an avg of per-leg efficiencies). "
                       "Efficiency is a RATIO not a quality multiple — read with full_is_sharpe "
                       "+ mean_oos_sharpe. Single ~1yr OOS windows have few trades; treat "
                       "per-window Sharpe as directional."),
        "scheme": scheme,
        "vol_window": vol_window if scheme in ("inverse_vol", "risk_parity") else None,
        "full_is_sharpe": round(full_is_sharpe, 3),
        "mean_oos_sharpe": round(mean_oos, 3),
        "wf_efficiency": wf_eff,
        "n_positive_oos": int(n_positive),
        "n_windows": int(len([w for w in win_rows if w["oos_days"] > 1])),
        "windows": win_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Portfolio-level walk-forward on combined returns.")
    ap.add_argument("paths", nargs="+", help="leg result JSON files (full-history, not OOS-only)")
    ap.add_argument("--scheme", default="inverse_vol",
                    choices=["equal_dollar", "inverse_vol", "risk_parity"])
    ap.add_argument("--vol-window", type=int, default=20)
    args = ap.parse_args()
    block = portfolio_walkforward(args.paths, args.scheme, args.vol_window)
    print(json.dumps(block, indent=2))


if __name__ == "__main__":
    main()
