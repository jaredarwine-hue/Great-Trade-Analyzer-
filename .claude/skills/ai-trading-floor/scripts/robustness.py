#!/usr/bin/env python3
"""Parameter-robustness scorer — reads the EXISTING results grid, runs NO new backtests.

Anti-overfit layer 2 (per the team's LAYERED validation framework): a strategy whose edge
sits on a single lucky parameter cell is overfit; a strategy whose neighbors ALSO work is
trustworthy. This tool quantifies that for every (ticker, strategy) by reading the parameter
neighborhood already on disk:

    results/<TICKER>_<strategy>.json            <- the CENTER (default/spec) param cell
    results/grid/<TICKER>_<strategy>_<...>.json <- the neighboring param cells

For each (ticker, strategy) it computes, over the neighborhood's IS daily Sharpes:
    center_sharpe   : the spec/default cell's Sharpe
    best_neighbor   : max Sharpe in the neighborhood
    median_neighbor : median Sharpe in the neighborhood
    plateau_score   : median_neighbor / best_neighbor   (the team-lead formula)
    classification  : "plateau" (neighbors hold up), "spiky" (best >> neighbors -> overfit
                      risk), or "dead" (best_neighbor <= 0 -> no edge anywhere, score moot)

plateau_score reads as "how much of the best cell's edge survives at the typical neighbor."
~1.0 = a flat plateau (robust); near 0 = a lone spike (fragile). It is only meaningful when
best_neighbor > 0, so a non-positive neighborhood is classified "dead" and the numeric score
is reported as null (don't divide two non-positive numbers into a misleading ratio).

OUTPUT: results/robustness_summary.json — one record per (ticker, strategy), sorted worst
plateau first, plus a per-strategy roll-up. The orchestrator uses this to PREFER plateau
strategies when selecting the final portfolio. Nothing here re-runs a backtest; it is a pure
read over results/ + results/grid/.

Usage:
    python robustness.py
    python robustness.py --results-dir results --out results/robustness_summary.json
    python robustness.py --spiky-below 0.5     # plateau_score < this => "spiky"
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

# A neighborhood needs at least this many cells (center + neighbors) for the plateau score
# to mean anything. breakout only has 3 cells total; we still score it but flag low-N.
MIN_NEIGHBORHOOD = 3

# A best-neighbor Sharpe at/below this is "no real edge anywhere" — the plateau ratio is
# unstable when the denominator is ~0, so we classify these "dead" instead of emitting a
# wild ratio (e.g. median -0.3 / best 0.02 = -15, which over-reads a non-edge).
DEAD_BEST_EPS = 0.10

# Files that are NOT single-strategy parameter neighborhoods and must be skipped.
SKIP_PREFIXES = ("_", "PORT_", "portfolio_", "robustness")


def _load_sharpe(path: Path) -> tuple[str, str, float] | None:
    """Return (ticker, strategy, sharpe_daily) from a results JSON, or None if unreadable."""
    try:
        with open(path) as f:
            p = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    stats = p.get("stats") or {}
    if "sharpe_daily" not in stats:
        return None
    return p.get("ticker", path.stem), p.get("strategy", "?"), float(stats["sharpe_daily"])


def collect_neighborhoods(results_dir: Path) -> dict[tuple[str, str], dict]:
    """Group center + grid cells into one neighborhood per (ticker, strategy).

    Center cells live at the top level (results/*.json, excluding PORT_* portfolios and
    underscore-prefixed summaries); neighbors live in results/grid/. Both contribute their
    IS Sharpe to the neighborhood Sharpe list.
    """
    neigh: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"center": None, "sharpes": []}
    )

    # Center (default/spec) cells — top level only.
    for path in sorted(results_dir.glob("*.json")):
        if path.name.startswith(SKIP_PREFIXES):
            continue  # skip batch summaries, rotation books, combined portfolios, our own out
        rec = _load_sharpe(path)
        if rec is None:
            continue
        ticker, strategy, sharpe = rec
        key = (ticker, strategy)
        neigh[key]["center"] = sharpe
        neigh[key]["sharpes"].append(sharpe)

    # Grid (neighbor) cells.
    grid_dir = results_dir / "grid"
    if grid_dir.is_dir():
        for path in sorted(grid_dir.glob("*.json")):
            if path.name.startswith(SKIP_PREFIXES):
                continue  # rotation grid is scored separately (single param family)
            rec = _load_sharpe(path)
            if rec is None:
                continue
            ticker, strategy, sharpe = rec
            neigh[(ticker, strategy)]["sharpes"].append(sharpe)

    return neigh


def score_neighborhood(center: float | None, sharpes: list[float], spiky_below: float) -> dict:
    """Compute the plateau score + classification for one neighborhood."""
    n = len(sharpes)
    best = max(sharpes)
    median = statistics.median(sharpes)
    worst = min(sharpes)

    if best <= DEAD_BEST_EPS:
        # No cell clears a meaningful positive Sharpe -> no edge to be robust ABOUT, and the
        # ratio would be unstable (tiny denominator). Don't emit a misleading number.
        classification = "dead"
        plateau = None
    else:
        plateau = median / best  # in (-inf, 1]; ~1 = flat plateau, near 0/neg = lone spike
        if n < MIN_NEIGHBORHOOD:
            classification = "low_n"
        elif plateau < spiky_below:
            classification = "spiky"
        else:
            classification = "plateau"

    return {
        "n_cells": n,
        "center_sharpe": round(center, 3) if center is not None else None,
        "best_neighbor": round(best, 3),
        "median_neighbor": round(median, 3),
        "worst_neighbor": round(worst, 3),
        "plateau_score": round(plateau, 3) if plateau is not None else None,
        "classification": classification,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Score parameter robustness from the existing results grid (no new backtests).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out", default="results/robustness_summary.json")
    p.add_argument("--spiky-below", type=float, default=0.5,
                   help="plateau_score below this => classified 'spiky' (overfit risk).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_dir = Path(args.results_dir)

    neigh = collect_neighborhoods(results_dir)
    if not neigh:
        print(f"No (ticker, strategy) neighborhoods found under {results_dir}/. "
              "Did you run run_universe.py (which writes results/ + results/grid/)?")
        return 1

    records = []
    for (ticker, strategy), data in neigh.items():
        score = score_neighborhood(data["center"], data["sharpes"], args.spiky_below)
        records.append({"ticker": ticker, "strategy": strategy, **score})

    # Sort: worst plateau first among scored ones, dead/low_n at the end.
    def sort_key(r):
        order = {"spiky": 0, "low_n": 1, "plateau": 2, "dead": 3}
        ps = r["plateau_score"] if r["plateau_score"] is not None else 1e9
        return (order.get(r["classification"], 9), ps)

    records.sort(key=sort_key)

    # Per-strategy roll-up: how many tickers are plateau vs spiky vs dead.
    by_strategy: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    plateau_scores_by_strat: dict[str, list] = defaultdict(list)
    for r in records:
        by_strategy[r["strategy"]][r["classification"]] += 1
        if r["plateau_score"] is not None:
            plateau_scores_by_strat[r["strategy"]].append(r["plateau_score"])
    rollup = {}
    for strat, counts in by_strategy.items():
        scores = plateau_scores_by_strat[strat]
        rollup[strat] = {
            "counts": dict(counts),
            "median_plateau_score": round(statistics.median(scores), 3) if scores else None,
            "n_plateau": counts.get("plateau", 0),
            "n_spiky": counts.get("spiky", 0),
        }

    out = {
        "note": "Parameter-robustness from the existing results grid (no new backtests). "
                "plateau_score = median_neighbor_Sharpe / best_neighbor_Sharpe over each "
                "(ticker,strategy) param neighborhood. ~1 = robust plateau; near 0 = lone "
                "spike (overfit risk). 'dead' = no positive-Sharpe cell (score N/A).",
        "spiky_below": args.spiky_below,
        "n_neighborhoods": len(records),
        "per_strategy_rollup": rollup,
        "neighborhoods": records,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # Console summary.
    print(f"Scored {len(records)} (ticker, strategy) neighborhoods -> {out_path}\n")
    print("Per-strategy robustness roll-up (median plateau score; plateau/spiky counts):")
    print(f"  {'STRATEGY':18} {'medPlateau':>11} {'plateau':>8} {'spiky':>6} {'dead':>5}")
    for strat in sorted(rollup):
        r = rollup[strat]
        mp = r["median_plateau_score"]
        print(f"  {strat:18} {(f'{mp:.2f}' if mp is not None else 'n/a'):>11} "
              f"{r['n_plateau']:>8} {r['n_spiky']:>6} {r['counts'].get('dead', 0):>5}")

    spiky = [r for r in records if r["classification"] == "spiky"]
    if spiky:
        print(f"\nMost overfit-prone (spiky, best >> neighbors) — top 10:")
        print(f"  {'TICKER':6} {'STRATEGY':18} {'plateau':>8} {'center':>7} {'best':>6} {'median':>7}")
        for r in spiky[:10]:
            print(f"  {r['ticker']:6} {r['strategy']:18} {r['plateau_score']:8.2f} "
                  f"{(r['center_sharpe'] if r['center_sharpe'] is not None else 0):7.2f} "
                  f"{r['best_neighbor']:6.2f} {r['median_neighbor']:7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
