#!/usr/bin/env python3
"""Cross-STRATEGY correlation of daily-return books — the diversification map for the combine.

The Sharpe>2 goal is a diversification problem: portfolio Sharpe only scales with N when the
legs are UNCORRELATED. The auditor's structural finding was that per-ticker momentum gates are
just long beta. This tool builds each strategy's combined equal-weight daily-return BOOK (its
equity curve across all the tickers it ran on), then reports the pairwise correlation matrix,
each strategy's average |corr| to the others (lower = better diversifier), and a simple
cluster grouping — so the orchestrator can weight ACROSS clusters instead of stacking
correlated legs.

Reads ONLY existing results/*.json (no new backtests). Single-ticker strategies are pooled
into one equal-weight book each; each PORT_rotation_* book is its own leg. Emits
results/strategy_correlation.json (matrix + avg|corr| ranking + clusters), reusing the kit's
daily-return convention (resample to calendar day, pct_change).

Usage:
    python strategy_correlation.py
    python strategy_correlation.py --window is      # results/   (default)
    python strategy_correlation.py --window oos     # results/oos/
    python strategy_correlation.py --cluster-thresh 0.5
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

SINGLE_STRATS = ["sma_crossover", "rsi_reversion", "breakout",
                 "bollinger_meanrev", "donchian_trend", "vol_gate_trend"]


def _daily_ret(path: str) -> pd.Series | None:
    """Daily-return series from a results JSON's equity curve (kit convention)."""
    try:
        ec = json.load(open(path)).get("equity_curve", [])
    except (OSError, json.JSONDecodeError):
        return None
    if len(ec) < 3:
        return None
    s = pd.Series([x["equity"] for x in ec],
                  index=pd.to_datetime([x["date"] for x in ec]))
    return s.resample("1D").last().dropna().pct_change().dropna()


def build_books(results_dir: Path) -> dict[str, pd.Series]:
    """One daily-return book per strategy: single-ticker strats pooled equal-weight; each
    rotation pool variant its own book."""
    books: dict[str, pd.Series] = {}
    for strat in SINGLE_STRATS:
        rets = []
        for f in glob.glob(str(results_dir / f"*_{strat}.json")):
            if os.path.basename(f).startswith("PORT_"):
                continue
            r = _daily_ret(f)
            if r is not None:
                rets.append(r)
        if rets:
            # Equal-weight book = mean of per-ticker daily returns aligned on dates.
            books[strat] = pd.concat(rets, axis=1).mean(axis=1)
    for f in glob.glob(str(results_dir / "PORT_rotation_*.json")):
        name = os.path.basename(f)[len("PORT_"):-len(".json")]
        r = _daily_ret(f)
        if r is not None:
            books[name] = r
    return books


def cluster(corr: pd.DataFrame, thresh: float) -> list[list[str]]:
    """Greedy clustering: strategies join a cluster if corr to its seed >= thresh."""
    remaining = list(corr.columns)
    clusters: list[list[str]] = []
    while remaining:
        seed = remaining.pop(0)
        grp = [seed]
        for o in list(remaining):
            if corr.loc[seed, o] >= thresh:
                grp.append(o)
                remaining.remove(o)
        clusters.append(grp)
    return clusters


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cross-strategy correlation map of daily-return books (no new backtests).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results-dir", default="results")
    p.add_argument("--window", choices=["is", "oos"], default="is",
                   help="is = results/ ; oos = results/oos/")
    p.add_argument("--out", default=None, help="Default: results/strategy_correlation[_oos].json")
    p.add_argument("--cluster-thresh", type=float, default=0.5,
                   help="corr >= this groups strategies into the same cluster.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base = Path(args.results_dir)
    rdir = base / "oos" if args.window == "oos" else base
    out_path = Path(args.out) if args.out else base / (
        "strategy_correlation_oos.json" if args.window == "oos" else "strategy_correlation.json")

    books = build_books(rdir)
    if len(books) < 2:
        print(f"Need >=2 strategy books under {rdir}/ — found {len(books)}.")
        return 1

    df = pd.DataFrame(books).dropna()
    corr = df.corr()

    avg_abs = {c: round(float(np.mean([abs(corr.loc[c, o]) for o in corr.columns if o != c])), 3)
               for c in corr.columns}
    ranked = sorted(avg_abs, key=lambda c: avg_abs[c])
    clusters = cluster(corr, args.cluster_thresh)

    out = {
        "note": "Cross-strategy daily-return correlation. Each single-ticker strategy = one "
                "equal-weight book across its tickers; each PORT_rotation_* = its own book. "
                "avg_abs_corr: lower = better diversifier. clusters: corr>=thresh grouped — "
                "weight ACROSS clusters, don't stack within one.",
        "window": args.window,
        "cluster_thresh": args.cluster_thresh,
        "shared_days": int(len(df)),
        "strategies": list(corr.columns),
        "avg_abs_corr": avg_abs,
        "avg_abs_corr_ranked": [{"strategy": c, "avg_abs_corr": avg_abs[c]} for c in ranked],
        "clusters": clusters,
        "correlation_matrix": {r: {c: round(float(corr.loc[r, c]), 3) for c in corr.columns}
                               for r in corr.columns},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Cross-strategy correlation ({args.window}, {len(df)} shared days) -> {out_path}\n")
    print("avg |corr| to others (lower = better diversifier):")
    for c in ranked:
        print(f"  {c:20} {avg_abs[c]:.2f}")
    print(f"\nclusters (corr >= {args.cluster_thresh}):")
    for i, grp in enumerate(clusters, 1):
        print(f"  cluster {i}: {', '.join(grp)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
