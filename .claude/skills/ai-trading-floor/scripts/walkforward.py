#!/usr/bin/env python3
"""Walk-forward validation harness (anchored / expanding) — anti-overfit layer 1.

A single in-sample/out-of-sample split can be lucky. Walk-forward re-tests a strategy across
SEVERAL expanding train->test splits and asks: does the edge keep showing up out-of-sample,
window after window? It reports, per strategy:
  * per-window OOS (test-slice) Sharpe
  * walk-forward EFFICIENCY = mean(OOS Sharpe) / full-IS Sharpe   (1.0 = OOS matches IS;
    <<1 or negative = the IS edge doesn't generalize = overfit)
  * count of POSITIVE OOS windows (out of 5)

WINDOWS (anchored/expanding train, per the team spec):
    train inception -> 2021-12-31 | test 2022-01-01 -> 2022-12-31
    train inception -> 2022-12-31 | test 2023
    train inception -> 2023-12-31 | test 2024
    train inception -> 2024-12-31 | test 2025-01-01 -> 2025-05-22
    train inception -> 2025-05-22 | test 2025-05-23 -> 2026-05-23   (the held-out OOS)

LEAKAGE GUARD (the auditor checks this):
  * --mode fixed (default): the strategy runs with its REGISTERED spec params. There is NO
    parameter selection, so a window's test slice can never influence anything. Train Sharpe
    and test Sharpe are both read from ONE full-history equity curve (the engine's central
    shift(1) means a signal at any bar only sees PAST bars — no boundary leak), then sliced
    by date. This mode has zero selection leakage by construction.
  * --mode select: for EACH window, pick the best grid cell BY TRAIN-SLICE SHARPE ONLY, then
    evaluate THAT cell on the test slice. The selection function (``select_params``) is given
    ONLY the train end date and never receives the test slice; the test Sharpe is computed
    AFTER selection. An assertion records, per window, that selection used train data only.

No look-ahead beyond the engine's shift(1); the only thing this harness adds is the
date-slicing of an already-computed, leak-free equity curve into train/test Sharpes.

Usage:
    python walkforward.py                                   # all 6 single-ticker strategies, fixed mode
    python walkforward.py --mode select                    # per-window best-grid-cell selection
    python walkforward.py --tickers SPY,GLD,TLT --strategies donchian_trend,sma_crossover
    python walkforward.py --rotation                       # also walk-forward the rotation legs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402  (engine = single source of truth)
import run_universe as ru  # noqa: E402  (universe + param grids)

# Expanding windows: (train_end_inclusive, test_start, test_end_inclusive). Train always
# anchors at inception (None = from the start of the data).
WINDOWS = [
    ("2021-12-31", "2022-01-01", "2022-12-31"),
    ("2022-12-31", "2023-01-01", "2023-12-31"),
    ("2023-12-31", "2024-01-01", "2024-12-31"),
    ("2024-12-31", "2025-01-01", "2025-05-22"),
    ("2025-05-22", "2025-05-23", "2026-05-23"),  # the held-out OOS
]

# Efficiency = mean(OOS)/IS is only meaningful when there's a SUBSTANTIVE IS edge to
# generalize. Below this full-IS Sharpe the ratio is unstable AND misleading (a 0.14 IS
# Sharpe with mean OOS 0.9 yields "efficiency 6.7", which over-reads a near-non-edge), so
# we report efficiency as null there and rank such strategies by mean OOS instead.
EFFICIENCY_IS_EPS = 0.30


def _slice(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if start:
        mask &= df["Date"] >= pd.Timestamp(start)
    if end:
        mask &= df["Date"] <= pd.Timestamp(end)
    return df.loc[mask].reset_index(drop=True)


def _curve_sharpe(curve: list[dict], start: str | None, end: str | None) -> tuple[float, int]:
    """Annualized Sharpe of an equity curve restricted to [start, end]. Returns (sharpe, n_days).

    Reuses backtest._annualized_sharpe so the convention matches every other Sharpe in the kit.
    """
    if not curve:
        return 0.0, 0
    c = pd.DataFrame(curve)
    c["date"] = pd.to_datetime(c["date"])
    if start:
        c = c[c["date"] >= pd.Timestamp(start)]
    if end:
        c = c[c["date"] <= pd.Timestamp(end)]
    if len(c) < 2:
        return 0.0, len(c)
    daily = c.set_index("date")["equity"].resample("1D").last().dropna()
    ret = daily.pct_change().dropna()
    return round(bt._annualized_sharpe(ret), 3), int(len(ret))


def full_curve(df: pd.DataFrame, strategy: str, params: dict, capital: float) -> list[dict]:
    """Run the strategy ONCE over the whole (inception->end) frame; return its equity curve.

    The engine's shift(1) means the signal at any bar uses only PAST bars, so slicing this
    one curve into train/test windows introduces NO look-ahead at the boundary.
    """
    long_signal, stop = bt.split_signal_stop(bt.STRATEGIES[strategy].fn(df, **params))
    result = bt.simulate(df, long_signal, capital, stop=stop)
    return result["equity_curve"]


def select_params(df: pd.DataFrame, strategy: str, train_end: str, capital: float) -> tuple[dict, dict]:
    """Pick the best grid cell BY TRAIN-SLICE SHARPE ONLY (no test data touched).

    Returns (chosen_params, leakage_proof). ``leakage_proof`` records the train window the
    selection saw and asserts no test date entered the decision — the auditor can verify the
    selection never reads beyond ``train_end``.
    """
    train_df = _slice(df, None, train_end)
    grid = ru.PARAM_GRIDS[strategy]
    best_params, best_sharpe = None, -1e9
    cells_tried = 0
    for params in ru.iter_param_sets(strategy, use_grid=True):
        # Run on the TRAIN slice only; score on the train slice only.
        curve = full_curve(train_df, strategy, params, capital)
        sh, _ = _curve_sharpe(curve, None, None)
        cells_tried += 1
        if sh > best_sharpe:
            best_sharpe, best_params = sh, params
    # Leakage proof: the only frame selection ever saw ended at train_end.
    assert train_df["Date"].max() <= pd.Timestamp(train_end), "selection saw data past train_end"
    proof = {
        "train_end_seen": str(train_df["Date"].max().date()),
        "train_end_requested": train_end,
        "cells_tried": cells_tried,
        "chosen_train_sharpe": round(best_sharpe, 3),
    }
    return best_params, proof


def walk_one(df: pd.DataFrame, strategy: str, mode: str, capital: float) -> dict:
    """Run all expanding windows for one (df, strategy). Returns the walk-forward record."""
    spec_default = {k: v[0] for k, v in ru.PARAM_GRIDS[strategy].items()}

    windows_out = []
    oos_sharpes = []
    for train_end, test_start, test_end in WINDOWS:
        if mode == "select":
            params, proof = select_params(df, strategy, train_end, capital)
        else:  # fixed: registered spec params, no selection => no leakage possible
            params, proof = spec_default, {"selection": "none (fixed spec params)"}

        # ONE full-history run with the chosen params; slice into train/test Sharpes.
        # (Test Sharpe computed AFTER selection, so selection never depended on it.)
        curve = full_curve(df, strategy, params, capital)
        train_sharpe, _ = _curve_sharpe(curve, None, train_end)
        oos_sharpe, oos_days = _curve_sharpe(curve, test_start, test_end)
        oos_sharpes.append(oos_sharpe)
        windows_out.append({
            "train_end": train_end, "test_start": test_start, "test_end": test_end,
            "params": params, "train_sharpe": train_sharpe,
            "oos_sharpe": oos_sharpe, "oos_days": oos_days,
            "leakage_proof": proof,
        })

    # Full-IS Sharpe (inception -> last train_end) with the spec default, the efficiency base.
    full_is_curve = full_curve(df, strategy, spec_default, capital)
    full_is_sharpe, _ = _curve_sharpe(full_is_curve, None, WINDOWS[-1][0])
    mean_oos = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
    # Efficiency only meaningful when there's a real IS edge to generalize (denominator
    # not ~0); otherwise null so a 0.02 IS Sharpe doesn't produce a wild ratio.
    efficiency = (mean_oos / full_is_sharpe) if full_is_sharpe > EFFICIENCY_IS_EPS else None
    n_positive = int(sum(1 for s in oos_sharpes if s > 0))

    return {
        "strategy": strategy,
        "mode": mode,
        "full_is_sharpe": round(full_is_sharpe, 3),
        "mean_oos_sharpe": round(mean_oos, 3),
        "wf_efficiency": round(efficiency, 3) if efficiency is not None else None,
        "n_positive_oos": n_positive,
        "n_windows": len(WINDOWS),
        "windows": windows_out,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Anchored/expanding walk-forward validation (leak-guarded).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="results/walkforward_summary.json")
    p.add_argument("--tickers", default=",".join(ru.DEFAULT_UNIVERSE))
    p.add_argument("--strategies", default=",".join(ru.PARAM_GRIDS.keys()))
    p.add_argument("--mode", choices=["fixed", "select"], default="fixed",
                   help="fixed = registered spec params (no selection). "
                        "select = per-window best grid cell by TRAIN Sharpe only.")
    p.add_argument("--capital", type=float, default=10000.0)
    p.add_argument("--rotation", action="store_true",
                   help="Also walk-forward the rotation legs (run_rotation per window).")
    return p


def walk_rotation(data_dir: Path, capital: float) -> list[dict]:
    """Walk-forward the rotation legs: re-run run_rotation per test window (fixed params).

    The rotation has fixed spec params (252/21/3), so there's no selection leakage — we just
    evaluate each test window's Sharpe. Uses run_rotation's own windowed run with warmup.
    """
    import run_rotation as rr
    out = []
    for pool_name in ("allasset", "equity", "etf"):
        pool = rr.POOL_PRESETS[pool_name]
        tickers = pool + [rr.CASH]
        if any(not (data_dir / f"{t}.parquet").exists() for t in tickers):
            continue
        close_full, open_full, _ = rr.load_closes_opens(data_dir, tickers)
        oos_sharpes, windows_out = [], []
        for train_end, test_start, test_end in WINDOWS:
            # Fixed spec rotation params (lookback 252, skip 21, top_k 3); run_window keeps
            # 252-bar warmup before test_start so the first month-end momentum is valid.
            res = rr.run_window(close_full, open_full, test_start, test_end,
                                252, 21, 3, capital, pool)
            sh, days = _curve_sharpe(res["equity_curve"], None, None)
            oos_sharpes.append(sh)
            windows_out.append({"train_end": train_end, "test_start": test_start,
                                "test_end": test_end, "oos_sharpe": sh, "oos_days": days})
        # Full-IS rotation Sharpe (inception -> last train_end).
        is_res = rr.run_window(close_full, open_full, None, WINDOWS[-1][0], 252, 21, 3, capital, pool)
        full_is, _ = _curve_sharpe(is_res["equity_curve"], None, None)
        mean_oos = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
        out.append({
            "strategy": f"rotation_{pool_name}", "mode": "fixed",
            "full_is_sharpe": round(full_is, 3), "mean_oos_sharpe": round(mean_oos, 3),
            "wf_efficiency": round(mean_oos / full_is, 3) if full_is > 0 else None,
            "n_positive_oos": int(sum(1 for s in oos_sharpes if s > 0)),
            "n_windows": len(WINDOWS), "windows": windows_out,
        })
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    for s in strategies:
        if s not in ru.PARAM_GRIDS:
            sys.exit(f"ERROR: unknown strategy '{s}'. Known: {sorted(ru.PARAM_GRIDS)}")

    records = []
    for ticker in tickers:
        path = data_dir / f"{ticker}.parquet"
        if not path.exists():
            print(f"  WARNING: {path} missing — skipping {ticker}.")
            continue
        df = pd.read_parquet(path)
        for strategy in strategies:
            try:
                rec = walk_one(df, strategy, args.mode, args.capital)
            except Exception as e:
                print(f"  {ticker} {strategy}: ERROR {e}")
                continue
            rec = {"ticker": ticker, **rec}
            records.append(rec)

    if args.rotation:
        for rec in walk_rotation(data_dir, args.capital):
            records.append({"ticker": "PORT", **rec})

    # Rank: strategies with a SUBSTANTIVE IS edge (efficiency computable) first, ordered by
    # efficiency; then strategies whose IS edge was too small to score, ordered by mean OOS.
    # This avoids inflated ratios (tiny-IS denominators) dominating the top of the table.
    def sort_key(r):
        has_eff = r["wf_efficiency"] is not None
        return (not has_eff, -(r["wf_efficiency"] or 0.0), -r["mean_oos_sharpe"])
    records.sort(key=sort_key)

    out = {
        "note": "Anchored/expanding walk-forward. wf_efficiency = mean(OOS window Sharpe) / "
                "full-IS Sharpe; n_positive_oos out of n_windows. mode 'fixed' = no selection "
                "(zero leakage); mode 'select' = per-window best grid cell chosen on TRAIN "
                "Sharpe only (leakage_proof per window). CAVEAT (auditor): wf_efficiency is a "
                "RATIO, not a quality multiple — efficiency 2-3 usually means a MODEST IS edge "
                "(~0.5 Sharpe) that held up OOS, NOT a 2-3x strong edge. Read it together with "
                "full_is_sharpe + mean_oos_sharpe, never alone.",
        "mode": args.mode,
        "windows": [{"train_end": w[0], "test_start": w[1], "test_end": w[2]} for w in WINDOWS],
        "n_records": len(records),
        "records": records,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # Console summary: best + worst generalizers.
    print(f"Walk-forward ({args.mode} mode): {len(records)} records -> {out_path}\n")
    print(f"  {'TICKER':6} {'STRATEGY':18} {'fullIS':>7} {'meanOOS':>8} {'wf_eff':>7} {'+OOS':>5}")
    for r in records[:15]:
        eff = f"{r['wf_efficiency']:.2f}" if r["wf_efficiency"] is not None else "n/a"
        print(f"  {r['ticker']:6} {r['strategy']:18} {r['full_is_sharpe']:7.2f} "
              f"{r['mean_oos_sharpe']:8.2f} {eff:>7} {r['n_positive_oos']:d}/{r['n_windows']}")
    if len(records) > 15:
        print(f"  ... ({len(records) - 15} more in the JSON)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
