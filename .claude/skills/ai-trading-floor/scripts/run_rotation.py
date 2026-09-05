#!/usr/bin/env python3
"""Cross-sectional dual-momentum ROTATION runner (spec strategy 2 — the diversifier).

A per-ticker boolean cannot express this strategy: it RANKS assets against each other and
against a cash proxy, so it needs to see the whole pool at once. This runner builds the
monthly target-weight matrix across the pool, fills at the next trading day's open, marks
to market daily, and emits the SAME results.json shape the rest of the toolkit consumes
(equity_curve + trades + stats), reusing ``backtest.compute_stats`` unchanged so the daily-
aggregated Sharpe convention is identical to every other leg.

RULES (reports/STRATEGY_SPECS.md strategy 2):
  * Pool (14): SPY QQQ IWM XLK XLE XLF XLV XLU TLT IEF GLD SLV DBC UUP.  Cash proxy = SHY.
  * Rebalance: the LAST trading day of each calendar month (decision bar).
  * Momentum (12-1 skip): mom = Close[t-skip] / Close[t-lookback] - 1   (default skip=21,
    lookback=252 -> "12 months ago to 1 month ago").
  * Absolute gate: an asset is eligible only if its mom > SHY's mom on the decision bar.
  * Rank: hold the TOP-K eligible by mom, EQUAL-WEIGHT (1/K). Fewer than K eligible -> fill
    the empty slots with SHY. Zero eligible -> 100% SHY.
  * Fill: at the NEXT trading day's OPEN after the decision bar (no look-ahead — the
    decision uses only closes <= the decision bar; execution is the following open).
  * Catastrophe stop: a HELD sleeve whose Close falls >20% below its rotation-in price
    (entry*0.80) is rotated to SHY at the next open (stop < entry holds).
  * Hold until the next monthly rebalance; otherwise no intra-month trading (besides the
    catastrophe stop).

NO LOOK-AHEAD: momentum on the decision bar uses Close[t-skip]/Close[t-lookback] (strictly
past); the resulting weights are applied at the next open. Daily MTM uses that day's close.

Usage:
    python run_rotation.py                                  # full cross-asset pool, IS+OOS -> results/
    python run_rotation.py --pool equity --label rotation_equity   # all-equity subset leg
    python run_rotation.py --pool etf    --label rotation_etf      # de-correlator ETF subset leg
    python run_rotation.py --grid                           # default pool + param-robustness grid
    python run_rotation.py --label rotation_l189_s0_k2 --lookback 189 --skip 0 --top-k 2  # a grid cell
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402  (reuse compute_stats — single source of truth)

POOL = ["SPY", "QQQ", "IWM", "XLK", "XLE", "XLF", "XLV", "XLU",
        "TLT", "IEF", "GLD", "SLV", "DBC", "UUP"]
CASH = "SHY"
CATASTROPHE_DD = 0.20  # 20% below rotation-in price -> sleeve to cash

# Named pool presets so we can emit a few rotation legs (the spec's "sensible universe
# subsets") with one flag each. The spec default ("allasset") is the full cross-asset pool.
# Subsets give the orchestrator MORE candidate legs with different correlation profiles:
# all-equity is high-beta (less diversifying), all-ETF is the de-correlator-heavy mix.
POOL_PRESETS: dict[str, list[str]] = {
    "allasset": POOL,
    "equity": ["SPY", "QQQ", "IWM", "XLK", "XLE", "XLF", "XLV", "XLU"],
    "etf": ["SPY", "QQQ", "IWM", "TLT", "IEF", "GLD", "SLV", "DBC", "UUP"],
}


def load_closes_opens(data_dir: Path, tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """Load aligned Close and Open matrices (index = shared trading dates, cols = tickers)."""
    closes, opens = {}, {}
    for t in tickers:
        df = pd.read_parquet(data_dir / f"{t}.parquet")
        df = df.set_index(pd.to_datetime(df["Date"]))
        closes[t] = df["Close"]
        opens[t] = df["Open"]
    close_df = pd.DataFrame(closes).sort_index()
    open_df = pd.DataFrame(opens).sort_index()
    # Keep only dates where every ticker has data (clean cross-sectional alignment).
    close_df = close_df.dropna(how="any")
    open_df = open_df.reindex(close_df.index)
    return close_df, open_df, close_df.index


def month_end_decision_dates(dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Last trading day of each calendar month present in ``dates``."""
    s = pd.Series(dates, index=dates)
    return list(s.groupby([dates.year, dates.month]).last().values)


def compute_target_weights(
    close_df: pd.DataFrame,
    decision_date: pd.Timestamp,
    lookback: int,
    skip: int,
    top_k: int,
    pool: list[str],
) -> dict[str, float]:
    """Top-K equal-weight target weights decided on ``decision_date`` (uses past closes only)."""
    idx = close_df.index.get_loc(decision_date)
    if idx < lookback:
        return {CASH: 1.0}  # not enough history yet -> sit in cash

    # 12-1 skip momentum: Close[t-skip] / Close[t-lookback] - 1, for each pool member + cash.
    px_skip = close_df.iloc[idx - skip]
    px_look = close_df.iloc[idx - lookback]
    mom = (px_skip / px_look - 1.0)

    cash_mom = mom[CASH]
    eligible = {t: mom[t] for t in pool if mom[t] > cash_mom and not np.isnan(mom[t])}
    ranked = sorted(eligible, key=lambda t: eligible[t], reverse=True)[:top_k]

    weights: dict[str, float] = {}
    w = 1.0 / top_k
    for t in ranked:
        weights[t] = w
    # Fill empty slots (and the all-empty case) with cash.
    cash_slots = top_k - len(ranked)
    if cash_slots > 0:
        weights[CASH] = weights.get(CASH, 0.0) + cash_slots * w
    return weights


def simulate_rotation(
    close_df: pd.DataFrame,
    open_df: pd.DataFrame,
    lookback: int,
    skip: int,
    top_k: int,
    capital: float,
    pool: list[str],
) -> dict:
    """Run the monthly rotation, daily MTM, with a 20% catastrophe stop. Returns engine-shaped dict."""
    dates = close_df.index
    decision_dates = month_end_decision_dates(dates)
    # Map each decision date to its EXECUTION date = next trading day's open.
    exec_for_decision: dict[pd.Timestamp, pd.Timestamp] = {}
    for d in decision_dates:
        loc = dates.get_loc(d)
        if loc + 1 < len(dates):
            exec_for_decision[dates[loc + 1]] = d  # keyed by exec date

    equity = float(capital)
    # Holdings: ticker -> {"shares": float, "entry_px": float} ; cash tracked separately.
    holdings: dict[str, dict] = {}
    cash = float(capital)  # uninvested $ (the SHY sleeve is held as the SHY ticker too)

    equity_curve: list[dict] = []
    trades: list[dict] = []
    n_rebalances = 0

    def portfolio_value(i: int) -> float:
        val = cash
        for t, h in holdings.items():
            val += h["shares"] * float(close_df.iloc[i][t])
        return val

    def _trade_record(t, entry_px, exit_px, entry_date, exit_date, reason) -> dict:
        ret = exit_px / entry_px - 1.0
        # R relative to the strategy's only hard risk band — the 20% catastrophe stop.
        return {
            "ticker": t,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": round(entry_px, 4),
            "exit_price": round(exit_px, 4),
            "exit_reason": reason,
            "return_pct": round(ret * 100.0, 4),
            "r_multiple": round(ret / CATASTROPHE_DD, 4),
        }

    def liquidate_at_open(t: str, i: int, reason: str):
        nonlocal cash
        h = holdings.pop(t)
        exit_px = float(open_df.iloc[i][t])
        cash += h["shares"] * exit_px
        trades.append(_trade_record(
            t, h["entry_px"], exit_px, h["entry_date"], dates[i].isoformat(), reason))

    def buy_at_open(t: str, dollars: float, i: int):
        nonlocal cash
        entry_px = float(open_df.iloc[i][t])
        shares = dollars / entry_px if entry_px > 0 else 0.0
        holdings[t] = {"shares": shares, "entry_px": entry_px, "entry_date": dates[i].isoformat()}
        cash -= dollars

    for i in range(len(dates)):
        today = dates[i]

        # --- Catastrophe stop check at TODAY's open: any held sleeve >20% below entry close? ---
        # Decision uses YESTERDAY's close vs entry; executed at today's open.
        if i > 0:
            for t in list(holdings.keys()):
                if t == CASH:
                    continue
                entry_px = holdings[t]["entry_px"]
                if float(close_df.iloc[i - 1][t]) <= entry_px * (1.0 - CATASTROPHE_DD):
                    liquidate_at_open(t, i, "catastrophe_stop")

        # --- Monthly rebalance executes at today's open if today is an exec date ---
        if today in exec_for_decision:
            n_rebalances += 1
            target = compute_target_weights(close_df, exec_for_decision[today], lookback, skip, top_k, pool)
            # Liquidate ALL sleeves to cash at today's open, then re-buy to exact target
            # weights. (Simplest provably-correct path; turnover cost is out of scope here.)
            for t in list(holdings.keys()):
                liquidate_at_open(t, i, "rebalance")
            total = cash  # everything is now cash; deploy to the target weights
            for t, w in target.items():
                if w <= 0:
                    continue
                buy_at_open(t, total * w, i)

        # --- Daily mark-to-market at today's close ---
        equity = portfolio_value(i)
        equity_curve.append({"date": today.isoformat(), "equity": round(float(equity), 4)})

    # Close any open sleeves at the final close.
    last = len(dates) - 1
    for t in list(holdings.keys()):
        h = holdings.pop(t)
        exit_px = float(close_df.iloc[last][t])
        cash += h["shares"] * exit_px
        trades.append(_trade_record(
            t, h["entry_px"], exit_px, h["entry_date"], dates[last].isoformat(), "eod"))
    equity_curve[-1]["equity"] = round(float(cash), 4)

    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "final_equity": float(cash),
        "ambiguous_stop_bars": 0,
        "inverted_stop_skips": 0,
        "n_rebalances": n_rebalances,
    }


def run_window(close_full, open_full, start, end, lookback, skip, top_k, capital, pool):
    """Slice to [start,end] (keeping `lookback` warmup before start) and run the rotation."""
    idx = close_full.index
    end_ts = pd.Timestamp(end) if end else idx[-1]
    start_ts = pd.Timestamp(start) if start else idx[0]
    # Keep warmup bars before the window so month-1 momentum can compute.
    start_loc = idx.searchsorted(start_ts)
    warmup_loc = max(0, start_loc - lookback - skip - 5)
    sl = slice(idx[warmup_loc], end_ts)
    close_w = close_full.loc[sl]
    open_w = open_full.loc[sl]
    res = simulate_rotation(close_w, open_w, lookback, skip, top_k, capital, pool)
    # Trim the equity curve to the actual reporting window (drop warmup region from stats).
    curve = pd.DataFrame(res["equity_curve"])
    curve["date"] = pd.to_datetime(curve["date"])
    mask = curve["date"] >= start_ts
    # Rebase equity so the window starts at `capital` (warmup shouldn't inflate returns).
    in_win = curve[mask].reset_index(drop=True)
    if len(in_win):
        base = in_win["equity"].iloc[0]
        scale = capital / base if base > 0 else 1.0
        in_win["equity"] = in_win["equity"] * scale
        res = dict(res)
        res["equity_curve"] = [
            {"date": r["date"].isoformat(), "equity": round(float(r["equity"]), 4)}
            for _, r in in_win.iterrows()
        ]
        res["final_equity"] = float(in_win["equity"].iloc[-1])
        # Trades inside the window only.
        res["trades"] = [t for t in res["trades"]
                         if pd.Timestamp(t["entry_date"]) >= start_ts]
    return res


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cross-sectional monthly dual-momentum rotation (spec strategy 2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--label", default="rotation_dualmom",
                   help="Result file stem (e.g. 'rotation_dualmom' -> results/PORT_rotation_dualmom.json).")
    p.add_argument("--lookback", type=int, default=252)
    p.add_argument("--skip", type=int, default=21)
    p.add_argument("--top-k", type=int, dest="top_k", default=3)
    p.add_argument("--capital", type=float, default=10000.0)
    p.add_argument("--is-start", default=None)
    p.add_argument("--is-end", default="2025-05-22")
    p.add_argument("--oos-start", default="2025-05-23")
    p.add_argument("--oos-end", default="2026-05-23")
    p.add_argument("--pool", default="allasset", choices=sorted(POOL_PRESETS),
                   help="Rotation pool preset: allasset (full cross-asset), equity (high-beta), etf (de-correlator mix).")
    p.add_argument("--grid", action="store_true",
                   help="Also write the param-robustness grid (lookback{189,252} x skip{0,21} x top_k{2,3,4}) into grid/.")
    return p


def write_result(res, label, lookback, skip, top_k, pool, start, end, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = bt.compute_stats(res, res["equity_curve"][0]["equity"] if res["equity_curve"] else 10000.0)
    stats["n_rebalances"] = res.get("n_rebalances", 0)
    payload = {
        "ticker": "PORT",  # a portfolio-level book, not a single ticker
        # strategy == the leg label so (ticker, strategy) uniquely identifies each rotation
        # variant AND matches the filename stem PORT_<label> — the orchestrator keys IS<->OOS
        # twins on (ticker, strategy), so the variants must NOT collide on one shared name.
        "strategy": label,
        "label": label,
        "params": {"lookback": lookback, "skip": skip, "top_k": top_k,
                   "pool": pool, "cash": CASH, "catastrophe_dd": CATASTROPHE_DD},
        "note": "cross-sectional monthly top-K dual-momentum rotation; weights decided on last "
                "trading day of month, filled next open; daily MTM; 20% catastrophe stop.",
        "bars": len(res["equity_curve"]),
        "date_range": [res["equity_curve"][0]["date"], res["equity_curve"][-1]["date"]] if res["equity_curve"] else [],
        "stats": stats,
        "trades": res["trades"],
        "equity_curve": res["equity_curve"],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return stats


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    pool = POOL_PRESETS[args.pool]

    tickers = pool + [CASH]
    missing = [t for t in tickers if not (data_dir / f"{t}.parquet").exists()]
    if missing:
        sys.exit(f"ERROR: missing parquets for rotation pool: {missing}")

    close_full, open_full, _ = load_closes_opens(data_dir, tickers)

    def run_and_write(label, lb, sk, tk, default_cell: bool):
        is_res = run_window(close_full, open_full, args.is_start, args.is_end, lb, sk, tk, args.capital, pool)
        oos_res = run_window(close_full, open_full, args.oos_start, args.oos_end, lb, sk, tk, args.capital, pool)
        is_dir = results_dir if default_cell else results_dir / "grid"
        oos_dir = (results_dir / "oos") if default_cell else (results_dir / "oos" / "grid")
        is_stats = write_result(is_res, label, lb, sk, tk, pool, args.is_start, args.is_end, is_dir / f"PORT_{label}.json")
        oos_stats = write_result(oos_res, label, lb, sk, tk, pool, args.oos_start, args.oos_end, oos_dir / f"PORT_{label}.json")
        return is_stats, oos_stats

    # Default (center) cell -> top level (drives dashboard + portfolio combiner).
    is_stats, oos_stats = run_and_write(args.label, args.lookback, args.skip, args.top_k, default_cell=True)
    print(f"=== ROTATION {args.label}  pool={args.pool}({len(pool)})  lookback={args.lookback} skip={args.skip} top_k={args.top_k} ===")
    print(f"IS : Sharpe {is_stats['sharpe_daily']:+.2f}  CAGR {is_stats['cagr_pct']:+.1f}%  "
          f"maxDD {is_stats['max_drawdown_pct']:.1f}%  rebalances {is_stats['n_rebalances']}  trades {is_stats['num_trades']}")
    print(f"OOS: Sharpe {oos_stats['sharpe_daily']:+.2f}  CAGR {oos_stats['cagr_pct']:+.1f}%  "
          f"maxDD {oos_stats['max_drawdown_pct']:.1f}%  rebalances {oos_stats['n_rebalances']}  trades {oos_stats['num_trades']}")

    # Robustness grid -> grid/ subfolders (kept out of the dashboard glob).
    if args.grid:
        print("\n--- robustness grid (IS Sharpe) ---")
        for lb in (189, 252):
            for sk in (0, 21):
                for tk in (2, 3, 4):
                    if (lb, sk, tk) == (args.lookback, args.skip, args.top_k):
                        continue
                    lab = f"rotation_l{lb}_s{sk}_k{tk}"
                    gis, _ = run_and_write(lab, lb, sk, tk, default_cell=False)
                    print(f"  l{lb} s{sk} k{tk}: IS Sharpe {gis['sharpe_daily']:+.2f}  "
                          f"CAGR {gis['cagr_pct']:+.1f}%  rebal {gis['n_rebalances']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
