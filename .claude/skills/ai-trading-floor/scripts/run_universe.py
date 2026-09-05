#!/usr/bin/env python3
"""Batch runner: backtest every (ticker x strategy x param-set) over IN-SAMPLE and
OUT-OF-SAMPLE windows, writing reproducible results JSONs.

This is the Data Engineer's reproducible harness for the ai-trading-floor team. It
reuses the bundled engine in ``backtest.py`` as the single source of truth — it imports
the strategy functions and ``simulate`` / ``compute_stats`` directly, so the numbers it
writes are identical to what ``backtest.py`` produces on the same slice.

Anti-overfit design (mandatory per the team charter):
  * IN-SAMPLE  window  (inception -> --is-end, default 2025-05-22) -> ./results/*.json
  * OUT-OF-SAMPLE window (--oos-start -> --oos-end, default 2025-05-23 -> 2026-05-23)
      -> ./results/oos/*.json   (a subfolder so it doesn't clutter the main dashboard,
         which globs ./results/*.json at the top level only)
  * Each strategy runs its DEFAULT param-set PLUS a small robustness GRID of neighbors,
    so a setting only "qualifies" if its neighbors also work. Filenames encode the key
    params so nothing overwrites.

This runner handles the SINGLE-TICKER strategies (Bollinger MR, Donchian, vol-gate trend,
plus the 3 built-ins). The cross-sectional dual-momentum ROTATION leg is a separate runner
(``run_rotation.py``) because it ranks tickers against each other and cannot be a per-ticker
boolean. Strategies that emit a (signal, stop) tuple have their intrabar stop honored via
the engine (Engine Change A).

Result layout (unique per ticker + strategy + params):
    CENTER (default) param-set — drives the dashboard (globs results/*.json, top level only):
        ./results/<TICKER>_<strategy>.json                 (in-sample, clean name)
        ./results/oos/<TICKER>_<strategy>.json             (out-of-sample, clean name)
    GRID neighbors — kept for the auditor's robustness checks; in a grid/ subfolder so they
    don't bloat the dashboard:
        ./results/grid/<TICKER>_<strategy>_<k1><v1>...json        (in-sample)
        ./results/oos/grid/<TICKER>_<strategy>_<k1><v1>...json    (out-of-sample)

Usage:
    python run_universe.py                       # full spec universe x all strategies, IS+OOS
    python run_universe.py --tickers SPY,QQQ     # subset
    python run_universe.py --strategies donchian_trend,bollinger_meanrev
    python run_universe.py --no-grid             # center params only (no robustness grid)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Reuse the bundled engine as the single source of truth.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402

# ---------------------------------------------------------------------------
# Default universe (from the team charter) and window definitions.
# ---------------------------------------------------------------------------

# Spec universe (reports/STRATEGY_SPECS.md (a)): 30 names. CAT/HD/WMT/V were dropped
# from the spec list; DIA/XLI/XLB/SHY were added. SHY is the rotation cash proxy (run by
# run_rotation.py, not here) but is fine to backtest single-name too.
EQUITY_INDEX = ["SPY", "QQQ", "IWM", "DIA"]
SECTORS = ["XLK", "XLE", "XLF", "XLV", "XLP", "XLU", "XLI", "XLB"]
BONDS = ["TLT", "IEF", "SHY"]
COMMODITIES = ["GLD", "SLV", "DBC"]
FX = ["UUP"]
STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "JPM", "XOM", "UNH", "JNJ", "PG", "KO"]
DEFAULT_UNIVERSE = EQUITY_INDEX + SECTORS + BONDS + COMMODITIES + FX + STOCKS

# ---------------------------------------------------------------------------
# Parameter grids for robustness ("a center qualifies only if its neighbors work").
# The FIRST entry of each list is the spec CENTER; the full Cartesian product is the grid.
# Grids mirror reports/STRATEGY_SPECS.md (c) and backtest.py's registered defaults so the
# default (center) run reproduces the registered strategy exactly.
#
# dual_momentum is intentionally EXCLUDED here: the real rotation leg is the cross-sectional
# run_rotation.py. (Add it back as a per-ticker sanity baseline only if explicitly needed.)
# ---------------------------------------------------------------------------

PARAM_GRIDS: dict[str, dict[str, list]] = {
    "sma_crossover": {"fast": [20, 10, 30], "slow": [50, 100, 200]},
    "rsi_reversion": {"rsi_period": [14], "buy_below": [30.0, 25.0, 35.0],
                      "sell_above": [55.0, 50.0, 60.0]},
    "breakout": {"lookback": [20, 40, 55]},
    # Bollinger MR: center 20 / 2.0 / 35  (bb_period x num_std x rsi_filter)
    "bollinger_meanrev": {"bb_period": [20, 15, 25], "num_std": [2.0, 1.5, 2.5],
                          "rsi_filter": [35.0, 30.0, 40.0], "time_stop_bars": [15]},
    # Donchian: center 55 / 20; grid sweeps the immediate neighborhood {40,55,70}x{15,20,25}.
    # (The distant 20/10 "low neighbor" the spec mentions is a sanity comparison, not a
    #  robustness neighbor — run it ad hoc: backtest.py --strategy donchian_trend
    #  --entry-window 20 --exit-window 10.)
    "donchian_trend": {"entry_window": [55, 40, 70], "exit_window": [20, 15, 25],
                       "atr_mult": [2.0]},
    # Vol-gate (unsized v1): center ema_fast 50, vol_cap 0.20 (annualized), vol_window 20
    "vol_gate_trend": {"ema_fast": [50, 30, 75], "ema_slow": [200],
                       "vol_cap": [0.20, 0.15, 0.10], "vol_window": [20, 15, 30],
                       "atr_mult": [3.0]},
    # Anchored-VWAP trend (Shannon): center swing_lookback 10 / trend_sma 50; grid sweeps
    # tighter/looser swing detection {5,10,20} and the trend gate {off, 50, 100}.
    "anchored_vwap_trend": {"swing_lookback": [10, 5, 20], "trend_sma": [50, 0, 100]},
}


def iter_param_sets(strategy: str, use_grid: bool) -> list[dict]:
    """Yield param dicts for a strategy. The first item is always the default set.

    Default = first value of every param list. Grid = full Cartesian product (which
    INCLUDES the default as one of its members). With ``use_grid=False`` only the
    default set is returned.
    """
    grid = PARAM_GRIDS[strategy]
    keys = list(grid.keys())
    default = {k: grid[k][0] for k in keys}
    if not use_grid:
        return [default]

    sets: list[dict] = []
    # Cartesian product, but keep the default FIRST so its file gets the clean name.
    import itertools
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        sets.append(params)
    sets.sort(key=lambda p: (p != default))  # default sorts first
    return sets


# Explicit, collision-free abbreviations for every grid param key. Two keys that
# would naively abbreviate the same (e.g. entry_lookback / exit_lookback both -> "el")
# get distinct tokens here so filenames stay unique and unambiguous.
PARAM_ABBR: dict[str, str] = {
    "fast": "f", "slow": "sl",
    "rsi_period": "rp", "buy_below": "bb", "sell_above": "sa",
    "lookback": "lb",
    "bb_period": "bp", "num_std": "ns", "rsi_filter": "rf", "time_stop_bars": "ts",
    "entry_window": "ent", "exit_window": "exw", "atr_mult": "am",
    "ema_fast": "ef", "ema_slow": "es", "vol_cap": "vc", "vol_window": "vw",
    "sma_filter": "smf", "skip": "sk",
    "swing_lookback": "swl", "trend_sma": "tsm",
}


def param_suffix(params: dict, default: dict) -> str:
    """Build a filename suffix encoding params. Empty string for the default set.

    Encodes EVERY param for grid neighbors so names are unambiguous + unique. Uses the
    explicit ``PARAM_ABBR`` table (no two keys collide). Floats render with the dot
    turned into 'p' and minus into 'm' so the token is filesystem-safe.
    """
    if params == default:
        return ""
    parts = []
    for k, v in params.items():
        token = str(v).replace(".", "p").replace("-", "m")
        abbr = PARAM_ABBR.get(k, "".join(w[0] for w in k.split("_")))
        parts.append(f"{abbr}{token}")
    return "_" + "_".join(parts)


# ---------------------------------------------------------------------------
# Window slicing.
# ---------------------------------------------------------------------------


def slice_window(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    """Return rows with Date in [start, end] inclusive, as a fresh RangeIndex frame.

    Keeps the kit schema contract: Date stays a COLUMN, index is a plain RangeIndex.
    """
    mask = pd.Series(True, index=df.index)
    if start:
        mask &= df["Date"] >= pd.Timestamp(start)
    if end:
        mask &= df["Date"] <= pd.Timestamp(end)
    return df.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# One backtest -> one results JSON.
# ---------------------------------------------------------------------------


def run_one(
    ticker: str,
    strategy: str,
    params: dict,
    df: pd.DataFrame,
    out_path: Path,
    capital: float,
) -> dict:
    """Compute the signal (+stop), simulate, and write the results JSON.

    The bundled engine in ``backtest.py`` is the single source of truth: this splits the
    strategy's (signal, stop) output, feeds the stop into ``simulate`` for honest intrabar
    drawdowns, and reuses ``compute_stats`` unchanged.
    """
    long_signal, stop = bt.split_signal_stop(bt.STRATEGIES[strategy].fn(df, **params))

    result = bt.simulate(df, long_signal, capital, stop=stop)
    stats = bt.compute_stats(result, capital)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ticker": ticker,
        "strategy": strategy,
        "params": params,
        "has_stop": stop is not None,
        "data_file": f"data/{ticker}.parquet",
        "bars": len(df),
        "date_range": [df["Date"].min().isoformat(), df["Date"].max().isoformat()],
        "stats": stats,
        "trades": result["trades"],
        "equity_curve": result["equity_curve"],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return stats


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Backtest every (ticker x strategy x params) over IS + OOS windows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default="data", help="Folder of <TICKER>.parquet files.")
    p.add_argument("--results-dir", default="results", help="IS results folder (OOS -> <dir>/oos).")
    p.add_argument("--tickers", default=",".join(DEFAULT_UNIVERSE),
                   help="Comma-separated tickers (default: full universe).")
    p.add_argument("--strategies", default=",".join(PARAM_GRIDS.keys()),
                   help="Comma-separated strategies (default: all registered).")
    p.add_argument("--is-start", default=None, help="In-sample start (default: inception).")
    p.add_argument("--is-end", default="2025-05-22", help="In-sample end (inclusive).")
    p.add_argument("--oos-start", default="2025-05-23", help="Out-of-sample start.")
    p.add_argument("--oos-end", default="2026-05-23", help="Out-of-sample end (inclusive).")
    p.add_argument("--capital", type=float, default=10000.0, help="Starting capital.")
    p.add_argument("--no-grid", action="store_true", help="Default (center) params only (no robustness grid).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    oos_dir = results_dir / "oos"

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    for s in strategies:
        if s not in PARAM_GRIDS:
            sys.exit(f"ERROR: unknown/ungridded strategy '{s}'. Known: {sorted(PARAM_GRIDS)}")

    use_grid = not args.no_grid

    # Pre-load every ticker once; slice per window.
    frames: dict[str, pd.DataFrame] = {}
    for t in tickers:
        path = data_dir / f"{t}.parquet"
        if not path.exists():
            print(f"  WARNING: {path} missing — skipping {t}.")
            continue
        frames[t] = pd.read_parquet(path)

    windows = [
        ("IS", results_dir, args.is_start, args.is_end),
        ("OOS", oos_dir, args.oos_start, args.oos_end),
    ]

    written = 0
    best: dict[str, tuple] = {"IS": (None, -1e9), "OOS": (None, -1e9)}
    summary_rows: list[dict] = []

    for label, out_dir, w_start, w_end in windows:
        for ticker in tickers:
            if ticker not in frames:
                continue
            df = slice_window(frames[ticker], w_start, w_end)
            if len(df) < 30:
                print(f"  [{label}] {ticker}: only {len(df)} bars — skipping.")
                continue

            for strategy in strategies:
                grid = PARAM_GRIDS[strategy]
                default = {k: grid[k][0] for k in grid}
                for params in iter_param_sets(strategy, use_grid):
                    suffix = param_suffix(params, default)
                    # The default (center) cell goes in the clean window dir (drives the
                    # dashboard, which globs results/*.json at the top level only). Grid
                    # neighbors go in a grid/ subfolder so they stay reproducible for the
                    # auditor's robustness check without bloating the dashboard.
                    is_clean_default = (params == default)
                    dest_dir = out_dir if is_clean_default else (out_dir / "grid")
                    out_path = dest_dir / f"{ticker}_{strategy}{suffix}.json"
                    try:
                        stats = run_one(ticker, strategy, params, df, out_path, args.capital)
                    except Exception as e:  # keep the batch going; report the failure
                        print(f"  [{label}] {ticker} {strategy} {params}: ERROR {e}")
                        continue
                    written += 1

                    if is_clean_default:  # rank only the clean default (center) runs
                        sh = stats["sharpe_daily"]
                        summary_rows.append({
                            "window": label, "ticker": ticker, "strategy": strategy,
                            "sharpe": sh, "cagr": stats["cagr_pct"],
                            "trades": stats["num_trades"], "max_dd": stats["max_drawdown_pct"],
                        })
                        if sh > best[label][1]:
                            best[label] = (f"{ticker} {strategy}", sh)

    # ---- Summary -----------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"BATCH COMPLETE — {written} results written")
    print(f"  IS  -> {results_dir}/*.json")
    print(f"  OOS -> {oos_dir}/*.json")
    print("=" * 72)
    for label in ("IS", "OOS"):
        name, sh = best[label]
        print(f"  Best {label} default Sharpe: {name}  ({sh:.3f})")

    # Top 10 default-param results by IS Sharpe (the cleanest cross-section view).
    is_rows = sorted([r for r in summary_rows if r["window"] == "IS"],
                     key=lambda r: r["sharpe"], reverse=True)
    print("\n  Top 10 IS (default params) by daily Sharpe:")
    print(f"    {'TICKER':6} {'STRATEGY':18} {'SHARPE':>7} {'CAGR%':>8} {'TRADES':>7} {'MAXDD%':>8}")
    for r in is_rows[:10]:
        print(f"    {r['ticker']:6} {r['strategy']:18} {r['sharpe']:7.2f} "
              f"{r['cagr']:8.1f} {r['trades']:7d} {r['max_dd']:8.1f}")

    # Persist the cross-section summary so the orchestrator/auditor can read it.
    summary_path = results_dir / "_batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "windows": {"IS": [args.is_start, args.is_end],
                        "OOS": [args.oos_start, args.oos_end]},
            "tickers": tickers, "strategies": strategies,
            "use_grid": use_grid,
            "results_written": written,
            "best": {k: {"name": v[0], "sharpe": v[1]} for k, v in best.items()},
            "default_rows": summary_rows,
        }, f, indent=2)
    print(f"\n  Cross-section summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
