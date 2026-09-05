#!/usr/bin/env python3
"""Self-contained mini backtest engine for the AI Trading Floor toolkit.

Reads ONE parquet of daily bars (kit schema: Date, Open, High, Low, Close, Volume),
runs a long-only strategy chosen via --strategy, and writes a results.json containing
the trade list, the per-bar equity curve, and summary stats. No repository imports;
the only local dependency is the sibling ``indicators.py`` for indicator math.

NO LOOK-AHEAD: every strategy turns prices into a per-bar POSITION Series — +1 (long),
-1 (short), or 0 (flat) — and the engine shifts it by one bar (``position.shift(1)``)
before trading. A position decided at the close of bar i is acted on at the OPEN of bar
i+1 — you can never trade on information you couldn't have had yet. Long vs short is
decided ENTIRELY by the strategy's own conditions; there is no engine/CLI direction
switch. (A plain boolean Series still works: it's the long-only case, True -> +1.)

Stats use the real conventions:
  * Sharpe is computed on DAILY-AGGREGATED returns of the equity curve (not per-trade),
    then annualized by sqrt(252). Per-trade Sharpe is inflated by ~sqrt(trades/day) and
    is the wrong number; this engine does it right.
  * "R" per trade = trade return / the strategy's risk-per-trade. For these long-only
    templates risk is approximated as the entry-to-stop distance where a stop exists,
    otherwise as the realized loss unit; see ``_compute_trade`` for the exact rule.

Built-in strategies (all long-only, all shift(1)):
  * sma_crossover     — long while fast SMA > slow SMA.            (--fast --slow)
  * rsi_reversion     — buy when RSI dips below --buy-below,
                        sell when RSI rises above --sell-above.    (--rsi-period --buy-below --sell-above)
  * breakout          — long while close > prior --lookback-bar high (excl. current bar). (--lookback)
  * bollinger_meanrev — buy dip < lower band inside an uptrend (Close>SMA200, RSI<35);
                        exit at mid band or 15-bar time stop; entry_low-0.5*ATR stop.
  * donchian_trend    — Turtle System 2: enter > prior 55-bar high, exit < prior 20-bar
                        low, 2*ATR initial stop.                  (--entry-window --exit-window --atr-mult)
  * vol_gate_trend    — vol-target ENTRY-GATE approx (unsized v1): EMA50>EMA200 & Close>EMA50
                        & ann.vol<0.20; 3*ATR stop.              (--ema-fast --ema-slow --vol-cap)
  * dual_momentum     — per-ticker 12-1 momentum sanity baseline (the real rotation leg is
                        the cross-sectional run_rotation.py).    (--mom-lookback --mom-skip)
  * anchored_vwap_trend — long while Close > VWAP anchored to the most recent confirmed
                        swing-pivot low (re-anchors on each new low; optional SMA trend
                        gate).                                   (--swing-lookback --trend-sma)

Strategies may return a bare boolean Series OR a (long_signal, stop_price_series) tuple;
the engine shifts BOTH by one bar (no look-ahead) and runs an intrabar stop if present.

Usage:
    python backtest.py --data data/AAPL.parquet --strategy sma_crossover --fast 20 --slow 50
    python backtest.py --data data/AAPL.parquet --strategy rsi_reversion --buy-below 30 --sell-above 55
    python backtest.py --data data/AAPL.parquet --strategy breakout --lookback 20 --out results/AAPL_breakout.json

------------------------------------------------------------------------------------
HOW TO ADD YOUR OWN STRATEGY (the extensible pattern)
------------------------------------------------------------------------------------
A strategy is ONE function that takes the price DataFrame + its params and returns a
per-bar POSITION Series aligned to df.index: +1 where its conditions say go LONG, -1
where they say go SHORT, 0 to stay flat. The side is decided ENTIRELY by the conditions
— there is NO engine/CLI direction switch and NO separate "short" version of a strategy.
(Returning a plain boolean Series still works: it's the long-only case, True -> +1.) Do
NOT shift it yourself — the engine shifts once, centrally, so no strategy can leak future
data.

1. Long-only is just a boolean (every classic template does this):

       def my_long(df: pd.DataFrame, threshold: float = 1.0) -> pd.Series:
           ema_fast = ind.ema(df["Close"], 10)
           ema_slow = ind.ema(df["Close"], 30)
           return (ema_fast > ema_slow * (1 + threshold / 100))   # bool -> long/flat

2. Long AND short from the same conditions — emit +1 / -1 / 0:

       def my_long_short(df: pd.DataFrame) -> pd.Series:
           fast, slow = ind.sma(df["Close"], 20), ind.sma(df["Close"], 50)
           pos = np.zeros(len(df), dtype=int)
           pos[(fast > slow).to_numpy()] = 1     # long while in an uptrend
           pos[(fast < slow).to_numpy()] = -1    # short while in a downtrend
           return pd.Series(pos, index=df.index)

3. Save it as its OWN file ``scripts/strategies/<name>.py`` with the function PLUS a
   ``SPEC`` (its defaults + CLI flags). It auto-registers as strategy ``<name>`` — NO edit
   to this engine, NO central list to touch:

       # scripts/strategies/my_long_short.py
       import numpy as np, pandas as pd
       import indicators as ind
       from ._spec import Param, StrategySpec

       def my_long_short(df, fast=20, slow=50):
           f, s = ind.sma(df["Close"], int(fast)), ind.sma(df["Close"], int(slow))
           pos = np.zeros(len(df), dtype=int)
           pos[(f > s).to_numpy()] = 1     # long in an uptrend
           pos[(f < s).to_numpy()] = -1    # short in a downtrend
           return pd.Series(pos, index=df.index)

       SPEC = StrategySpec(my_long_short, {"fast": 20, "slow": 50}, params=[
           Param("--fast", "fast", int, "my_long_short: fast SMA"),
           Param("--slow", "slow", int, "my_long_short: slow SMA"),
       ])

That's it — ``backtest.py --strategy my_long_short`` (and run_pipeline / run_universe / the
dashboard) all pick it up automatically. The engine executes whatever position the conditions
emit (next-open fills, long OR short mark-to-market, side-aware stops). Full guide +
the "from a video/article/screenshot" workflow: scripts/strategies/README.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# Import the sibling indicators module whether run as a script or imported.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import indicators as ind  # noqa: E402

SCHEMA_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
TRADING_DAYS = 252

# Strategy templates live one-per-file in ``strategies/`` and auto-register — see
# strategies/__init__.py. STRATEGIES maps name -> StrategySpec; ALL_PARAMS is the set of
# CLI flags they expose. Add a strategy by dropping a <name>.py in that folder (no edit here).
from strategies import ALL_PARAMS, STRATEGIES, Param, SignalFn, StrategySpec  # noqa: E402,F401


def split_signal_stop(
    fn_output: pd.Series | tuple[pd.Series, pd.Series],
) -> tuple[pd.Series, pd.Series | None]:
    """Normalize a strategy's return into (long_signal, stop_or_None).

    A strategy may return either a bare boolean Series (no stop) or a
    ``(long_signal, stop)`` tuple. This lets stop-based strategies (Bollinger,
    Donchian, vol-gate) coexist with the signal-only built-ins through one code path.
    """
    if isinstance(fn_output, tuple):
        signal, stop = fn_output
        return signal, stop
    return fn_output, None


# ---------------------------------------------------------------------------
# The engine: turn a POSITION Series (+1 long / -1 short / 0 flat) into trades,
# equity, and stats. A boolean Series is accepted as the long-only case (True -> +1).
# ---------------------------------------------------------------------------


def simulate(
    df: pd.DataFrame,
    position: pd.Series,
    capital: float,
    stop: pd.Series | None = None,
) -> dict:
    """Run an all-in simulation with next-bar-open execution from a POSITION series.

    ``position`` is the strategy's per-bar intent and is the ONLY thing that decides
    the side: ``+1`` = long, ``-1`` = short, ``0`` = flat. A boolean Series is accepted
    as the long-only case (True -> +1, False -> 0), so every classic long template keeps
    working unchanged. Direction is never chosen by the engine or the CLI — a strategy's
    own conditions emit the sign (``if long-conditions: +1; elif short-conditions: -1;
    else 0``).

    NO LOOK-AHEAD: the series is shifted one bar here (the single, central guard) — the
    value at bar i means "hold this position during bar i+1", filled at bar i+1's OPEN.
    When the target side changes the engine exits at the next open and, if the new side
    is nonzero, opens it the SAME bar (a +1 -> -1 flip closes the long and opens the
    short at one open).

    Mark-to-market: a long marks at ``shares*price``; a short at
    ``entry_equity + shares*(entry_px - price)`` (it gains as price falls).

    INTRABAR STOP (optional): ``stop`` is a per-bar Series of stop-PRICE levels
    (NaN = none), read ``shift(1)`` like the position; the level captured on the entry
    bar is held FIXED. A LONG stop sits BELOW entry and fires when ``Low <= stop`` (fill
    ``min(stop, Open)``); a SHORT stop sits ABOVE entry and fires when ``High >= stop``
    (fill ``max(stop, Open)``) — a gap-through never fills better than the level. A stop
    on the wrong side of entry is skipped (``inverted_stop_skips``), never a fake win. If
    a stop and a side change want the same bar, the stop (worse) wins, counted in
    ``ambiguous_stop_bars``. With ``stop=None`` and a boolean signal this is byte-identical
    to the original long-only engine.
    """
    # Signed target position, shifted one bar. Booleans collapse to the long-only case
    # (True -> +1, False -> 0); any nonzero magnitude is normalized to all-in (sign).
    shifted = pd.to_numeric(position.shift(1), errors="coerce").fillna(0.0).to_numpy()
    target_pos = np.sign(shifted).astype(int)
    if stop is not None:
        stop_arr = stop.shift(1).to_numpy(dtype=float)  # stop level governing each bar
    else:
        stop_arr = np.full(len(df), np.nan)
    open_px = df["Open"].to_numpy(dtype=float)
    high_px = df["High"].to_numpy(dtype=float)
    low_px = df["Low"].to_numpy(dtype=float)
    close_px = df["Close"].to_numpy(dtype=float)
    dates = pd.to_datetime(df["Date"])

    equity = float(capital)
    shares = 0.0
    side = 0                       # current side: +1 long, -1 short, 0 flat
    entry_i = 0
    entry_px = 0.0
    entry_equity = float(capital)  # equity at entry (baseline for short mark-to-market)
    stop_at_entry = np.nan         # fixed initial stop level for the live trade

    trades: list[dict] = []
    equity_curve: list[dict] = []
    ambiguous_stop_bars = 0
    inverted_stop_skips = 0

    for i in range(len(df)):
        # --- Exit path first: a stop, or a change in the target side, closes a trade ---
        if side != 0:
            want_side_change = target_pos[i] != side
            if np.isnan(stop_at_entry):
                stop_hit = False
            elif side > 0:
                stop_hit = low_px[i] <= stop_at_entry    # long stop is BELOW entry
            else:
                stop_hit = high_px[i] >= stop_at_entry   # short stop is ABOVE entry

            if stop_hit and want_side_change:
                ambiguous_stop_bars += 1  # both want this bar -> take the WORSE (stop)
            if stop_hit:
                # Gap-through fills at the open, never better than the stop level.
                exit_px = min(stop_at_entry, open_px[i]) if side > 0 else max(stop_at_entry, open_px[i])
                equity = shares * exit_px if side > 0 else entry_equity + shares * (entry_px - exit_px)
                trades.append(_compute_trade(
                    df, dates, entry_i, i, entry_px, exit_px, side,
                    stop_at_entry=stop_at_entry, exit_reason="stop"))
                side, shares, stop_at_entry = 0, 0.0, np.nan
            elif want_side_change:
                exit_px = open_px[i]
                equity = shares * exit_px if side > 0 else entry_equity + shares * (entry_px - exit_px)
                trades.append(_compute_trade(
                    df, dates, entry_i, i, entry_px, exit_px, side,
                    stop_at_entry=stop_at_entry, exit_reason="signal"))
                side, shares, stop_at_entry = 0, 0.0, np.nan

        # --- Entry path: open the target side at this bar's open if flat and target != 0 ---
        if side == 0 and target_pos[i] != 0:
            new_side = int(target_pos[i])
            entry_px = open_px[i]
            candidate_stop = stop_arr[i]
            if not np.isnan(candidate_stop):
                wrong_side = candidate_stop >= entry_px if new_side > 0 else candidate_stop <= entry_px
                if wrong_side:
                    # Stop on the wrong side of entry: skip it, don't fake a win.
                    candidate_stop = np.nan
                    inverted_stop_skips += 1
            stop_at_entry = candidate_stop
            shares = equity / entry_px if entry_px > 0 else 0.0
            entry_equity = equity
            side, entry_i = new_side, i

        # Mark-to-market the equity at this bar's close.
        if side == 0:
            bar_equity = equity
        elif side > 0:
            bar_equity = shares * close_px[i]
        else:
            bar_equity = entry_equity + shares * (entry_px - close_px[i])
        equity_curve.append(
            {"date": dates.iloc[i].isoformat(), "equity": round(float(bar_equity), 4)}
        )

    # Close any open position at the final bar's close.
    if side != 0:
        i = len(df) - 1
        exit_px = close_px[i]
        equity = shares * exit_px if side > 0 else entry_equity + shares * (entry_px - exit_px)
        trades.append(_compute_trade(
            df, dates, entry_i, i, entry_px, exit_px, side,
            stop_at_entry=stop_at_entry, exit_reason="eod"))
        equity_curve[-1]["equity"] = round(float(equity), 4)

    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "final_equity": float(equity),
        "ambiguous_stop_bars": ambiguous_stop_bars,
        "inverted_stop_skips": inverted_stop_skips,
    }


def _compute_trade(
    df: pd.DataFrame,
    dates: pd.Series,
    entry_i: int,
    exit_i: int,
    entry_px: float,
    exit_px: float,
    side: int = 1,
    stop_at_entry: float = float("nan"),
    exit_reason: str = "signal",
) -> dict:
    """Build one trade record including an R-multiple and the exit reason.

    ``side`` is +1 for a long trade or -1 for a short. PnL per share is ``exit - entry``
    for a long and ``entry - exit`` for a short. When the trade had a REAL initial stop
    (``stop_at_entry`` finite), risk = the entry-to-stop distance on the correct side
    (asserted ``stop < entry`` for longs, ``stop > entry`` for shorts). Without a stop we
    fall back to the entry-bar's True-Range as a stop-free proxy for "one unit of adverse
    move" (the original template behavior).

    ``exit_reason`` is one of: "stop" (intrabar stop fired), "signal" (the target side
    changed, next-open exit), or "eod" (open position closed at the final bar). Recording
    it makes the inverted-stop tell visible: a high win-rate dominated by "stop" exits is
    the CONVENTIONS §4 red flag.
    """
    is_short = side < 0
    pnl_per_share = (entry_px - exit_px) if is_short else (exit_px - entry_px)
    if not np.isnan(stop_at_entry):
        if is_short:
            assert stop_at_entry > entry_px, (
                f"inverted short stop: stop {stop_at_entry} <= entry {entry_px}")
            risk = stop_at_entry - entry_px
        else:
            assert stop_at_entry < entry_px, (
                f"inverted long stop: stop {stop_at_entry} >= entry {entry_px}")
            risk = entry_px - stop_at_entry
    else:
        high = float(df["High"].iloc[entry_i])
        low = float(df["Low"].iloc[entry_i])
        prev_close = float(df["Close"].iloc[entry_i - 1]) if entry_i > 0 else low
        true_range = max(high - low, abs(high - prev_close), abs(low - prev_close))
        risk = true_range if true_range > 0 else max(entry_px * 0.01, 1e-9)
    return {
        "entry_index": int(entry_i),
        "exit_index": int(exit_i),
        "entry_date": dates.iloc[entry_i].isoformat(),
        "exit_date": dates.iloc[exit_i].isoformat(),
        "entry_price": round(entry_px, 4),
        "exit_price": round(exit_px, 4),
        "stop_at_entry": round(float(stop_at_entry), 4) if not np.isnan(stop_at_entry) else None,
        "exit_reason": exit_reason,
        "bars_held": int(exit_i - entry_i),
        "return_pct": round(
            ((entry_px - exit_px) / entry_px if is_short else (exit_px / entry_px - 1.0)) * 100.0, 4
        ) if entry_px else 0.0,
        "r_multiple": round(pnl_per_share / risk, 4),
    }


def _annualized_sharpe(daily_ret: pd.Series) -> float:
    """Annualized Sharpe from a daily-return series (mean/std * sqrt(252)).

    Returns 0.0 when there are too few points or zero variance — the single, shared
    definition used by the headline Sharpe AND the per-year slices so they're consistent.
    """
    if len(daily_ret) > 1 and daily_ret.std(ddof=1) > 0:
        return float(daily_ret.mean() / daily_ret.std(ddof=1) * np.sqrt(TRADING_DAYS))
    return 0.0


def compute_stats(result: dict, capital: float) -> dict:
    """Compute summary stats. Sharpe uses DAILY-aggregated equity returns."""
    trades = result["trades"]
    curve = pd.DataFrame(result["equity_curve"])
    curve["date"] = pd.to_datetime(curve["date"])

    num_trades = len(trades)
    rets = np.array([t["return_pct"] / 100.0 for t in trades], dtype=float)
    wins = int((rets > 0).sum()) if num_trades else 0
    win_rate = (wins / num_trades * 100.0) if num_trades else 0.0

    final_equity = result["final_equity"]
    total_return = (final_equity / capital - 1.0) * 100.0

    # CAGR over the actual calendar span of the equity curve.
    span_days = max((curve["date"].iloc[-1] - curve["date"].iloc[0]).days, 1)
    years = span_days / 365.25
    cagr = ((final_equity / capital) ** (1.0 / years) - 1.0) * 100.0 if years > 0 else 0.0

    # DAILY-aggregated Sharpe: aggregate equity to one value per calendar day, take the
    # day-over-day percent change, annualize. This is the correct convention.
    daily_equity = curve.set_index("date")["equity"].resample("1D").last().dropna()
    daily_ret = daily_equity.pct_change().dropna()
    sharpe = _annualized_sharpe(daily_ret)

    # Per-YEAR Sharpe (subperiod stability): annualized Sharpe of each calendar year's
    # daily returns. Years with < 60 trading days (e.g. a partial first/last year) are
    # still reported but should be read as low-confidence. This is the same daily-return
    # convention as the headline Sharpe, just sliced by calendar year.
    sharpe_by_year: dict[str, float] = {}
    for year, grp in daily_ret.groupby(daily_ret.index.year):
        sharpe_by_year[str(int(year))] = round(_annualized_sharpe(grp), 3)

    # Max drawdown from the equity curve peak.
    eq = curve["equity"].to_numpy(dtype=float)
    running_max = np.maximum.accumulate(eq)
    drawdowns = (eq - running_max) / running_max
    max_dd = float(drawdowns.min() * 100.0) if len(drawdowns) else 0.0

    avg_r = float(np.mean([t["r_multiple"] for t in trades])) if num_trades else 0.0

    # Exit-reason breakdown (visible inverted-stop tell: high WR + mostly stop exits).
    stop_exits = sum(1 for t in trades if t.get("exit_reason") == "stop")
    signal_exits = sum(1 for t in trades if t.get("exit_reason") == "signal")
    ambiguous = int(result.get("ambiguous_stop_bars", 0))
    ambiguous_pct = (ambiguous / num_trades * 100.0) if num_trades else 0.0

    return {
        "num_trades": num_trades,
        "win_rate_pct": round(win_rate, 2),
        "total_return_pct": round(total_return, 2),
        "cagr_pct": round(cagr, 2),
        "sharpe_daily": round(sharpe, 3),
        "sharpe_by_year": sharpe_by_year,
        "max_drawdown_pct": round(max_dd, 2),
        "avg_r_multiple": round(avg_r, 3),
        "final_equity": round(final_equity, 2),
        "starting_capital": round(capital, 2),
        "stop_exits": stop_exits,
        "signal_exits": signal_exits,
        "ambiguous_stop_bars": ambiguous,
        "ambiguous_stop_pct": round(ambiguous_pct, 2),
        "inverted_stop_skips": int(result.get("inverted_stop_skips", 0)),
    }


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a built-in long-only strategy on one parquet of daily bars.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="Path to a <TICKER>.parquet file.")
    parser.add_argument(
        "--strategy", default="sma_crossover", choices=sorted(STRATEGIES),
        help="Which built-in strategy to run.",
    )
    parser.add_argument("--capital", type=float, default=10000.0, help="Starting capital.")
    parser.add_argument("--out", help="Output results.json path (default: ./results/<ticker>_<strategy>.json).")
    # Strategy params: one CLI flag per param, generated from each strategy module in
    # scripts/strategies/ (deduped across strategies that legitimately share a flag — e.g.
    # --lookback for breakout + short_breakdown, --atr-mult for donchian + vol_gate). Add a
    # strategy file and its flags appear here automatically; nothing to edit in this parser.
    for p in ALL_PARAMS:
        parser.add_argument(p.flag, dest=p.dest, type=p.type, default=None, help=p.help)
    return parser


def resolve_params(spec: StrategySpec, args: argparse.Namespace) -> dict:
    """Overlay any CLI-provided params on top of the strategy's defaults."""
    params = dict(spec.defaults)
    for key in params:
        val = getattr(args, key, None)
        if val is not None:
            params[key] = val
    return params


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"ERROR: data file not found: {data_path}")

    df = pd.read_parquet(data_path)
    if list(df.columns) != SCHEMA_COLUMNS:
        sys.exit(f"ERROR: {data_path} columns {list(df.columns)} != schema {SCHEMA_COLUMNS}")
    if len(df) < 30:
        sys.exit(f"ERROR: only {len(df)} rows — need at least 30 to backtest.")

    spec = STRATEGIES[args.strategy]
    params = resolve_params(spec, args)
    long_signal, stop = split_signal_stop(spec.fn(df, **params))

    result = simulate(df, long_signal, args.capital, stop=stop)
    stats = compute_stats(result, args.capital)

    ticker = data_path.stem.replace("_15min", "")
    out_path = (
        Path(args.out)
        if args.out
        else Path("results") / f"{ticker}_{args.strategy}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "ticker": ticker,
        "strategy": args.strategy,
        "params": params,
        "data_file": str(data_path),
        "bars": len(df),
        "date_range": [df["Date"].min().isoformat(), df["Date"].max().isoformat()],
        "stats": stats,
        "trades": result["trades"],
        "equity_curve": result["equity_curve"],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n=== {ticker} — {args.strategy} {params} ===")
    print(f"Bars              : {len(df)}  ({df['Date'].min().date()} -> {df['Date'].max().date()})")
    print(f"Trades            : {stats['num_trades']}")
    print(f"Win rate          : {stats['win_rate_pct']:.1f}%")
    print(f"Total return      : {stats['total_return_pct']:+.2f}%")
    print(f"CAGR              : {stats['cagr_pct']:+.2f}%")
    print(f"Sharpe (daily)    : {stats['sharpe_daily']:.2f}")
    print(f"Max drawdown      : {stats['max_drawdown_pct']:.2f}%")
    print(f"Avg R / trade     : {stats['avg_r_multiple']:+.2f}")
    print(f"Final equity      : ${stats['final_equity']:,.2f}  (from ${stats['starting_capital']:,.2f})")
    if stop is not None:
        print(f"Exits stop/signal : {stats['stop_exits']} / {stats['signal_exits']}  "
              f"(ambiguous bars {stats['ambiguous_stop_bars']} = {stats['ambiguous_stop_pct']:.1f}%, "
              f"inverted-stop skips {stats['inverted_stop_skips']})")
    print(f"\nResults written to: {out_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
