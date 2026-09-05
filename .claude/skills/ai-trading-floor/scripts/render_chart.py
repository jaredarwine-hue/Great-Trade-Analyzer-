#!/usr/bin/env python3
"""Render a dark-theme candlestick PNG for the AI Trading Floor toolkit (self-contained).

Reads ONE parquet of bars (kit schema: Date, Open, High, Low, Close, Volume) and draws a
candlestick price panel + a volume panel, with optional SMA/EMA overlays and optional
entry/exit markers pulled from a backtest results.json. Matplotlib only — no project
imports, nothing reads or writes the repository.

Style follows the kit's chart standard (see CHART_RENDERER.md): dark background
(#1a1a2e / #16213e), green candles #26a69a (close >= open) / red #ef5350 (close < open),
bodies via plt.Rectangle + wicks via plt.plot, gridspec height ratios [3, 1],
figsize=(12, 7) at dpi=150 with bbox_inches="tight" (stays under 2000px on BOTH axes).
Date labels on the x-axis, rotated 45.

The --cutoff option is the no-future-leakage teaching example from CHART_RENDERER.md
Section 9: it slices the data to bars on/before a date so the chart only ever shows what
a decision at that moment could have seen, and draws a dashed marker at the cutoff bar.

Usage:
    python render_chart.py --data data/AAPL.parquet --out chart.png
    python render_chart.py --data data/AAPL.parquet --sma 20,50 --ema 9 --out chart.png
    python render_chart.py --data data/AAPL.parquet --results results/AAPL_sma_crossover.json --out chart.png
    python render_chart.py --data data/AAPL.parquet --cutoff 2024-03-15 --out chart.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Import the sibling indicators module for MA overlays.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import indicators as ind  # noqa: E402

SCHEMA_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
GREEN = "#26a69a"
RED = "#ef5350"
BG = "#1a1a2e"
PANEL = "#16213e"
MA_COLORS = ["#ffeb3b", "#ff9800", "#42a5f5", "#ab47bc", "#26c6da"]


def parse_periods(spec: str | None) -> list[int]:
    """Parse '20,50' -> [20, 50]; None/'' -> []."""
    if not spec:
        return []
    return [int(p.strip()) for p in spec.split(",") if p.strip()]


def load_markers(results_path: Path) -> list[dict]:
    """Pull entry/exit (index, price) marker points from a backtest results.json."""
    with open(results_path) as f:
        payload = json.load(f)
    return payload.get("trades", [])


def render(
    df: pd.DataFrame,
    sma_periods: list[int],
    ema_periods: list[int],
    trades: list[dict],
    title: str,
    save_path: Path,
    cutoff_marker: bool = False,
) -> None:
    """Draw the candlestick + volume chart and save it to ``save_path``.

    If ``cutoff_marker`` is True, a dashed vertical line is drawn at the last bar to
    signal "this is everything the decision could see" (the df is already sliced to the
    cutoff by the caller — see CHART_RENDERER.md Section 9).
    """
    n = len(df)
    x = range(n)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    fig.patch.set_facecolor(BG)

    # Candles: wick via plot, body via Rectangle.
    for i in range(n):
        row = df.iloc[i]
        color = GREEN if row["Close"] >= row["Open"] else RED
        ax1.plot([i, i], [row["Low"], row["High"]], color=color, linewidth=0.6)
        body_low, body_high = min(row["Open"], row["Close"]), max(row["Open"], row["Close"])
        ax1.add_patch(plt.Rectangle((i - 0.35, body_low), 0.7, body_high - body_low, color=color))

    # Overlays.
    ci = 0
    for period in sma_periods:
        ax1.plot(ind.sma(df["Close"], period).values, color=MA_COLORS[ci % len(MA_COLORS)],
                 linewidth=1.2, label=f"SMA {period}", alpha=0.9)
        ci += 1
    for period in ema_periods:
        ax1.plot(ind.ema(df["Close"], period).values, color=MA_COLORS[ci % len(MA_COLORS)],
                 linewidth=1.2, linestyle="--", label=f"EMA {period}", alpha=0.9)
        ci += 1

    # Trade markers — placed in CLEAR SPACE (buys below the bar's low, sells above the high)
    # in VIVID colors distinct from the candles, so they never blend into green/red bars.
    span = float(df["High"].max() - df["Low"].min())
    moff = span * 0.02 if span > 0 else 0.0
    bx, by, sx, sy = [], [], [], []
    for t in trades:
        ei, xi = t.get("entry_index"), t.get("exit_index")
        if ei is not None and 0 <= ei < n:
            bx.append(ei); by.append(float(df["Low"].iloc[ei]) - moff)
        if xi is not None and 0 <= xi < n:
            sx.append(xi); sy.append(float(df["High"].iloc[xi]) + moff)
    if bx:
        ax1.scatter(bx, by, marker="^", color="#00e676", s=90,
                    edgecolors="#0b0f16", linewidths=0.8, zorder=5, label="Buy")
    if sx:
        ax1.scatter(sx, sy, marker="v", color="#ff4d6d", s=90,
                    edgecolors="#0b0f16", linewidths=0.8, zorder=5, label="Sell")

    # No-future-leakage cutoff marker: the right edge IS the cutoff (df pre-sliced).
    if cutoff_marker and n > 0:
        ax1.axvline(n - 1, color="#ffeb3b", linewidth=1.0, linestyle="--", alpha=0.7)

    ax1.set_facecolor(PANEL)
    ax1.set_title(title, color="white", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Price ($)", color="white")
    ax1.tick_params(colors="white")
    if sma_periods or ema_periods:
        ax1.legend(loc="upper left", fontsize=8, facecolor=PANEL, edgecolor="#333", labelcolor="white")
    ax1.grid(True, alpha=0.15, color="white")
    for spine in ax1.spines.values():
        spine.set_color("#333")

    # Volume panel.
    vol_colors = [GREEN if df.iloc[i]["Close"] >= df.iloc[i]["Open"] else RED for i in range(n)]
    ax2.bar(x, df["Volume"].values, color=vol_colors, alpha=0.7, width=0.7)
    ax2.set_facecolor(PANEL)
    ax2.set_ylabel("Volume", color="white")
    ax2.tick_params(colors="white")
    ax2.grid(True, alpha=0.15, color="white")
    for spine in ax2.spines.values():
        spine.set_color("#333")

    # Date labels on the x-axis (rotated 45), sampled so they don't overlap.
    dates = pd.to_datetime(df["Date"])
    step = max(1, n // 12)
    ticks = list(range(0, n, step))
    fmt = "%Y-%m-%d %H:%M" if (dates.dt.hour.nunique() > 1) else "%Y-%m-%d"
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([dates.iloc[i].strftime(fmt) for i in ticks],
                        rotation=45, ha="right", color="white", fontsize=8)
    ax2.set_xlabel("Date", color="white")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a dark-theme candlestick chart from a kit parquet file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="Path to a <TICKER>.parquet file.")
    parser.add_argument("--results", help="Optional backtest results.json for trade markers.")
    parser.add_argument("--sma", help="Comma-separated SMA periods to overlay, e.g. 20,50.")
    parser.add_argument("--ema", help="Comma-separated EMA periods to overlay, e.g. 9,21.")
    parser.add_argument(
        "--cutoff",
        help="Only plot bars on/before this date (YYYY-MM-DD). No-future-leakage slice.",
    )
    parser.add_argument("--out", default="chart.png", help="Output PNG path.")
    args = parser.parse_args(argv)

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"ERROR: data file not found: {data_path}")
    df = pd.read_parquet(data_path)
    if list(df.columns) != SCHEMA_COLUMNS:
        sys.exit(f"ERROR: {data_path} columns {list(df.columns)} != schema {SCHEMA_COLUMNS}")

    # No-future-leakage slice: keep only bars on/before the cutoff (CHART_RENDERER.md §9).
    if args.cutoff:
        df = df[pd.to_datetime(df["Date"]) <= pd.Timestamp(args.cutoff)].reset_index(drop=True)
        if df.empty:
            sys.exit(f"ERROR: no bars on/before cutoff {args.cutoff}.")

    trades: list[dict] = []
    if args.results:
        results_path = Path(args.results)
        if not results_path.exists():
            sys.exit(f"ERROR: results file not found: {results_path}")
        trades = load_markers(results_path)

    sma_periods = parse_periods(args.sma)
    ema_periods = parse_periods(args.ema)
    # Default to a single 50 SMA if no overlays were requested, so the chart is useful.
    if not sma_periods and not ema_periods and len(df) > 50:
        sma_periods = [50]

    ticker = data_path.stem.replace("_15min", "")
    title = f"{ticker} — {len(df)} bars"
    if args.cutoff:
        title += f" — cutoff {args.cutoff}"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render(df, sma_periods, ema_periods, trades, title, out_path, cutoff_marker=bool(args.cutoff))
    print(f"Chart saved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
