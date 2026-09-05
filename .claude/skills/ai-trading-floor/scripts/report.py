#!/usr/bin/env python3
"""Build a single interactive, OFFLINE Plotly HTML report for the AI Trading Floor kit.

Combines one parquet of bars + one backtest results.json into ONE self-contained .html
file: a candlestick price chart with indicator overlays and entry/exit markers, an
equity-curve panel, and a stats table. The Plotly JS is embedded inline
(``include_plotlyjs="inline"``) so the file is fully double-clickable with NO internet
connection. No project imports; nothing reads or writes the repository.

Usage:
    python report.py --results results/AAPL_sma_crossover.json --data data/AAPL.parquet --out report.html
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import the sibling indicators module for overlays.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import indicators as ind  # noqa: E402

SCHEMA_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
BG = "#1a1a2e"
PANEL = "#16213e"
GREEN = "#26a69a"
RED = "#ef5350"
TEXT = "#c9d1d9"


def build_figure(df: pd.DataFrame, payload: dict) -> go.Figure:
    """Assemble the 3-row figure: price+markers, equity curve, stats table."""
    dates = pd.to_datetime(df["Date"])
    trades = payload.get("trades", [])
    stats = payload.get("stats", {})
    strategy = payload.get("strategy", "strategy")

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.5, 0.28, 0.22],
        vertical_spacing=0.06,
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]],
        subplot_titles=("Price & Trades", "Equity Curve", "Summary Stats"),
    )

    # Row 1 — candlesticks.
    fig.add_trace(
        go.Candlestick(
            x=dates, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            increasing_line_color=GREEN, decreasing_line_color=RED, name="Price",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Indicator overlays: 50 SMA (and 20 if there's room) for context.
    for period, color in [(20, "#ffeb3b"), (50, "#ff9800")]:
        if len(df) > period:
            fig.add_trace(
                go.Scatter(x=dates, y=ind.sma(df["Close"], period), mode="lines",
                           line=dict(color=color, width=1.2), name=f"SMA {period}"),
                row=1, col=1,
            )

    # Entry/exit markers from results.json.
    for label, key, marker, color in [
        ("Entry", "entry_index", "triangle-up", GREEN),
        ("Exit", "exit_index", "triangle-down", RED),
    ]:
        xs, ys = [], []
        price_key = "entry_price" if key == "entry_index" else "exit_price"
        for t in trades:
            idx = t.get(key)
            if idx is not None and 0 <= idx < len(df):
                xs.append(dates.iloc[idx])
                ys.append(t[price_key])
        if xs:
            fig.add_trace(
                go.Scatter(x=xs, y=ys, mode="markers", name=label,
                           marker=dict(symbol=marker, size=11, color=color,
                                       line=dict(color="white", width=1))),
                row=1, col=1,
            )

    # Row 2 — equity curve.
    curve = pd.DataFrame(payload.get("equity_curve", []))
    if not curve.empty:
        curve["date"] = pd.to_datetime(curve["date"])
        fig.add_trace(
            go.Scatter(x=curve["date"], y=curve["equity"], mode="lines",
                       line=dict(color="#42a5f5", width=1.6), name="Equity",
                       fill="tozeroy", fillcolor="rgba(66,165,245,0.12)"),
            row=2, col=1,
        )

    # Row 3 — stats table.
    label_map = [
        ("Trades", "num_trades"), ("Win rate %", "win_rate_pct"),
        ("Total return %", "total_return_pct"), ("CAGR %", "cagr_pct"),
        ("Sharpe (daily)", "sharpe_daily"), ("Max drawdown %", "max_drawdown_pct"),
        ("Avg R / trade", "avg_r_multiple"), ("Final equity $", "final_equity"),
        ("Starting capital $", "starting_capital"),
    ]
    metrics = [lbl for lbl, _ in label_map]
    values = [stats.get(key, "") for _, key in label_map]
    fig.add_trace(
        go.Table(
            header=dict(values=["Metric", "Value"], fill_color=PANEL,
                        font=dict(color="white", size=13), align="left"),
            cells=dict(values=[metrics, values], fill_color=BG,
                       font=dict(color=TEXT, size=12), align="left", height=24),
        ),
        row=3, col=1,
    )

    ticker = payload.get("ticker", "")
    fig.update_layout(
        title=dict(text=f"{ticker} — {strategy} — AI Trading Floor Report",
                   font=dict(color="white", size=20)),
        paper_bgcolor=BG, plot_bgcolor=PANEL, font=dict(color=TEXT, family="Inter"),
        xaxis_rangeslider_visible=False, showlegend=True,
        legend=dict(bgcolor=PANEL, bordercolor="#333", borderwidth=1),
        height=1000, margin=dict(l=60, r=40, t=80, b=40),
    )
    for axis in ("xaxis", "xaxis2", "yaxis", "yaxis2"):
        fig.update_layout(**{axis: dict(gridcolor="#2a2f45", zerolinecolor="#2a2f45")})
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build an interactive, offline Plotly HTML report from a backtest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results", required=True, help="Backtest results.json path.")
    parser.add_argument("--data", required=True, help="The parquet file the backtest used.")
    parser.add_argument("--out", default="report.html", help="Output HTML path.")
    args = parser.parse_args(argv)

    results_path, data_path = Path(args.results), Path(args.data)
    if not results_path.exists():
        sys.exit(f"ERROR: results file not found: {results_path}")
    if not data_path.exists():
        sys.exit(f"ERROR: data file not found: {data_path}")

    with open(results_path) as f:
        payload = json.load(f)
    df = pd.read_parquet(data_path)
    if list(df.columns) != SCHEMA_COLUMNS:
        sys.exit(f"ERROR: {data_path} columns {list(df.columns)} != schema {SCHEMA_COLUMNS}")

    fig = build_figure(df, payload)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # include_plotlyjs="inline" embeds the full library so the file works OFFLINE.
    fig.write_html(str(out_path), include_plotlyjs="inline", full_html=True)
    print(f"Interactive report saved to: {out_path}")
    print("Open it by double-clicking — it works fully offline (no internet needed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
