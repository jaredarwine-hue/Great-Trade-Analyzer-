# Rendering clean trading charts

A beginner's guide to drawing readable, honest candlestick charts with matplotlib.
This doc teaches the *visual standard* used across the AI Trading Floor kit:
dark-theme candles, a volume panel, indicator overlays from
`scripts/indicators.py`, non-overlapping price labels, and one safety rule that
keeps charts trustworthy — never plot data the decision couldn't have seen.

A runnable reference lives at `scripts/render_chart.py`. Read this doc, then read
that script — they teach the same thing, one in prose, one in code.

---

## 1. What a good chart looks like

A chart is a decision aid, not decoration. The standard here optimizes for two
readers: a human glancing at it, and an LLM "vision" agent ingesting the PNG. Both
need the same things — high contrast, real candles (not a squiggly line), volume
context, and labels you can actually read.

The recipe, top to bottom:

- **Two stacked panels:** price on top, volume below, sharing the x-axis.
- **Dark theme** so colored overlays pop.
- **OHLC candles** drawn as rectangles + wicks (green up, red down).
- **A few indicator lines** (EMAs, VWAP, Bollinger/ATR bands) — not twenty.
- **Right-edge price labels** that don't pile on top of each other.
- **Timestamps** on the x-axis, rotated so they don't collide.
- **Every axis under 2000px** (a hard cap — see Section 7).

---

## 2. The dark theme

Set these colors once and reuse them. The figure background is darkest, the plot
area slightly lighter, text white, grid faint.

```python
import matplotlib
matplotlib.use("Agg")          # headless backend — render to file, no GUI window
import matplotlib.pyplot as plt

COLOR_FIG    = "#1a1a2e"        # figure background (darkest)
COLOR_PLOT   = "#16213e"        # plot-area background (slightly lighter)
COLOR_TEXT   = "#ffffff"        # axis labels, ticks, title
COLOR_GRID   = "#ffffff"        # grid lines (drawn at low alpha)
COLOR_SPINE  = "#333333"        # axis borders
COLOR_UP     = "#26a69a"        # green candle (close >= open)
COLOR_DOWN   = "#ef5350"        # red candle (close < open)

fig.patch.set_facecolor(COLOR_FIG)
ax.set_facecolor(COLOR_PLOT)
ax.tick_params(colors=COLOR_TEXT)
ax.grid(True, color=COLOR_GRID, alpha=0.15, linewidth=0.5)
for spine in ax.spines.values():
    spine.set_color(COLOR_SPINE)
```

Grid `alpha=0.15` is the sweet spot — visible enough to read levels off, faint
enough not to compete with the candles.

---

## 3. Drawing OHLC candles (not a line chart)

**Do not** `plt.plot(df["Close"])`. A close-only line throws away the open, high,
and low — the whole point of a candle. Draw each bar as a thin **wick** line
(high to low) plus a **body** rectangle (open to close), colored by direction.

The x-axis is the bar's integer position (`0, 1, 2, ...`), not its timestamp —
that keeps candles evenly spaced even across weekends/gaps. We relabel the ticks
with timestamps afterward (Section 6).

```python
import matplotlib.pyplot as plt

def draw_candles(ax, df) -> None:
    """Draw OHLC candles: a wick line + a body rectangle per bar."""
    width = 0.6                                   # body width in x-units (<1 = gap between bars)
    for i, (_, bar) in enumerate(df.iterrows()):
        up = bar["Close"] >= bar["Open"]
        color = "#26a69a" if up else "#ef5350"

        # Wick: a vertical line from the bar's low to its high.
        ax.plot([i, i], [bar["Low"], bar["High"]], color=color, linewidth=0.8, zorder=2)

        # Body: a rectangle from open to close. Guard the zero-height (doji) case.
        lower = min(bar["Open"], bar["Close"])
        height = abs(bar["Close"] - bar["Open"]) or 1e-9
        ax.add_patch(
            plt.Rectangle((i - width / 2, lower), width, height,
                          facecolor=color, edgecolor=color, zorder=3)
        )
```

`zorder` keeps bodies above wicks above the grid. The `or 1e-9` stops a flat
open==close bar from vanishing (a zero-height rectangle draws nothing).

---

## 4. The volume panel

Volume goes in a second panel below price, sharing the x-axis, sized about a third
of the price panel. Color each volume bar to match its candle so up/down days read
at a glance.

```python
fig, (ax_price, ax_vol) = plt.subplots(
    2, 1, sharex=True,
    figsize=(12, 7), dpi=150,
    gridspec_kw={"height_ratios": [3, 1]},        # price 3x taller than volume
)

colors = ["#26a69a" if c >= o else "#ef5350"
          for o, c in zip(df["Open"], df["Close"])]
ax_vol.bar(range(len(df)), df["Volume"], color=colors, alpha=0.7, width=0.6)
```

`sharex=True` links the panels so a level lines up vertically across both. `alpha
=0.7` keeps the volume bars from overpowering the price panel above.

---

## 5. Indicator overlays

Compute overlays with `scripts/indicators.py` (which matches the engine) and draw
them on the price panel. Keep it to a handful — a chart with 4 clean lines beats
one with 15. Pick lines that answer the question you're looking at: EMAs for trend,
VWAP for intraday fair value, Bollinger or ATR-based bands for stretch.

```python
import indicators as ind

ema9  = ind.ema(df["Close"], 9)
ema21 = ind.ema(df["Close"], 21)
mid, upper, lower = ind.bollinger_bands(df["Close"], 20, 2.0)

x = range(len(df))
ax_price.plot(x, ema9,  color="#ffeb3b", linewidth=1.2, label="EMA 9")    # yellow
ax_price.plot(x, ema21, color="#ff7043", linewidth=1.2, label="EMA 21")   # orange
ax_price.plot(x, upper, color="#90caf9", linewidth=0.9, linestyle="--", label="BB upper")
ax_price.plot(x, lower, color="#90caf9", linewidth=0.9, linestyle="--", label="BB lower")
```

For **intraday** data, `ind.vwap_session(df)` gives the session-anchored VWAP
(resets each day) — plot it in cyan (`#00e5ff`). For an **ATR-distance** reference
(e.g. a candidate stop a fixed ATR below the last close), draw a single horizontal
line rather than a band sweeping the whole chart:

```python
atr = ind.atr(df["High"], df["Low"], df["Close"], 14)
stop_ref = df["Close"].iloc[-1] - 1.0 * atr.iloc[-1]
ax_price.axhline(stop_ref, color="#4fc3f7", linewidth=0.9, linestyle=":")
```

A consistent color-per-indicator-family convention makes charts scannable across a
batch: EMA9 yellow, EMA21 orange, EMA50 white, VWAP cyan, bands light-blue, swing
levels magenta. Pick a scheme and keep it.

---

## 6. Readable right-edge price labels (de-collision)

Every horizontal level (each EMA's last value, VWAP, a band, the last close) is
worth labeling at the **right margin** so the reader gets the number without
eyeballing the y-axis. The problem: when two levels are pennies apart, their labels
overlap into an unreadable smear. The fix is a simple **de-collision** pass — walk
the labels top-to-bottom and push any that are too close down by a fixed pixel gap,
drawing a thin **leader line** back to the true level when a label gets nudged.

The technique, in plain steps:

1. Collect every label as `{"y": price, "text": "EMA21 264.40", "color": ...}`.
2. Give the plot a right-margin gutter so labels have room to live:
   `ax.set_xlim(-0.5, n - 0.5 + max(8, n * 0.22))` (extend ~22% past the last bar).
3. Force `fig.canvas.draw()` once so the axes have a real pixel size, then convert
   a fixed pixel gap into data units: `gap = 13 * (y_range / axes_height_px)`.
4. Sort labels by `y` descending. Walk down; if a label is within `gap` of the one
   above it, snap it to `previous_y - gap`. Clamp inside the axis.
5. If a label moved more than a hair from its true level, draw a faint leader line
   from the level to the label so the reader can still trace it.

```python
def draw_right_edge_labels(ax, fig, items, n_bars) -> None:
    """Stack right-margin price labels without overlap; add leader lines."""
    fig.canvas.draw()                                  # ensure a valid bbox exists
    y0, y1 = ax.get_ylim()
    bbox = ax.get_window_extent()
    gap = 13 * (y1 - y0) / bbox.height                 # 13px gap, in data units

    items = sorted(items, key=lambda d: d["y"], reverse=True)
    label_x = n_bars - 0.5 + max(1.0, n_bars * 0.012)  # just inside the gutter
    prev_y = None
    for it in items:
        y = it["y"]
        if prev_y is not None and prev_y - y < gap:
            y = prev_y - gap                           # nudge down to clear the one above
        y = min(max(y, y0 + 0.3 * gap), y1 - 0.3 * gap)
        if abs(y - it["y"]) > 0.05 * gap:              # moved enough to need a leader line
            ax.plot([n_bars - 0.7, label_x], [it["y"], y],
                    color=it["color"], linewidth=0.5, alpha=0.6)
        ax.text(label_x, y, it["text"], color=it["color"],
                va="center", ha="left", fontsize=9, fontweight="bold")
        prev_y = y
```

You don't need this for a chart with two far-apart lines — reach for it once labels
start to crowd. Keep the label count modest; even perfect de-collision runs out of
vertical room if you stack 15 levels into a tight price range.

---

## 7. The 2000-pixel cap (a hard rule)

**Both** the width and height of every PNG must stay **under 2000px.** LLM vision
agents fatal-error on images past that, so an oversized chart is worse than no
chart. The safe combo:

```python
fig = plt.figure(figsize=(12, 7), dpi=150)   # nominal 1800 x 1050
fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
# bbox_inches="tight" trims whitespace -> ~1784 x 1038, comfortably under 2000.
```

Do **not** use `figsize=(14, 8)` — at dpi=150 that's 2100×1200, over the cap.
Always pass `facecolor=fig.get_facecolor()` to `savefig` or the saved margin
reverts to white and breaks the dark theme.

Verify it after rendering (pure dimension read, no GUI):

```python
from PIL import Image
with Image.open(path) as im:
    w, h = im.size
assert w < 2000 and h < 2000, f"{path}: {w}x{h} exceeds the 2000px cap"
```

---

## 8. Timestamps on the x-axis

Bars are positioned by integer index (Section 3), so relabel a handful of ticks
with their real timestamps and rotate them 45° so they don't overlap. Show every
Nth tick, not all of them.

```python
n = len(df)
step = max(1, n // 10)                          # ~10 labels regardless of bar count
ticks = list(range(0, n, step))
ax_vol.set_xticks(ticks)
ax_vol.set_xticklabels(
    [df["Date"].iloc[i].strftime("%Y-%m-%d") for i in ticks],   # use "%m-%d %H:%M" for intraday
    rotation=45, ha="right",
)
```

Put the labels on the bottom (volume) panel only — with `sharex=True` the price
panel inherits the same x-grid without doubling the text.

---

## 9. No future leakage (the one principle that makes a chart honest)

A chart drawn to support a decision must only show data available **at or before**
the moment of that decision. If you're illustrating "what did this look like when
we decided to enter on 2024-03-15 at 10:30?", the chart must end at that bar — not
run to the end of the file. Showing even one future bar makes the setup look
obvious in hindsight and quietly corrupts any study built on it.

The mechanism is dead simple: **slice your DataFrame to the cutoff before you
plot.**

```python
cutoff = pd.Timestamp("2024-03-15 10:30:00")
visible = df[df["Date"] <= cutoff]              # everything up to and including the cutoff bar
# ...compute indicators and render on `visible`, never on the full df.
```

Compute indicators on the *sliced* frame too — an EMA over the full series and then
truncated is fine (EMAs only look backward), but anything that peeks forward (a
centered swing pivot needs bars on both sides) must be computed only where it had
the future bars at the time. When in doubt, slice first, compute second.

If you want to make the cutoff visually obvious, draw a vertical marker at the last
bar:

```python
ax_price.axvline(len(visible) - 1, color="#ffeb3b", linewidth=1.0, linestyle="--")
```

---

## 10. Viewing the PNG (do NOT launch a GUI)

To inspect a rendered chart, use the **Read tool** — it ingests the PNG directly
into the agent's vision context. **Never** run `open`, `xdg-open`, or `start` on a
PNG: those launch a GUI window on the user's machine, which is unwanted. A pure
dimension check via `PIL.Image.open(path).size` is fine (it doesn't display
anything); just never call `.show()`.

---

## 11. Putting it together

The full flow for one clean chart:

1. Load an engine-schema parquet (`[Date, Open, High, Low, Close, Volume]`).
2. Slice to your cutoff date (Section 9) — honesty first.
3. Make a 2-panel figure at `figsize=(12, 7), dpi=150` (Section 4, 7).
4. Apply the dark theme (Section 2).
5. Draw candles (Section 3) and volume (Section 4).
6. Overlay a few indicators from `scripts/indicators.py` (Section 5).
7. Add de-collided right-edge labels (Section 6).
8. Relabel x-ticks with rotated timestamps (Section 8).
9. `savefig(..., bbox_inches="tight")`, then assert both dims < 2000px (Section 7).
10. Inspect with the Read tool (Section 10).

`scripts/render_chart.py` does exactly this end-to-end — run it, read the PNG,
then adapt it. The indicator math it draws comes straight from
`scripts/indicators.py`, so the chart and your backtest agree on the numbers.
