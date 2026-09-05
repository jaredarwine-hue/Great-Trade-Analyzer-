"""Teaching-quality technical-indicator library for the AI Trading Floor kit.

Every function here follows ONE contract so beginners can learn the *pattern*:

    in:  a price Series (e.g. ``close``) OR an OHLCV DataFrame
    out: a pd.Series (or tuple of Series) ALIGNED to the input's index

"Aligned" means the output has the same length and the same index labels as the
input. Bars that don't have enough history to compute a value are filled with
``NaN`` (the warmup region) rather than dropped — that way the result lines up
1-for-1 with the bars you fed in, and you can ``df["rsi"] = rsi(df["Close"])``
without a length mismatch.

These implementations are intentionally STANDALONE (pandas + numpy only) and are
written to MATCH the standard conventions used by production backtest engines so
you learn numbers consistent with the real backtester. The biggest "gotcha" to
internalize:

  * Simple smoothing  = ``.rolling(period).mean()``  (flat window, equal weights)
  * Wilder smoothing  = ``.ewm(alpha=1/period, adjust=False).mean()``
    (recursive, recent bars weighted more; the classic RSI / "RMA" smoothing)

RSI here uses Wilder smoothing (matching the engine). ATR here uses the SIMPLE
rolling mean of True Range — this matches the standard engine ``atr`` indicator
exactly (a common engine convention), even though "textbook" ATR is
Wilder-smoothed. ``atr_wilder`` is also provided if you want
the textbook version. When the engine and the textbook disagree, this kit follows
the engine so your study numbers reproduce the backtester.

Data schema expected by the OHLCV-DataFrame functions (see DATA_CONTRACT.md):
    RangeIndex; columns [Date, Open, High, Low, Close, Volume];
    Date = naive datetime64 (Eastern for intraday); OHLCV float64.

Read scripts/README.md for the full "how to add your own indicator" walkthrough.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Moving averages — the two smoothing families everything else builds on.
# ---------------------------------------------------------------------------


def sma(close: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average: the flat mean of the last ``period`` closes.

    Formula: ``SMA_t = mean(close[t-period+1 .. t])``. Every bar in the window
    gets equal weight. The classic "is price above its 50-day average?" line.

    Warmup/NaN: the first ``period - 1`` bars are NaN (not enough history to
    fill the window). Output is aligned to ``close``'s index.
    """
    return close.rolling(period).mean()


def ema(close: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average: a recursive average weighting recent bars more.

    Formula: ``EMA_t = alpha*close_t + (1-alpha)*EMA_{t-1}`` with
    ``alpha = 2/(period+1)`` (the standard "span" convention). Reacts faster to
    new prices than SMA. Used for trend lines (EMA 9/21/50) and as the building
    block for MACD and Keltner channels.

    Warmup/NaN: with ``adjust=False`` the recursion seeds on the first bar, so
    there are no NaNs after the first value — but treat the first ``period`` or
    so bars as "still warming up" since the average hasn't seen a full window.
    Output is aligned to ``close``'s index.
    """
    return close.ewm(span=period, adjust=False).mean()


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (a.k.a. RMA) — the recursive average used by RSI/ATR.

    Formula: an EMA with ``alpha = 1/period`` (NOT ``2/(period+1)``). This is
    the smoothing J. Welles Wilder used in his original indicators. ``min_periods
    = period`` means the result is NaN until ``period`` bars exist, so the warmup
    region matches a textbook RSI/ATR.

    This is a SHARED helper rather than an indicator you'd plot directly — RSI
    and the Wilder ATR both call it so their smoothing stays identical.
    """
    return series.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Oscillators — bounded "how stretched is price?" indicators.
# ---------------------------------------------------------------------------


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index with Wilder smoothing. Range [0, 100].

    Formula: split bar-to-bar changes into gains (up moves) and losses (down
    moves), Wilder-smooth each, then ``RSI = 100 - 100/(1 + avg_gain/avg_loss)``.
    Reads as "what fraction of recent movement was upward?" Conventional
    thresholds: <= 30 oversold, >= 70 overbought.

    Matches the standard engine RSI (Wilder via ewm ``alpha=1/period``), except
    this returns NaN during warmup instead of the engine's
    ``fillna(50.0)`` — NaN keeps the output honestly "unknown" before
    enough history exists, which is the right default for a teaching library.

    Warmup/NaN: first ``period`` bars are NaN. Output aligned to ``close``.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)      # positive moves, negatives -> 0
    loss = -delta.clip(upper=0)     # magnitude of negative moves, positives -> 0

    avg_gain = wilder_smooth(gain, period)
    avg_loss = wilder_smooth(loss, period)

    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic oscillator -> (%K, %D). Range [0, 100].

    Formula:
        %K_t = 100 * (close_t - lowest_low(k_period)) /
                     (highest_high(k_period) - lowest_low(k_period))
        %D_t = SMA(%K, d_period)        # the smoothed "signal" line
    %K asks "where in the recent high-low range did we close?" — 0 = at the
    range low, 100 = at the range high. %D smooths %K for crossover signals.

    Warmup/NaN: %K is NaN for the first ``k_period - 1`` bars; %D adds another
    ``d_period - 1`` NaNs on top. A flat range (high == low) yields NaN for that
    bar (division by zero is masked, not forced to a number). Both Series are
    aligned to the input index.
    """
    rolling_high = high.rolling(k_period).max()
    rolling_low = low.rolling(k_period).min()
    rng = rolling_high - rolling_low
    percent_k = 100.0 * (close - rolling_low) / rng
    percent_k = percent_k.where(rng > 0)   # NaN on zero-range bars, not inf
    percent_d = percent_k.rolling(d_period).mean()
    return percent_k, percent_d


# ---------------------------------------------------------------------------
# Trend / momentum composites.
# ---------------------------------------------------------------------------


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD -> (macd_line, signal_line, histogram).

    Formula:
        macd_line   = EMA(fast) - EMA(slow)      # fast vs slow trend spread
        signal_line = EMA(macd_line, signal)     # smoothed trigger line
        histogram   = macd_line - signal_line    # momentum of the spread
    A rising histogram means the fast trend is pulling away from the slow trend
    (accelerating momentum); histogram crossing zero is the classic signal.

    Matches the engine's ``macd`` indicator (same spans, ``adjust=False`` EMAs).
    The engine exposes one component at a time via a ``component`` param; this
    teaching version returns all three at once so the relationship is visible.

    Warmup/NaN: EMAs seed immediately (no NaNs with ``adjust=False``), but the
    first ~``slow`` bars are still warming up. All three Series align to ``close``.
    """
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ---------------------------------------------------------------------------
# Volatility / range.
# ---------------------------------------------------------------------------


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range = max of (today's range, gap-up vs prior close, gap-down).

    TR_t = max(high-low, |high-prev_close|, |low-prev_close|). It captures the
    full move INCLUDING overnight gaps, which a plain high-low range misses.
    Shared helper for both ATR variants. NaN on the first bar (no prior close).
    """
    prev_close = close.shift(1)
    return pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range (engine convention: SIMPLE rolling mean of True Range).

    Formula: ``ATR = SMA(true_range, period)``. A pure volatility gauge in price
    units — bigger ATR = wider bars. Used to size stops (e.g. "stop = entry -
    1.5*ATR") and to build Keltner channels.

    This matches the standard engine ATR (rolling-mean smoothing). If you want
    the textbook Wilder-smoothed version, call
    ``atr_wilder``. They differ in the smoothing family, not the True Range math.

    Warmup/NaN: first ``period`` bars are NaN (TR is NaN on bar 0, and the
    rolling mean needs ``period`` TR values). Output aligned to the input index.
    """
    tr = _true_range(high, low, close)
    return tr.rolling(period).mean()


def atr_wilder(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Textbook ATR using Wilder smoothing (RMA) instead of a flat mean.

    Same True Range, but smoothed with ``wilder_smooth`` (``alpha=1/period``).
    Most charting platforms show THIS version. The engine uses the simple-mean
    ``atr`` above; pick whichever your study should reproduce and be consistent.

    Warmup/NaN: first ``period`` bars are NaN. Output aligned to the input index.
    """
    tr = _true_range(high, low, close)
    return wilder_smooth(tr, period)


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands -> (middle, upper, lower).

    Formula:
        middle = SMA(close, period)
        upper  = middle + num_std * rolling_std(close, period)
        lower  = middle - num_std * rolling_std(close, period)
    The bands widen when volatility rises and pinch in ("squeeze") when it falls.
    Price tagging the upper/lower band is a stretched-vs-mean read.

    Matches the engine's ``bb_upper`` / ``bb_lower`` math (SMA mid +/- num_std *
    rolling std). Uses pandas' default sample std (ddof=1), same as the engine.

    Warmup/NaN: first ``period - 1`` bars are NaN. All three Series align to
    ``close``.
    """
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner Channels -> (middle, upper, lower). An ATR-based envelope.

    Formula:
        middle = EMA(close, ema_period)
        upper  = middle + multiplier * ATR(atr_period)
        lower  = middle - multiplier * ATR(atr_period)
    Like Bollinger Bands but built on ATR (true range) instead of close std, so
    it accounts for gaps. A Bollinger squeeze INSIDE the Keltner channel is the
    classic low-volatility-coil setup.

    Matches the engine's ``keltner_upper`` / ``keltner_lower`` (EMA mid, simple
    rolling-mean ATR via the ``atr`` above).

    Warmup/NaN: governed by the ATR warmup (first ``atr_period`` bars NaN). All
    three Series align to the input index.
    """
    mid = ema(close, ema_period)
    atr_val = atr(high, low, close, atr_period)
    upper = mid + multiplier * atr_val
    lower = mid - multiplier * atr_val
    return mid, upper, lower


def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Williams %R oscillator. Range [-100, 0].

    Formula: ``-100 * (highest_high(N) - close) / (highest_high(N) -
    lowest_low(N))``. It's the Stochastic's %K flipped onto a [-100, 0] scale:
    -100 = closed at the period low (most oversold), 0 = closed at the period
    high. Conventional thresholds: <= -80 oversold, >= -20 overbought. A
    different smoothing family from RSI, so it's useful as an independent read.

    Matches the standard engine williams_r.

    Warmup/NaN: first ``period - 1`` bars are NaN; a zero-range window yields NaN
    for that bar. Output aligned to the input index.
    """
    rolling_high = high.rolling(period).max()
    rolling_low = low.rolling(period).min()
    rng = rolling_high - rolling_low
    wr = -100.0 * (rolling_high - close) / rng
    return wr.where(rng > 0)


# ---------------------------------------------------------------------------
# Volume.
# ---------------------------------------------------------------------------


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume — a running volume tally signed by price direction.

    Formula: start at 0; on an up-close add the bar's volume, on a down-close
    subtract it, on an unchanged close add nothing. The cumulative line is meant
    to confirm trends (OBV making new highs alongside price) or warn of
    divergences (price up but OBV flat = weak buying).

    Warmup/NaN: no NaN — the series starts at 0.0 on bar 0 and accumulates.
    Output aligned to ``close``.
    """
    direction = np.sign(close.diff()).fillna(0.0)   # +1 up, -1 down, 0 flat/first
    signed_volume = direction * volume
    return signed_volume.cumsum()


# ---------------------------------------------------------------------------
# VWAP family — needs the full OHLCV DataFrame (volume + a Date column).
# ---------------------------------------------------------------------------


def _typical_price(df: pd.DataFrame) -> pd.Series:
    """Typical price = (High + Low + Close) / 3 — the per-bar "fair" price VWAP weights."""
    return (df["High"] + df["Low"] + df["Close"]) / 3.0


def vwap_session(df: pd.DataFrame) -> pd.Series:
    """Session-anchored intraday VWAP that RESETS every calendar date.

    Formula per bar: ``cumsum(typical_price * volume) / cumsum(volume)``, where
    the cumulative sums restart at each new calendar day (09:30 ET session open).
    It's the volume-weighted average price *so far today* — the intraday "fair
    value" line traders lean on. Resetting daily is the key rule: yesterday's
    flow shouldn't anchor today's VWAP.

    Matches the standard intraday session-VWAP (resets daily; rejects daily
    input), including its refusal to run on daily data (where every date has 1
    bar VWAP would just equal that bar's typical price, which is meaningless).

    Requires intraday data (>= 2 bars per calendar date) and a ``Date`` column.
    Warmup/NaN: NaN only on bars where cumulative volume is 0. Output aligned to
    ``df``'s index.
    """
    work = df.copy()
    work["_date"] = pd.to_datetime(work["Date"]).dt.date
    if work.groupby("_date").size().max() <= 1:
        raise ValueError(
            "vwap_session requires intraday data (>= 2 bars per calendar date). "
            "On daily data session VWAP collapses to the bar's typical price — "
            "use _typical_price / HLC3 instead."
        )
    typical = _typical_price(work)
    pv = typical * work["Volume"]
    cum_pv = pv.groupby(work["_date"]).cumsum()
    cum_vol = work["Volume"].groupby(work["_date"]).cumsum()
    vwap = cum_pv / cum_vol
    return vwap.where(cum_vol > 0).astype(float)


def anchored_vwap(df: pd.DataFrame, anchor_date: str | pd.Timestamp) -> pd.Series:
    """VWAP anchored to a SPECIFIC start date (cumulative from that point on).

    Unlike ``vwap_session`` (which resets daily), this anchors once at
    ``anchor_date`` and accumulates forward indefinitely — the running
    volume-weighted average price since a meaningful event (an earnings gap, a
    swing low, the start of a base). Reading: is price above or below "everyone's
    average cost since the anchor?"

    Formula from the anchor bar onward:
        ``cumsum(typical_price * volume) / cumsum(volume)``
    Bars BEFORE the anchor are NaN (the anchor hasn't happened yet).

    ``anchor_date`` is compared on calendar date, so passing "2024-03-01" anchors
    at the first bar on or after that day. Requires a ``Date`` column and Volume.
    Output aligned to ``df``'s index.
    """
    work = df.copy()
    dates = pd.to_datetime(work["Date"])
    anchor = pd.Timestamp(anchor_date)
    on_or_after = dates >= anchor
    if not on_or_after.any():
        # Anchor is past the end of the data — entire series is NaN (no anchor bar).
        return pd.Series(np.nan, index=df.index, dtype=float)

    typical = _typical_price(work)
    pv = (typical * work["Volume"]).where(on_or_after, 0.0)
    vol = work["Volume"].where(on_or_after, 0.0)
    cum_pv = pv.cumsum()
    cum_vol = vol.cumsum()
    vwap = cum_pv / cum_vol
    # NaN before the anchor (cum_vol is still 0 there) and on any zero-vol bar.
    return vwap.where((cum_vol > 0) & on_or_after).astype(float)


# ---------------------------------------------------------------------------
# Rolling extremes & prior-period levels.
# ---------------------------------------------------------------------------


def rolling_high(series: pd.Series, period: int) -> pd.Series:
    """Highest value over a trailing ``period``-bar window (INCLUDES current bar).

    ``rolling_high_t = max(series[t-period+1 .. t])``. Used for breakout reads
    ("is price at a 20-bar high?"). Because it includes the current bar, ``close
    == rolling_high(close, 20)`` flags a new high. If you need the level a
    breakout must CLEAR (current bar excluded), use ``prior_period_high``.

    Warmup/NaN: first ``period - 1`` bars are NaN. Aligned to ``series``.
    """
    return series.rolling(period).max()


def rolling_low(series: pd.Series, period: int) -> pd.Series:
    """Lowest value over a trailing ``period``-bar window (INCLUDES current bar).

    Mirror of ``rolling_high``. For the breakdown level a bar must undercut
    (current bar excluded), use ``prior_period_low``.

    Warmup/NaN: first ``period - 1`` bars are NaN. Aligned to ``series``.
    """
    return series.rolling(period).min()


def prior_period_high(df: pd.DataFrame, window: int) -> pd.Series:
    """Highest High over the prior ``window`` bars, STRICTLY EXCLUDING the current bar.

    ``shift(1)`` before ``rolling`` so bar i sees only bars [i-window, i-1]. This
    is the level a breakout-above-N-bar-high entry must clear — excluding the
    current bar avoids the trivial "today is its own high" trap.

    Matches the standard engine prior_period_high (takes a ``window`` param).
    Note: the engine's separate ``prior_day_extreme`` handles the literal "prior
    calendar day's high"; this is the prior-N-BARS high.

    Warmup/NaN: first ``window`` bars are NaN. Aligned to ``df``'s index.
    """
    return df["High"].shift(1).rolling(window).max()


def prior_period_low(df: pd.DataFrame, window: int) -> pd.Series:
    """Lowest Low over the prior ``window`` bars, STRICTLY EXCLUDING the current bar.

    Mirror of ``prior_period_high``: ``shift(1).rolling(window).min()``. The level
    a failed-breakdown / undercut entry references.

    Matches the standard engine prior_period_low.

    Warmup/NaN: first ``window`` bars are NaN. Aligned to ``df``'s index.
    """
    return df["Low"].shift(1).rolling(window).min()


# ---------------------------------------------------------------------------
# Swing pivots — discrete structural points, not a per-bar Series.
# ---------------------------------------------------------------------------


def swing_pivot_lows(
    low: pd.Series,
    lookback: int = 3,
) -> tuple[list[int], list[float]]:
    """Find swing-pivot LOWS: bars lower than the ``lookback`` bars on EACH side.

    A bar i is a pivot low iff ``low[i]`` is strictly less than every Low in
    [i-lookback, i-1] AND every Low in [i+1, i+lookback]. These mark local
    support turns — the building block for trend structure and stop placement.

    Returns ``(positions, levels)``:
        positions: integer POSITIONS (0-based) of each pivot bar
        levels:    the pivot Low value at each of those positions
    (Positions, not index labels, so the result works regardless of the index
    type. Map back via ``low.index[pos]`` if you need the timestamp.)

    Edge case: if ``n < 2*lookback + 1`` no pivot can have full neighbors on both
    sides, so both lists are returned EMPTY (never raises). Uses strict ``<`` so
    flat/tied bars are not pivots — matching the engine's zigzag convention.
    """
    return _swing_pivots(low, lookback, find_lows=True)


def swing_pivot_highs(
    high: pd.Series,
    lookback: int = 3,
) -> tuple[list[int], list[float]]:
    """Find swing-pivot HIGHS: bars higher than the ``lookback`` bars on EACH side.

    Mirror of ``swing_pivot_lows`` — local resistance turns. Same return shape
    ``(positions, levels)`` and the same short-input safety (empty lists when
    ``n < 2*lookback + 1``). Strict ``>`` so ties are not pivots.
    """
    return _swing_pivots(high, lookback, find_lows=False)


def _swing_pivots(
    series: pd.Series,
    lookback: int,
    find_lows: bool,
) -> tuple[list[int], list[float]]:
    """Shared 3-bar (``lookback``-bar) pivot scanner. See the public wrappers."""
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    values = series.to_numpy()
    n = len(values)
    positions: list[int] = []
    levels: list[float] = []
    # A pivot needs `lookback` neighbors on BOTH sides, so the scan is bounded.
    for i in range(lookback, n - lookback):
        center = values[i]
        left = values[i - lookback:i]
        right = values[i + 1:i + lookback + 1]
        if find_lows:
            is_pivot = center < left.min() and center < right.min()
        else:
            is_pivot = center > left.max() and center > right.max()
        if is_pivot:
            positions.append(i)
            levels.append(float(center))
    return positions, levels


# ---------------------------------------------------------------------------
# Trend-strength (ADX) — straightforward Wilder-smoothed directional movement.
# ---------------------------------------------------------------------------


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average Directional Index. Range [0, 100]. Measures trend STRENGTH only.

    ADX says how strongly price is trending, NOT which way. Build-up:
        +DM = up-move if it exceeds the down-move, else 0   (directional movement)
        -DM = down-move if it exceeds the up-move, else 0
        +DI = 100 * wilder(+DM) / wilder(TR)               (directional indicators)
        -DI = 100 * wilder(-DM) / wilder(TR)
        DX  = 100 * |+DI - -DI| / (+DI + -DI)
        ADX = wilder(DX)                                    (smoothed DX)
    Reading: ADX < 20 = choppy/range; ADX > 25 = a real trend (up or down). Pair
    with +DI vs -DI to get direction. All smoothing is Wilder (``alpha=1/period``),
    the standard for ADX.

    Warmup/NaN: roughly the first ``2*period`` bars are NaN (DX is smoothed twice).
    Output is the ADX line only, aligned to the input index.
    """
    up_move = high.diff()
    down_move = -low.diff()

    # Directional movement: a side "wins" only when its move is the larger AND positive.
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )

    tr = _true_range(high, low, close)
    atr_w = wilder_smooth(tr, period)
    plus_di = 100.0 * wilder_smooth(plus_dm, period) / atr_w
    minus_di = 100.0 * wilder_smooth(minus_dm, period) / atr_w

    di_sum = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    dx = dx.where(di_sum > 0)   # NaN when there's no directional movement at all
    return wilder_smooth(dx, period)
