"""anchored_vwap_trend — long while price holds above the VWAP anchored to the most recent
confirmed swing-pivot low (re-anchors as new lows form). Leak-free anchor confirmation."""
from __future__ import annotations

import numpy as np
import pandas as pd

import indicators as ind
from ._spec import Param, StrategySpec


def anchored_vwap_trend(
    df: pd.DataFrame,
    swing_lookback: int = 10,
    trend_sma: int = 50,
) -> pd.Series:
    """Brian Shannon-style anchored-VWAP trend: long while price holds above the VWAP
    anchored to the most recent swing low (re-anchors as new lows form — "handoffs").

    The anchored VWAP is the average participant's cost basis since the anchor; above it
    buyers are in control, below it sellers are. Rule: long when ``Close > AVWAP(anchor)``
    (and, if ``trend_sma > 0``, also ``Close > SMA(trend_sma)`` so we only buy in an
    uptrend). Exit when price closes back below the anchor's VWAP.

    Anchor selection (leak-free): the anchor is the most recent CONFIRMED swing-pivot low.
    A pivot at bar ``p`` (lower than ``swing_lookback`` bars on each side) can only be known
    at bar ``p + swing_lookback`` — so we don't adopt it as the anchor until that bar. As
    newer lows confirm, the anchor hands off forward, keeping the VWAP close to price. The
    AVWAP itself uses only typical price * volume up to the current bar, and the engine
    applies the single central ``shift(1)`` to the returned signal. No look-ahead.
    """
    n = len(df)
    lb = max(int(swing_lookback), 1)

    # Confirmed swing-pivot lows -> the bar position of each, knowable lb bars later.
    piv_pos, _ = ind.swing_pivot_lows(df["Low"], lb)
    confirms = sorted((p + lb, p) for p in piv_pos)  # (confirm_bar, anchor_bar)

    # For each bar, the anchor position currently in force (-1 = none confirmed yet).
    anchor_arr = np.full(n, -1, dtype=int)
    current, ci = -1, 0
    for i in range(n):
        while ci < len(confirms) and confirms[ci][0] <= i:
            current = confirms[ci][1]
            ci += 1
        anchor_arr[i] = current

    # Cumulative typical*volume and volume, so AVWAP from any anchor a to bar i is
    # (cumPV[i]-cumPV[a-1]) / (cumVol[i]-cumVol[a-1]) — O(n) overall.
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    cum_pv = (typical * df["Volume"]).cumsum().to_numpy(dtype=float)
    cum_vol = df["Volume"].cumsum().to_numpy(dtype=float)

    avwap = np.full(n, np.nan)
    for i in range(n):
        a = anchor_arr[i]
        if a < 0:
            continue
        base_pv = cum_pv[a - 1] if a > 0 else 0.0
        base_vol = cum_vol[a - 1] if a > 0 else 0.0
        vol = cum_vol[i] - base_vol
        if vol > 0:
            avwap[i] = (cum_pv[i] - base_pv) / vol

    close = df["Close"].to_numpy(dtype=float)
    long_signal = np.zeros(n, dtype=bool)
    have_avwap = ~np.isnan(avwap)
    long_signal[have_avwap] = close[have_avwap] > avwap[have_avwap]

    if int(trend_sma) > 0:
        sma_trend = ind.sma(df["Close"], int(trend_sma)).to_numpy(dtype=float)
        gate = ~np.isnan(sma_trend) & (close > sma_trend)
        long_signal = long_signal & gate

    return pd.Series(long_signal, index=df.index)


SPEC = StrategySpec(
    fn=anchored_vwap_trend,
    defaults={"swing_lookback": 10, "trend_sma": 50},
    params=[
        Param("--swing-lookback", "swing_lookback", int, "anchored_vwap_trend: swing-pivot-low lookback (bars each side)."),
        Param("--trend-sma", "trend_sma", int, "anchored_vwap_trend: SMA trend gate (0 = off)."),
    ],
)
