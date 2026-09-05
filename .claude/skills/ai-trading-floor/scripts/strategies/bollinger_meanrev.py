"""bollinger_meanrev — buy band-dips inside an uptrend; mid-band/time-stop exit; ATR stop.

Returns a (long_signal, stop) tuple. NEITHER is shifted here — the engine applies the single
central shift(1) to BOTH, so no strategy can leak future data. The stop is built below the
entry (``Low - 0.5*ATR``) so ``stop < entry`` holds (asserted in the engine).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import indicators as ind
from ._spec import Param, StrategySpec


def bollinger_meanrev(
    df: pd.DataFrame,
    bb_period: int = 20,
    num_std: float = 2.0,
    rsi_filter: float = 35.0,
    time_stop_bars: int = 15,
) -> tuple[pd.Series, pd.Series]:
    """Bollinger mean-reversion with trend + RSI regime filters (spec strategy 1).

    Entry (stateful): long when on a completed bar ``Close < lower band`` AND
    ``Close > SMA200`` (only buy dips inside an uptrend) AND ``RSI(14) < rsi_filter``.
    Exit (signal): flat when ``Close >= middle band`` OR after ``time_stop_bars`` bars
    held (time stop), whichever first.
    Stop: ``entry-bar Low - 0.5*ATR(14)`` (below the entry bar -> stop < entry).

    The SMA200 + RSI<35 filters are the key change from the naive band-touch: they cut
    the left-tail of buying dips in downtrends. All indicators read closes <= the signal
    bar; the engine shifts the boolean AND the stop once, so no look-ahead.
    """
    close = df["Close"]
    mid, upper, lower = ind.bollinger_bands(close, int(bb_period), float(num_std))
    sma200 = ind.sma(close, 200)
    rsi14 = ind.rsi(close, 14)
    atr14 = ind.atr(df["High"], df["Low"], close, 14)

    c = close.to_numpy()
    lo_band = lower.to_numpy()
    mid_band = mid.to_numpy()
    sma2 = sma200.to_numpy()
    rsi = rsi14.to_numpy()

    long_signal = np.zeros(len(df), dtype=bool)
    holding = False
    bars_held = 0
    for i in range(len(df)):
        if np.isnan(lo_band[i]) or np.isnan(mid_band[i]) or np.isnan(sma2[i]) or np.isnan(rsi[i]):
            long_signal[i] = False
            holding, bars_held = False, 0
            continue
        if not holding:
            if c[i] < lo_band[i] and c[i] > sma2[i] and rsi[i] < float(rsi_filter):
                holding, bars_held = True, 0
        else:
            bars_held += 1
            if c[i] >= mid_band[i] or bars_held >= int(time_stop_bars):
                holding = False
        long_signal[i] = holding

    # Stop level per bar = that bar's Low - 0.5*ATR (captured on the entry bar by the engine).
    stop = (df["Low"] - 0.5 * atr14)
    return pd.Series(long_signal, index=df.index), stop


SPEC = StrategySpec(
    fn=bollinger_meanrev,
    defaults={"bb_period": 20, "num_std": 2.0, "rsi_filter": 35.0, "time_stop_bars": 15},
    params=[
        Param("--bb-period", "bb_period", int, "bollinger_meanrev: band SMA/std lookback."),
        Param("--num-std", "num_std", float, "bollinger_meanrev: band width in std devs."),
        Param("--rsi-filter", "rsi_filter", float, "bollinger_meanrev: only buy when RSI(14) below this."),
        Param("--time-stop-bars", "time_stop_bars", int, "bollinger_meanrev: time stop (bars held)."),
    ],
)
