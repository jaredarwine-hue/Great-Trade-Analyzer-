"""donchian_trend — Donchian-channel trend (Turtle System 2): channel breakout + ATR stop.

Returns a (long_signal, stop) tuple; the engine shifts BOTH once (no look-ahead). Channel
levels exclude the current bar (shift(1) inside the helper) and the stop is built below entry.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import indicators as ind
from ._spec import Param, StrategySpec


def donchian_trend(
    df: pd.DataFrame,
    entry_window: int = 55,
    exit_window: int = 20,
    atr_mult: float = 2.0,
) -> tuple[pd.Series, pd.Series]:
    """Donchian-channel trend, Turtle System 2 (spec strategy 3).

    Entry (signal): long when ``Close > prior_period_high(df, entry_window)`` (channel top,
    current bar excluded). Exit (signal): flat when ``Close < prior_period_low(df,
    exit_window)`` (trailing exit channel). Stop: ``entry - atr_mult*ATR(14)`` (2N Turtle
    stop); the engine takes whichever fires first (intrabar stop or the channel exit).

    Spec center is 55/20 with a 2*ATR initial stop (20/10 is only a grid neighbor). Channel
    levels exclude the current bar (shift(1) inside the helper); the engine shifts the
    boolean + stop once. No look-ahead.
    """
    close = df["Close"]
    chan_high = ind.prior_period_high(df, int(entry_window))
    chan_low = ind.prior_period_low(df, int(exit_window))
    atr14 = ind.atr(df["High"], df["Low"], close, 14)

    c = close.to_numpy()
    hi = chan_high.to_numpy()
    lo = chan_low.to_numpy()
    long_signal = np.zeros(len(df), dtype=bool)
    holding = False
    for i in range(len(df)):
        if np.isnan(hi[i]) or np.isnan(lo[i]):
            long_signal[i] = False
            continue
        if not holding and c[i] > hi[i]:
            holding = True
        elif holding and c[i] < lo[i]:
            holding = False
        long_signal[i] = holding

    # 2N initial stop below the entry: reference the bar's close (entry proxy) - k*ATR.
    stop = (close - float(atr_mult) * atr14)
    return pd.Series(long_signal, index=df.index), stop


SPEC = StrategySpec(
    fn=donchian_trend,
    defaults={"entry_window": 55, "exit_window": 20, "atr_mult": 2.0},
    params=[
        Param("--entry-window", "entry_window", int, "donchian_trend: entry channel (prior-high) window."),
        Param("--exit-window", "exit_window", int, "donchian_trend: exit channel (prior-low) window."),
        Param("--atr-mult", "atr_mult", float, "donchian_trend / vol_gate_trend: ATR stop multiple."),
    ],
)
