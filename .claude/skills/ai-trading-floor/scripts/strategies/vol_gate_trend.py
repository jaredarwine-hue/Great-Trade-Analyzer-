"""vol_gate_trend — trend filter that sits out high-vol regimes (UNSIZED vol-target approx).

Returns a (long_signal, stop) tuple; the engine shifts BOTH once (no look-ahead). Results from
this leg MUST be labeled "vol-target (entry-gate approx, unsized)".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import indicators as ind
from ._spec import Param, StrategySpec

TRADING_DAYS = 252


def vol_gate_trend(
    df: pd.DataFrame,
    ema_fast: int = 50,
    ema_slow: int = 200,
    vol_cap: float = 0.20,
    vol_window: int = 20,
    atr_mult: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """Vol-target trend, ENTRY-GATE approximation (UNSIZED v1, spec strategy 4 fallback).

    This is the labeled v1 fallback for the volatility-targeted trend leg. Real per-bar
    vol SIZING (Engine Change C) is DEFERRED; here we approximate it as an all-in entry
    gate that simply SITS OUT high-vol regimes (where a true vol-target system would scale
    size toward zero):
      long-eligible when ``EMA(fast) > EMA(slow)`` AND ``Close > EMA(fast)`` AND
      annualized realized vol ``daily_ret.rolling(vol_window).std()*sqrt(252) < vol_cap``.
    Exit (signal): flat when the trend filter turns off. Stop: ``entry - atr_mult*ATR(14)``
    (3N, wider because we are not vol-scaling size).

    Results from this leg MUST be labeled "vol-target (entry-gate approx, unsized)" so the
    auditor and portfolio analyst know it is the approximation, not real vol targeting.
    EMAs/vol/ATR read closes <= the signal bar; engine shifts boolean + stop once. No leak.
    """
    close = df["Close"]
    ema_f = ind.ema(close, int(ema_fast))
    ema_s = ind.ema(close, int(ema_slow))
    atr14 = ind.atr(df["High"], df["Low"], close, 14)
    daily_ret = close.pct_change()
    ann_vol = daily_ret.rolling(int(vol_window)).std() * np.sqrt(TRADING_DAYS)

    uptrend = (ema_f > ema_s) & (close > ema_f)
    calm = ann_vol < float(vol_cap)
    long_signal = (uptrend & calm).fillna(False)

    stop = (close - float(atr_mult) * atr14)
    return long_signal, stop


SPEC = StrategySpec(
    fn=vol_gate_trend,
    defaults={"ema_fast": 50, "ema_slow": 200, "vol_cap": 0.20, "vol_window": 20, "atr_mult": 3.0},
    params=[
        Param("--ema-fast", "ema_fast", int, "vol_gate_trend: fast EMA period."),
        Param("--ema-slow", "ema_slow", int, "vol_gate_trend: slow EMA period."),
        Param("--vol-cap", "vol_cap", float, "vol_gate_trend: annualized realized-vol cap (e.g. 0.20)."),
        Param("--vol-window", "vol_window", int, "vol_gate_trend: realized-vol lookback in bars."),
        Param("--atr-mult", "atr_mult", float, "donchian_trend / vol_gate_trend: ATR stop multiple."),
    ],
)
