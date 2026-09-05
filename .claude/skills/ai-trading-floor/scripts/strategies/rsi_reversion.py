"""rsi_reversion — buy oversold RSI, hold until overbought (stateful in/out rule)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import indicators as ind
from ._spec import Param, StrategySpec


def rsi_reversion(
    df: pd.DataFrame,
    rsi_period: int = 14,
    buy_below: float = 30.0,
    sell_above: float = 55.0,
) -> pd.Series:
    """Buy when RSI dips below ``buy_below``; hold until RSI rises above ``sell_above``.

    This is a stateful "in/out" rule, so we walk the RSI once: enter on an oversold
    reading, stay long until an overbought-enough reading, then flat.
    """
    rsi_vals = ind.rsi(df["Close"], int(rsi_period))
    long_signal = np.zeros(len(df), dtype=bool)
    holding = False
    for i, r in enumerate(rsi_vals.to_numpy()):
        if np.isnan(r):
            long_signal[i] = False
            continue
        if not holding and r < buy_below:
            holding = True
        elif holding and r > sell_above:
            holding = False
        long_signal[i] = holding
    return pd.Series(long_signal, index=df.index)


SPEC = StrategySpec(
    fn=rsi_reversion,
    defaults={"rsi_period": 14, "buy_below": 30.0, "sell_above": 55.0},
    params=[
        Param("--rsi-period", "rsi_period", int, "rsi_reversion: RSI lookback."),
        Param("--buy-below", "buy_below", float, "rsi_reversion: buy when RSI below this."),
        Param("--sell-above", "sell_above", float, "rsi_reversion: sell when RSI above this."),
    ],
)
