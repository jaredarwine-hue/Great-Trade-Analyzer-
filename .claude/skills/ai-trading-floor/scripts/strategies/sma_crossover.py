"""sma_crossover — long while the fast SMA is above the slow SMA (classic trend follow)."""
from __future__ import annotations

import pandas as pd

import indicators as ind
from ._spec import Param, StrategySpec


def sma_crossover(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """Long while the fast SMA is above the slow SMA (classic trend follow)."""
    fast_ma = ind.sma(df["Close"], int(fast))
    slow_ma = ind.sma(df["Close"], int(slow))
    return (fast_ma > slow_ma).fillna(False)


SPEC = StrategySpec(
    fn=sma_crossover,
    defaults={"fast": 20, "slow": 50},
    params=[
        Param("--fast", "fast", int, "sma_crossover: fast SMA period."),
        Param("--slow", "slow", int, "sma_crossover: slow SMA period."),
    ],
)
