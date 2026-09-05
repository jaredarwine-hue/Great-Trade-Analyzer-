"""dual_momentum — 12-1 absolute-momentum gate (per-ticker SANITY BASELINE only).

The real dual-momentum PORTFOLIO leg is the cross-sectional monthly rotation in
``run_rotation.py``; this per-ticker boolean is kept only as a sanity baseline.
"""
from __future__ import annotations

import pandas as pd

import indicators as ind
from ._spec import Param, StrategySpec


def dual_momentum(
    df: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    sma_filter: int = 200,
) -> pd.Series:
    """Absolute-momentum (time-series) gate — OPTIONAL single-ticker SANITY BASELINE only.

    NOTE: the real dual-momentum PORTFOLIO leg is the CROSS-SECTIONAL monthly top-3
    rotation implemented in ``run_rotation.py`` (it must rank tickers against each other +
    a SHY cash gate, which no per-ticker boolean can express). This function is kept ONLY
    as a per-ticker sanity baseline, NOT the portfolio leg.

    Baseline rule: long when the 12-1 skip momentum ``Close[t-skip]/Close[t-lookback]-1`` is
    positive AND ``Close > SMA(sma_filter)``. Returns signal only (no stop). No look-ahead.
    """
    n, s = int(lookback), int(skip)
    mom = df["Close"].shift(s) / df["Close"].shift(n) - 1.0
    trend_ma = ind.sma(df["Close"], int(sma_filter))
    return ((mom > 0.0) & (df["Close"] > trend_ma)).fillna(False)


SPEC = StrategySpec(
    fn=dual_momentum,
    defaults={"lookback": 252, "skip": 21, "sma_filter": 200},
    params=[
        Param("--mom-lookback", "lookback", int, "dual_momentum: 12-1 momentum lookback in bars."),
        Param("--mom-skip", "skip", int, "dual_momentum: skip-most-recent bars (12-1 = 21)."),
        Param("--sma-filter", "sma_filter", int, "dual_momentum: regime SMA-filter lookback."),
    ],
)
