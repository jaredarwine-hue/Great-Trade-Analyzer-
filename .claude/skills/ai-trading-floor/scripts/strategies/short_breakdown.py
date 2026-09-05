"""short_breakdown — SHORT-side example: be short (-1) on a downside breakout, flat otherwise.

The signed mirror of ``breakout`` and a worked example of the contract: the side lives in the
strategy's CONDITIONS (it emits -1), not in any engine/CLI flag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import indicators as ind
from ._spec import Param, StrategySpec


def short_breakdown(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """SHORT-only: be short (-1) while the close is below the prior ``lookback``-bar low
    (excl. the current bar) — a downside breakout; flat (0) otherwise. The signed mirror
    of ``breakout``, expressing the short side purely through its own conditions."""
    prior_low = ind.prior_period_low(df, int(lookback))
    return pd.Series(np.where((df["Close"] < prior_low).to_numpy(), -1, 0), index=df.index)


SPEC = StrategySpec(
    fn=short_breakdown,
    defaults={"lookback": 20},
    params=[Param("--lookback", "lookback", int, "breakout / short_breakdown: prior-extreme lookback in bars.")],
)
