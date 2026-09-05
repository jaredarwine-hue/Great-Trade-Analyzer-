"""breakout — long while the close is above the prior N-bar high (current bar excluded)."""
from __future__ import annotations

import pandas as pd

import indicators as ind
from ._spec import Param, StrategySpec


def breakout(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Long while the close is above the prior ``lookback``-bar high (excl. current bar)."""
    prior_high = ind.prior_period_high(df, int(lookback))
    return (df["Close"] > prior_high).fillna(False)


SPEC = StrategySpec(
    fn=breakout,
    defaults={"lookback": 20},
    params=[Param("--lookback", "lookback", int, "breakout: prior-high lookback in bars.")],
)
