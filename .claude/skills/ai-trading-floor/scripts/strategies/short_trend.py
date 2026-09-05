"""short_trend — be SHORT (-1) while the asset is in a downtrend (fast EMA below slow EMA AND
close below the fast EMA); flat (0) otherwise. A trend-following short: it rides sustained
declines, the short-side analogue of an EMA trend long. Side lives in the conditions."""
from __future__ import annotations

import numpy as np
import pandas as pd

import indicators as ind
from ._spec import Param, StrategySpec


def short_trend(df: pd.DataFrame, ema_fast: int = 50, ema_slow: int = 200) -> pd.Series:
    """Short while ``EMA(fast) < EMA(slow)`` AND ``Close < EMA(fast)`` (a confirmed downtrend);
    flat otherwise. Emits -1 / 0 — a pure short trend-follower."""
    ef = ind.ema(df["Close"], int(ema_fast))
    es = ind.ema(df["Close"], int(ema_slow))
    down = (ef < es) & (df["Close"] < ef)
    return pd.Series(np.where(down.fillna(False).to_numpy(), -1, 0), index=df.index)


SPEC = StrategySpec(
    fn=short_trend,
    defaults={"ema_fast": 50, "ema_slow": 200},
    params=[
        Param("--ema-fast", "ema_fast", int, "vol_gate_trend / short_trend: fast EMA period."),
        Param("--ema-slow", "ema_slow", int, "vol_gate_trend / short_trend: slow EMA period."),
    ],
)
