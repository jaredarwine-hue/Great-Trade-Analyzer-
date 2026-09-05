"""Shared strategy contract: ``StrategySpec`` + the CLI ``Param`` descriptor.

Kept deliberately dependency-free (stdlib + pandas typing only) so every strategy module
AND the engine can import it without any circular import.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

SignalFn = Callable[..., pd.Series]


@dataclass
class Param:
    """One CLI parameter a strategy exposes.

    ``flag`` is the argparse option string (e.g. ``--fast``); ``dest`` is the key in the
    strategy's ``defaults`` it overrides; ``type`` is ``int`` or ``float``; ``help`` is the
    CLI help text. Strategies that legitimately SHARE a flag (e.g. ``breakout`` and
    ``short_breakdown`` both use ``--lookback``; ``donchian_trend`` and ``vol_gate_trend``
    both use ``--atr-mult``) are de-duplicated by flag in the loader.
    """

    flag: str
    dest: str
    type: type
    help: str = ""


@dataclass
class StrategySpec:
    """A built-in strategy: the signal function + its default params + the CLI it exposes.

    ``fn`` returns a per-bar POSITION Series (+1 long / -1 short / 0 flat; a plain boolean is
    the long-only case). ``defaults`` are the param values used when no CLI flag overrides
    them. ``params`` lists the CLI flags this strategy adds (see ``Param``).
    """

    fn: SignalFn
    defaults: dict
    params: list[Param] = field(default_factory=list)
