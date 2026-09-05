"""Strategy registry — auto-discovers every strategy module in this package.

THE EXTENSIBLE PATTERN: each ``<name>.py`` in this folder defines a strategy function and
exports ``SPEC = StrategySpec(...)``. Drop a new file in here and it is registered
automatically as strategy ``<name>`` — no edit to the engine, no central list to maintain.
The engine (``backtest.py``) just does ``from strategies import STRATEGIES, ALL_PARAMS``.

A strategy's ``fn`` returns a per-bar POSITION Series: +1 long, -1 short, 0 flat (a plain
boolean is the long-only case, True -> +1). The side lives ENTIRELY in the strategy's
conditions — there is no engine/CLI direction switch.
"""
from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path

# Put the scripts/ dir (this package's parent) on sys.path so strategy modules can
# ``import indicators`` (the sibling indicator module) no matter how they are loaded.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from ._spec import Param, SignalFn, StrategySpec  # noqa: E402  (re-exported for the engine)

STRATEGIES: dict[str, StrategySpec] = {}
_params_by_flag: dict[str, Param] = {}

for _m in pkgutil.iter_modules([str(Path(__file__).resolve().parent)]):
    if _m.name.startswith("_"):
        continue  # skip _spec and any private helpers
    _mod = importlib.import_module(f"{__name__}.{_m.name}")
    _spec = getattr(_mod, "SPEC", None)
    if _spec is None:
        continue
    STRATEGIES[_m.name] = _spec
    for _p in _spec.params:
        _params_by_flag.setdefault(_p.flag, _p)  # first declaration of a shared flag wins

# Deterministic flag order so the CLI help is stable run-to-run.
ALL_PARAMS = [_params_by_flag[f] for f in sorted(_params_by_flag)]

__all__ = ["STRATEGIES", "ALL_PARAMS", "StrategySpec", "Param", "SignalFn"]
