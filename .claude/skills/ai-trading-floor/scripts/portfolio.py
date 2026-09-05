#!/usr/bin/env python3
"""Portfolio combiner — blend N single-strategy backtests into one portfolio.

This is the Orchestrator Analyst's tool. It does NOT re-decide any strategy's
internal trades; it loads the per-bar equity curves the Data Engineer produced
(``results/<ticker>_<strategy>.json``), turns each into daily returns aligned on
a shared date index, and combines them under a selectable, **1-day-lagged**
weighting scheme.

Why the lag matters (the one rule that keeps this honest)
---------------------------------------------------------
The capital split for day ``T`` must be decided using information available only
through ``T-1``. Setting day ``T``'s weights from day ``T``'s own returns (e.g.
inverse-vol computed from a window that *includes* T) is look-ahead bias. Every
weighting scheme here builds a weight Series and then ``shift(1)`` it before it
multiplies returns. The first day after the lag has no weights yet, so it earns
zero — that is correct, not a bug.

Sharpe convention (matches scripts/backtest.py::compute_stats)
--------------------------------------------------------------
Daily returns -> ``mean / std(ddof=1) * sqrt(252)``. We resample the combined
equity curve to one value per calendar day first, exactly like the engine, so
the portfolio Sharpe is directly comparable to each single-strategy Sharpe.

Modes
-----
* ``combine`` — combine an explicit list of result files.
* ``search``  — given a pool of result files, search for a low-correlation
  subset that maximizes combined Sharpe (greedy add or correlation-capped
  selection). Subset size is capped and the number of combos tried is reported,
  so this is selection, not brute-force data-snooping.

Usage
-----
    python portfolio.py combine results/A.json results/B.json \
        --scheme inverse_vol --name growth_blend

    python portfolio.py search results/*.json \
        --scheme inverse_vol --max-size 5 --corr-cap 0.5 --name best

    # Validate a chosen subset out-of-sample with the SAME scheme:
    python portfolio.py combine results/oos/A.json results/oos/B.json \
        --scheme inverse_vol --name best_oos
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252  # matches scripts/backtest.py
DEFAULT_VOL_WINDOW = 20  # lookback (in days) for inverse-vol / risk-parity weights


# ---------------------------------------------------------------------------
# Loading & alignment.
# ---------------------------------------------------------------------------
@dataclass
class StrategyReturns:
    """One sub-strategy reduced to a daily return Series on a shared index."""

    label: str
    daily_ret: pd.Series  # index = calendar day, value = day-over-day pct change
    source: str
    num_trades: int  # from stats.num_trades; used to reject thin-trade artifacts


def _label_for(result: dict, path: str) -> str:
    """Human label = the FILE STEM, which is unique on disk and is what callers
    type on the CLI (``results/<stem>.json``).

    We deliberately do NOT use ``<ticker>_<strategy>`` from the JSON body: several
    files can share the same ticker+strategy (e.g. three rotation variants all
    carrying ticker=PORT, strategy=rotation_dualmom). That collision silently
    collapses them to one column in the correlation matrix / alignment. The
    filename stem is guaranteed distinct and matches how members are referenced,
    so it's the safe label. For normal files stem == "<ticker>_<strategy>" anyway.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem:
        return stem
    ticker = result.get("ticker")
    strategy = result.get("strategy")
    return f"{ticker}_{strategy}" if ticker and strategy else "unknown"


def load_daily_returns(path: str) -> StrategyReturns:
    """Load one result JSON and reduce its equity_curve to DAILY returns.

    We resample the equity curve to one value per calendar day (last obs of the
    day) before differencing — same as the engine — so a sub-strategy's own
    Sharpe reproduces when combined alone.
    """
    with open(path, "r") as fh:
        result = json.load(fh)
    curve = result.get("equity_curve")
    if not curve:
        raise ValueError(f"{path}: no equity_curve to combine")

    df = pd.DataFrame(curve)
    df["date"] = pd.to_datetime(df["date"])
    daily_equity = df.set_index("date")["equity"].resample("1D").last().dropna()
    daily_ret = daily_equity.pct_change().dropna()
    if daily_ret.empty:
        raise ValueError(f"{path}: equity_curve too short to form daily returns")

    # Trade count drives the thin-trade guard. Prefer stats.num_trades; fall back
    # to len(trades). A 1-2 trade "strategy" is really buy-and-hold over the
    # window — its daily series can show a spuriously high Sharpe (audited: the
    # OOS Sharpe>=2 single names were all 1-5 trade artifacts), so the search
    # must be able to exclude it.
    num_trades = result.get("stats", {}).get("num_trades")
    if num_trades is None:
        num_trades = len(result.get("trades", []))

    return StrategyReturns(
        label=_label_for(result, path),
        daily_ret=daily_ret,
        source=os.path.relpath(path),
        num_trades=int(num_trades),
    )


def load_many(paths: list[str], strict: bool) -> list[StrategyReturns]:
    """Load several result files, skipping non-result files when not strict.

    Globs like ``results/*.json`` can sweep in sidecar files (``_batch_summary``,
    a previously written ``portfolio_*``) that have no ``equity_curve``. In
    ``search`` we skip those with a warning; in an explicit ``combine`` we are
    strict, because every named file is presumed to be a real strategy leg.
    """
    out: list[StrategyReturns] = []
    for p in paths:
        try:
            out.append(load_daily_returns(p))
        except (ValueError, KeyError) as exc:
            if strict:
                raise
            print(f"  skip {os.path.relpath(p)}: {exc}", file=__import__("sys").stderr)
    if not out:
        raise ValueError("No loadable result files (none had an equity_curve).")
    return out


def align_returns(strategies: list[StrategyReturns]) -> pd.DataFrame:
    """Align all sub-strategies onto a shared date index.

    Inner join on dates: every column has a real return on every retained day, so
    a missing strategy never silently contributes a 0% (which would understate
    its vol and distort the blend). We report the retained span to the caller.
    """
    labels = [s.label for s in strategies]
    dupes = {x for x in labels if labels.count(x) > 1}
    if dupes:
        raise ValueError(
            f"Duplicate strategy labels would collapse columns: {sorted(dupes)}. "
            f"Labels come from filenames — rename the colliding result files."
        )
    series = {s.label: s.daily_ret for s in strategies}
    aligned = pd.DataFrame(series).dropna(how="any")
    if aligned.shape[0] < 2:
        raise ValueError(
            "After aligning on shared dates fewer than 2 common days remain — "
            "the strategies' equity curves barely overlap."
        )
    return aligned


# ---------------------------------------------------------------------------
# Weighting schemes — each returns LAGGED daily weights (already shift(1)'d).
# ---------------------------------------------------------------------------
def _equal_dollar_weights(rets: pd.DataFrame) -> pd.DataFrame:
    """Equal capital to every strategy, every day. Constant, so the lag is a
    formality, but we still shift(1) so the rule is uniform across schemes.

    Caveat worth stating: equal-DOLLAR is not equal-RISK. A high-vol strategy
    dominates the blend's variance here. Use inverse_vol if you want each
    strategy to contribute comparable risk.
    """
    n = rets.shape[1]
    raw = pd.DataFrame(1.0 / n, index=rets.index, columns=rets.columns)
    return raw.shift(1)


def _inverse_vol_weights(rets: pd.DataFrame, window: int) -> pd.DataFrame:
    """Risk-parity-lite: weight inversely to each strategy's trailing volatility.

    ONE ``shift(1)`` on the final weights is sufficient and correct (audited).
    ``rolling(window).std()`` at row r covers rows [r-window+1 .. r] — a window
    that ENDS at r. The single ``shift(1)`` maps the weight USED on day T to the
    row computed at T-1, whose window therefore ends at T-1. So
    ``weight[T] = f(vol over [T-window .. T-1])`` — knowable before day T's
    return exists. A second shift would over-lag to T-2 for no honesty benefit.
    """
    trailing_vol = rets.rolling(window, min_periods=max(2, window // 2)).std(ddof=1)
    inv = 1.0 / trailing_vol.replace(0.0, np.nan)
    raw = inv.div(inv.sum(axis=1), axis=0)  # normalize to sum 1 each day
    return raw.shift(1)  # day-T weight comes from the T-1 row (window ends T-1)


def build_weights(rets: pd.DataFrame, scheme: str, vol_window: int) -> pd.DataFrame:
    if scheme == "equal_dollar":
        return _equal_dollar_weights(rets)
    if scheme in ("inverse_vol", "risk_parity"):
        return _inverse_vol_weights(rets, vol_window)
    raise ValueError(f"unknown scheme {scheme!r} (use equal_dollar or inverse_vol)")


# ---------------------------------------------------------------------------
# Combine & score.
# ---------------------------------------------------------------------------
def active_index(weights: pd.DataFrame) -> pd.DatetimeIndex:
    """Days the portfolio is actually allocated — weights not all-NaN.

    The lag (and, for inverse_vol, the rolling window) leaves leading warm-up
    days with no weights yet. We trim those rather than score them as flat 0%:
    a flat day would understate volatility and isn't a day the portfolio traded.
    Crucially, per-strategy comparison legs are scored on this SAME index, so the
    'combined vs best single' comparison is apples-to-apples (identical days).
    """
    return weights.dropna(how="all").index


def combine_returns(rets: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """portfolio_ret(T) = Σ weight_i(T) * ret_i(T), weights already lagged.

    Restricted to the active (post-warm-up) window. Within that window any
    individual NaN weight (a single leg still warming up) is treated as 0 for
    that leg only, which is the conservative "not yet allocated to that leg".
    """
    idx = active_index(weights)
    w = weights.reindex(rets.index).loc[idx].fillna(0.0)
    port = (w * rets.loc[idx]).sum(axis=1)
    return port


def equity_from_returns(port_ret: pd.Series, capital: float) -> pd.Series:
    return capital * (1.0 + port_ret).cumprod()


def sharpe_ci(port_ret: pd.Series) -> tuple[float, float, float]:
    """Annualized Sharpe with a 95% CI (Lo 2002 IID approximation).

    SE(daily Sharpe) ~= sqrt((1 + 0.5*S_d^2) / n), annualized by sqrt(252). On
    ~200 OOS days this band is ~+-1 Sharpe — wide enough that two OOS point
    estimates are usually statistically indistinguishable. Report it so nobody
    over-reads a point estimate. Returns (sharpe, lo, hi), all annualized.

    CAVEAT (auditor): this is the IID SE. Trend/momentum daily returns are often
    positively autocorrelated, which makes the TRUE SE LARGER than this. So treat
    this CI as a FLOOR on the uncertainty — the real band is at least this wide,
    never narrower. Don't claim two Sharpes differ unless the IID CIs already
    fail to overlap (and even then, prefer the paired t-test on daily diffs).
    """
    n = len(port_ret)
    if n <= 1 or port_ret.std(ddof=1) == 0:
        return 0.0, 0.0, 0.0
    s_d = port_ret.mean() / port_ret.std(ddof=1)  # daily
    se_d = np.sqrt((1.0 + 0.5 * s_d**2) / n)
    ann = np.sqrt(TRADING_DAYS)
    s = s_d * ann
    return float(s), float((s_d - 1.96 * se_d) * ann), float((s_d + 1.96 * se_d) * ann)


def stats_from_daily(port_ret: pd.Series, capital: float) -> dict:
    """Same stat definitions as scripts/backtest.py::compute_stats, on daily."""
    equity = equity_from_returns(port_ret, capital)
    final_equity = float(equity.iloc[-1])
    total_return = (final_equity / capital - 1.0) * 100.0

    span_days = max((equity.index[-1] - equity.index[0]).days, 1)
    years = span_days / 365.25
    cagr = ((final_equity / capital) ** (1.0 / years) - 1.0) * 100.0 if years > 0 else 0.0

    sharpe, sharpe_lo, sharpe_hi = sharpe_ci(port_ret)

    eq = equity.to_numpy(dtype=float)
    running_max = np.maximum.accumulate(eq)
    drawdowns = (eq - running_max) / running_max
    max_dd = float(drawdowns.min() * 100.0) if len(drawdowns) else 0.0

    return {
        "total_return_pct": round(total_return, 2),
        "cagr_pct": round(cagr, 2),
        "sharpe_daily": round(sharpe, 3),
        "sharpe_ci95": [round(sharpe_lo, 3), round(sharpe_hi, 3)],
        "max_drawdown_pct": round(max_dd, 2),
        "final_equity": round(final_equity, 2),
        "starting_capital": round(capital, 2),
        "trading_days": int(len(port_ret)),
    }


def avg_weight_distribution(weights: pd.DataFrame) -> dict:
    """Mean realized weight per strategy over the ACTIVE (post-warm-up) window.

    We average each leg over EVERY active day (a leg not yet allocated counts as
    0 that day), not over only that leg's own non-NaN days. The latter inflates
    legs that enter the window late and makes the reported shares sum to >1 even
    though the daily book is fully invested (each day's weights sum to 1). With
    NaN->0 over the shared active window the reported shares sum to ~1, matching
    the actual average capital allocation.
    """
    valid = weights.dropna(how="all")
    if valid.empty:
        return {}
    means = valid.fillna(0.0).mean(axis=0)
    return {col: round(float(means[col]), 4) for col in means.index}


def correlation_stability(rets: pd.DataFrame, n_subperiods: int = 3) -> dict:
    """Recompute the leg correlation matrix on each of N equal subperiods.

    Low correlation ONCE (over the whole sample) can be an average that hides a
    subperiod where the legs moved together — diversification that exists on
    paper but not when it's needed. We split the shared window into N contiguous
    subperiods, compute each pair's correlation in each, and report the MAX
    absolute pairwise correlation seen in ANY subperiod (``worst_pair_abs_corr``)
    plus the per-subperiod max. A combo whose diversification only holds in one
    subperiod shows a high worst-case here even if the full-sample corr is low.
    """
    cols = list(rets.columns)
    if len(cols) < 2 or len(rets) < n_subperiods * 2:
        return {"subperiods": [], "worst_pair_abs_corr": 0.0}
    chunks = np.array_split(np.arange(len(rets)), n_subperiods)
    sub = []
    worst = 0.0
    for ci, idx in enumerate(chunks):
        block = rets.iloc[idx]
        c = block.corr().abs()
        # max off-diagonal abs corr in this subperiod
        pair_max = 0.0
        pair_name = None
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                v = float(c.iloc[i, j])
                if v > pair_max:
                    pair_max, pair_name = v, f"{cols[i]} / {cols[j]}"
        worst = max(worst, pair_max)
        sub.append({
            "subperiod": ci + 1,
            "start": block.index[0].strftime("%Y-%m-%d"),
            "end": block.index[-1].strftime("%Y-%m-%d"),
            "max_abs_pair_corr": round(pair_max, 3),
            "max_pair": pair_name,
        })
    return {"subperiods": sub, "worst_pair_abs_corr": round(worst, 3)}


def deflated_sharpe(observed_sharpe: float, n_trials: int, n_days: int) -> dict:
    """Selection-bias haircut: how much of a searched Sharpe is likely luck.

    When you pick the best of ``n_trials`` combos, the winner's Sharpe is
    upward-biased — even pure noise produces a max-of-N that grows with N. The
    expected MAX Sharpe of N independent noise strategies (each ~N(0, SE)) is
    approximately ``SE * E[max of N standard normals]``, where
    ``SE = sqrt(1/n_days)`` (annualization cancels in the ratio we report). We
    return that expected-noise-max as a benchmark and the deflated Sharpe =
    observed - expected_noise_max. If deflated <= 0 the observed Sharpe is within
    what searching N combos would produce from noise alone. (Bailey/Lopez de
    Prado 'Deflated Sharpe Ratio', simplified to the IID case.)
    """
    if n_trials < 1 or n_days < 2:
        return {"expected_noise_max": 0.0, "deflated_sharpe": observed_sharpe}
    # E[max of N standard normals] approx (Gumbel): sqrt(2 ln N) adjusted.
    n = max(n_trials, 1)
    if n == 1:
        e_max_z = 0.0
    else:
        ln_n = np.log(n)
        # Standard approximation for the expected max of N iid N(0,1):
        e_max_z = (1 - np.euler_gamma) * _inv_norm(1 - 1.0 / n) + np.euler_gamma * _inv_norm(
            1 - 1.0 / (n * np.e)
        )
    se_daily = np.sqrt(1.0 / n_days)
    expected_noise_max = float(e_max_z * se_daily * np.sqrt(TRADING_DAYS))
    return {
        "n_trials": int(n_trials),
        "expected_noise_max": round(expected_noise_max, 3),
        "deflated_sharpe": round(observed_sharpe - expected_noise_max, 3),
    }


def _inv_norm(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation, no SciPy)."""
    p = min(max(p, 1e-9), 1 - 1e-9)
    a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
         1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0]
    b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
         6.680131188771972e1, -1.328068155288572e1]
    c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
         -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0]
    d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0,
         3.754408661907416e0]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / (
            (((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / (
            (((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / (
        ((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ---------------------------------------------------------------------------
# Top-level combine of a chosen file set.
# ---------------------------------------------------------------------------
def combine_files(
    paths: list[str],
    scheme: str,
    capital: float,
    vol_window: int = DEFAULT_VOL_WINDOW,
) -> dict:
    strategies = load_many(paths, strict=True)
    rets = align_returns(strategies)
    weights = build_weights(rets, scheme, vol_window)
    port_ret = combine_returns(rets, weights)
    idx = port_ret.index  # the active (post-warm-up) traded window

    # Correlation on the full shared window (uses all overlap, more robust).
    corr = rets.corr()
    corr_stability = correlation_stability(rets, n_subperiods=3)
    combined_stats = stats_from_daily(port_ret, capital)

    # Single-strategy comparison: re-score each leg on the SAME active window the
    # portfolio actually trades, so 'combined vs best single' is apples-to-apples.
    per_strategy = {}
    for col in rets.columns:
        per_strategy[col] = stats_from_daily(rets.loc[idx, col], capital)
    best_single_label = max(per_strategy, key=lambda k: per_strategy[k]["sharpe_daily"])

    combined_equity = equity_from_returns(port_ret, capital)
    equity_curve = [
        {"date": d.strftime("%Y-%m-%dT%H:%M:%S"), "equity": round(float(v), 4)}
        for d, v in combined_equity.items()
    ]

    corr_labels = list(corr.columns)
    avg_w = avg_weight_distribution(weights)

    return {
        "type": "portfolio",  # so dashboard.py classifies it without a filename check
        "scheme": scheme,
        "vol_window": vol_window if scheme in ("inverse_vol", "risk_parity") else None,
        "lag_rule": "weights shift(1): day T uses data through T-1",
        "members": [{"label": s.label, "source": s.source} for s in strategies],
        "shared_window": {
            "start": rets.index[0].strftime("%Y-%m-%d"),
            "end": rets.index[-1].strftime("%Y-%m-%d"),
            "days": int(rets.shape[0]),
        },
        "correlation_matrix": {
            row: {col: round(float(corr.loc[row, col]), 3) for col in corr.columns}
            for row in corr.index
        },
        "correlation_stability": corr_stability,
        # dashboard.py heatmap aliases (labels + 2D matrix in the same order).
        "corr_labels": corr_labels,
        "corr_matrix": [
            [round(float(corr.loc[r, c]), 3) for c in corr_labels]
            for r in corr_labels
        ],
        "avg_weight_distribution": avg_w,
        "weights": avg_w,  # dashboard.py reads "weights" for the constituent bars
        # dashboard.py reads top-level "stats" for the headline Sharpe/Calmar card.
        "stats": combined_stats,
        "combined": combined_stats,
        "per_strategy": per_strategy,
        "best_single": {"label": best_single_label, **per_strategy[best_single_label]},
        "beats_best_single": (
            combined_stats["sharpe_daily"]
            > per_strategy[best_single_label]["sharpe_daily"]
        ),
        "equity_curve": equity_curve,
    }


# ---------------------------------------------------------------------------
# Subset search — low-correlation combo maximizing Sharpe (NOT brute force).
# ---------------------------------------------------------------------------
def _combined_sharpe(
    rets: pd.DataFrame, cols: list[str], scheme: str, vol_window: int, capital: float
) -> float:
    sub = rets[cols]
    weights = build_weights(sub, scheme, vol_window)
    port_ret = combine_returns(sub, weights)
    return stats_from_daily(port_ret, capital)["sharpe_daily"]


def _strategy_suffix(label: str, strategy_names: list[str]) -> str | None:
    """Return which of ``strategy_names`` a ``<ticker>_<strategy>`` label ends in.

    Labels and tickers both contain underscores (XLF_vol_target_trend), so we
    can't naive-split on '_'. Match on the strategy SUFFIX instead.
    """
    for s in strategy_names:
        if label == s or label.endswith(f"_{s}"):
            return s
    return None


def search_subset(
    paths: list[str],
    scheme: str,
    capital: float,
    max_size: int = 5,
    corr_cap: float = 0.6,
    vol_window: int = DEFAULT_VOL_WINDOW,
    method: str = "greedy",
    min_trades: int = 10,
    exclude_strategy: list[str] | None = None,
) -> dict:
    """Find a low-correlation subset that maximizes combined Sharpe.

    Four anti-snooping / quality guardrails:
      * ``min_trades`` — legs with fewer than this many trades are dropped from
        the candidate pool BEFORE searching. A 1-5 trade "strategy" is really
        buy-and-hold; its daily series can show a spurious Sharpe>=2 (audited:
        the OOS single-name Sharpe>=2 results were all 1-5 trade artifacts). We
        exclude them so a combo can't ride a thin-trade leg, and report which.
      * ``exclude_strategy`` — drop whole strategy families by suffix (e.g.
        ``breakout``, which has NEGATIVE mean IS Sharpe and shouldn't pollute the
        pool). Matched on the strategy suffix so XLF_breakout etc. are caught.
      * ``corr_cap`` — a strategy is only ADDED if its max pairwise correlation
        with already-chosen members stays <= cap. Diversification is the point;
        stacking near-duplicates inflates in-sample Sharpe and fails OOS.
      * ``max_size`` — caps how large the combo can grow, and (for the
        exhaustive method) bounds the number of combos tried.

    ``method='greedy'`` (default): start from the single best Sharpe, then
    repeatedly add the candidate that (a) respects the corr cap and (b) most
    improves combined Sharpe; stop when nothing improves it or max_size is hit.
    O(N^2) combos tried — cheap and resistant to data-snooping.

    ``method='exhaustive'``: try every combo up to max_size whose internal
    pairwise correlations all respect the cap. Use only for small pools; the
    combo count is reported so the search space is auditable.
    """
    all_strategies = load_many(paths, strict=False)
    exclude_strategy = exclude_strategy or []

    # Strategy-family exclusion: drop whole strategies (e.g. negative-edge
    # 'breakout') by suffix before the thin-trade check.
    excluded_by_strategy = [
        {"label": s.label}
        for s in all_strategies
        if _strategy_suffix(s.label, exclude_strategy) is not None
    ]
    pool = [
        s
        for s in all_strategies
        if _strategy_suffix(s.label, exclude_strategy) is None
    ]

    # Thin-trade guard: drop legs below min_trades from the candidate pool.
    excluded = [
        {"label": s.label, "num_trades": s.num_trades}
        for s in pool
        if s.num_trades < min_trades
    ]
    strategies = [s for s in pool if s.num_trades >= min_trades]
    if len(strategies) < 1:
        raise ValueError(
            f"No strategy has >= {min_trades} trades after exclusions — every "
            f"candidate is a thin-trade artifact. Lower --min-trades only if you "
            f"accept that risk."
        )

    rets = align_returns(strategies)
    corr = rets.corr()
    labels = list(rets.columns)
    trades_by_label = {s.label: s.num_trades for s in strategies}

    combos_tried = 0

    if method == "greedy":
        # Seed with the best standalone Sharpe.
        singles = {c: _combined_sharpe(rets, [c], scheme, vol_window, capital) for c in labels}
        combos_tried += len(labels)
        chosen = [max(singles, key=singles.get)]
        best_sharpe = singles[chosen[0]]

        improved = True
        while improved and len(chosen) < max_size:
            improved = False
            best_candidate = None
            best_candidate_sharpe = best_sharpe
            for cand in labels:
                if cand in chosen:
                    continue
                # Enforce the correlation cap against everything already chosen.
                if any(abs(corr.loc[cand, c]) > corr_cap for c in chosen):
                    continue
                combos_tried += 1
                trial_sharpe = _combined_sharpe(
                    rets, chosen + [cand], scheme, vol_window, capital
                )
                if trial_sharpe > best_candidate_sharpe:
                    best_candidate = cand
                    best_candidate_sharpe = trial_sharpe
            if best_candidate is not None:
                chosen.append(best_candidate)
                best_sharpe = best_candidate_sharpe
                improved = True

    elif method == "exhaustive":
        best_sharpe = -np.inf
        chosen = []
        for size in range(1, max_size + 1):
            for combo in itertools.combinations(labels, size):
                # Skip combos that violate the correlation cap internally.
                if size > 1:
                    pairs = itertools.combinations(combo, 2)
                    if any(abs(corr.loc[a, b]) > corr_cap for a, b in pairs):
                        continue
                combos_tried += 1
                s = _combined_sharpe(rets, list(combo), scheme, vol_window, capital)
                if s > best_sharpe:
                    best_sharpe = s
                    chosen = list(combo)
    else:
        raise ValueError(f"unknown search method {method!r} (greedy or exhaustive)")

    chosen_paths = [s.source for s in strategies if s.label in chosen]
    detail = combine_files(chosen_paths, scheme, capital, vol_window)
    # Selection-bias haircut: deflate the winning Sharpe by what searching this
    # many combos would produce from noise alone.
    deflated = deflated_sharpe(
        detail["combined"]["sharpe_daily"],
        combos_tried,
        detail["combined"]["trading_days"],
    )
    detail["search"] = {
        "method": method,
        "pool_size": len(labels),
        "max_size": max_size,
        "corr_cap": corr_cap,
        "min_trades": min_trades,
        "exclude_strategy": exclude_strategy,
        "combos_tried": combos_tried,
        "chosen": chosen,
        "chosen_num_trades": {c: trades_by_label[c] for c in chosen},
        "excluded_thin_trade": excluded,
        "excluded_by_strategy_count": len(excluded_by_strategy),
        "deflated_sharpe": deflated,
    }
    return detail


# ---------------------------------------------------------------------------
# Matched-window comparison — score combos APPLES-TO-APPLES.
# ---------------------------------------------------------------------------
def compare_combos(
    combos: dict[str, list[str]],
    base_dir: str,
    scheme: str,
    capital: float,
    vol_window: int = DEFAULT_VOL_WINDOW,
) -> dict:
    """Score several named combos on the SHARED COMMON window so any 'winner' is
    apples-to-apples.

    Why this exists: ``combine_returns`` trims each combo to its OWN active
    window (post-warm-up). Two combos with different warm-up lengths (e.g. one
    has a leg whose rolling-vol window fills later) then get scored on different
    day-counts — which can make a combo look better purely because it ran a few
    extra days. (This is exactly how a combo read 1.45 on 224 days but 1.00 on
    the 195-day window a rival used.) Here we intersect all combos' active
    indices to a single common window and re-score everyone on it, and we print
    the 95% Sharpe CI so nobody over-reads a point estimate on a short window.
    """
    # First pass: build each combo's active-window daily return series.
    series: dict[str, pd.Series] = {}
    for name, members in combos.items():
        paths = [os.path.join(base_dir, f"{m}.json") for m in members]
        strategies = load_many(paths, strict=True)
        rets = align_returns(strategies)
        weights = build_weights(rets, scheme, vol_window)
        series[name] = combine_returns(rets, weights)

    # Common window = intersection of every combo's active index.
    common = None
    for s in series.values():
        common = s.index if common is None else common.intersection(s.index)
    if common is None or len(common) < 2:
        raise ValueError("Combos share fewer than 2 common days — cannot compare.")

    rows = {}
    for name, s in series.items():
        on_common = s.loc[common]
        st = stats_from_daily(on_common, capital)
        rows[name] = {
            "members": combos[name],
            "sharpe_daily": st["sharpe_daily"],
            "sharpe_ci95": st["sharpe_ci95"],
            "cagr_pct": st["cagr_pct"],
            "max_drawdown_pct": st["max_drawdown_pct"],
        }

    # Pairwise paired t-test on daily-return differences (are they distinguishable?).
    names = list(series.keys())
    pairwise = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = series[names[i]].loc[common]
            b = series[names[j]].loc[common]
            d = a - b
            if d.std(ddof=1) > 0:
                t = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
            else:
                t = 0.0
            pairwise[f"{names[i]} vs {names[j]}"] = {
                "sharpe_diff": round(rows[names[i]]["sharpe_daily"]
                                     - rows[names[j]]["sharpe_daily"], 3),
                "paired_t": round(t, 3),
                "distinguishable": abs(t) > 1.96,
            }

    return {
        "scheme": scheme,
        "common_window": {
            "start": common[0].strftime("%Y-%m-%d"),
            "end": common[-1].strftime("%Y-%m-%d"),
            "days": int(len(common)),
        },
        "combos": rows,
        "pairwise": pairwise,
    }


# ---------------------------------------------------------------------------
# Output helpers.
# ---------------------------------------------------------------------------
def _print_compare(report: dict) -> None:
    cw = report["common_window"]
    print(f"\n=== Compare ({report['scheme']}) on SHARED window "
          f"{cw['start']} -> {cw['end']} ({cw['days']} days) ===")
    print(f"{'combo':<16}{'Sharpe':>8}{'95% CI':>20}{'CAGR%':>9}{'maxDD%':>9}")
    for name, row in report["combos"].items():
        lo, hi = row["sharpe_ci95"]
        print(f"{name:<16}{row['sharpe_daily']:>8.3f}"
              f"{f'[{lo:.2f}, {hi:.2f}]':>20}"
              f"{row['cagr_pct']:>9.2f}{row['max_drawdown_pct']:>9.2f}")
    print("\nPairwise (paired t on daily diffs; |t|>1.96 => distinguishable):")
    for pair, info in report["pairwise"].items():
        verdict = "DISTINGUISHABLE" if info["distinguishable"] else "indistinguishable"
        print(f"  {pair}: ΔSharpe {info['sharpe_diff']:+.3f}  "
              f"t={info['paired_t']:+.3f}  {verdict}")
    print()


def _print_summary(report: dict) -> None:
    print(f"\n=== Portfolio: {report['scheme']} ===")
    sw = report["shared_window"]
    print(f"Shared window: {sw['start']} -> {sw['end']} ({sw['days']} days)")
    if "search" in report:
        s = report["search"]
        print(
            f"Search ({s['method']}): pool={s['pool_size']} "
            f"max_size={s['max_size']} corr_cap={s['corr_cap']} "
            f"min_trades={s['min_trades']} combos_tried={s['combos_tried']}"
        )
        chosen_str = ", ".join(
            f"{c} ({s['chosen_num_trades'][c]}t)" for c in s["chosen"]
        )
        print(f"Chosen: {chosen_str}")
        if s["excluded_thin_trade"]:
            ex = ", ".join(
                f"{e['label']} ({e['num_trades']}t)" for e in s["excluded_thin_trade"]
            )
            print(f"Excluded (< {s['min_trades']} trades, thin-trade artifacts): {ex}")
    print("\nCorrelation matrix:")
    corr = report["correlation_matrix"]
    cols = list(corr.keys())
    header = "".ljust(22) + "".join(c[:10].rjust(11) for c in cols)
    print(header)
    for row in cols:
        line = row[:21].ljust(22) + "".join(
            f"{corr[row][c]:>11.3f}" for c in cols
        )
        print(line)
    print("\nAvg weight distribution:")
    for label, w in report["avg_weight_distribution"].items():
        print(f"  {label:<28} {w:>7.4f}")
    c = report["combined"]
    print("\nCombined:")
    print(
        f"  Sharpe={c['sharpe_daily']}  CAGR={c['cagr_pct']}%  "
        f"maxDD={c['max_drawdown_pct']}%  totalRet={c['total_return_pct']}%"
    )
    bs = report["best_single"]
    print(
        f"Best single ({bs['label']}): Sharpe={bs['sharpe_daily']}  "
        f"CAGR={bs['cagr_pct']}%  maxDD={bs['max_drawdown_pct']}%"
    )
    verdict = "BEATS" if report["beats_best_single"] else "DOES NOT BEAT"
    print(f"Combined {verdict} the best single strategy on Sharpe.\n")


def _write(report: dict, name: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"portfolio_{name}.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def _expand(paths: list[str]) -> list[str]:
    out: list[str] = []
    for p in paths:
        matches = glob.glob(p)
        out.extend(sorted(matches) if matches else [p])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine single-strategy backtests.")
    sub = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("paths", nargs="+", help="result JSON files (globs ok)")
    common.add_argument(
        "--scheme",
        default="inverse_vol",
        choices=["equal_dollar", "inverse_vol", "risk_parity"],
        help="weighting scheme (all 1-day-lagged)",
    )
    common.add_argument("--capital", type=float, default=10000.0)
    common.add_argument("--vol-window", type=int, default=DEFAULT_VOL_WINDOW)
    common.add_argument("--name", default="combo", help="output name suffix")
    common.add_argument("--out-dir", default="results", help="output directory")

    p_comb = sub.add_parser("combine", parents=[common], help="combine an explicit set")

    p_srch = sub.add_parser("search", parents=[common], help="search low-corr subset")
    p_srch.add_argument("--max-size", type=int, default=5)
    p_srch.add_argument("--corr-cap", type=float, default=0.6)
    p_srch.add_argument(
        "--min-trades",
        type=int,
        default=10,
        help="drop legs with fewer trades (thin-trade buy-and-hold artifacts)",
    )
    p_srch.add_argument(
        "--exclude-strategy",
        nargs="*",
        default=[],
        help="drop whole strategy families by suffix (e.g. breakout)",
    )
    p_srch.add_argument(
        "--method", default="greedy", choices=["greedy", "exhaustive"]
    )

    # compare: score several named combos on the SHARED common window.
    p_cmp = sub.add_parser(
        "compare",
        help="score named combos on a shared common window (apples-to-apples)",
    )
    p_cmp.add_argument(
        "combos",
        nargs="+",
        help="each as name=member1,member2,... (members are <ticker>_<strategy>)",
    )
    p_cmp.add_argument("--base-dir", default="results",
                       help="dir holding the member result JSONs (e.g. results/oos)")
    p_cmp.add_argument("--scheme", default="inverse_vol",
                       choices=["equal_dollar", "inverse_vol", "risk_parity"])
    p_cmp.add_argument("--capital", type=float, default=10000.0)
    p_cmp.add_argument("--vol-window", type=int, default=DEFAULT_VOL_WINDOW)
    p_cmp.add_argument("--name", default="compare")
    p_cmp.add_argument("--out-dir", default="results")

    args = parser.parse_args()

    if args.mode == "compare":
        combos = {}
        for spec in args.combos:
            if "=" not in spec:
                parser.error(f"combo spec must be name=members: got {spec!r}")
            name, members = spec.split("=", 1)
            combos[name] = [m.strip() for m in members.split(",") if m.strip()]
        report = compare_combos(
            combos, args.base_dir, args.scheme, args.capital, args.vol_window
        )
        _print_compare(report)
        out_path = _write(report, args.name, args.out_dir)
        print(f"Wrote {out_path}")
        return

    paths = _expand(args.paths)
    if args.mode == "combine":
        report = combine_files(paths, args.scheme, args.capital, args.vol_window)
    else:
        report = search_subset(
            paths,
            args.scheme,
            args.capital,
            max_size=args.max_size,
            corr_cap=args.corr_cap,
            vol_window=args.vol_window,
            method=args.method,
            min_trades=args.min_trades,
            exclude_strategy=args.exclude_strategy,
        )

    _print_summary(report)
    out_path = _write(report, args.name, args.out_dir)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
