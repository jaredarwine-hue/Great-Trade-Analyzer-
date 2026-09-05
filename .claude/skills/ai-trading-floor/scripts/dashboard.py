#!/usr/bin/env python3
"""Build ONE interactive, OFFLINE dashboard HTML for ALL backtests in ./results.

Aggregates every ``results/*.json`` (single-strategy backtests) AND every
``results/portfolio_*.json`` (combined portfolios) into a single self-contained
``reports/dashboard.html``:

- LEFT: a compressed, collapsible tree — Portfolios pinned on top, then
  single strategies grouped by ASSET CLASS -> TICKER -> strategy. Search + sort
  (Sharpe / Calmar / Return) keep it scannable at hundreds of strategies.
- RIGHT: stat cards (Sharpe + Calmar headline) and a chart with
  STRATEGY-SPECIFIC overlays (Bollinger bands, Donchian channels, RSI panel,
  realized-vol panel, SMAs) + entry/exit markers + the equity curve. Combined
  portfolios render their equity curve + constituent weights + correlation heatmap.

Plotly is embedded inline so the file is fully double-clickable OFFLINE. Price
series are embedded ONCE per ticker (shared map) so big parameter sweeps stay
small. No project imports; nothing reads or writes the repo.

Usage:
    python dashboard.py
    python dashboard.py --results-dir results --out reports/dashboard.html
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from plotly.offline import get_plotlyjs

SCHEMA_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]

# Map each universe ticker to an asset class (drives the sidebar grouping).
ASSET_CLASS = {
    **{t: "Equity Index" for t in ["SPY", "QQQ", "IWM", "DIA"]},
    **{t: "Sectors" for t in ["XLK", "XLE", "XLF", "XLV", "XLP", "XLU", "XLI", "XLB"]},
    **{t: "Stocks" for t in ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "JPM", "XOM",
                              "UNH", "JNJ", "PG", "KO", "CAT", "HD", "WMT", "V"]},
    **{t: "Bonds" for t in ["TLT", "IEF", "SHY"]},
    **{t: "Metals" for t in ["GLD", "SLV"]},
    **{t: "Commodities" for t in ["DBC"]},
    **{t: "FX / Dollar" for t in ["UUP"]},
}
# Order asset-class sections appear in (Portfolios is always first, handled in JS).
CLASS_ORDER = ["Universe / Rotation", "Equity Index", "Sectors", "Stocks", "Bonds",
               "Metals", "Commodities", "FX / Dollar", "Other"]


def _param_str(params: dict) -> str:
    if not params:
        return ""
    return " · ".join(f"{k} {v}" for k, v in params.items())


def _calmar(stats: dict) -> float:
    """Calmar = CAGR / |max drawdown| (both in %). 0 if drawdown is ~0."""
    cagr = stats.get("cagr_pct")
    mdd = stats.get("max_drawdown_pct")
    if cagr is None or not mdd:
        return 0.0
    return round(cagr / abs(mdd), 3)


def _price_payload(data_file: str | None, ticker: str) -> dict:
    """Load the price parquet for candle + overlay rendering (best-effort)."""
    candidates = []
    if data_file:
        candidates.append(Path(data_file))
    candidates.append(Path("data") / f"{ticker}.parquet")
    for path in candidates:
        if path.exists():
            df = pd.read_parquet(path)
            if list(df.columns) != SCHEMA_COLUMNS:
                continue
            dates = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
            return {
                "dates": dates.tolist(),
                "open": df["Open"].round(2).tolist(),
                "high": df["High"].round(2).tolist(),
                "low": df["Low"].round(2).tolist(),
                "close": df["Close"].round(2).tolist(),
                "volume": df["Volume"].round(0).tolist(),
            }
    return {}


def _equity(payload: dict) -> dict:
    curve = payload.get("equity_curve", [])
    return {
        "dates": [pt["date"] for pt in curve],
        "values": [round(float(pt["equity"]), 2) for pt in curve],
    }


def _is_portfolio(payload: dict, path: Path) -> bool:
    return (path.name.startswith("portfolio_")
            or payload.get("type") == "portfolio"
            or "legs" in payload or "constituents" in payload)


def load_strategy(results_path: Path) -> dict | None:
    """Read one results.json into the compact dict the dashboard embeds."""
    try:
        with open(results_path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  WARNING: skipping {results_path} ({exc})")
        return None
    if not payload.get("equity_curve"):
        return None  # batch summaries / manifests have no curve

    stats = dict(payload.get("stats", {}))
    stats["calmar"] = _calmar(stats)
    equity = _equity(payload)

    if _is_portfolio(payload, results_path):
        # Combined portfolio: stats live under "combined"; carry constituent detail.
        combined = payload.get("combined") or payload.get("stats") or {}
        stats = dict(combined)
        stats["calmar"] = _calmar(stats)

        # correlation_matrix is a nested {label:{label:val}} dict -> labels + matrix.
        corr = payload.get("correlation_matrix") or payload.get("correlation")
        corr_out = None
        if isinstance(corr, dict) and corr and isinstance(next(iter(corr.values())), dict):
            labels = list(corr.keys())
            corr_out = {"labels": labels,
                        "matrix": [[corr[r].get(c) for c in labels] for r in labels]}
        members = payload.get("members") or payload.get("legs") or payload.get("constituents") or []
        leg_labels = [m.get("label") if isinstance(m, dict) else m for m in members]
        details = {
            "legs": leg_labels,
            "weights": payload.get("avg_weight_distribution") or payload.get("weights"),
            "scheme": payload.get("scheme"),
            "correlation": corr_out,
            "beats_best_single": payload.get("beats_best_single"),
            "best_single": payload.get("best_single"),
        }
        sw = payload.get("shared_window") or {}
        date_range = payload.get("date_range") or (
            [sw.get("start"), sw.get("end")] if sw.get("start") else
            ([equity["dates"][0][:10], equity["dates"][-1][:10]] if equity["dates"] else []))
        name = payload.get("name") or results_path.stem.replace("portfolio_", "")
        toks = name.lower().replace("-", "_").split("_")
        window = ("out-of-sample" if "oos" in toks
                  else "in-sample" if "is" in toks else "")
        return {
            "kind": "portfolio",
            "id": f"portfolio::{name}",
            "name": name,
            "window": window,
            "asset_class": "Portfolios",
            "ticker": "",
            "strategy": "portfolio",
            "params": {},
            "param_str": payload.get("scheme", ""),
            "bars": combined.get("trading_days", len(equity["values"])),
            "date_range": date_range,
            "stats": stats,
            "sharpe": stats.get("sharpe_daily", 0.0),
            "calmar": stats.get("calmar", 0.0),
            "total_return_pct": stats.get("total_return_pct", 0.0),
            "equity": equity,
            "details": details,
        }

    ticker = payload.get("ticker", results_path.stem)
    strategy = payload.get("strategy", "strategy")
    params = payload.get("params", {})
    # Universe/rotation strategies trade many names — group them on their own.
    if ticker in ("PORT", "UNIV") or "rotation" in strategy:
        asset_class = "Universe / Rotation"
    else:
        asset_class = ASSET_CLASS.get(ticker, "Other")
    trades = [{
        "entry_date": t.get("entry_date"), "exit_date": t.get("exit_date"),
        "entry_price": t.get("entry_price"), "exit_price": t.get("exit_price"),
        "return_pct": t.get("return_pct"), "r_multiple": t.get("r_multiple"),
    } for t in payload.get("trades", [])]
    return {
        "kind": "strategy",
        "id": f"{ticker}_{strategy}_{_param_str(params)}",
        "ticker": ticker,
        "asset_class": asset_class,
        "strategy": strategy,
        "params": params,
        "param_str": _param_str(params),
        "bars": payload.get("bars", len(equity["values"])),
        "date_range": payload.get("date_range", []),
        "stats": stats,
        "sharpe": stats.get("sharpe_daily", 0.0),
        "calmar": stats.get("calmar", 0.0),
        "total_return_pct": stats.get("total_return_pct", 0.0),
        "equity": equity,
        "trades": trades,
        "_data_file": payload.get("data_file"),
    }


def _recover_portfolios(current: list[dict], args: argparse.Namespace) -> list[dict]:
    """Persist portfolios to a cache and restore any that are momentarily absent.

    Other tools (e.g. run_universe.py) regenerate the dashboard right after the
    single-strategy batch — a window where the Orchestrator's portfolio_*.json may
    not exist yet. Without this, the Combined Portfolios section flickers away.

    Fix: cache each portfolio (self-contained) and ACCUMULATE — a portfolio saved on a
    prior run keeps showing even after its live ``results/`` file is overwritten or a rerun
    blanks the folder, so good portfolios never disappear. Cached-only entries are tagged
    ``cached`` (the user curates them with the ★ Save / ✕ remove controls). We deliberately
    do NOT auto-prune; removing a portfolio is an explicit user action.
    """
    if args.results:  # explicit subset mode — don't inject the cache
        return []
    cache = Path(args.results_dir) / ".portfolio_cache"
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError:
        return []
    now = set()
    for p in current:  # refresh the cache copy of every live portfolio
        now.add(p["name"])
        try:
            (cache / f"{p['name']}.json").write_text(json.dumps(p), encoding="utf-8")
        except OSError:
            pass
    extra = []  # surface portfolios saved earlier that aren't in results/ right now
    for f in sorted(cache.glob("*.json")):
        if f.stem in now:
            continue
        try:
            p = json.loads(f.read_text())
            p["cached"] = True
            extra.append(p)
        except (OSError, json.JSONDecodeError):
            pass
    return extra


def _view_of(p: dict) -> dict:
    """The renderable slice of a per-window portfolio (stats/equity/details)."""
    return {
        "stats": p.get("stats", {}),
        "equity": p.get("equity", {}),
        "details": p.get("details", {}),
        "bars": p.get("bars", 0),
        "date_range": p.get("date_range", []),
    }


def _merge_portfolio_pairs(ports: list[dict]) -> list[dict]:
    """Collapse ``<base>_is`` and ``<base>_oos`` into ONE entry with both views.

    A single strategy is only tested in-sample; the IS/OOS split lives at the
    PORTFOLIO level. So each portfolio becomes one row carrying both an
    'in-sample' and 'out-of-sample' view that the UI toggles between.
    """
    groups: dict[str, dict[str, dict]] = {}
    for p in ports:
        base = re.sub(r"_(is|oos)$", "", p["name"], flags=re.IGNORECASE)
        win = p.get("window") or "in-sample"  # unlabeled portfolios = in-sample
        groups.setdefault(base, {})[win] = p

    merged = []
    for base, by_win in groups.items():
        is_p, oos_p = by_win.get("in-sample"), by_win.get("out-of-sample")
        views = {}
        if is_p:
            views["in-sample"] = _view_of(is_p)
        if oos_p:
            views["out-of-sample"] = _view_of(oos_p)
        if not views:
            continue
        primary = is_p or oos_p
        head = views.get("in-sample") or views.get("out-of-sample")
        merged.append({
            "kind": "portfolio",
            "id": f"portfolio::{base}",
            "name": base,
            "ticker": "",
            "strategy": "portfolio",
            "asset_class": "Portfolios",
            "param_str": primary.get("param_str", ""),
            "cached": bool((is_p and is_p.get("cached")) or (oos_p and oos_p.get("cached"))),
            # Sidebar/sort keys use the honest IN-SAMPLE numbers.
            "sharpe": head["stats"].get("sharpe_daily", 0.0),
            "calmar": head["stats"].get("calmar", 0.0),
            "total_return_pct": head["stats"].get("total_return_pct", 0.0),
            "is_sharpe": views["in-sample"]["stats"].get("sharpe_daily") if "in-sample" in views else None,
            "oos_sharpe": views["out-of-sample"]["stats"].get("sharpe_daily") if "out-of-sample" in views else None,
            "views": views,
        })
    return merged


def collect_strategies(args: argparse.Namespace) -> list[dict]:
    if args.results:
        paths = [Path(p) for p in args.results]
    else:
        paths = sorted(p for p in Path(args.results_dir).glob("*.json")
                       if not p.name.startswith("_"))
    out = []
    for path in paths:
        loaded = load_strategy(path)
        if loaded:
            out.append(loaded)
    # Keep portfolios from vanishing when another process regenerated us mid-rerun.
    out.extend(_recover_portfolios([s for s in out if s["kind"] == "portfolio"], args))
    # Merge each portfolio's IS + OOS files into one toggle-able entry.
    strat = [s for s in out if s["kind"] != "portfolio"]
    merged = _merge_portfolio_pairs([s for s in out if s["kind"] == "portfolio"])
    out = strat + merged
    out.sort(key=lambda s: s.get("sharpe", 0.0), reverse=True)
    return out


# ----------------------------------------------------------------------------- #
#  HTML template — placeholder tokens (NOT str.format) so JS braces stay single.
# ----------------------------------------------------------------------------- #
PAGE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark"><meta name="theme-color" content="#0a0e14">
<title>AI Trading Floor — Strategy Dashboard</title>
<script>__PLOTLYJS__</script>
<style>
:root{
  --bg:#0a0e14;--panel:#10151e;--panel-2:#141b26;--raise:#19212e;
  --row-hover:#161d28;--line:#222b39;--line-soft:#1a2230;
  --txt:#e7edf4;--txt-2:#aeb9c7;--dim:#7e8a9a;--faint:#586273;
  --up:#2fd49d;--up-bg:rgba(47,212,157,.12);
  --down:#f26d78;--down-bg:rgba(242,109,120,.12);
  --blue:#4c9fff;--gold:#ffc24b;--radius:10px;
  --mono:ui-monospace,'SF Mono',SFMono-Regular,'JetBrains Mono',Menlo,Monaco,'Cascadia Code',Consolas,monospace;
  --sans:system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
}
*{box-sizing:border-box}
html{color-scheme:dark}
body{margin:0;height:100vh;overflow:hidden;color:var(--txt);
  font-family:var(--sans);font-size:13px;line-height:1.45;
  font-variant-numeric:tabular-nums;-webkit-font-smoothing:antialiased;
  background:radial-gradient(1100px 560px at 82% -12%,#121a28 0%,transparent 58%),var(--bg)}

.top{display:flex;align-items:center;gap:14px;padding:0 18px;height:56px;
  background:linear-gradient(180deg,#121925,#0d121b);
  border-bottom:1px solid var(--line);position:relative;z-index:20}
.brand{display:flex;align-items:center;gap:10px;flex:none}
.brand .mark{width:20px;height:20px;display:block}
.top h1{font-size:14px;font-weight:650;margin:0;color:#fff;letter-spacing:.2px;white-space:nowrap}
.top h1 .accent{color:var(--blue)}
.meta{color:var(--dim);font-size:12px;font-family:var(--mono);letter-spacing:.2px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0}
.meta b{color:var(--txt-2);font-weight:600}
.spacer{margin-left:auto}
.search{background:var(--panel);border:1px solid var(--line);border-radius:8px;
  color:var(--txt);font:12px var(--mono);padding:6px 10px;width:190px;outline:none}
.search:focus{border-color:var(--blue)}
.sel{background:var(--panel);border:1px solid var(--line);border-radius:8px;
  color:var(--txt-2);font:600 12px var(--sans);padding:6px 8px;outline:none;cursor:pointer}
.filters{display:inline-flex;background:var(--panel);border:1px solid var(--line);
  border-radius:9px;padding:3px;gap:2px;flex:none}
.btn{background:transparent;color:var(--dim);border:0;border-radius:7px;
  padding:6px 12px;cursor:pointer;font:600 12px/1 var(--sans);letter-spacing:.2px;
  transition:background .18s,color .18s}
.btn:hover{color:var(--txt-2);background:var(--row-hover)}
.btn.on{background:var(--raise);color:#fff;box-shadow:inset 0 1px 0 rgba(255,255,255,.05)}
.btn:focus-visible,.tk-row:focus-visible,.srow:focus-visible{outline:2px solid var(--blue);outline-offset:-2px}

.wrap{display:flex;height:calc(100vh - 56px)}
.left{width:344px;min-width:344px;overflow-y:auto;
  background:linear-gradient(180deg,#0c1118,#0a0e14);border-right:1px solid var(--line)}
.right{flex:1;display:flex;flex-direction:column;padding:16px 18px 10px;overflow-y:auto}

.sec{padding:13px 16px 5px;font:700 10px/1 var(--sans);letter-spacing:.13em;
  text-transform:uppercase;color:var(--faint);position:sticky;top:0;
  background:linear-gradient(180deg,#0c1118,#0c1118f0);backdrop-filter:blur(4px);z-index:2}
.tk-row{display:flex;align-items:center;gap:9px;padding:9px 14px;cursor:pointer;
  border-bottom:1px solid var(--line-soft);transition:background .16s}
.tk-row:hover{background:var(--row-hover)}
.chev{color:var(--faint);font-size:9px;width:9px;transition:transform .16s;flex:none}
.tk-row.open .chev{transform:rotate(90deg)}
.tk-name{font:600 13px var(--mono);color:#fff;letter-spacing:.2px}
.tk-right{margin-left:auto;display:flex;align-items:center;gap:9px}
.tk-count{color:var(--faint);font:11px var(--mono)}
.spill{font:600 11px var(--mono);color:var(--dim)}
.children{display:none}
.children.open{display:block}
.srow{display:flex;align-items:center;gap:8px;padding:8px 14px 8px 32px;cursor:pointer;
  border-bottom:1px solid var(--line-soft);border-left:3px solid transparent;
  transition:background .14s,border-color .14s}
.srow:hover{background:var(--row-hover)}
.srow.sel{background:linear-gradient(90deg,rgba(76,159,255,.10),transparent 72%);border-left-color:var(--gold)}
.srow.port{padding-left:14px}
.srow .nm{font:600 12.5px var(--mono);color:var(--txt)}
.srow .nm .strat{color:var(--blue);font-weight:500}
.srow .pm{color:var(--faint);font:11px var(--mono);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:2px}
.srow .info{min-width:0}
.srow .sp{margin-left:auto;display:flex;align-items:center;gap:7px;flex:none}
.ret{font:600 12px var(--mono);padding:3px 7px;border-radius:6px;white-space:nowrap}
.rpos{color:var(--up)} .rneg{color:var(--down)}
.ret.rpos{background:var(--up-bg)} .ret.rneg{background:var(--down-bg)}
.star{cursor:pointer;color:var(--faint);font-size:14px;line-height:1;flex:none;transition:color .15s,transform .15s}
.star:hover{color:var(--gold);transform:scale(1.18)}
.star.on{color:var(--gold)}

.hd{color:#fff;font:650 18px/1.3 var(--sans);padding:2px 2px 0;letter-spacing:.2px}
.hd .tk{font-family:var(--mono)} .hd .params{color:var(--dim);font-weight:400;font-size:13px;font-family:var(--mono)}
.sub{color:var(--dim);font-size:12px;font-family:var(--mono);padding:6px 2px 12px}
.vtog{display:inline-flex;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:2px;gap:2px;vertical-align:middle}
.vbtn{background:transparent;color:var(--dim);border:0;border-radius:6px;padding:4px 12px;cursor:pointer;font:600 11px var(--sans);letter-spacing:.2px;transition:background .15s,color .15s}
.vbtn:hover{color:var(--txt-2)} .vbtn.on{background:var(--raise);color:#fff}
.vbtn:focus-visible{outline:2px solid var(--blue);outline-offset:2px}
.cmp{margin-left:12px;font-family:var(--mono);color:var(--dim)}
.cmp b{color:var(--txt)} .cmp .gp{color:var(--up)} .cmp .gn{color:var(--down)}
.cmp .warn{color:var(--gold)}
.legsline{margin-top:7px;color:var(--faint);font-family:var(--mono);font-size:11px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(132px,1fr));gap:10px;padding:0 0 14px}
.card{position:relative;overflow:hidden;border-radius:var(--radius);
  background:linear-gradient(180deg,var(--panel-2),var(--panel));border:1px solid var(--line);
  padding:12px 14px;transition:border-color .18s,box-shadow .18s}
.card::before{content:'';position:absolute;left:0;top:0;width:100%;height:2px;
  background:linear-gradient(90deg,var(--blue),transparent);opacity:.55}
.card.hero::before{background:linear-gradient(90deg,var(--gold),transparent);opacity:.8}
.card:hover{border-color:#2c3a4e;box-shadow:0 6px 18px rgba(0,0,0,.35)}
.card .k{color:var(--dim);font-size:10.5px;font-weight:600;text-transform:uppercase;letter-spacing:.07em}
.card .v{font:650 19px/1.1 var(--mono);color:#fff;margin-top:6px;letter-spacing:.3px}
#chart{flex:1;min-height:560px;background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:6px 8px}
.empty{color:var(--dim);padding:48px;text-align:center}

.left::-webkit-scrollbar,.right::-webkit-scrollbar{width:10px}
.left::-webkit-scrollbar-thumb,.right::-webkit-scrollbar-thumb{
  background:#232d3c;border-radius:7px;border:2px solid transparent;background-clip:content-box}
.left::-webkit-scrollbar-thumb:hover,.right::-webkit-scrollbar-thumb:hover{background:#2f3c4f}
.left::-webkit-scrollbar-track,.right::-webkit-scrollbar-track{background:transparent}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style></head><body>
<div class="top">
<div class="brand">
<svg class="mark" viewBox="0 0 24 24" fill="none" aria-hidden="true">
<rect x="1" y="1" width="22" height="22" rx="6" fill="#10151e" stroke="#222b39"/>
<path d="M4 16 L9 11 L13 14 L20 6" stroke="#4c9fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="20" cy="6" r="2" fill="#ffc24b"/></svg>
<h1>AI Trading Floor <span class="accent">/</span> Dashboard</h1></div>
<span class="meta" id="summary"></span>
<span class="spacer"></span>
<input class="search" id="q" type="search" placeholder="filter ticker / strategy" aria-label="Filter">
<select class="sel" id="sortBy" aria-label="Sort by">
<option value="sharpe">Sort: Sharpe</option>
<option value="calmar">Sort: Calmar</option>
<option value="ret">Sort: Return</option>
</select>
<div class="filters" role="group" aria-label="Filter">
<button class="btn on" id="fAll" onclick="setF('all')">All</button>
<button class="btn" id="fWin" onclick="setF('win')">Profitable</button>
<button class="btn" id="fLose" onclick="setF('lose')">Losing</button>
</div></div>
<div class="wrap">
<div class="left"><div id="list"></div></div>
<div class="right">
<div class="hd" id="hd">Select a strategy</div>
<div class="sub" id="sub"></div>
<div class="cards" id="cards"></div>
<div id="chart"></div>
</div></div>
<script>
const DATA = __DATA__;
const PRICES = __PRICES__;
const CLASS_ORDER = __CLASSORDER__;
const G="#26a69a",R="#ef5350",BLUE="#4c9fff",GOLD="#ffc24b",ORANGE="#ff9f45",
PURPLE="#b48cff",TXT="#aeb9c7",DIM="#7e8a9a",LINE="#222b39",PAPER="rgba(0,0,0,0)",PLOT="#0c121b",
MONO="ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace";
let filt="all", sortKey="sharpe", query="", selId=null, openTk=new Set(), portView={};
let saved=new Set();try{saved=new Set(JSON.parse(localStorage.getItem('aitf_saved')||'[]'));}catch(e){}
function toggleSave(name,ev){if(ev)ev.stopPropagation();
  saved.has(name)?saved.delete(name):saved.add(name);
  try{localStorage.setItem('aitf_saved',JSON.stringify([...saved]));}catch(e){}
  render();}

/* ---------- indicators (computed in-browser from shared price arrays) ------ */
function sma(v,p){const o=Array(v.length).fill(null);let s=0;for(let i=0;i<v.length;i++){s+=v[i];if(i>=p)s-=v[i-p];if(i>=p-1)o[i]=s/p;}return o;}
function ema(v,p){const o=Array(v.length).fill(null);const k=2/(p+1);let e=v[0];for(let i=0;i<v.length;i++){e=i?v[i]*k+e*(1-k):v[0];if(i>=p-1)o[i]=e;}return o;}
function rstd(v,p){const o=Array(v.length).fill(null);for(let i=p-1;i<v.length;i++){let m=0;for(let j=i-p+1;j<=i;j++)m+=v[j];m/=p;let s=0;for(let j=i-p+1;j<=i;j++)s+=(v[j]-m)**2;o[i]=Math.sqrt(s/(p-1));}return o;}
function priorHigh(h,p){const o=Array(h.length).fill(null);for(let i=p;i<h.length;i++){let m=-1e18;for(let j=i-p;j<i;j++)if(h[j]>m)m=h[j];o[i]=m;}return o;}
function priorLow(l,p){const o=Array(l.length).fill(null);for(let i=p;i<l.length;i++){let m=1e18;for(let j=i-p;j<i;j++)if(l[j]<m)m=l[j];o[i]=m;}return o;}
function rsi(c,p){const o=Array(c.length).fill(null);let g=0,l=0;for(let i=1;i<c.length;i++){const d=c[i]-c[i-1],u=Math.max(d,0),w=Math.max(-d,0);if(i<=p){g+=u;l+=w;if(i===p){g/=p;l/=p;o[i]=100-100/(1+(l?g/l:1e9));}}else{g=(g*(p-1)+u)/p;l=(l*(p-1)+w)/p;o[i]=100-100/(1+(l?g/l:1e9));}}return o;}
function realizedVol(c,p){const r=[null];for(let i=1;i<c.length;i++)r.push(c[i]/c[i-1]-1);const o=Array(c.length).fill(null);for(let i=p;i<c.length;i++){let m=0;for(let j=i-p+1;j<=i;j++)m+=r[j];m/=p;let s=0;for(let j=i-p+1;j<=i;j++)s+=(r[j]-m)**2;o[i]=Math.sqrt(s/(p-1))*Math.sqrt(252)*100;}return o;}
/* Anchored VWAP re-anchored to the most recent CONFIRMED swing-pivot low (mirrors
   backtest.py anchored_vwap_trend: pivot lower than `lb` bars each side, confirmed lb
   bars later; AVWAP = cumsum(((H+L+C)/3)*Vol)/cumsum(Vol) from the anchor forward). */
function avwapSwing(h,l,c,v,lb){const n=c.length,o=Array(n).fill(null);
  if(!v||v.length!==n)return o;
  const piv=[];
  for(let i=lb;i<n-lb;i++){const ci=l[i];let ok=true;
    for(let j=i-lb;j<i&&ok;j++)if(l[j]<=ci)ok=false;
    for(let j=i+1;j<=i+lb&&ok;j++)if(l[j]<=ci)ok=false;
    if(ok)piv.push(i);}
  const cf=piv.map(p=>[p+lb,p]).sort((a,b)=>a[0]-b[0]);
  const anc=Array(n).fill(-1);let cur=-1,k=0;
  for(let i=0;i<n;i++){while(k<cf.length&&cf[k][0]<=i){cur=cf[k][1];k++;}anc[i]=cur;}
  const cpv=Array(n),cv=Array(n);let spv=0,sv=0;
  for(let i=0;i<n;i++){const tp=(h[i]+l[i]+c[i])/3;spv+=tp*v[i];sv+=v[i];cpv[i]=spv;cv[i]=sv;}
  for(let i=0;i<n;i++){const a=anc[i];if(a<0)continue;const bpv=a>0?cpv[a-1]:0,bv=a>0?cv[a-1]:0,vv=cv[i]-bv;if(vv>0)o[i]=(cpv[i]-bpv)/vv;}
  return o;}

/* ---------- formatting ----------------------------------------------------- */
function fmt(n,d){d=d==null?2:d;return (n>=0?'+':'')+Number(n).toFixed(d);}
function num(n,d){d=d==null?2:d;return Number(n).toFixed(d);}
function money(n){return '$'+Number(n).toLocaleString(undefined,{maximumFractionDigits:0});}
function pget(p,keys,def){for(const k of keys)if(p&&p[k]!=null)return p[k];return def;}
function shortLabel(l){ // "AAPL_vol_gate_trend" -> "AAPL·volgate"
  if(!l)return l;const i=l.indexOf('_');if(i<0)return l;
  const tk=l.slice(0,i),st=l.slice(i+1);
  const m={vol_gate_trend:'volgate',vol_target_trend:'voltgt',donchian_trend:'donch',
    bollinger_meanrev:'boll',sma_crossover:'sma',rsi_reversion:'rsi',breakout:'brk',
    rotation_dualmom:'rotate',dual_momentum:'dualmom',
    rotation_equity:'rot-eq',rotation_allasset:'rot-all',rotation_etf:'rot-etf'};
  return tk+'·'+(m[st]||st.replace(/_/g,''));
}
function metric(s){return sortKey==='ret'?s.total_return_pct:(sortKey==='calmar'?s.calmar:s.sharpe);}
function metricStr(s){return sortKey==='ret'?fmt(s.total_return_pct)+'%':num(metric(s))+(sortKey==='sharpe'?'':'');}

/* ---------- left tree ------------------------------------------------------ */
function passFilter(s){
  if(filt==='win'&&s.total_return_pct<=0)return false;
  if(filt==='lose'&&s.total_return_pct>0)return false;
  if(query){const h=(s.ticker+' '+s.strategy+' '+(s.name||'')+' '+s.param_str).toLowerCase();
    if(!h.includes(query))return false;}
  return true;
}
function render(){
  const L=document.getElementById('list');L.innerHTML='';
  const items=DATA.filter(passFilter);
  // group: portfolios first, then asset class -> ticker
  const ports=items.filter(s=>s.kind==='portfolio');
  const strat=items.filter(s=>s.kind!=='portfolio');
  if(ports.length){
    L.appendChild(sec('Combined Portfolios'));
    ports.sort((a,b)=>(saved.has(b.name)-saved.has(a.name))||metric(b)-metric(a))
      .forEach(p=>L.appendChild(stratRow(p,true)));
  }
  const byClass={};
  strat.forEach(s=>{(byClass[s.asset_class]=byClass[s.asset_class]||{});
    (byClass[s.asset_class][s.ticker]=byClass[s.asset_class][s.ticker]||[]).push(s);});
  const classes=Object.keys(byClass).sort((a,b)=>{
    const ia=CLASS_ORDER.indexOf(a),ib=CLASS_ORDER.indexOf(b);
    return (ia<0?99:ia)-(ib<0?99:ib);});
  classes.forEach(cls=>{
    L.appendChild(sec(cls));
    const tks=Object.keys(byClass[cls]).map(tk=>{
      const arr=byClass[cls][tk].slice().sort((a,b)=>metric(b)-metric(a));
      return {tk,arr,best:metric(arr[0])};
    }).sort((a,b)=>b.best-a.best);
    tks.forEach(({tk,arr})=>{
      const open=openTk.has(tk);
      const row=document.createElement('div');row.className='tk-row'+(open?' open':'');
      row.tabIndex=0;row.setAttribute('role','button');
      const best=arr[0];
      row.innerHTML=`<span class="chev">▶</span><span class="tk-name">${tk}</span>`+
        `<span class="tk-right"><span class="tk-count">${arr.length}</span>`+
        `<span class="spill">S ${num(best.sharpe)}</span>`+
        `<span class="ret ${best.total_return_pct>0?'rpos':'rneg'}">${fmt(best.total_return_pct)}%</span></span>`;
      const kids=document.createElement('div');kids.className='children'+(open?' open':'');
      arr.forEach(s=>kids.appendChild(stratRow(s,false)));
      const toggle=()=>{openTk.has(tk)?openTk.delete(tk):openTk.add(tk);
        row.classList.toggle('open');kids.classList.toggle('open');};
      row.onclick=toggle;
      row.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();toggle();}};
      L.appendChild(row);L.appendChild(kids);
    });
  });
  if(!items.length)L.innerHTML='<div class="empty">No matches.</div>';
}
function sec(t){const d=document.createElement('div');d.className='sec';d.textContent=t;return d;}
function stratRow(s,isPort){
  const e=document.createElement('div');e.className='srow'+(isPort?' port':'')+(s.id===selId?' sel':'');
  e.tabIndex=0;e.setAttribute('role','button');e.dataset.id=s.id;
  const label=isPort?`<span class="strat">⬡ ${s.name}</span>`
    :`${s.ticker} <span class="strat">${s.strategy}</span>`;
  const oosCls=s.oos_sharpe==null?'':(s.oos_sharpe>=1?'rpos':(s.oos_sharpe<0?'rneg':''));
  const pm=isPort
    ? `IS S${s.is_sharpe!=null?num(s.is_sharpe):'—'} → `
      +`<span class="${oosCls}">OOS S${s.oos_sharpe!=null?num(s.oos_sharpe):'—'}</span>`
    : `${s.param_str||'&nbsp;'} · S ${num(s.sharpe)} · Cal ${num(s.calmar)}`;
  const star=isPort
    ? `<span class="star ${saved.has(s.name)?'on':''}" title="Save / lock this portfolio so it stays" `
      +`onclick="toggleSave('${s.name}',event)">${saved.has(s.name)?'★':'☆'}</span>`
    : '';
  e.innerHTML=`<div class="info"><div class="nm">${label}</div>`+
    `<div class="pm">${pm}</div></div>`+
    `<div class="sp">${star}<span class="ret ${s.total_return_pct>0?'rpos':'rneg'}">${fmt(s.total_return_pct)}%</span></div>`;
  const sel=()=>pick(s.id);
  e.onclick=sel;
  e.onkeydown=ev=>{if(ev.key==='Enter'||ev.key===' '){ev.preventDefault();sel();}};
  return e;
}

/* ---------- right pane ----------------------------------------------------- */
function card(k,v,cls,hero){return `<div class="card${hero?' hero':''}"><div class="k">${k}</div>`+
  `<div class="v ${cls||''}">${v}</div></div>`;}
function cardsHTML(st){
  const ret=st.total_return_pct||0, dd=st.max_drawdown_pct||0;
  return card('Sharpe',num(st.sharpe_daily||0),(st.sharpe_daily||0)>=1?'rpos':'',true)+
    card('Calmar',num(st.calmar||0),(st.calmar||0)>=1?'rpos':'',true)+
    card('Total Return',fmt(ret)+'%',ret>0?'rpos':'rneg')+
    card('CAGR',fmt(st.cagr_pct||0)+'%',(st.cagr_pct||0)>0?'rpos':'rneg')+
    card('Max Drawdown',fmt(dd)+'%','rneg')+
    card('Win Rate',(st.win_rate_pct==null?'—':st.win_rate_pct+'%'))+
    card('Trades',st.num_trades==null?'—':st.num_trades)+
    card('Final Equity',money(st.final_equity||0),(st.final_equity||0)>=(st.starting_capital||0)?'rpos':'rneg');
}
function shClass(v){return v==null?'':(v>=1?'gp':(v<0?'gn':''));}
function portToggle(s,w){
  const mk=(win,lab)=> s.views[win]
    ? `<button class="vbtn${w===win?' on':''}" onclick="setPortView('${s.id}','${win}')">${lab}</button>`:'';
  return `<span class="vtog">${mk('in-sample','In-Sample')}${mk('out-of-sample','Out-of-Sample')}</span>`;
}
function setPortView(id,win){portView[id]=win;pick(id);}

function pick(id){
  selId=id;
  document.querySelectorAll('.srow').forEach(x=>x.classList.toggle('sel',x.dataset.id===id));
  const cur=document.querySelector('.srow.sel');if(cur)cur.scrollIntoView({block:'nearest'});
  const s=DATA.find(x=>x.id===id);if(!s)return;

  if(s.kind==='portfolio'){
    const wins=Object.keys(s.views||{});
    let w=portView[s.id]||(wins.includes('in-sample')?'in-sample':wins[0]);
    portView[s.id]=w;
    const view=s.views[w], st=view.stats||{};
    const rng=(view.date_range&&view.date_range.length===2)?`${view.date_range[0]} → ${view.date_range[1]}`:'';
    document.getElementById('hd').innerHTML=`⬡ <span class="tk">${s.name}</span> <span class="params">(combined portfolio${s.param_str?' · '+s.param_str:''})</span>`;
    const isSh=s.is_sharpe, oosSh=s.oos_sharpe;
    document.getElementById('sub').innerHTML=
      portToggle(s,w)+
      `<span class="cmp">IS Sharpe <b class="${shClass(isSh)}">${isSh!=null?num(isSh):'—'}</b>`+
      ` · OOS Sharpe <b class="${shClass(oosSh)}">${oosSh!=null?num(oosSh):'—'}</b></span>`+
      `<div class="legsline">${view.bars} bars · ${rng} · ${legsSummary(view)}`+
      (w==='out-of-sample'?' · <span class="cmp warn">⚠ single OOS window (wide CI)</span>':'')+
      (saved.has(s.name)?' · <span class="cmp" style="color:var(--gold)">★ saved</span>':
        (s.cached?' · saved copy (not in current results)':''))+`</div>`;
    document.getElementById('cards').innerHTML=cardsHTML(st);
    drawPortfolioView(view);
    return;
  }

  const st=s.stats||{};
  const rng=(s.date_range&&s.date_range.length===2)?`${s.date_range[0]} → ${s.date_range[1]}`:'';
  document.getElementById('hd').innerHTML=`<span class="tk">${s.ticker}</span> — ${s.strategy}`+
    (s.param_str?` <span class="params">(${s.param_str})</span>`:'');
  document.getElementById('sub').textContent=`${s.bars} bars · ${rng} · in-sample`;
  document.getElementById('cards').innerHTML=cardsHTML(st);
  drawChart(s);
}

/* ---------- axis builder (variable stacked panels) ------------------------- */
const GRIDY={gridcolor:'#1a2230',zerolinecolor:'#243042',color:DIM,tickfont:{family:MONO,size:10,color:DIM}};
const GRIDX={gridcolor:'#1a2230',zerolinecolor:'#1a2230',color:DIM,tickfont:{family:MONO,size:10,color:DIM}};
function buildAxes(panels){ // panels top->bottom: {key,h,title}
  const gap=0.07,n=panels.length,tot=panels.reduce((a,p)=>a+p.h,0),usable=1-gap*(n-1);
  const layout={},ax={};let top=1;
  panels.forEach((p,i)=>{
    const h=usable*p.h/tot, bot=Math.max(top-h,0);
    const xa=i?'xaxis'+(i+1):'xaxis', ya=i?'yaxis'+(i+1):'yaxis';
    layout[ya]=Object.assign({domain:[bot,top],title:{text:p.title||'',font:{size:11,color:DIM}}},GRIDY);
    layout[xa]=Object.assign({domain:[0,1],showticklabels:i===n-1,rangeslider:{visible:false}},GRIDX);
    if(i){layout[xa].matches='x';layout[xa].anchor=ya;}else{layout[xa].anchor='y';}
    ax[p.key]={x:i?'x'+(i+1):'x',y:i?'y'+(i+1):'y'};
    top=bot-gap;
  });
  return {layout,ax};
}
function baseLayout(){return{paper_bgcolor:PAPER,plot_bgcolor:PLOT,font:{color:TXT,family:MONO,size:11},
  margin:{l:58,r:18,t:30,b:26},showlegend:true,hovermode:'x unified',
  legend:{bgcolor:'rgba(12,18,27,.65)',bordercolor:LINE,borderwidth:1,orientation:'h',y:1.06,x:0,font:{size:11,color:TXT}},
  hoverlabel:{bgcolor:'#10151e',bordercolor:LINE,font:{family:MONO,size:11,color:'#e7edf4'}}};}

/* ---------- single-strategy chart with strategy-specific overlays ---------- */
function drawChart(s){
  const p=PRICES[s.ticker]||{}, hasPrice=p.dates&&p.dates.length;
  const pr=s.params||{}, strat=s.strategy, traces=[];
  // does this strategy want a dedicated oscillator panel?
  let osc=null;
  if(strat==='rsi_reversion')osc='rsi';
  else if(strat.indexOf('vol_')===0||strat.indexOf('vol_target')===0||strat.indexOf('vol_gate')===0)osc='vol';
  const panels=[];
  if(hasPrice){panels.push({key:'price',h:osc?2.4:3,title:'Price ($)'});
    if(osc)panels.push({key:'osc',h:1,title:osc==='rsi'?'RSI':'Realized Vol %'});}
  panels.push({key:'eq',h:hasPrice?1.3:3,title:'Equity ($)'});
  const {layout,ax}=buildAxes(panels);
  Object.assign(layout,baseLayout());

  if(hasPrice){
    const A=ax.price, x=p.dates;
    traces.push({type:'candlestick',x,open:p.open,high:p.high,low:p.low,close:p.close,
      increasing:{line:{color:G}},decreasing:{line:{color:R}},name:'Price',xaxis:A.x,yaxis:A.y,showlegend:false});
    // ---- strategy-specific price overlays ----
    if(strat==='sma_crossover'){
      const f=pget(pr,['fast'],20),sl=pget(pr,['slow'],50);
      line(traces,x,sma(p.close,f),GOLD,'SMA '+f,A);line(traces,x,sma(p.close,sl),ORANGE,'SMA '+sl,A);
    }else if(strat==='bollinger_meanrev'){
      const pd=pget(pr,['period','length','window'],20),k=pget(pr,['num_std','std','k'],2);
      const mid=sma(p.close,pd),sd=rstd(p.close,pd);
      const up=mid.map((m,i)=>m==null?null:m+k*sd[i]),lo=mid.map((m,i)=>m==null?null:m-k*sd[i]);
      line(traces,x,up,'rgba(180,140,255,.55)','Upper band',A);
      band(traces,x,lo,'Lower band',A,'rgba(180,140,255,.10)');
      line(traces,x,mid,PURPLE,'Mid (SMA'+pd+')',A,1);
    }else if(strat==='donchian_trend'){
      const eL=pget(pr,['entry_lookback','entry','lookback','n'],55),xL=pget(pr,['exit_lookback','exit','m'],20);
      line(traces,x,priorHigh(p.high,eL),GOLD,'Donchian high '+eL,A,1.3,'hv');
      line(traces,x,priorLow(p.low,xL),'#7fb2ff','Donchian low '+xL,A,1.3,'hv');
    }else if(strat==='breakout'){
      const lb=pget(pr,['lookback','n'],20);
      line(traces,x,priorHigh(p.high,lb),GOLD,'Breakout high '+lb,A,1.3,'hv');
    }else if(osc==='vol'){
      const tp=pget(pr,['trend_period','trend','sma','ema'],100);
      line(traces,x,ema(p.close,tp),ORANGE,'EMA '+tp+' (trend)',A);
    }else if(strat==='dual_momentum'){
      const sp=pget(pr,['sma','trend_period'],200);line(traces,x,sma(p.close,sp),ORANGE,'SMA '+sp,A);
    }else if(strat==='anchored_vwap_trend'){
      const lb=pget(pr,['swing_lookback'],10),ts=pget(pr,['trend_sma'],50);
      line(traces,x,avwapSwing(p.high,p.low,p.close,p.volume,lb),GOLD,'Anchored VWAP (swing '+lb+')',A,1.6);
      if(ts>0)line(traces,x,sma(p.close,ts),ORANGE,'SMA '+ts+' (trend gate)',A,1);
    }
    // entry/exit markers — placed in CLEAR SPACE (buys below the bar's low, sells above the
    // high) in VIVID distinct colors with a dark outline, so they never blend into the candles.
    let _pmin=1e18,_pmax=-1e18;
    for(let i=0;i<p.low.length;i++){if(p.low[i]<_pmin)_pmin=p.low[i];if(p.high[i]>_pmax)_pmax=p.high[i];}
    const moff=(_pmax-_pmin)*0.022||0;
    const loAt={},hiAt={};for(let i=0;i<x.length;i++){loAt[x[i]]=p.low[i];hiAt[x[i]]=p.high[i];}
    const dkey=d=>(d||'').slice(0,10);
    const et=(s.trades||[]).filter(t=>t.entry_date), xt=(s.trades||[]).filter(t=>t.exit_date);
    if(et.length)traces.push({type:'scatter',mode:'markers',name:'▲ Buy',
      x:et.map(t=>t.entry_date),
      y:et.map(t=>{const lo=loAt[dkey(t.entry_date)];return (lo!=null?lo:t.entry_price)-moff;}),
      marker:{symbol:'triangle-up',size:12,color:'#00e676',line:{color:'#06131f',width:1.5}},
      text:et.map(t=>'Buy $'+t.entry_price),hovertemplate:'%{text}<extra></extra>',xaxis:A.x,yaxis:A.y});
    if(xt.length)traces.push({type:'scatter',mode:'markers',name:'▼ Sell',
      x:xt.map(t=>t.exit_date),
      y:xt.map(t=>{const hi=hiAt[dkey(t.exit_date)];return (hi!=null?hi:t.exit_price)+moff;}),
      marker:{symbol:'triangle-down',size:12,color:'#ff4d6d',line:{color:'#06131f',width:1.5}},
      text:xt.map(t=>'Sell $'+t.exit_price),hovertemplate:'%{text}<extra></extra>',xaxis:A.x,yaxis:A.y});
    // ---- oscillator panel ----
    if(osc==='rsi'){
      const O=ax.osc, rp=pget(pr,['rsi_period','period'],14);
      line(traces,x,rsi(p.close,rp),'#c9d1d9','RSI '+rp,O,1.4);
      hline(traces,x,pget(pr,['buy_below','buy'],30),G,'buy',O);
      hline(traces,x,pget(pr,['sell_above','sell'],55),R,'sell',O);
      layout[O.y.replace('y','yaxis')]&&(layout[oscYaxis(O)].range=[0,100]);
    }else if(osc==='vol'){
      const O=ax.osc, vp=pget(pr,['vol_period','vol'],20);
      line(traces,x,realizedVol(p.close,vp),'#ffd27f','Realized vol '+vp,O,1.4);
      let cap=pget(pr,['vol_cap_pct','vol_cap','cap'],null);
      if(cap!=null){if(cap<1)cap=cap*100;else if(cap<5)cap=cap*Math.sqrt(252);hline(traces,x,cap,R,'cap',O);}
    }
  }
  // equity (always)
  const E=ax.eq;
  traces.push({type:'scatter',mode:'lines',x:s.equity.dates,y:s.equity.values,
    line:{color:BLUE,width:1.8,shape:'spline',smoothing:.4},fill:'tozeroy',fillcolor:'rgba(76,159,255,.13)',
    name:'Equity',xaxis:E.x,yaxis:E.y});
  Plotly.newPlot('chart',traces,layout,{responsive:true,displayModeBar:false});
}
function oscYaxis(O){return O.y==='y'?'yaxis':'yaxis'+O.y.slice(1);}
function line(tr,x,y,color,name,A,w,shape){tr.push({type:'scatter',mode:'lines',x,y,
  line:{color,width:w||1.2,shape:shape||'linear'},name,xaxis:A.x,yaxis:A.y});}
function band(tr,x,y,name,A,fill){tr.push({type:'scatter',mode:'lines',x,y,line:{color:'rgba(180,140,255,.55)',width:1},
  name,fill:'tonexty',fillcolor:fill,xaxis:A.x,yaxis:A.y});}
function hline(tr,x,val,color,name,A){if(val==null)return;tr.push({type:'scatter',mode:'lines',
  x:[x[0],x[x.length-1]],y:[val,val],line:{color,width:1,dash:'dot'},name,showlegend:false,xaxis:A.x,yaxis:A.y});}

/* ---------- combined-portfolio chart -------------------------------------- */
function legsSummary(view){
  const d=view.details||{};
  const w=d.weights, legs=d.legs||d.constituents||d.members;
  let parts=[];
  if(legs)parts.push((Array.isArray(legs)?legs.length:Object.keys(legs).length)+' legs');
  if(d.scheme)parts.push(d.scheme);
  if(w){const ws=Array.isArray(w)?w:Object.entries(w).map(([k,v])=>shortLabel(k)+' '+(v*100).toFixed(0)+'%');
    parts.push(ws.slice(0,8).join('  '));}
  if(d.combos_tried!=null)parts.push(d.combos_tried+' combos tried');
  return parts.join(' · ');
}
function drawPortfolioView(view){
  const d=view.details||{};
  const corr=d.correlation||d.corr_matrix, labels=d.corr_labels||(corr&&corr.labels);
  const matrix=corr&&(corr.matrix||corr.data||Array.isArray(corr)&&corr);
  const hasCorr=Array.isArray(matrix)&&matrix.length;
  // Equity curve dominates; correlation matrix is a compact panel beneath it.
  const panels=[{key:'eq',h:3,title:'Combined equity ($)'}];
  if(hasCorr)panels.push({key:'corr',h:1.35,title:''});
  const {layout,ax}=buildAxes(panels);Object.assign(layout,baseLayout());
  const traces=[{type:'scatter',mode:'lines',x:view.equity.dates,y:view.equity.values,
    line:{color:BLUE,width:2,shape:'spline',smoothing:.4},fill:'tozeroy',fillcolor:'rgba(76,159,255,.13)',
    name:'Portfolio',xaxis:ax.eq.x,yaxis:ax.eq.y}];
  if(hasCorr){
    const labs=(labels||matrix.map((_,i)=>'#'+(i+1))).map(shortLabel);
    const hm={type:'heatmap',z:matrix,x:labs,y:labs,xaxis:ax.corr.x,yaxis:ax.corr.y,
      zmin:-1,zmax:1,xgap:2,ygap:2,
      colorscale:[[0,'#1f8f6e'],[0.4,'#0e1622'],[0.5,'#0e1622'],[0.6,'#2a1822'],[1,'#e0556a']],
      text:matrix,texttemplate:'%{z:.2f}',textfont:{family:MONO,size:10,color:'#e7edf4'},
      hovertemplate:'%{y}  ↔  %{x}<br>corr %{z:.2f}<extra></extra>',showscale:true};
    traces.push(hm);
    const xk=ax.corr.x==='x'?'xaxis':'xaxis'+ax.corr.x.slice(1);
    const yk=ax.corr.y==='y'?'yaxis':'yaxis'+ax.corr.y.slice(1);
    // Keep the matrix compact (left ~45% width) so it doesn't dwarf the equity curve.
    layout[xk].domain=[0,0.45];
    layout[xk].tickangle=-25;layout[xk].tickfont={family:MONO,size:9,color:DIM};layout[xk].showgrid=false;layout[xk].ticks='';
    layout[yk].autorange='reversed';layout[yk].tickfont={family:MONO,size:9,color:DIM};layout[yk].showgrid=false;layout[yk].ticks='';
    const yd=layout[yk].domain||[0,0.3];
    hm.colorbar={x:0.47,xanchor:'left',y:(yd[0]+yd[1])/2,yanchor:'middle',len:yd[1]-yd[0],
      thickness:7,outlinewidth:0,tickfont:{size:8,color:DIM},tickvals:[-1,0,1]};
    layout.margin={l:104,r:18,t:30,b:64};
  }
  Plotly.newPlot('chart',traces,layout,{responsive:true,displayModeBar:false});
}

/* ---------- controls ------------------------------------------------------- */
function setF(f){filt=f;[['All','all'],['Win','win'],['Lose','lose']].forEach(([id,v])=>
  document.getElementById('f'+id).classList.toggle('on',v===f));render();}
document.getElementById('q').addEventListener('input',e=>{query=e.target.value.trim().toLowerCase();render();});
document.getElementById('sortBy').addEventListener('change',e=>{sortKey=e.target.value;render();});

/* ---------- boot ----------------------------------------------------------- */
(function(){
  const n=DATA.length, strat=DATA.filter(s=>s.kind!=='portfolio'), ports=DATA.filter(s=>s.kind==='portfolio');
  const wins=strat.filter(s=>s.total_return_pct>0).length;
  const best=strat.length?strat.slice().sort((a,b)=>b.sharpe-a.sharpe)[0]:null;
  document.getElementById('summary').innerHTML = n
    ? `<b>${strat.length}</b> strateg${strat.length===1?'y':'ies'}`+
      (ports.length?` · <b>${ports.length}</b> portfolio${ports.length===1?'':'s'}`:'')+
      ` · <span class="rpos">${wins}▲</span>/<span class="rneg">${strat.length-wins}▼</span>`+
      (best?` · best Sharpe <b>${num(best.sharpe)}</b> (${best.ticker} ${best.strategy})`:'')
    : 'No results yet — run a backtest.';
  if(!n){document.getElementById('hd').textContent='No results in ./results yet.';return;}
  render();
  // auto-open + select a sensible default: best non-stale portfolio (by in-sample
  // Sharpe) if any, else the best strategy. The portfolio opens on its IS view.
  const anyLive = ports.filter(p=>!p.cached);
  const pool = anyLive.length?anyLive:ports;
  const firstPort = pool.slice().sort((a,b)=>b.sharpe-a.sharpe)[0];
  const first = firstPort || best;
  if(first){if(first.kind!=='portfolio'){openTk.add(first.ticker);render();}pick(first.id);}
})();
</script></body></html>"""


def build_html(strategies: list[dict], out_path: Path) -> None:
    # Embed each ticker's candles ONCE (shared map) — keeps big sweeps small.
    prices: dict[str, dict] = {}
    for s in strategies:
        tk = s.get("ticker")
        if tk and tk not in prices:
            payload = _price_payload(s.get("_data_file"), tk)
            if payload:
                prices[tk] = payload
        s.pop("_data_file", None)

    data_json = json.dumps(strategies).replace("</", "<\\/")
    prices_json = json.dumps(prices).replace("</", "<\\/")
    class_json = json.dumps(CLASS_ORDER)
    html = (PAGE
            .replace("__DATA__", data_json)
            .replace("__PRICES__", prices_json)
            .replace("__CLASSORDER__", class_json)
            .replace("__PLOTLYJS__", get_plotlyjs()))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate all backtests in ./results into ONE offline dashboard HTML.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", default="results",
                        help="Folder of backtest *.json files to aggregate.")
    parser.add_argument("--results", nargs="*",
                        help="Explicit results.json paths (overrides --results-dir).")
    parser.add_argument("--out", default="reports/dashboard.html", help="Output HTML path.")
    args = parser.parse_args(argv)

    strategies = collect_strategies(args)
    n_port = sum(1 for s in strategies if s["kind"] == "portfolio")
    n_strat = len(strategies) - n_port
    if not strategies:
        print(f"No results found in {args.results_dir or args.results}. Run a backtest first.")
    build_html(strategies, Path(args.out))
    print(f"Dashboard saved to: {args.out}  ({n_strat} strategies, {n_port} portfolios)")
    print("Open it by double-clicking — it works fully offline (no internet needed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
