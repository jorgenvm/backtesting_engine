"""
Dashboard: one self-contained HTML file over every run in results/runs/.

  python run.py dashboard        (or: python dashboard.py)

Pick a run, optionally a second one to show side by side (e.g. research vs holdout).
Panels: verdicts, key metrics, equity (net / before costs / buy & hold, fold boundaries),
drawdown, in- vs out-of-sample Sharpe per fold, parameter heatmap, yearly returns,
rolling 1-year Sharpe, data health. Needs internet once for the Plotly script (CDN).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from bt import metrics
from bt.config import RESULTS_DIR

PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"


def _r(x, nd: int = 6):
    """JSON-safe rounding (NaN/inf → None)."""
    if isinstance(x, (list, tuple, np.ndarray, pd.Series, pd.Index)):
        return [_r(v, nd) for v in x]
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return round(float(x), nd) if isinstance(x, (float, np.floating)) else x


def _days(idx) -> list[str]:
    return [f"{d:%Y-%m-%d}" for d in pd.DatetimeIndex(idx)]


def _heatmap(grid: pd.DataFrame, params: dict) -> dict:
    names = list(params)
    x = names[0]
    y = names[1] if len(names) > 1 else None
    g = grid.assign(_y=grid[y] if y else "")
    z = g.pivot_table(index="_y", columns=x, values="sharpe", aggfunc="mean", sort=False)
    n = g.pivot_table(index="_y", columns=x, values="chosen", aggfunc="sum", sort=False)
    return dict(x=[str(v) for v in z.columns], y=[str(v) for v in z.index], xname=x, yname=y or "",
                z=[_r(row, 3) for row in z.to_numpy()], chosen=n.to_numpy().astype(int).tolist(),
                other=names[2:])


def load_run(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    d = pd.read_parquet(run_dir / "daily.parquet")
    folds = pd.read_parquet(run_dir / "folds.parquet")
    grid = pd.read_parquet(run_dir / "grid.parquet")
    yr, yb = metrics.yearly(d["ret"]), metrics.yearly(d["bench_ret"])
    rs, rb = metrics.rolling_sharpe(d["ret"]), metrics.rolling_sharpe(d["bench_ret"])
    created = pd.Timestamp(meta["created"])
    return dict(
        id=run_dir.name, meta=meta,
        label=f"{created:%Y-%m-%d %H:%M} · {meta['strategy']} · {meta['mode']} · {meta['symbol']} {meta['bar_size']}",
        daily=dict(date=_days(d.index), net=_r(d["equity"], 2), gross=_r(d["equity_gross"], 2),
                   bench=_r(d["bench_equity"], 2), dd=_r(d["drawdown"], 5)),
        folds=[dict(fold=int(f.fold), test=f"{f.test_start:%Y-%m-%d}", end=f"{f.test_end:%Y-%m-%d}",
                    train=f"{f.train_start:%Y-%m-%d} → {f.train_end:%Y-%m-%d}", params=f.params,
                    is_sharpe=_r(f.is_sharpe, 3), oos_sharpe=_r(f.oos_sharpe, 3)) for f in folds.itertuples()],
        heat=_heatmap(grid, meta["params"]),
        yearly=dict(year=[int(y) for y in yr.index], net=_r(yr, 5), bench=_r(yb.reindex(yr.index), 5)),
        rolling=dict(date=_days(rs.index), net=_r(rs, 3), bench=_r(rb, 3)),
    )


def build(out: Path = RESULTS_DIR / "dashboard.html") -> Path:
    runs_dir = RESULTS_DIR / "runs"
    dirs = sorted((p for p in runs_dir.iterdir() if (p / "meta.json").exists()), reverse=True) \
        if runs_dir.exists() else []
    runs = [load_run(p) for p in dirs]
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(runs, default=str, separators=(",", ":")).replace("</", "<\\/")
    out.write_text(_TEMPLATE.replace("__PLOTLY__", PLOTLY).replace("__DATA__", payload), encoding="utf-8")
    return out


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Backtest Dashboard</title>
<script src="__PLOTLY__"></script>
<style>
:root {
  color-scheme: light;
  --page: #f4f4f4; --surface: #ffffff; --ink: #111111; --ink-2: #555555; --muted: #8a8a8a;
  --grid: #e8e8e8; --axis: #c8c8c8; --ring: rgba(0,0,0,0.10);
  --blue: #2a78d6; --blue-2: #86b6ef; --red: #d03b3b; --gray: #9a9a9a; --gray-2: #c4c4c4; --mid: #f2f2f2;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
       font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { display: flex; flex-wrap: wrap; gap: 12px 20px; align-items: center; padding: 16px; }
header h1 { font-size: 17px; margin: 0 auto 0 0; font-weight: 650; }
label { color: var(--ink-2); font-size: 13px; display: flex; gap: 8px; align-items: center;
        flex: 1 1 280px; max-width: 520px; min-width: 0; }
label select { flex: 1; min-width: 0; }
select { font: inherit; color: var(--ink); background: var(--surface); border: 1px solid var(--ring);
                 border-radius: 8px; padding: 6px 10px; max-width: 100%; }
main { display: grid; gap: 16px; padding: 0 16px 24px; grid-template-columns: minmax(0, 1fr); }
main.two { grid-template-columns: repeat(2, minmax(0, 1fr)); }
@media (max-width: 1100px) { main.two { grid-template-columns: minmax(0, 1fr); } }
.run { display: flex; flex-direction: column; gap: 16px; min-width: 0; }
.run h2 { font-size: 15px; margin: 0; font-weight: 600; }
.run .sub { color: var(--ink-2); font-size: 13px; margin-top: 2px; }
.card { background: var(--surface); border: 1px solid var(--ring); border-radius: 12px; padding: 14px 16px; min-width: 0; }
.card h3 { font-size: 13px; font-weight: 600; margin: 0 0 8px; color: var(--ink-2); }
.cards { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr)); }
.verdicts { display: grid; gap: 8px; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); }
.v { border: 1px solid var(--ring); border-radius: 10px; padding: 8px 10px; background: var(--surface); }
.v .top { display: flex; gap: 6px; align-items: center; font-size: 12px; color: var(--ink-2); }
.v .val { font-size: 17px; font-weight: 600; margin-top: 2px; }
.v .note { font-size: 11.5px; color: var(--muted); }
.ic { width: 18px; height: 18px; border-radius: 50%; display: inline-grid; place-items: center;
      font-size: 11px; font-weight: 700; color: #fff; flex: none; }
.ic.pass { background: var(--blue); } .ic.warn { background: var(--ink-2); }
.ic.fail { background: var(--red); } .ic.info { background: var(--gray-2); }
.tiles { display: grid; gap: 8px; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); }
.tile .k { font-size: 12px; color: var(--ink-2); }
.tile .n { font-size: 22px; font-weight: 600; }
.tile .c { font-size: 12px; color: var(--muted); }
.ehead { display: flex; justify-content: space-between; align-items: center; gap: 12px;
         padding-bottom: 12px; border-bottom: 1px solid var(--grid); }
.ehead h3 { margin: 0; color: var(--ink); font-size: 12px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
.seg { display: flex; gap: 6px; }
.seg button { font: inherit; min-width: 36px; height: 32px; border-radius: 4px; cursor: pointer;
              border: 1px solid var(--axis); background: var(--surface); color: var(--ink); }
.seg button.on { background: var(--ink); border-color: var(--ink); color: #fff; }
.seg button:disabled { opacity: .35; cursor: not-allowed; }
.erows { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 340px), 1fr)); column-gap: 40px; }
.erow { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 11px 0;
        border-bottom: 1px solid var(--grid); }
.ek { color: var(--ink-2); }
.ev { font-weight: 700; text-align: right; font-variant-numeric: tabular-nums; }
.ev.pos { color: var(--blue); } .ev.neg { color: var(--red); }
.chip { display: inline-grid; place-items: center; width: 22px; height: 22px; margin-left: 4px; border-radius: 3px;
        color: #fff; font-size: 11px; font-weight: 700; }
.chip.W { background: var(--blue); } .chip.L { background: var(--red); } .chip.B { background: var(--gray); }
.muted { color: var(--muted); font-weight: 400; }
.plot { width: 100%; height: 300px; }
.plot.tall { height: 360px; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 4px 8px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th:first-child, td:first-child, td.l, th.l { text-align: left; }
th { color: var(--ink-2); font-weight: 600; }
.scroll { overflow-x: auto; }
details summary { cursor: pointer; color: var(--ink-2); font-size: 13px; margin-top: 8px; }
.empty { padding: 48px 16px; text-align: center; color: var(--ink-2); }
code { font-size: 12.5px; }
</style></head>
<body>
<header>
  <h1>Backtest Dashboard</h1>
  <label>Run <select id="runA"></select></label>
  <label>Compare <select id="runB"></select></label>
</header>
<main id="main"></main>
<script>
const RUNS = __DATA__;
const byId = Object.fromEntries(RUNS.map(r => [r.id, r]));
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const pct = (x, d = 1) => x == null ? '–' : (x * 100).toFixed(d) + '%';
const num = (x, d = 2) => x == null ? '–' : Number(x).toFixed(d);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const ICON = {pass: '✓', warn: '!', fail: '✕', info: '·'};

function base(extra = {}) {
  const ax = {gridcolor: css('--grid'), linecolor: css('--axis'), zerolinecolor: css('--axis'),
              tickfont: {color: css('--muted'), size: 11}, automargin: true};
  return Object.assign({
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: {family: 'system-ui, -apple-system, "Segoe UI", sans-serif', color: css('--ink-2'), size: 12},
    margin: {l: 8, r: 8, t: 8, b: 8}, hovermode: 'x unified',
    hoverlabel: {bgcolor: css('--surface'), bordercolor: css('--ring'), font: {color: css('--ink')}},
    legend: {orientation: 'h', x: 0, y: 1.12, font: {color: css('--ink-2')}},
    xaxis: Object.assign({}, ax), yaxis: Object.assign({}, ax),
  }, extra);
}
const CFG = {displayModeBar: false, responsive: true};
const line = (x, y, name, color, extra = {}) =>
  Object.assign({x, y, name, type: 'scatter', mode: 'lines', line: {color, width: 2}}, extra);

function verdicts(m) {
  return m.verdicts.map(v => `<div class="v"><div class="top"><span class="ic ${v.status}" aria-hidden="true">${ICON[v.status]}</span>
    ${esc(v.check)}</div><div class="val">${esc(v.value)} <span class="note">${v.status.toUpperCase()}</span></div>
    <div class="note">${esc(v.note)}</div></div>`).join('');
}
function tiles(m) {
  const k = m.metrics;
  const rows = [
    ['CAGR', pct(k.cagr), `buy & hold ${pct(k.bench_cagr)}`],
    ['Sharpe', num(k.sharpe), `buy & hold ${num(k.bench_sharpe)}`],
    ['Max drawdown', pct(k.max_dd), `Calmar ${num(k.calmar)}`],
    ['Volatility', pct(k.vol), `Sortino ${num(k.sortino)}`],
    ['Cost drag', pct(k.cost_drag, 2) + '/yr', `before costs ${pct(k.cagr_gross)}`],
    ['Turnover', Math.round(k.turnover_per_year).toLocaleString() + '/yr', 'units traded per year'],
  ];
  return rows.map(([a, b, c]) => `<div class="tile"><div class="k">${a}</div><div class="n">${b}</div><div class="c">${c}</div></div>`).join('');
}
function evaluation(card, e, unit) {
  if (!e) { card.innerHTML = '<div class="ehead"><h3>Evaluation</h3></div><p class="muted">No closed trades.</p>'; return; }
  const u = unit === 'r' && e.r ? 'r' : 'usd', v = e[u];
  const money = x => (x < 0 ? '−$' : '$') + Math.abs(x).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
  const rr = x => (x > 0 ? '+' : x < 0 ? '−' : '') + Math.abs(x).toFixed(2) + 'R';
  const f = u === 'r' ? rr : money, sign = x => x > 0 ? 'pos' : x < 0 ? 'neg' : '';
  const rows = [
    ['Total number of trades', e.trades.toLocaleString() + (e.open ? ` <span class="muted">+${e.open} open</span>` : '')],
    ['Avg. profit per trading day', f(v.per_day), sign(v.per_day)],
    ['Biggest winner', f(v.best), sign(v.best)],
    ['Biggest loser', f(v.worst), sign(v.worst)],
    ['Total fees, modelled', f(v.fees), 'neg'],
    ['Avg. holding time (D / H / M)', `${num(e.hold_days)} / ${num(e.hold_hours)} / ${Math.round(e.hold_minutes).toLocaleString()}`],
    ['Winrate w/o BE', pct(e.winrate, 2)],
    u === 'r' ? ['Total R', rr(v.total), sign(v.total)] : ['ROI', pct(v.roi, 2), sign(v.roi)],
    ['Max drawdown', u === 'r' ? v.max_dd.toFixed(2) + 'R' : pct(v.max_dd, 2)],
    ['Winning / losing days', `${e.win_days} / ${e.loss_days}`],
    ['Trades per active day / week', `${num(e.per_day)} / ${num(e.per_week)}`],
    ['Current streak', e.streak.map(s => `<span class="chip ${s}" title="${{W: 'win', L: 'loss', B: 'breakeven'}[s]}">${s}</span>`).join('')],
  ];
  card.innerHTML = `<div class="ehead"><h3>Evaluation · out-of-sample trades</h3><div class="seg" role="group" aria-label="Units">
      <button data-u="usd" class="${u === 'usd' ? 'on' : ''}" aria-pressed="${u === 'usd'}">$</button>
      <button data-u="r" class="${u === 'r' ? 'on' : ''}" aria-pressed="${u === 'r'}" ${e.r ? '' : 'disabled title="R needs stop prices (bracket strategies)"'}>R</button></div></div>
    <div class="erows">${rows.map(([k, val, c]) => `<div class="erow"><span class="ek">${k}</span><span class="ev ${c || ''}">${val}</span></div>`).join('')}</div>`;
  card.querySelectorAll('.seg button').forEach(b => b.onclick = () => evaluation(card, e, b.dataset.u));
}
function health(h) {
  const rows = [['Source', h.source], ['Rows', h.rows.toLocaleString()], ['Bad rows dropped', `${h.bad_rows.toLocaleString()} (${h.bad_pct}%)`],
    ['Bars', h.bars.toLocaleString()], ['Range', `${h.first.slice(0, 10)} → ${h.last.slice(0, 10)}`],
    ['Quotes', h.quotes ? 'present (half spread charged)' : 'none (fixed slippage only)'],
    [`Gaps > ${h.gap_threshold_h} h`, `${h.gaps} (weekends excluded)`], ['Largest gap', `${h.largest_gap_h} h`]];
  const top = h.top_gaps.map(g => `<tr><td class="l">${g[0].slice(0, 16)}</td><td class="l">${g[1].slice(0, 16)}</td><td>${g[2]}</td></tr>`).join('');
  return `<table>${rows.map(r => `<tr><td class="l">${r[0]}</td><td>${esc(r[1])}</td></tr>`).join('')}</table>` +
    (top ? `<details><summary>Largest gaps</summary><div class="scroll"><table><tr><th class="l">from</th><th class="l">to</th><th>hours</th></tr>${top}</table></div></details>` : '');
}
function foldTable(run) {
  return `<details><summary>Table view</summary><div class="scroll"><table><tr><th class="l">test window</th><th class="l">train window</th>
    <th class="l">chosen parameters</th><th>IS Sharpe</th><th>OOS Sharpe</th></tr>${run.folds.map(f =>
    `<tr><td class="l">${f.test} → ${f.end}</td><td class="l">${f.train}</td><td class="l"><code>${esc(f.params)}</code></td>
     <td>${num(f.is_sharpe)}</td><td>${num(f.oos_sharpe)}</td></tr>`).join('')}</table></div></details>`;
}
function yearTable(run) {
  const y = run.yearly;
  return `<details><summary>Table view</summary><div class="scroll"><table><tr><th class="l">year</th><th>strategy</th><th>buy & hold</th></tr>${
    y.year.map((yr, i) => `<tr><td class="l">${yr}</td><td>${pct(y.net[i])}</td><td>${pct(y.bench[i])}</td></tr>`).join('')}</table></div></details>`;
}

function renderRun(el, run) {
  const m = run.meta;
  const id = s => `${run.id}-${s}`;
  el.innerHTML = `
    <div><h2>${esc(m.strategy)} · ${m.mode} · ${m.symbol} ${m.bar_size}</h2>
      <div class="sub">${m.mode === 'holdout' ? 'Holdout' : 'Walk-forward out-of-sample'} ${m.oos_start} → ${m.oos_end} ·
        ${m.kind} · delay ${m.delay} bar · code ${m.code_hash}${m.contaminated ? ' · <b>contaminated holdout</b>' : ''}</div></div>
    <div class="verdicts">${verdicts(m)}</div>
    <div class="card"><h3>Key metrics (net of costs)</h3><div class="tiles">${tiles(m)}</div></div>
    <div class="card" id="${id('eval')}"></div>
    <div class="card"><h3>Equity, USD — vertical lines mark walk-forward folds</h3><div class="plot tall" id="${id('eq')}"></div></div>
    <div class="cards">
      <div class="card"><h3>Drawdown</h3><div class="plot" id="${id('dd')}"></div></div>
      <div class="card"><h3>Sharpe per fold: in-sample (train) vs out-of-sample (test)</h3><div class="plot" id="${id('folds')}"></div>${foldTable(run)}</div>
      <div class="card"><h3>Parameter grid: ${m.mode === 'holdout' ? 'train-window' : 'research-period'} Sharpe, ×n = times chosen</h3><div class="plot" id="${id('heat')}"></div>
        ${run.heat.other.length ? `<div class="sub">Averaged over ${esc(run.heat.other.join(', '))}</div>` : ''}</div>
      <div class="card"><h3>Yearly returns</h3><div class="plot" id="${id('yr')}"></div>${yearTable(run)}</div>
      <div class="card"><h3>Rolling 1-year Sharpe</h3><div class="plot" id="${id('roll')}"></div></div>
      <div class="card"><h3>Data health</h3>${health(m.health)}</div>
    </div>`;

  evaluation(document.getElementById(id('eval')), m.evaluation, 'usd');
  const d = run.daily, blue = css('--blue'), red = css('--red'), gray = css('--gray');
  const folds = run.folds.slice(1).map(f => ({type: 'line', xref: 'x', yref: 'paper', x0: f.test, x1: f.test, y0: 0, y1: 1,
                                            line: {color: css('--axis'), width: 1}}));
  Plotly.newPlot(id('eq'), [
    line(d.date, d.net, 'Net', blue, {hovertemplate: '%{y:$,.0f}'}),
    line(d.date, d.gross, 'Before costs', css('--blue-2'), {hovertemplate: '%{y:$,.0f}'}),
    line(d.date, d.bench, 'Buy & hold', gray, {hovertemplate: '%{y:$,.0f}'}),
  ], base({shapes: folds, yaxis: Object.assign(base().yaxis, {tickformat: '$,.0f'})}), CFG);

  Plotly.newPlot(id('dd'), [line(d.date, d.dd, 'Drawdown', red,
    {fill: 'tozeroy', fillcolor: red + '1f', hovertemplate: '%{y:.1%}', showlegend: false})],
    base({yaxis: Object.assign(base().yaxis, {tickformat: '.0%'})}), CFG);

  const fx = run.folds.map(f => f.test.slice(0, 7)), fp = run.folds.map(f => f.params);
  Plotly.newPlot(id('folds'), [
    {x: fx, y: run.folds.map(f => f.is_sharpe), name: 'In-sample', type: 'bar', marker: {color: css('--gray-2')},
     customdata: fp, hovertemplate: '%{y:.2f}<br>%{customdata}<extra>In-sample</extra>'},
    {x: fx, y: run.folds.map(f => f.oos_sharpe), name: 'Out-of-sample', type: 'bar',
     marker: {color: run.folds.map(f => f.oos_sharpe < 0 ? red : blue)},
     hovertemplate: '%{y:.2f}<extra>Out-of-sample</extra>'},
  ], base({barmode: 'group', bargap: 0.3, bargroupgap: 0.08, hovermode: 'closest',
           xaxis: Object.assign(base().xaxis, {type: 'category'})}), CFG);

  const h = run.heat, zs = h.z.flat().filter(v => v != null), lim = Math.max(0.1, ...zs.map(Math.abs));
  Plotly.newPlot(id('heat'), [{
    type: 'heatmap', x: h.x, y: h.y, z: h.z, zmin: -lim, zmax: lim, zmid: 0, xgap: 2, ygap: 2,
    colorscale: [[0, red], [0.5, css('--mid')], [1, blue]],
    text: h.chosen.map(r => r.map(n => n ? `×${n}` : '')), texttemplate: '%{text}',
    textfont: {color: css('--ink')}, customdata: h.chosen,
    hovertemplate: `${h.xname} %{x}${h.yname ? `<br>${h.yname} %{y}` : ''}<br>Sharpe %{z:.2f}<br>chosen %{customdata}×<extra></extra>`,
    colorbar: {thickness: 10, outlinewidth: 0, tickfont: {color: css('--muted'), size: 11}},
  }], base({hovermode: 'closest', xaxis: Object.assign(base().xaxis, {type: 'category', title: {text: h.xname}}),
            yaxis: Object.assign(base().yaxis, {type: 'category', title: {text: h.yname}})}), CFG);

  const y = run.yearly;
  Plotly.newPlot(id('yr'), [
    {x: y.year, y: y.net, name: 'Strategy', type: 'bar', marker: {color: y.net.map(v => v < 0 ? red : blue)}, hovertemplate: '%{y:.1%}'},
    {x: y.year, y: y.bench, name: 'Buy & hold', type: 'bar', marker: {color: css('--gray-2')}, hovertemplate: '%{y:.1%}'},
  ], base({barmode: 'group', bargap: 0.3, bargroupgap: 0.08, yaxis: Object.assign(base().yaxis, {tickformat: '.0%'}),
           xaxis: Object.assign(base().xaxis, {type: 'category'})}), CFG);

  const r = run.rolling;
  Plotly.newPlot(id('roll'), [line(r.date, r.net, 'Strategy', blue, {hovertemplate: '%{y:.2f}'}),
                              line(r.date, r.bench, 'Buy & hold', gray, {hovertemplate: '%{y:.2f}'})],
    base({yaxis: Object.assign(base().yaxis, {zeroline: true})}), CFG);
}

function render() {
  const main = document.getElementById('main');
  if (!RUNS.length) {
    main.className = '';
    main.innerHTML = '<div class="empty">No runs yet. Run <code>python run.py research --strategy ma_cross</code>, then rebuild.</div>';
    return;
  }
  const ids = [document.getElementById('runA').value, document.getElementById('runB').value].filter(Boolean);
  main.className = ids.length > 1 ? 'two' : '';
  main.innerHTML = ids.map(() => '<section class="run"></section>').join('');
  [...main.children].forEach((el, i) => renderRun(el, byId[ids[i]]));
  try { localStorage.setItem('dash-sel', JSON.stringify(ids)); } catch (e) {}
}

const selA = document.getElementById('runA'), selB = document.getElementById('runB');
const opts = RUNS.map(r => `<option value="${r.id}">${esc(r.label)}</option>`).join('');
selA.innerHTML = opts;
selB.innerHTML = '<option value="">—</option>' + opts;
try {
  const [a, b] = JSON.parse(localStorage.getItem('dash-sel') || '[]');
  if (byId[a]) selA.value = a;
  if (byId[b]) selB.value = b;
} catch (e) {}
selA.onchange = selB.onchange = render;
render();
</script>
</body></html>
"""

if __name__ == "__main__":
    print(f"  Wrote {build()}")
