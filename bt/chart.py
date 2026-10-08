"""
One day of a run, TradingView Lightweight Charts: the strategy's bars (left) and the exit bars
(right) with session windows shaded, volume underneath and the run's trades drawn on the right:
entry arrow, stop / entry / BE / TP levels until the exit, and the result with net R.
Times are config timezone.

  python run.py chart --date 2023-06-14 [--run <run folder>]
"""
from __future__ import annotations

import json

import pandas as pd

from bt import data, sessions
from bt.config import CFG, RESULTS_DIR, TZ

LIB = "https://cdn.jsdelivr.net/npm/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js"


def _wall(ts) -> list[int]:
    """UTC → wall-clock seconds (the chart library only renders UTC)."""
    t = pd.DatetimeIndex(pd.to_datetime(ts, utc=True)).tz_convert(TZ).tz_localize(None)
    return (t.as_unit("s").asi8).tolist()


def _candles(df: pd.DataFrame) -> list[dict]:
    return [dict(time=t, open=o, high=h, low=l, close=c)
            for t, o, h, l, c in zip(_wall(df.index), df["open"], df["high"], df["low"], df["close"])]


def _hm(t) -> str:
    return "" if pd.isna(t) else f"{pd.Timestamp(t).tz_convert(TZ):%H:%M}"


def _num(x) -> float | None:
    return None if pd.isna(x) else round(float(x), 2)


def _f(x, fmt: str) -> str:
    return "" if pd.isna(x) else format(x, fmt)


def build(left: pd.DataFrame, right: pd.DataFrame, title: str, trades: pd.DataFrame | None = None,
          labels: tuple[str, str] = ("", "")) -> str:
    """HTML page. left / right: bars already cut to the day; trades: engine trade list."""
    trades = pd.DataFrame() if trades is None else trades
    fibs = []
    for r in trades.itertuples():
        end = r.exit_time if pd.notna(r.exit_time) else right.index[-1]
        ent, ext = _wall([r.entry_time, end])
        fibs.append(dict(dir=r.direction, ent=ent, ext=max(ext, ent + 60), result=r.result,
                         r=_num(r.r_net), usd=_num(r.pnl), stop=_num(r.stop_price), entry=_num(r.entry_price),
                         be=_num(r.be_price), tp=_num(r.tp_price)))
    rows = "".join(
        f"<tr class='{r.direction}'><td>{_hm(r.entry_time)}</td><td>{r.session}</td><td>{r.direction}</td>"
        f"<td>{r.entry_price:.2f}</td><td>{_f(r.stop_price, '.2f')}</td><td>{_f(r.be_price, '.2f')}</td>"
        f"<td>{_f(r.tp_price, '.2f')}</td><td>{_hm(r.exit_time)}</td><td>{r.result}</td>"
        f"<td>{_f(r.r_net, '+.2f')}</td><td>{_f(r.pnl, ',.0f')}</td></tr>" for r in trades.itertuples())

    def bg(df):
        return [dict(time=t, value=1 if s else 0) for t, s in zip(_wall(df.index), sessions.label(df.index))]

    vol = [dict(time=t, value=float(v)) for t, v in zip(_wall(right.index), right["volume"])]
    payload = json.dumps(dict(left=_candles(left), right=_candles(right), bgl=bg(left), bgr=bg(right),
                              vol=vol, fibs=fibs))
    head = title + f" · {len(trades)} trades" + (f" (${trades['pnl'].sum():+,.0f} net)" if len(trades) else "")
    return (_TEMPLATE.replace("__TITLE__", head).replace("__LIB__", LIB).replace("__TZ__", TZ)
            .replace("__DATA__", payload).replace("__TRADES__", rows)
            .replace("__HIDE__", "" if len(trades) else "hidden")
            .replace("__LB__", labels[0]).replace("__RB__", labels[1]))


def for_run(run: str, day: str):
    """Write the chart for one day of a saved run and return its path."""
    runs = sorted(p for p in (RESULTS_DIR / "runs").iterdir() if p.is_dir())
    run_dir = runs[-1] if run == "latest" else RESULTS_DIR / "runs" / run
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    mode = "research" if meta["mode"] == "research" else "full"
    d0 = pd.Timestamp(day, tz=TZ)
    d1 = d0 + pd.Timedelta(days=1)

    def cut(bar_size):
        b = data.load_bars(meta["symbol"], bar_size, mode)[0]
        return b[(b.index >= d0) & (b.index < d1)]

    left, right = cut(meta["bar_size"]), cut(CFG["exit_bar_size"])
    if right.empty:
        raise SystemExit(f"No {meta['symbol']} data on {day}.")
    trades = None
    if (run_dir / "trades.parquet").exists():
        t = pd.read_parquet(run_dir / "trades.parquet")
        trades = t[(t["entry_time"] >= d0) & (t["entry_time"] < d1)]
    out = RESULTS_DIR / "charts" / f"{run_dir.name}_{day}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    title = f"{meta['strategy']} · {meta['symbol']} {day}"
    out.write_text(build(left, right, title, trades, (meta["bar_size"], CFG["exit_bar_size"])),
                   encoding="utf-8")
    return out


_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"><title>Trade chart</title>
<script src="__LIB__"></script>
<style>
 body{margin:0;font:13px system-ui,sans-serif;background:#f4f4f4;color:#222}
 header{padding:8px 16px;font-weight:600}
 .row{display:flex;gap:8px;padding:0 8px;height:70vh}
 .row>div{flex:1;background:#fff;position:relative}
 .tag{position:absolute;top:4px;left:8px;z-index:3;color:#888}
 table{margin:0 16px 12px;border-collapse:collapse} h4{margin:12px 16px 4px}
 td,th{padding:2px 10px;text-align:right}
 tr.long td:nth-child(3){color:#1f5fd6} tr.short td:nth-child(3){color:#d6283a}
 .hidden{display:none}
</style></head><body>
<header>__TITLE__ — times __TZ__ · shaded = session · levels: <b style="color:#d6283a">stop</b>
 <b style="color:#555">entry</b> <b style="color:#e08a00">BE</b> <b style="color:#1a9a4a">TP</b></header>
<div class="row"><div id="left"><span class="tag">__LB__</span></div><div id="right"><span class="tag" id="legend">__RB__</span></div></div>
<div class="__HIDE__"><h4>Trades</h4>
<table><tr><th>entry</th><th>session</th><th>dir</th><th>price</th><th>stop</th><th>BE</th><th>TP</th>
<th>exit</th><th>result</th><th>R net</th><th>USD net</th></tr>__TRADES__</table></div>
<script>
const D = __DATA__;
function make(el, candles, bg) {
  const ch = LightweightCharts.createChart(el, {autoSize: true,
    layout: {background: {color: '#fff'}, textColor: '#555'},
    grid: {vertLines: {visible: false}, horzLines: {color: '#eee'}},
    timeScale: {timeVisible: true, secondsVisible: false}});
  ch.addHistogramSeries({priceScaleId: 'bg', color: 'rgba(31,95,214,0.07)', lastValueVisible: false,
    priceLineVisible: false}).setData(bg);
  ch.priceScale('bg').applyOptions({scaleMargins: {top: 0, bottom: 0}, visible: false});
  const s = ch.addCandlestickSeries({upColor: '#cfd2da', downColor: '#666a75', borderVisible: false,
    wickUpColor: '#999', wickDownColor: '#555'});
  s.setData(candles);
  return [ch, s];
}
function trades(ch) {
  const m = [], lvl = {stop: ['#d6283a', 2], entry: ['#555', 0], be: ['#e08a00', 2], tp: ['#1a9a4a', 0]};
  for (const F of D.fibs) {
    for (const [k, [color, style]] of Object.entries(lvl)) {
      if (F[k] === null) continue;
      ch.addLineSeries({color, lineWidth: 1, lineStyle: style, lastValueVisible: false,
        priceLineVisible: false, crosshairMarkerVisible: false})
        .setData([{time: F.ent, value: F[k]}, {time: F.ext, value: F[k]}]);
    }
    const long = F.dir === 'long';
    m.push({time: F.ent, position: long ? 'belowBar' : 'aboveBar', color: long ? '#1f5fd6' : '#d6283a',
            shape: long ? 'arrowUp' : 'arrowDown', text: long ? 'LONG' : 'SHORT'});
    if (F.result !== 'open') m.push({time: F.ext, position: long ? 'aboveBar' : 'belowBar',
            color: F.result === 'win' ? '#1a9a4a' : F.result === 'be' ? '#e08a00' : '#d6283a',
            shape: 'circle', text: `${F.result.toUpperCase()} ` + (F.r === null
              ? `${F.usd > 0 ? '+' : ''}$${Math.round(F.usd)}` : `${F.r > 0 ? '+' : ''}${F.r}R`)});
  }
  return m.sort((a, b) => a.time - b.time);
}
const [cl] = make(document.getElementById('left'), D.left, D.bgl);
const [cr, sr] = make(document.getElementById('right'), D.right, D.bgr);
sr.setMarkers(trades(cr));
const vol = cr.addHistogramSeries({priceScaleId: 'vol', color: '#c8cbd2', lastValueVisible: false, priceLineVisible: false});
vol.setData(D.vol);
cr.priceScale('vol').applyOptions({scaleMargins: {top: 0.82, bottom: 0}, visible: false});
cr.priceScale('right').applyOptions({scaleMargins: {top: 0.05, bottom: 0.2}});
const legend = document.getElementById('legend'), tag = legend.textContent;
cr.subscribeCrosshairMove(p => {
  const b = p.time && p.seriesData.get(sr), v = p.time && p.seriesData.get(vol);
  if (!b) { legend.textContent = tag; return; }
  const t = new Date(p.time * 1000).toISOString().slice(11, 16);
  legend.textContent = `${tag} ${t}  O ${b.open} H ${b.high} L ${b.low} C ${b.close}  Vol ${v ? v.value : ''}`;
});
cl.timeScale().fitContent(); cr.timeScale().fitContent();
</script></body></html>
"""
