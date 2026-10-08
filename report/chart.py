"""
One-day chart, TradingView Lightweight Charts: a higher-timeframe panel (left, plain M5 by
default) and M1 (right), session windows shaded, volume under M1, and the trades from
backtest/engine.py drawn on M1 — entry arrow, stop / entry / BE / TP levels until the exit,
and the result with net R. Strategy-agnostic: a strategy adds its own layers through build().

  python -m report.chart --instrument NQ --date 2026-06-10
  python -m report.chart --date 2026-06-10 --trades trades.csv     # engine output, any range
  python -m report.chart --date 2026-06-10 --out chart.html --no-open

Strategy layers (all optional, see build()):
  htf         candles for the left panel instead of plain M5 (e.g. Heikin Ashi)
  lines       horizontal levels: price, start, end, color [, stub_from, label, direction, panel]
  bar_colors  per M1 bar CSS color, "" = default (e.g. mark signal candles)
  pane        per M1 bar values shown under M1 instead of volume [, pane_level threshold]

Times on the chart and in the tables are Europe/Berlin (config.SESSION_TZ).
Hover an M1 bar to read its OHLC, volume and pane value.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import webbrowser
from pathlib import Path

import pandas as pd

import config
from core import sessions
from data.loader import load_m1
from data.resample import to_m5

LIB = "https://cdn.jsdelivr.net/npm/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js"
LONG, SHORT, NEUTRAL = "#1f5fd6", "#d6283a", "#c8cbd2"


def _wall(ts: pd.Series | pd.DatetimeIndex) -> list[int]:
    """UTC → Berlin wall-clock seconds (the chart library only renders UTC)."""
    t = pd.DatetimeIndex(ts).tz_convert(config.SESSION_TZ).tz_localize(None)
    return (t.asi8 // 10**9).tolist()


def _num(x) -> float | None:
    return None if pd.isna(x) else float(x)


def _candles(df: pd.DataFrame, colors=None) -> list[dict]:
    out = [dict(time=t, open=o, high=h, low=l, close=c)
           for t, o, h, l, c in zip(_wall(df.index), df["open"], df["high"], df["low"], df["close"])]
    if colors is not None:
        for bar, col in zip(out, colors):
            if col:
                bar["color"] = bar["wickColor"] = col
    return out


def _hm(t) -> str:
    return "" if pd.isna(t) else f"{pd.Timestamp(t).tz_convert(config.SESSION_TZ):%H:%M}"


def build(m1: pd.DataFrame, code: str, day: str,
          trades: pd.DataFrame | None = None,
          lines: pd.DataFrame | None = None,
          bar_colors: pd.Series | None = None,
          htf: pd.DataFrame | None = None, htf_label: str = "M5",
          pane: pd.Series | None = None, pane_label: str = "Vol", pane_level: float | None = None,
          title: str = "") -> str:
    """
    HTML page for one Berlin calendar day. m1 may cover more than the day (warm-up);
    everything is cut to the day here. trades = backtest.engine.run output.
    """
    tz = config.SESSION_TZ
    d0 = pd.Timestamp(day, tz=tz)
    d1 = d0 + pd.Timedelta(days=1)
    m1d = m1.loc[d0:d1 - pd.Timedelta(minutes=1)]
    if m1d.empty:
        raise SystemExit(f"No {code} data for {day}.")
    left = (to_m5(m1) if htf is None else htf).loc[d0:d1 - pd.Timedelta(minutes=1)]

    trades = pd.DataFrame() if trades is None else trades
    if len(trades):
        et = pd.to_datetime(trades["entry_time"], utc=True)
        trades = trades[(et >= d0) & (et < d1)]
    lines = pd.DataFrame() if lines is None else lines
    if len(lines):
        lines = lines[(pd.to_datetime(lines["end"], utc=True) > d0)
                      & (pd.to_datetime(lines["start"], utc=True) < d1)]

    segs = []
    for r in lines.to_dict("records"):
        src = r.get("stub_from", r["start"])
        src = r["start"] if pd.isna(src) else src
        s, p, e = _wall(pd.DatetimeIndex(pd.to_datetime([src, r["start"], r["end"]], utc=True)))
        segs.append(dict(price=float(r["price"]), src=s, prt=p, exp=e, color=r["color"],
                         label=r.get("label") or "", dir=r.get("direction") or "",
                         panel=r.get("panel") or "both"))

    fibs = []
    for r in trades.itertuples():
        end = r.exit_time if pd.notna(r.exit_time) else m1d.index[-1]
        ent, ext = _wall(pd.DatetimeIndex(pd.to_datetime([r.entry_time, end], utc=True)))
        fibs.append(dict(dir=r.direction, ent=ent, ext=max(ext, ent + 60), result=r.result,
                         r=None if pd.isna(r.r_net) else round(r.r_net, 2),
                         stop=r.stop_price, entry=r.entry_price, be=_num(r.be_price), tp=r.tp_price))

    ttable = "".join(
        f"<tr class='{r.direction}'><td>{_hm(r.entry_time)}</td><td>{r.session}</td>"
        f"<td>{r.direction}</td><td>{r.entry_price:.2f}</td><td>{r.stop_price:.2f}</td>"
        f"<td>{'' if pd.isna(r.be_price) else f'{r.be_price:.2f}'}</td><td>{r.tp_price:.2f}</td>"
        f"<td>{_hm(r.exit_time)}</td><td>{r.result}</td>"
        f"<td>{'' if pd.isna(r.r_gross) else f'{r.r_gross:+.2f}'}</td>"
        f"<td>{'' if pd.isna(r.r_net) else f'{r.r_net:+.2f}'}</td></tr>"
        for r in trades.itertuples())
    ltable = "".join(
        f"<tr><td>{_hm(r['start'])}</td><td>{_hm(r['end'])}</td><td>{r.get('label') or ''}</td>"
        f"<td>{r['price']:.2f}</td></tr>"
        for r in lines.to_dict("records"))

    t1 = _wall(m1d.index)
    colors = None if bar_colors is None else bar_colors.reindex(m1d.index).fillna("").tolist()
    vals = (m1d["volume"] if pane is None else pane.reindex(m1d.index)).fillna(0).astype(float)
    sub = [dict(time=t, value=round(v, 3), color=c or NEUTRAL)
           for t, v, c in zip(t1, vals, colors or [""] * len(t1))]
    info = {t: [int(v), round(p, 2)] for t, v, p in zip(t1, m1d["volume"], vals)}
    bg = lambda df: [dict(time=t, value=1 if s else 0) for t, s in zip(_wall(df.index), sessions.label(df.index))]

    data = json.dumps(dict(htf=_candles(left), m1=_candles(m1d, colors), lines=segs, bg5=bg(left),
                           bg1=bg(m1d), sub=sub, level=pane_level, info=info, pane=pane_label,
                           long=LONG, short=SHORT, fibs=fibs))
    closed = trades[trades["result"] != "open"] if len(trades) else trades
    head = (f"{code} {day}" + (f" · {title}" if title else "") + f" · {len(trades)} trades"
            + (f" ({closed['r_net'].sum():+.2f}R net)" if len(closed) else ""))
    return (_TEMPLATE.replace("__TITLE__", head).replace("__LIB__", LIB).replace("__HTF__", htf_label)
            .replace("__DATA__", data).replace("__TRADES__", ttable).replace("__LINES__", ltable)
            .replace("__LINESHOW__", "" if len(lines) else "hidden"))


_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"><title>Backtest chart</title>
<script src="__LIB__"></script>
<style>
 body{margin:0;font:13px system-ui,sans-serif;background:#f4f4f4;color:#222}
 header{padding:8px 16px;font-weight:600}
 .row{display:flex;gap:8px;padding:0 8px;height:70vh}
 .row>div{flex:1;background:#fff;position:relative}
 .tag{position:absolute;top:4px;left:8px;z-index:3;color:#888}
 table{margin:0 16px 12px;border-collapse:collapse}
 h4{margin:12px 16px 4px}
 td,th{padding:2px 10px;text-align:right}
 tr.long td:nth-child(3){color:#1f5fd6} tr.short td:nth-child(3){color:#d6283a}
 .hidden{display:none}
</style></head><body>
<header>__TITLE__ — __HTF__ | M1 · times Europe/Berlin · shaded = session · trade levels: <b style="color:#d6283a">stop</b>
 <b style="color:#555">entry</b> <b style="color:#e08a00">BE</b> <b style="color:#1a9a4a">TP</b></header>
<div class="row"><div id="htf"><span class="tag">__HTF__</span></div><div id="m1"><span class="tag" id="legend">M1</span></div></div>
<h4>Trades</h4>
<table><tr><th>entry</th><th>session</th><th>dir</th><th>price</th><th>stop</th><th>BE</th><th>TP</th>
<th>exit</th><th>result</th><th>R gross</th><th>R net</th></tr>__TRADES__</table>
<div class="__LINESHOW__"><h4>Lines</h4>
<table><tr><th>start</th><th>end</th><th>label</th><th>price</th></tr>__LINES__</table></div>
<script>
const D = __DATA__;
const LONG = D.long, SHORT = D.short;
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
function lines(ch, panel) {
  const m = [];
  for (const L of D.lines) {
    if (L.panel !== 'both' && L.panel !== panel) continue;
    const opt = {color: L.color, lineWidth: 2, lastValueVisible: false, priceLineVisible: false,
                 crosshairMarkerVisible: false};
    ch.addLineSeries(opt).setData([{time: L.prt, value: L.price}, {time: L.exp, value: L.price}]);
    if (L.src < L.prt) ch.addLineSeries({...opt, lineWidth: 1, lineStyle: 1})
        .setData([{time: L.src, value: L.price}, {time: L.prt, value: L.price}]);
    if (L.label) m.push({time: L.src, color: L.color,
        position: L.dir === 'short' ? 'belowBar' : 'aboveBar',
        shape: L.dir === 'long' ? 'arrowDown' : L.dir === 'short' ? 'arrowUp' : 'circle', text: L.label});
  }
  return m;
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
    m.push({time: F.ent, position: long ? 'belowBar' : 'aboveBar', color: long ? LONG : SHORT,
            shape: long ? 'arrowUp' : 'arrowDown', text: long ? 'LONG' : 'SHORT'});
    if (F.result !== 'open') m.push({time: F.ext, position: long ? 'aboveBar' : 'belowBar',
            color: F.result === 'win' ? '#1a9a4a' : F.result === 'be' ? '#e08a00' : '#d6283a',
            shape: 'circle', text: `${F.result.toUpperCase()} ${F.r > 0 ? '+' : ''}${F.r}R`});
  }
  return m;
}
const byTime = (a, b) => a.time - b.time;
const [c5, s5] = make(document.getElementById('htf'), D.htf, D.bg5);
const [c1, s1] = make(document.getElementById('m1'), D.m1, D.bg1);
s5.setMarkers(lines(c5, 'htf').sort(byTime));
lines(c1, 'm1'); s1.setMarkers(trades(c1).sort(byTime));
const sub = c1.addHistogramSeries({priceScaleId: 'sub', lastValueVisible: false, priceLineVisible: false});
sub.setData(D.sub);
if (D.level !== null) sub.createPriceLine({price: D.level, color: '#888', lineWidth: 1, lineStyle: 2,
                                           axisLabelVisible: false});
c1.priceScale('sub').applyOptions({scaleMargins: {top: 0.82, bottom: 0}, visible: false});
c1.priceScale('right').applyOptions({scaleMargins: {top: 0.05, bottom: 0.2}});
const legend = document.getElementById('legend');
c1.subscribeCrosshairMove(p => {
  const b = p.time && p.seriesData.get(s1), i = p.time && D.info[p.time];
  if (!b || !i) { legend.textContent = 'M1'; return; }
  const t = new Date(p.time * 1000).toISOString().slice(11, 16);
  legend.textContent = `M1 ${t}  O ${b.open} H ${b.high} L ${b.low} C ${b.close}  Vol ${i[0]}`
                     + (D.pane !== 'Vol' ? `  ${D.pane} ${i[1]}` : '');
});
c5.timeScale().fitContent(); c1.timeScale().fitContent();
</script></body></html>
"""


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    for k in ("entry_time", "exit_time", "be_time"):
        if k in df.columns:
            df[k] = pd.to_datetime(df[k], utc=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Chart one day of M1 with backtest trades")
    ap.add_argument("--instrument", default="NQ", type=str.upper, choices=list(config.INSTRUMENTS))
    ap.add_argument("--date", required=True, help="Berlin calendar day, YYYY-MM-DD")
    ap.add_argument("--trades", type=Path, help="trade list from backtest.engine (.parquet or .csv)")
    ap.add_argument("--out", help="HTML path (default: system temp dir)")
    ap.add_argument("--no-open", action="store_true", help="do not open the browser")
    args = ap.parse_args()

    d0 = pd.Timestamp(args.date, tz=config.SESSION_TZ)
    m1 = load_m1(args.instrument, d0.tz_convert("UTC"), (d0 + pd.Timedelta(days=1)).tz_convert("UTC"))
    trades = _read(args.trades) if args.trades else None
    out = Path(args.out or Path(tempfile.gettempdir()) / f"chart_{args.instrument.lower()}_{args.date}.html")
    out.write_text(build(m1, args.instrument, args.date, trades), encoding="utf-8")
    print(f"  Wrote {out}")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
