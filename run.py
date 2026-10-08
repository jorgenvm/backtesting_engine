"""
Backtester command line. Workflow: ingest → research (as often as you like; every attempt is
counted) → holdout ONCE at the end. Anything after the holdout is research again.

  python run.py ingest data/cache/m1_nq_*.parquet --symbol NQ          # any CSV / Parquet, ticks or bars
  python run.py ingest ticks.csv --symbol ES --tz America/Chicago --map "ts=Date || ' ' || Time"
  python run.py ingest --databento --symbol NQ --start-year 2011 --end-year 2026 [--schema tbbo] [--estimate]
  python run.py health   --symbol NQ [--bar-size 5min]
  python run.py research --strategy ma_cross [--symbol NQ] [--bar-size 5min]
  python run.py holdout  --strategy ma_cross [--force]
  python run.py dashboard [--no-open]
  python run.py chart --date 2023-06-14 [--run latest]                  # one day of a run's trades

A strategy may set SYMBOL, BAR_SIZE and DELAY (default 1) next to PARAMS; flags override them.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

from bt import data, engine, metrics, validate
from bt.config import CFG, RESULTS_DIR, holdout_start, instrument


def load_strategy(name: str):
    mod = importlib.import_module(f"strategies.{name}")
    if not (hasattr(mod, "signal") or hasattr(mod, "candidates")) or not hasattr(mod, "PARAMS"):
        raise SystemExit(f"strategies/{name}.py needs PARAMS and a signal() or candidates() function.")
    return mod


def evaluate(name: str, mode: str, symbol: str | None = None, bar_size: str | None = None,
             force: bool = False) -> Path:
    """Research (walk-forward before the holdout) or holdout (one fold: the holdout itself)."""
    strat = load_strategy(name)
    fn = getattr(strat, "signal", None) or strat.candidates
    symbol = (symbol or getattr(strat, "SYMBOL", None) or CFG["symbol"]).upper()
    bar_size = bar_size or getattr(strat, "BAR_SIZE", None) or CFG["bar_size"]
    delay, params = getattr(strat, "DELAY", 1), strat.PARAMS
    grid, wf_cfg, code = validate.expand(params), CFG["walk_forward"], validate.code_hash(strat.__file__)
    if len(grid) > wf_cfg["max_grid"]:
        print(f"  ! {len(grid)} parameter sets (> {wf_cfg['max_grid']}): every one counts as a trial.")

    views = validate.holdout_views(name, symbol)
    if mode == "holdout" and views and not force:
        raise SystemExit(f"The {symbol} holdout was already viewed for {name} ({len(views)}x, first "
                         f"{views[0]['time']}). Re-running needs --force and is flagged contaminated.")

    bars, health = data.load_bars(symbol, bar_size, "research" if mode == "research" else "full")
    exit_bars = bars
    if not hasattr(strat, "signal") and CFG["exit_bar_size"] != bar_size:
        exit_bars = data.load_bars(symbol, CFG["exit_bar_size"], "research" if mode == "research" else "full")[0]
    print(f"\n  {name} · {symbol} {bar_size} · {mode} · {len(bars):,} bars "
          f"{bars.index[0]:%Y-%m-%d} → {bars.index[-1]:%Y-%m-%d} · {len(grid)} parameter sets")

    v = CFG["validation"]
    causality = validate.causality_check(fn, bars, grid, v["causality_cuts"], v["seed"])
    print(f"  causality   ok ({len(causality)} cuts)")

    rets, n_trades = {}, []
    for k, p in enumerate(grid):
        daily, trades = engine.simulate(fn(bars, **p), bars, exit_bars, symbol, delay)
        rets[k] = engine.returns(daily)["ret"]
        n_trades.append(len(trades) if len(trades) else int((daily["turnover"] > 0).sum()))
    rets = pd.DataFrame(rets).fillna(0.0)
    hs = holdout_start().tz_localize(None)
    pre = rets[rets.index < hs]                               # the registry never sees holdout data

    known = set(validate.trials(name).get("key", []))
    entries = []
    for k, p in enumerate(grid):
        key = f"{code}|{symbol}|{bar_size}|{json.dumps(p, sort_keys=True)}"
        if mode == "research" or key not in known:
            entries.append(dict(time=pd.Timestamp.now().isoformat(timespec="seconds"), strategy=name,
                                code_hash=code, symbol=symbol, bar_size=bar_size, params=p,
                                sr_daily=metrics.sharpe(pre[k], annualise=False), days=len(pre)))
    if not instrument(symbol).get("null_test"):
        validate.register(entries)
    tri = validate.trials(name)
    n_trials, sr_var = len(tri), float(tri["sr_daily"].var(ddof=1)) if len(tri) > 1 else 0.0

    if mode == "research":
        fold_list = validate.folds(rets.index, wf_cfg["train_years"], wf_cfg["test_years"], wf_cfg["embargo_days"])
        if not fold_list:
            raise SystemExit("Not enough data for one walk-forward fold (train + embargo + test).")
    else:
        fold_list = validate.holdout_fold(rets.index, holdout_start(), wf_cfg["train_years"], wf_cfg["embargo_days"])
    wf = validate.walk_forward(rets, params, fold_list)

    # Stitch the out-of-sample windows: each uses the parameters its own train window chose.
    choice = validate.in_windows(engine.trading_day(bars.index), wf)
    parts = []
    for k in sorted(set(wf["choice"])):
        out = fn(bars, **grid[k])
        if isinstance(out, pd.DataFrame):
            parts.append(out[validate.in_windows(engine.trading_day(pd.to_datetime(out["time"], utc=True)), wf) == k])
        else:
            parts.append(out.reindex(bars.index).where(choice == k))
    if isinstance(parts[0], pd.DataFrame):
        stitched = pd.concat(parts).sort_values("time", kind="stable")
    else:
        stitched = pd.concat(parts, axis=1).max(axis=1)        # one non-NaN column per bar
    daily, trades = engine.simulate(stitched, bars, exit_bars, symbol, delay)
    oos_start = wf["test_start"].min()
    d = engine.returns(daily[daily.index >= oos_start])
    bench = engine.returns(engine.benchmark(bars, symbol).reindex(d.index).fillna(0.0))
    d["bench_ret"], d["bench_equity"] = bench["ret_gross"], bench["equity_gross"]
    d["drawdown"] = metrics.drawdown(d["ret"])

    sel = rets[rets.index < hs] if mode == "research" else \
        rets[(rets.index >= fold_list[0]["train_start"]) & (rets.index < fold_list[0]["train_end"])]
    grid_sr = np.array([metrics.sharpe(sel[k]) for k in range(len(grid))])
    gdf = pd.DataFrame(grid)
    gdf["sharpe"], gdf["plateau"] = grid_sr, validate.plateau(grid_sr, params)
    gdf["chosen"] = [int((wf["choice"] == k).sum()) for k in range(len(grid))]
    gdf["trades"] = n_trades
    wf["params"] = [json.dumps(grid[k]) for k in wf["choice"]]

    contaminated = any(e.get("contaminated") for e in views) or (mode == "holdout" and bool(views))
    m = metrics.summary(d, d["bench_ret"])
    meta = dict(
        strategy=name, mode=mode, symbol=symbol, bar_size=bar_size, delay=delay, code_hash=code,
        created=pd.Timestamp.now().isoformat(timespec="seconds"), params=params,
        kind="positions" if hasattr(strat, "signal") else "candidates",
        oos_start=str(oos_start.date()), oos_end=str(d.index[-1].date()),
        holdout_start=str(hs.date()), metrics=m, is_sharpe=float(wf["is_sharpe"].mean()),
        dsr=metrics.deflated_sharpe(d["ret"], n_trials, sr_var), trials=n_trials, sr_var=sr_var,
        grid_positive=float((grid_sr > 0).mean()), causality=causality, health=health,
        holdout_views=len(views) + (mode == "holdout"), contaminated=contaminated,
        evaluation=metrics.evaluation(trades, d, CFG["account"]["initial_balance"]),
        config={k: CFG[k] for k in ("costs", "account", "bracket", "walk_forward", "exit_bar_size")},
    )
    meta["verdicts"] = validate.verdicts(meta)

    run_dir = RESULTS_DIR / "runs" / f"{pd.Timestamp.now():%Y%m%d_%H%M%S}_{name}_{mode}"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    d.to_parquet(run_dir / "daily.parquet")
    wf.to_parquet(run_dir / "folds.parquet")
    gdf.to_parquet(run_dir / "grid.parquet")
    if len(trades):
        trades.to_parquet(run_dir / "trades.parquet")
    if mode == "holdout":
        validate.record_holdout(dict(time=meta["created"], strategy=name, symbol=symbol, bar_size=bar_size,
                                     code_hash=code, params=wf["params"].iloc[0], sharpe=m["sharpe"],
                                     forced=force, contaminated=contaminated, run=run_dir.name))
    _print(meta, wf, run_dir)
    return run_dir


def _print(meta: dict, wf: pd.DataFrame, run_dir: Path) -> None:
    m = meta["metrics"]
    print("  folds       " + "  ".join(f"{r.test_start:%Y}:{r.oos_sharpe:+.2f}" for r in wf.itertuples()))
    print(f"  OOS         {meta['oos_start']} → {meta['oos_end']}   CAGR {m['cagr']:+.1%}   Sharpe {m['sharpe']:.2f}"
          f" (buy & hold {m['bench_sharpe']:.2f})   max DD {m['max_dd']:.1%}   cost drag {m['cost_drag']:.1%}/yr")
    icon = {"pass": "+", "warn": "!", "fail": "x", "info": "·"}
    for v in meta["verdicts"]:
        print(f"  {icon[v['status']]} {v['check']:<17} {v['value']:<14} {v['note']}")
    print(f"  saved       {run_dir}\n  view        python run.py dashboard\n")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="store raw data (file / glob / Databento)")
    p.add_argument("path", nargs="?", help="CSV or Parquet file, globs allowed")
    p.add_argument("--symbol", required=True)
    p.add_argument("--kind", choices=data.KINDS, help="default: inferred from the columns")
    p.add_argument("--tz", default="UTC", help="timezone of tz-naive source timestamps")
    p.add_argument("--map", nargs="*", default=[], metavar="COL=SQL", help="column = SQL expression")
    p.add_argument("--databento", action="store_true")
    p.add_argument("--start-year", type=int)
    p.add_argument("--end-year", type=int)
    p.add_argument("--schema", default="ohlcv-1m", help="ohlcv-1m | trades | tbbo …")
    p.add_argument("--estimate", action="store_true", help="Databento cost only, download nothing")
    p.add_argument("--overwrite", action="store_true", help="Databento: fetch stored years again")

    p = sub.add_parser("health", help="data health report")
    p.add_argument("--symbol", default=CFG["symbol"])
    p.add_argument("--bar-size", default=CFG["bar_size"])

    for cmd in ("research", "holdout"):
        p = sub.add_parser(cmd)
        p.add_argument("--strategy", required=True, help="module name in strategies/")
        p.add_argument("--symbol")
        p.add_argument("--bar-size")
        if cmd == "holdout":
            p.add_argument("--force", action="store_true", help="view the holdout again (contaminated)")

    p = sub.add_parser("dashboard", help="write results/dashboard.html")
    p.add_argument("--no-open", action="store_true")

    p = sub.add_parser("chart", help="one day of a run on exit bars, trades drawn")
    p.add_argument("--date", required=True, help=f"calendar day in {CFG['timezone']}, YYYY-MM-DD")
    p.add_argument("--run", default="latest", help="run folder name (default: newest)")
    p.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    if args.cmd == "ingest":
        if args.databento:
            cost = data.ingest_databento(args.symbol, args.start_year, args.end_year or args.start_year,
                                         args.schema, args.estimate, args.overwrite)
            print(f"  Databento {'estimate' if args.estimate else 'billed (est.)'}: ${cost:,.2f}")
        elif args.path:
            mapping = dict(m.split("=", 1) for m in args.map)
            rows = data.ingest(args.path, args.symbol, args.kind, mapping, args.tz)
            for y, n in rows.items():
                print(f"  {args.symbol.upper()} {y}  {n:>12,} rows")
        else:
            ap.error("ingest needs a path or --databento")
    elif args.cmd == "health":
        print(json.dumps(data.load_bars(args.symbol, args.bar_size, "research")[1], indent=1))
    elif args.cmd in ("research", "holdout"):
        try:
            evaluate(args.strategy, args.cmd, args.symbol, args.bar_size, getattr(args, "force", False))
        except validate.CausalityError as e:
            raise SystemExit(f"  x causality check failed: {e}")
    elif args.cmd == "dashboard":
        import dashboard
        out = dashboard.build()
        print(f"  Wrote {out}")
        if not args.no_open:
            webbrowser.open(out.resolve().as_uri())
    elif args.cmd == "chart":
        from bt import chart
        out = chart.for_run(args.run, args.date)
        print(f"  Wrote {out}")
        if not args.no_open:
            webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
