"""
Backtest engine — strategy-agnostic. A strategy turns M1 bars into a table of trade
candidates; the engine decides which ones are taken, walks each to its exit on M1 and
costs the fills.

Candidates (one row per possible entry, DataFrame):
  required  time        UTC timestamp of the M1 bar the entry fills on (its close), must be in m1
            direction   "long" | "short"
            entry       entry price
            stop        stop price
            tp          target price
  optional  be          breakeven level: once reached, the stop moves to entry (NaN / absent = none)
            risk        planned risk in points (default: entry → stop distance)
            signal_id   candidates sharing one id trade at most once (one_per_signal)
            session     session name for session_stop_r (default: core/sessions.label of time)
  Any other column is carried through to the trade list unchanged.

Position state: candidates are taken in time order. A candidate is skipped while max_open
trades are open (a trade counts as open through its exit bar), with one_per_signal if its
signal already traded, and with session_stop_r once the trades of that session (same
Berlin day) closed so far sum to -session_stop_r R gross or worse — trading resumes in the
next session. max_open=None and one_per_signal=False take every candidate.

Exits walk M1 bars after the entry bar:
  1. until breakeven level or stop: stop first → "loss"
  2. breakeven reached → stop moves to entry from the NEXT bar (target on that bar → "win")
  3. then entry-stop or target → "be" | "win"
Without a breakeven level it is stop → "loss" or target → "win".
Same bar stop + level counts as the stop; a bar opening through a stop fills at its open.
A trade still open on the last bar before a contract roll exits at that bar's close ("roll");
at the end of the data it stays "open". Sessions never close a trade.
Each trade also carries mfe_r / mae_r: the best and worst the trade ever ran, in R,
over the bars it was open (run-up and drawdown before the exit).
Raw prices are costed by core/costs.py.

  python -m backtest.engine --instrument NQ --candidates signals.parquet
  python -m backtest.engine --candidates signals.csv --max-open 0 --out trades.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config
from core import costs, sessions
from data.loader import load_m1
from report import stats

REQUIRED = ["time", "direction", "entry", "stop", "tp"]
TRADE_COLUMNS = ["entry_time", "direction", "entry_price", "stop_price", "be_price", "tp_price",
                 "risk_pts", "be_time", "exit_time", "exit_price", "result", "r_gross",
                 "mfe_r", "mae_r", "signal_id", "session"]
TAIL_DAYS = 7     # CLI: M1 loaded past the last candidate so its trade can reach an exit


def next_contract_start(m1: pd.DataFrame) -> np.ndarray:
    """Per M1 bar, index of the first bar of the next contract (len(m1) if none/unknown).
    A trade still open on bar nxt-1 must exit at that bar's close (result "roll")."""
    n = len(m1)
    if "contract" not in m1.columns:
        return np.full(n, n)
    cid = m1["contract"].to_numpy()
    starts = np.flatnonzero(cid[1:] != cid[:-1]) + 1
    return np.append(starts, n)[np.searchsorted(starts, np.arange(n), side="right")]


def find_exit(o, h, l, start: int, end: int, long: bool, stop: float, tp: float) -> tuple[int, float, str]:
    """
    First bar in [start, end) touching stop or target → (index, price, result).
    Stop and target on the same bar counts as a stop. A bar that opens through
    the stop fills at its open, not the stop. Returns (-1, nan, "open") if neither.
    """
    k, chunk = start, 2048
    while k < end:
        e = min(end, k + chunk)
        sl = (l[k:e] <= stop) if long else (h[k:e] >= stop)
        tg = (h[k:e] >= tp)   if long else (l[k:e] <= tp)
        hit = np.flatnonzero(sl | tg)
        if hit.size:
            j = hit[0]
            m = k + j
            if sl[j]:
                return m, (min(o[m], stop) if long else max(o[m], stop)), "loss"
            return m, tp, "win"
        k, chunk = e, chunk * 2
    return -1, np.nan, "open"


def _exit(o, h, l, c, j: int, end: int, n: int, long: bool, entry: float, stop: float,
          be: float, tp: float) -> tuple[int, float, str, int]:
    """Walk bars (j, end) for one trade → (exit bar, exit price, result, breakeven bar or -1)."""
    be_bar = -1
    if np.isnan(be):
        k, px, res = find_exit(o, h, l, j + 1, end, long, stop, tp)
    else:
        k, px, res = find_exit(o, h, l, j + 1, end, long, stop, be)
        if res == "win":                                    # breakeven level reached on bar k
            be_bar = k
            if (h[k] >= tp) if long else (l[k] <= tp):
                return k, tp, "win", be_bar
            k, px, res = find_exit(o, h, l, k + 1, end, long, entry, tp)
            res = {"loss": "be"}.get(res, res)
    if res == "open" and end < n:
        return end - 1, c[end - 1], "roll", be_bar
    return k, px, res, be_bar


def _normalise(cand: pd.DataFrame) -> pd.DataFrame:
    """Check required columns and fill the optional ones; time order, stable."""
    missing = [k for k in REQUIRED if k not in cand.columns]
    if missing:
        raise ValueError(f"candidates missing columns: {missing}")
    c = cand.copy()
    c["time"] = pd.to_datetime(c["time"], utc=True)
    sign = np.where(c["direction"] == "long", 1.0, -1.0)
    if "be" not in c.columns:
        c["be"] = np.nan
    if "risk" not in c.columns:
        c["risk"] = sign * (c["entry"] - c["stop"])
    if "signal_id" not in c.columns:
        c["signal_id"] = np.arange(len(c))              # every candidate its own signal
    if "session" not in c.columns:
        c["session"] = sessions.label(pd.DatetimeIndex(c["time"]))
    return c.sort_values("time", kind="stable").reset_index(drop=True)


def run(m1: pd.DataFrame, candidates: pd.DataFrame, code: str,
        slippage_ticks: float = config.SLIPPAGE_TICKS,
        commission: float | None = config.COMMISSION,
        risk_usd: float = config.RISK_PER_TRADE,
        max_open: int | None = config.MAX_OPEN_TRADES,
        one_per_signal: bool = config.ONE_TRADE_PER_SIGNAL,
        session_stop_r: float | None = config.SESSION_STOP_R) -> pd.DataFrame:
    """Costed trade list for one instrument (open trades included, result == "open")."""
    cand = _normalise(candidates)
    extra = [k for k in cand.columns if k not in REQUIRED + ["be", "risk", "signal_id", "session"]]

    idx, n = m1.index, len(m1)
    o, h, l, c = (m1[k].to_numpy(dtype=np.float64) for k in ("open", "high", "low", "close"))
    nxt = next_contract_start(m1)
    bars = idx.get_indexer(cand["time"])
    if (bars < 0).any():
        raise ValueError(f"{int((bars < 0).sum())} candidate times are not M1 bars in m1, "
                         f"first {cand['time'][bars < 0].iloc[0]}")
    day = pd.DatetimeIndex(cand["time"]).tz_convert(config.SESSION_TZ).date
    exits, used, rows = [], set(), []                      # exit bar of every trade taken
    session_r: dict = {}                                   # (day, session) → [(exit bar, R gross)]

    for j, r, dy in zip(bars, cand.itertuples(index=False), day):
        if max_open is not None and sum(x >= j for x in exits) >= max_open:
            continue
        if one_per_signal and r.signal_id in used:
            continue
        key = (dy, r.session)
        if session_stop_r is not None and \
                sum(rg for x, rg in session_r.get(key, []) if x <= j) <= -session_stop_r:
            continue
        if nxt[j] <= j + 1:                                 # last bar before a roll: no entry
            continue
        long = r.direction == "long"
        x, px, res, be_bar = _exit(o, h, l, c, j, int(nxt[j]), n, long, r.entry, r.stop,
                                   float(r.be), r.tp)
        r_gross = np.nan if res == "open" else (px - r.entry) / r.risk * (1 if long else -1)
        # Run-up / drawdown in R over the bars held. The exit bar counts only through the
        # exit price: whatever it did after the fill happened without us in the trade.
        end_bar = x if x >= 0 else n - 1
        hi, lo = h[j + 1:end_bar], l[j + 1:end_bar]
        last_px = px if x >= 0 else c[end_bar]
        best  = max(hi.max(), last_px) if hi.size else last_px
        worst = min(lo.min(), last_px) if lo.size else last_px
        up   = (best - r.entry) if long else (r.entry - worst)
        down = (worst - r.entry) if long else (r.entry - best)
        mfe, mae = max(0.0, up / r.risk), min(0.0, down / r.risk)
        rows.append((idx[j], r.direction, r.entry, r.stop, r.be, r.tp, r.risk,
                     idx[be_bar] if be_bar >= 0 else pd.NaT, idx[x] if x >= 0 else pd.NaT, px, res,
                     r_gross, mfe, mae, r.signal_id, r.session)
                    + tuple(getattr(r, k) for k in extra))
        exits.append(x if x >= 0 else n)
        used.add(r.signal_id)
        if res != "open":
            session_r.setdefault(key, []).append((x, r_gross))

    trades = pd.DataFrame(rows, columns=TRADE_COLUMNS + extra)
    return costs.apply(trades, code, slippage_ticks, commission, risk_usd)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest a candidates file on cached M1 data")
    ap.add_argument("--instrument", default="NQ", type=str.upper, choices=list(config.INSTRUMENTS))
    ap.add_argument("--candidates", required=True, type=Path, help=".parquet or .csv, columns above")
    ap.add_argument("--max-open", type=int, default=config.MAX_OPEN_TRADES,
                    help="trades open at once; 0 = no limit")
    ap.add_argument("--all-signals", action="store_true",
                    help="no open-trade limit, a signal may trigger repeatedly")
    ap.add_argument("--out", type=Path, help="write the trade list here (.parquet or .csv)")
    args = ap.parse_args()

    cand = _read(args.candidates)
    t = pd.to_datetime(cand["time"], utc=True)
    m1 = load_m1(args.instrument, t.min(), t.max() + pd.Timedelta(days=TAIL_DAYS))
    trades = run(m1, cand, args.instrument,
                 max_open=None if args.all_signals else (args.max_open or None),
                 one_per_signal=False if args.all_signals else config.ONE_TRADE_PER_SIGNAL)
    closed = trades[trades["result"] != "open"]
    s = stats.compute(closed["r_net"], closed["entry_time"], config.RISK_PER_TRADE,
                      config.INITIAL_BALANCE)
    print(f"  {len(cand)} candidates -> {len(trades)} trades ({len(trades) - len(closed)} open)")
    for k, v in s.items():
        print(f"  {k:<16} {v:,.2f}" if isinstance(v, float) else f"  {k:<16} {v}")
    if args.out:
        if args.out.suffix == ".parquet":
            trades.to_parquet(args.out)
        else:
            trades.to_csv(args.out, index=False)
        print(f"  Wrote {args.out}")


if __name__ == "__main__":
    main()
