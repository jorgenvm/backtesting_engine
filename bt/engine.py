"""
Backtest engine. A strategy has one of two shapes; both end in the same daily P&L frame.

  positions   signal(bars, **params) -> Series: target units per bar (contracts, shares, coins);
              NaN = flat. Fractions are allowed.
  candidates  candidates(bars, **params) -> DataFrame of bracket orders, one row per entry:
                required  time        bar the decision was made on (index value of bars)
                          direction   "long" | "short"
                          stop, tp    stop and target prices
                optional  be          breakeven level: once reached the stop moves to entry
                          size        units (default: account.risk_per_trade / risk in USD)
                          signal_id   candidates sharing one id trade at most once (one_per_signal)
                          session     for bracket.session_stop_r (default: bt.sessions.label)
              Any other column is carried through to the trade list.

Timing contract (enforced here, never left to the strategy): a decision on bar t, made with
bars up to and including t's close, fills at the OPEN of bar t + delay. delay >= 1; 0 is rejected.

Costs are always on, per unit traded: half the round-trip commission, config slippage and half the
bar's measured bid/ask spread when the data has quotes. A resting bracket target fills without
slippage or spread. A contract roll under an open position counts as a close plus a re-open
(both charged), and the price jump between the two contracts is not P&L.

Bracket exits walk the exit bars (config exit_bar_size) from the entry bar on:
  1. until the breakeven level or stop: stop first → "loss"
  2. breakeven reached → stop moves to entry from the NEXT bar (target on that bar → "win")
  3. then entry-stop or target → "be" | "win"
Stop and target on the same bar count as the stop; a bar opening through the stop fills at its
open. A trade still open on the last bar before a roll exits at that bar's close ("roll"); one
open at the end of the data is marked at the last close ("open"). An entry whose fill price is
already through its stop or target is skipped.

Daily frame (index: trading day in config timezone):
  pnl_gross  USD before costs   cost  USD   pnl  USD net   turnover  units traded
Bracket P&L is booked on the exit day. Positions also get a trade list: one round trip from
opening to flat (or a flip), without stop / target / R columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bt import sessions
from bt.config import CFG, TZ, instrument

TRADE_COLUMNS = ["signal_time", "entry_time", "direction", "entry_price", "stop_price", "be_price",
                 "tp_price", "risk_pts", "size", "be_time", "exit_time", "exit_price", "result",
                 "r_gross", "r_net", "pnl_gross", "pnl", "mfe_r", "mae_r", "signal_id", "session"]


def _check_delay(delay: int) -> None:
    if int(delay) != delay or delay < 1:
        raise ValueError(f"delay must be an integer >= 1 (fill on a later bar than the decision), got {delay}")


def _unit_cost(bars: pd.DataFrame, inst: dict, costs: bool) -> np.ndarray:
    """USD to trade one unit at market on each bar: half commission, slippage, half spread."""
    if not costs:
        return np.zeros(len(bars))
    c = CFG["costs"]
    spread = bars["spread"].fillna(0.0).to_numpy() if c["half_spread"] else 0.0
    return (inst["commission"] / 2 + c["slippage_ticks"] * inst["tick"] * inst["point_value"]
            + 0.5 * spread * inst["point_value"]) * np.ones(len(bars))


_DAYS: dict = {}


def trading_day(index) -> pd.DatetimeIndex:
    """Calendar day in config timezone, tz-naive. Cached for the few big bar indexes reused per run."""
    hit = _DAYS.get(id(index))
    if hit is not None and hit[0] is index:
        return hit[1]
    days = pd.DatetimeIndex(index).tz_convert(TZ).tz_localize(None).normalize()
    if isinstance(index, pd.DatetimeIndex):
        if len(_DAYS) >= 4:
            _DAYS.pop(next(iter(_DAYS)))
        _DAYS[id(index)] = (index, days)
    return days


def _daily(index: pd.DatetimeIndex, gross, cost, traded) -> pd.DataFrame:
    d = pd.DataFrame({"pnl_gross": gross, "cost": cost, "turnover": traded},
                     index=trading_day(index)).groupby(level=0).sum()
    d["pnl"] = d["pnl_gross"] - d["cost"]
    d.index.name = "date"
    return d


def returns(daily: pd.DataFrame, initial: float = CFG["account"]["initial_balance"]) -> pd.DataFrame:
    """Add equity / ret (net) and equity_gross / ret_gross: each day's P&L over the prior equity."""
    d = daily.copy()
    for sfx, pnl in (("", "pnl"), ("_gross", "pnl_gross")):
        eq = initial + d[pnl].cumsum()
        prev = eq.shift(1).fillna(initial)
        d["equity" + sfx] = eq
        d["ret" + sfx] = np.where(prev > 0, d[pnl] / prev.where(prev > 0, 1.0), 0.0)
    return d


# ── Positions ─────────────────────────────────────────────────────────────────
def simulate_positions(bars: pd.DataFrame, target, symbol: str, delay: int = 1,
                       costs: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(daily frame, trade list) for target units decided per bar, filled at the open delay bars later."""
    _check_delay(delay)
    inst = instrument(symbol)
    if not isinstance(target, pd.Series):
        target = pd.Series(np.asarray(target, dtype=float), index=bars.index)
    held = target.astype(float).reindex(bars.index).fillna(0.0).shift(delay).fillna(0.0).to_numpy()
    prev = np.r_[0.0, held[:-1]]                       # units held into this bar's open
    o, c, cid = (bars[k].to_numpy() for k in ("open", "close", "contract"))
    roll = np.r_[False, cid[1:] != cid[:-1]]
    gap = np.where(roll, 0.0, prev * (o - np.r_[o[0], c[:-1]]))     # previous close → this open
    move = held * (c - o)
    traded = np.where(roll, np.abs(prev) + np.abs(held), np.abs(held - prev))
    unit = _unit_cost(bars, inst, costs)
    daily = _daily(bars.index, (gap + move) * inst["point_value"], traded * unit, traded)
    return daily, _position_trades(bars, held, prev, gap, move, traded, unit, inst["point_value"], delay)


def _position_trades(bars, held, prev, gap, move, traded, unit, pv, delay) -> pd.DataFrame:
    """Round trips: flat (or the other side) → position → flat (or flipped). Resizing and rolls
    stay inside one trade. A flip's cost is split between the closing and the opening trade."""
    start = (held != 0) & (np.sign(held) != np.sign(prev))
    k = int(start.sum())
    if not k:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    tid = np.where(held != 0, np.cumsum(start) - 1, -1)          # trade held through each bar
    ptid = np.r_[-1, tid[:-1]]                                    # trade held into each bar's open
    same = tid == ptid
    gross, cost, size = np.zeros(k), np.zeros(k), np.zeros(k)

    def add(ids, vals):
        m = ids >= 0
        return ids[m], vals[m]

    for arr, (ids, vals) in ((gross, add(ptid, gap * pv)), (gross, add(tid, move * pv)),
                             (cost, add(np.where(same, tid, -1), traded * unit)),
                             (cost, add(np.where(same, -1, ptid), np.abs(prev) * unit)),
                             (cost, add(np.where(same, -1, tid), np.abs(held) * unit))):
        np.add.at(arr, ids, vals)
    np.maximum.at(size, tid[tid >= 0], np.abs(held[tid >= 0]))

    idx, n = bars.index, len(bars)
    o, c = bars["open"].to_numpy(), bars["close"].to_numpy()
    entry = np.flatnonzero(start)
    exit_at = np.full(k, n - 1)
    closes = np.flatnonzero((ptid >= 0) & ~same)
    exit_at[ptid[closes]] = closes
    is_open = np.ones(k, dtype=bool)
    is_open[ptid[closes]] = False
    pnl = gross - cost
    t = pd.DataFrame({
        "signal_time": idx[np.maximum(entry - delay, 0)], "entry_time": idx[entry],
        "direction": np.where(held[entry] > 0, "long", "short"), "entry_price": o[entry],
        "size": size, "exit_time": idx[exit_at], "exit_price": np.where(is_open, c[n - 1], o[exit_at]),
        "result": np.where(is_open, "open", np.where(pnl > 0, "win", "loss")),
        "pnl_gross": gross, "pnl": pnl, "signal_id": np.arange(k),
        "session": sessions.label(idx[entry]),
    })
    return t.reindex(columns=TRADE_COLUMNS)


def benchmark(bars: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Buy-and-hold one unit from the first fillable bar, before costs (use its pnl_gross)."""
    return simulate_positions(bars, pd.Series(1.0, index=bars.index), symbol, costs=False)[0]


# ── Candidates (bracket orders) ───────────────────────────────────────────────
def next_contract_start(bars: pd.DataFrame) -> np.ndarray:
    """Per bar, index of the first bar of the next contract (len(bars) if none)."""
    n, cid = len(bars), bars["contract"].to_numpy()
    starts = np.flatnonzero(cid[1:] != cid[:-1]) + 1
    return np.append(starts, n)[np.searchsorted(starts, np.arange(n), side="right")]


def find_exit(o, h, l, start: int, end: int, long: bool, stop: float, tp: float) -> tuple[int, float, str]:
    """First bar in [start, end) touching stop or target → (index, price, result); stop wins ties,
    a bar opening through the stop fills at its open. (-1, nan, "open") if neither."""
    k, chunk = start, 2048
    while k < end:
        e = min(end, k + chunk)
        sl = (l[k:e] <= stop) if long else (h[k:e] >= stop)
        tg = (h[k:e] >= tp) if long else (l[k:e] <= tp)
        hit = np.flatnonzero(sl | tg)
        if hit.size:
            m = k + hit[0]
            if sl[hit[0]]:
                return m, (min(o[m], stop) if long else max(o[m], stop)), "loss"
            return m, tp, "win"
        k, chunk = e, chunk * 2
    return -1, np.nan, "open"


def _exit(o, h, l, c, j: int, end: int, n: int, long: bool, entry: float, stop: float,
          be: float, tp: float) -> tuple[int, float, str, int]:
    """Walk bars [j, end) for one trade → (exit bar, exit price, result, breakeven bar or -1)."""
    be_bar = -1
    if np.isnan(be):
        k, px, res = find_exit(o, h, l, j, end, long, stop, tp)
    else:
        k, px, res = find_exit(o, h, l, j, end, long, stop, be)
        if res == "win":                                    # breakeven level reached on bar k
            be_bar = k
            if (h[k] >= tp) if long else (l[k] <= tp):
                return k, tp, "win", be_bar
            k, px, res = find_exit(o, h, l, k + 1, end, long, entry, tp)
            res = {"loss": "be"}.get(res, res)
    if res == "open":
        return (end - 1, c[end - 1], "roll" if end < n else "open", be_bar)
    return k, px, res, be_bar


def simulate_candidates(bars: pd.DataFrame, cand: pd.DataFrame, symbol: str, bar_size: str,
                        delay: int = 1, costs: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(daily frame, trade list) for bracket candidates decided on bar_size bars, walked on `bars`."""
    _check_delay(delay)
    missing = [k for k in ("time", "direction", "stop", "tp") if k not in cand.columns]
    if missing:
        raise ValueError(f"candidates missing columns: {missing}")
    inst, b = instrument(symbol), CFG["bracket"]
    pv, risk_usd = inst["point_value"], CFG["account"]["risk_per_trade"]
    comm_half = inst["commission"] / 2 if costs else 0.0
    c = cand.copy()
    c["time"] = pd.to_datetime(c["time"], utc=True)
    for k, default in (("be", np.nan), ("size", np.nan), ("signal_id", np.arange(len(c)))):
        if k not in c.columns:
            c[k] = default
    if "session" not in c.columns:
        c["session"] = sessions.label(pd.DatetimeIndex(c["time"]))
    c = c.sort_values("time", kind="stable").reset_index(drop=True)
    extra = [k for k in c.columns if k not in ("time", "direction", "stop", "tp", "be", "size",
                                                 "signal_id", "session")]

    idx, n = bars.index, len(bars)
    o, h, l, cl = (bars[k].to_numpy(dtype=np.float64) for k in ("open", "high", "low", "close"))
    cid, nxt, unit = bars["contract"].to_numpy(), next_contract_start(bars), _unit_cost(bars, inst, costs)
    fills = idx.searchsorted(c["time"] + delay * pd.Timedelta(bar_size), side="left")
    day = trading_day(c["time"])
    exits, used, rows, session_r = [], set(), [], {}
    gross, cost, traded = np.zeros(n), np.zeros(n), np.zeros(n)

    for j, r, dy in zip(fills, c.itertuples(index=False), day):
        if j >= n or (j > 0 and cid[j] != cid[j - 1]):       # no bar left, or a roll before the fill
            continue
        exits = [x for x in exits if x >= j]               # trades still open at this bar
        if b["max_open"] is not None and len(exits) >= b["max_open"]:
            continue
        if b["one_per_signal"] and r.signal_id in used:
            continue
        key = (dy, r.session)
        if b["session_stop_r"] is not None and \
                sum(rg for x, rg in session_r.get(key, []) if x < j) <= -b["session_stop_r"]:
            continue
        long, entry = r.direction == "long", o[j]
        if not ((r.stop < entry < r.tp) if long else (r.tp < entry < r.stop)):
            continue                                         # filled through the stop or target
        sign, risk = (1.0 if long else -1.0), abs(entry - r.stop)
        x, px, res, be_bar = _exit(o, h, l, cl, j, int(nxt[j]), n, long, entry, r.stop, float(r.be), r.tp)
        size = r.size if pd.notna(r.size) else risk_usd / (risk * pv)
        # Entry and market exits pay unit cost; a target fill pays only its half commission.
        t_cost = size * (unit[j] + (comm_half if res == "win" else unit[x]))
        t_gross = size * sign * (px - entry) * pv
        # Run-up / drawdown in R while held; the exit bar counts only through the exit price.
        hi, lo = h[j:x], l[j:x]
        best, worst = hi.max(initial=px), lo.min(initial=px)
        up, down = ((best - entry), (worst - entry)) if long else ((entry - worst), (entry - best))
        rows.append((r.time, idx[j], r.direction, entry, r.stop, r.be, r.tp, risk, size,
                     idx[be_bar] if be_bar >= 0 else pd.NaT, idx[x], px, res,
                     sign * (px - entry) / risk, (t_gross - t_cost) / (size * risk * pv),
                     t_gross, t_gross - t_cost, max(0.0, up / risk), min(0.0, down / risk),
                     r.signal_id, r.session) + tuple(getattr(r, k) for k in extra))
        gross[x] += t_gross
        cost[x] += t_cost
        traded[j] += size
        traded[x] += size
        exits.append(x)
        used.add(r.signal_id)
        session_r.setdefault(key, []).append((x, sign * (px - entry) / risk))

    trades = pd.DataFrame(rows, columns=TRADE_COLUMNS + extra)
    return _daily(idx, gross, cost, traded), trades


# ── Dispatch ──────────────────────────────────────────────────────────────────
def orders(strategy, bars: pd.DataFrame, params: dict):
    """Run the strategy: a positions Series or a candidates DataFrame."""
    fn = getattr(strategy, "signal", None) or getattr(strategy, "candidates")
    return fn(bars, **params)


def simulate(out, bars: pd.DataFrame, exit_bars: pd.DataFrame, symbol: str,
             delay: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(daily frame, trade list) for a strategy's output."""
    if isinstance(out, pd.DataFrame):
        return simulate_candidates(exit_bars, out, symbol, bars.attrs["bar_size"], delay)
    return simulate_positions(bars, out, symbol, delay)
