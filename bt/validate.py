"""
Overfitting defences.

  causality_check   rerun the strategy on data cut at random points; everything it said about
                    bars before the cut must match the full-data run (catches centred windows,
                    shift(-1), full-sample statistics …)
  expand / plateau  parameter grid; each point scored as the mean of itself and its neighbours
                    (one step along one parameter) so selection lands on a stable region
  folds / walk_forward
                    rolling train / embargo / test windows; parameters are picked on the train
                    window only and the test windows are stitched into the reported result
  registry          results/trials.jsonl: every (strategy code hash, symbol, bar size, params)
                    ever run; editing the strategy file makes new trials
  holdout ledger    results/holdout_ledger.jsonl: every look at the holdout
  verdicts          pass / warn / fail checks shown on the dashboard
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bt import metrics
from bt.config import RESULTS_DIR

TRIALS, LEDGER = RESULTS_DIR / "trials.jsonl", RESULTS_DIR / "holdout_ledger.jsonl"


class CausalityError(RuntimeError):
    pass


# ── Causality ─────────────────────────────────────────────────────────────────
def _first_mismatch(full, part, cut_time) -> pd.Timestamp | None:
    if isinstance(full, pd.DataFrame):
        a = full[pd.to_datetime(full["time"], utc=True) < cut_time].reset_index(drop=True)
        b = part.reset_index(drop=True)
        if a.shape != b.shape or list(a.columns) != list(b.columns):
            t = pd.concat([a["time"], b["time"]])
            return pd.Timestamp(t.min()) if len(t) else cut_time
        for i in range(len(a)):
            if not all(x == y or (pd.isna(x) and pd.isna(y)) or
                       (isinstance(x, float) and np.isclose(x, y)) for x, y in zip(a.iloc[i], b.iloc[i])):
                return pd.Timestamp(a["time"].iloc[i])
        return None
    idx = part.index
    x, y = full.reindex(idx).to_numpy(dtype=float), part.to_numpy(dtype=float)
    bad = ~(np.isclose(x, y, equal_nan=True))
    return idx[np.argmax(bad)] if bad.any() else None


def causality_check(fn, bars: pd.DataFrame, grid: list[dict], cuts: int, seed=None) -> list[dict]:
    """fn(bars, **params) is the strategy's signal or candidates. Raises CausalityError."""
    rng = np.random.default_rng(seed)
    n, full_out, log = len(bars), {}, []
    for _ in range(cuts):
        k = int(rng.integers(len(grid)))
        cut = int(rng.integers(int(n * 0.1), int(n * 0.9)))
        if k not in full_out:
            full_out[k] = fn(bars, **grid[k])
        part = bars.iloc[:cut].copy()
        part.attrs = dict(bars.attrs)
        bad = _first_mismatch(full_out[k], fn(part, **grid[k]), bars.index[cut])
        log.append({"params": grid[k], "cut": str(bars.index[cut]), "ok": bad is None})
        if bad is not None:
            raise CausalityError(
                f"Look-ahead: with params {grid[k]} the output at {bad} changes when the data ends at "
                f"{bars.index[cut]}. The strategy uses data from after the bar it decides on "
                f"(centred windows, shift(-1), full-sample statistics …).")
    return log


# ── Grid ──────────────────────────────────────────────────────────────────────
def expand(params: dict[str, list]) -> list[dict]:
    return [dict(zip(params, combo)) for combo in itertools.product(*params.values())]


def plateau(scores, params: dict[str, list]) -> np.ndarray:
    """Each grid point's score averaged with its grid neighbours (expand() order)."""
    s = np.nan_to_num(np.asarray(scores, dtype=float)).reshape([len(v) for v in params.values()])
    total, count = s.copy(), np.ones_like(s)
    for ax in range(s.ndim):
        lo, hi = [slice(None)] * s.ndim, [slice(None)] * s.ndim
        lo[ax], hi[ax] = slice(None, -1), slice(1, None)
        total[tuple(hi)] += s[tuple(lo)]
        count[tuple(hi)] += 1
        total[tuple(lo)] += s[tuple(hi)]
        count[tuple(lo)] += 1
    return (total / count).ravel()


# ── Walk-forward ──────────────────────────────────────────────────────────────
def folds(days: pd.DatetimeIndex, train_years: float, test_years: float, embargo_days: int,
          min_test_days: int = 20) -> list[dict]:
    """[start, end) windows. Test windows tile the data after the first train + embargo period."""
    train, test = pd.DateOffset(years=train_years), pd.DateOffset(years=test_years)
    emb, end = pd.Timedelta(days=embargo_days), days[-1] + pd.Timedelta(days=1)
    out, t0 = [], days[0] + train + emb
    while t0 < end:
        t1 = min(t0 + test, end)
        if ((days >= t0) & (days < t1)).sum() >= min_test_days:
            out.append(dict(fold=len(out), train_start=t0 - emb - train, train_end=t0 - emb,
                            test_start=t0, test_end=t1))
        t0 += test
    return out


def holdout_fold(days: pd.DatetimeIndex, holdout: pd.Timestamp, train_years: float,
                 embargo_days: int) -> list[dict]:
    """One fold: train on the train_years before the holdout (less the embargo), test on the rest."""
    h, emb = holdout.tz_localize(None).normalize(), pd.Timedelta(days=embargo_days)
    return [dict(fold=0, train_start=h - emb - pd.DateOffset(years=train_years), train_end=h - emb,
                 test_start=h, test_end=days[-1] + pd.Timedelta(days=1))]


def walk_forward(rets: pd.DataFrame, params: dict, fold_list: list[dict]) -> pd.DataFrame:
    """rets: daily net returns, one column per grid point in expand() order. Per fold the
    plateau-best point on the TRAIN window is chosen; its test window is out-of-sample."""
    rows = []
    for f in fold_list:
        tr = rets[(rets.index >= f["train_start"]) & (rets.index < f["train_end"])]
        te = rets[(rets.index >= f["test_start"]) & (rets.index < f["test_end"])]
        is_sr = np.array([metrics.sharpe(tr[c]) for c in rets.columns])
        score = plateau(is_sr, params)
        k = int(np.argmax(score))
        rows.append({**f, "choice": k, "is_sharpe": float(is_sr[k]), "plateau": float(score[k]),
                     "oos_sharpe": metrics.sharpe(te.iloc[:, k])})
    return pd.DataFrame(rows)


def in_windows(days: pd.DatetimeIndex, wf: pd.DataFrame) -> np.ndarray:
    """Grid index chosen for each day (-1 outside every test window)."""
    out = np.full(len(days), -1)
    for f in wf.itertuples():
        out[(days >= f.test_start) & (days < f.test_end)] = f.choice
    return out


# ── Trial registry and holdout ledger ─────────────────────────────────────────
def code_hash(path: str | Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, default=str) + "\n")


def register(entries: list[dict]) -> None:
    """entries: strategy, code_hash, symbol, bar_size, params, sr_daily (+ anything else)."""
    _append(TRIALS, entries)


def trials(strategy: str) -> pd.DataFrame:
    """Distinct trials of a strategy (latest record per trial key)."""
    df = pd.DataFrame([e for e in _read(TRIALS) if e["strategy"] == strategy])
    if df.empty:
        return df
    df["key"] = (df["code_hash"] + "|" + df["symbol"] + "|" + df["bar_size"] + "|"
                 + df["params"].map(lambda p: json.dumps(p, sort_keys=True)))
    return df.drop_duplicates("key", keep="last")


def holdout_views(strategy: str, symbol: str) -> list[dict]:
    return [e for e in _read(LEDGER) if e["strategy"] == strategy and e["symbol"] == symbol]


def record_holdout(entry: dict) -> None:
    _append(LEDGER, [entry])


# ── Verdicts ──────────────────────────────────────────────────────────────────
def verdicts(m: dict) -> list[dict]:
    """[{check, status: pass|warn|fail|info, value, note}] from a run's meta."""
    def v(check, status, value, note):
        return dict(check=check, status=status, value=value, note=note)

    dsr, is_sr, oos_sr = m["dsr"], m["is_sharpe"], m["metrics"]["sharpe"]
    pos, n, views = m["grid_positive"], m["trials"], m["holdout_views"]
    out = [
        v("Deflated Sharpe", "pass" if dsr >= 0.95 else "warn" if dsr >= 0.5 else "fail", f"{dsr:.2f}",
          "Edge survives the search" if dsr >= 0.95 else "Could be luck from searching"),
        v("OOS vs IS Sharpe", "pass" if oos_sr > 0 and oos_sr >= 0.5 * is_sr else "warn" if oos_sr > 0 else "fail",
          f"{oos_sr:.2f} / {is_sr:.2f}", "Out-of-sample over the mean chosen in-sample Sharpe"),
        v("Grid positive", "pass" if pos >= 0.6 else "warn" if pos >= 0.4 else "fail", f"{pos:.0%}",
          "Share of parameter sets with Sharpe > 0"),
        v("Holdout", "info" if views == 0 else "pass" if views == 1 and not m.get("contaminated") else "fail",
          "not run" if views == 0 else "clean" if views == 1 and not m.get("contaminated") else "contaminated",
          f"{views} view(s) of the holdout for this strategy"),
        v("Trials", "info" if n <= 100 else "warn", str(n), "Distinct configurations ever tested"),
        v("Causality", "pass", f"{len(m['causality'])} cuts", "Signal unchanged when future data is removed"),
    ]
    h = m["health"]
    out.append(v("Data", "pass" if h["quotes"] and h["gaps"] == 0 else "warn",
                 f"{h['gaps']} gaps", "quotes present" if h["quotes"] else "no quotes: costs use fixed slippage"))
    return out
