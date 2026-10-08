"""Known-answer tests. Run: python -m pytest"""
import numpy as np
import pandas as pd
import pytest

from bt import data, engine, metrics, validate
from bt.config import CFG, holdout_start, instrument

SYM = "SYNTH"


def make_bars(n=2000, seed=0, gaps=True) -> pd.DataFrame:
    """Random-walk 5-minute bars; with gaps=False each open equals the previous close."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-06", periods=n, freq="5min", tz="UTC")
    close = 10_000 + np.cumsum(rng.normal(0, 5, n))
    open_ = np.r_[close[0], close[:-1]] + (rng.normal(0, 1, n) if gaps else 0)
    wick = np.abs(rng.normal(0, 3, n))
    bars = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + wick,
                         "low": np.minimum(open_, close) - wick, "close": close, "volume": 1.0,
                         "ticks": np.nan, "spread": np.nan, "contract": 0}, index=idx)
    bars.attrs.update(symbol=SYM, bar_size="5min")
    return bars


def entry_cost() -> float:
    inst = instrument(SYM)
    return inst["commission"] / 2 + CFG["costs"]["slippage_ticks"] * inst["tick"] * inst["point_value"]


# ── Engine timing and costs ───────────────────────────────────────────────────
def test_always_long_equals_buy_and_hold_minus_one_entry_cost():
    bars = make_bars()
    daily, trades = engine.simulate_positions(bars, pd.Series(1.0, index=bars.index), SYM)
    bench = engine.benchmark(bars, SYM)
    assert daily["pnl"].sum() == pytest.approx(bench["pnl_gross"].sum() - entry_cost())
    assert daily["turnover"].sum() == 1.0
    assert len(trades) == 1 and trades["pnl"].iloc[0] == pytest.approx(daily["pnl"].sum())


def test_position_trades_split_round_trips_and_their_costs():
    bars = make_bars(n=40)
    target = pd.Series(0.0, index=bars.index)
    target.iloc[5:15], target.iloc[15:25], target.iloc[30:35] = 1.0, -2.0, 1.0   # long, flip short, flat, long
    daily, trades = engine.simulate_positions(bars, target, SYM)
    assert list(trades["direction"]) == ["long", "short", "long"] and list(trades["size"]) == [1, 2, 1]
    assert trades["pnl"].sum() == pytest.approx(daily["pnl"].sum())
    assert trades["pnl_gross"].sum() - trades["pnl"].sum() == pytest.approx(daily["cost"].sum())
    assert trades["entry_time"].iloc[1] == bars.index[16] and trades["exit_time"].iloc[0] == bars.index[16]


def test_perfect_foresight_is_acted_on_one_bar_later():
    bars = make_bars(gaps=False)
    move = (bars["close"] - bars["open"]).to_numpy() * instrument(SYM)["point_value"]
    seer = np.sign(bars["close"] - bars["open"])           # knows its own bar's direction
    daily, _ = engine.simulate_positions(bars, seer, SYM)
    delayed = np.sum(seer.shift(1).fillna(0).to_numpy() * move)
    same_bar = np.sum(seer.to_numpy() * move)
    assert daily["pnl_gross"].sum() == pytest.approx(delayed)
    assert abs(delayed) < 0.2 * same_bar                   # the foresight is worth ~nothing a bar late
    with pytest.raises(ValueError):
        engine.simulate_positions(bars, seer, SYM, delay=0)


def test_roll_jump_is_not_pnl_and_costs_a_round_trip():
    bars = make_bars(n=100)
    bars.loc[bars.index[50]:, ["open", "high", "low", "close"]] += 500   # next contract trades 500 higher
    bars.loc[bars.index[50]:, "contract"] = 1
    flat = make_bars(n=100)
    a, ta = engine.simulate_positions(bars, pd.Series(1.0, index=bars.index), SYM)
    b, _ = engine.simulate_positions(flat, pd.Series(1.0, index=flat.index), SYM)
    gap = (flat["open"].iloc[50] - flat["close"].iloc[49]) * instrument(SYM)["point_value"]
    assert a["pnl_gross"].sum() == pytest.approx(b["pnl_gross"].sum() - gap)
    assert a["turnover"].sum() == 3.0                      # entry + close and re-open at the roll
    assert len(ta) == 1 and ta["result"].iloc[0] == "open"  # a roll does not end the trade


def test_bracket_fills_next_open_and_stop_wins_ties():
    bars = make_bars(n=50, gaps=False)
    t = bars.index[10]
    o = bars["open"].iloc[11]
    bars.iloc[11, bars.columns.get_loc("high")] = o + 50   # bar 11 touches both stop and target
    bars.iloc[11, bars.columns.get_loc("low")] = o - 50
    cand = pd.DataFrame({"time": [t], "direction": ["long"], "stop": [o - 20], "tp": [o + 20]})
    _, trades = engine.simulate_candidates(bars, cand, SYM, "5min")
    tr = trades.iloc[0]
    assert tr["entry_time"] == bars.index[11] and tr["entry_price"] == o
    assert tr["result"] == "loss" and tr["exit_price"] == o - 20 and tr["r_gross"] == -1


def test_bracket_skips_entry_filled_through_stop():
    bars = make_bars(n=50)
    o = bars["open"].iloc[11]
    cand = pd.DataFrame({"time": [bars.index[10]], "direction": ["long"], "stop": [o + 1], "tp": [o + 20]})
    assert engine.simulate_candidates(bars, cand, SYM, "5min")[1].empty


# ── Overfitting defences ──────────────────────────────────────────────────────
def test_causality_check_catches_full_sample_statistics_and_shift():
    bars = make_bars()
    grid = [{"w": 20}]
    with pytest.raises(validate.CausalityError):
        validate.causality_check(lambda b, w: np.sign(b["close"] - b["close"].mean()), bars, grid, 3, seed=1)
    with pytest.raises(validate.CausalityError):
        validate.causality_check(lambda b, w: np.sign(b["close"].shift(-1) - b["close"]), bars, grid, 3, seed=1)
    with pytest.raises(validate.CausalityError):
        validate.causality_check(lambda b, w: np.sign(b["close"] - b["close"].rolling(w, center=True).mean()),
                                 bars, grid, 3, seed=1)
    ok = validate.causality_check(lambda b, w: np.sign(b["close"] - b["close"].rolling(w).mean()),
                                  bars, grid, 3, seed=1)
    assert all(c["ok"] for c in ok)


def test_more_trials_lower_the_deflated_sharpe():
    r = np.random.default_rng(3).normal(0.0008, 0.01, 1500)
    values = [metrics.deflated_sharpe(r, n, sr_var=0.0005) for n in (1, 10, 100, 1000)]
    assert values == sorted(values, reverse=True) and values[0] > values[-1]


def test_walk_forward_never_uses_test_data_to_pick_parameters():
    params = {"a": [1, 2, 3], "b": [1, 2, 3]}
    days = pd.bdate_range("2010-01-01", "2019-12-31")
    rets = pd.DataFrame(np.random.default_rng(4).normal(0, 0.01, (len(days), 9)), index=days)
    fold_list = validate.folds(days, 3, 1, 5)
    base = validate.walk_forward(rets, params, fold_list)
    for f in fold_list:                                   # rig everything from the train end on
        rigged = rets.copy()
        later = rigged.index >= f["train_end"]
        rigged.loc[later] = -0.01
        rigged.loc[later, (base["choice"].iloc[f["fold"]] + 4) % 9] = 0.05
        assert validate.walk_forward(rigged, params, [f])["choice"].iloc[0] == base["choice"].iloc[f["fold"]]


def test_folds_have_an_embargo_and_tile_the_test_period():
    days = pd.bdate_range("2010-01-01", "2019-12-31")
    fl = validate.folds(days, 3, 1, 5)
    assert all(f["test_start"] - f["train_end"] == pd.Timedelta(days=5) for f in fl)
    assert all(a["test_end"] == b["test_start"] for a, b in zip(fl, fl[1:]))


def test_plateau_prefers_a_stable_region_over_a_spike():
    params = {"a": [1, 2, 3], "b": [1, 2, 3]}
    scores = np.array([3.0, 0, 0, 0, 1.5, 1.5, 0, 1.5, 1.5])  # lone peak at 0, broad hill opposite
    assert np.argmax(scores) == 0
    assert np.argmax(validate.plateau(scores, params)) in (4, 5, 7, 8)


# ── Data layer ────────────────────────────────────────────────────────────────
def test_ingest_builds_bars_drops_bad_ticks_and_hides_the_holdout(tmp_path, monkeypatch):
    for name, path in (("DATA_DIR", tmp_path), ("RAW", tmp_path / "raw"), ("BARS", tmp_path / "bars")):
        monkeypatch.setattr(data, name, path)
    monkeypatch.setitem(CFG["data"], "start", "2000-01-01")
    h = holdout_start()
    ts = pd.date_range(h - pd.Timedelta(minutes=10), h + pd.Timedelta(minutes=10), freq="20s")
    price = np.full(len(ts), 100.0) + np.arange(len(ts)) * 0.25
    price[5] = 1000.0                                      # spike: far from both agreeing neighbours
    ticks = pd.DataFrame({"price": price, "size": 1.0, "bid": price - 0.25, "ask": price + 0.25}, index=ts)
    ticks.index.name = "ts"
    data.ingest(ticks, SYM, "ticks")
    data.ingest(ticks, SYM, "ticks")                       # re-ingest replaces, no duplicates
    stored = pd.concat(pd.read_parquet(f) for f in data.raw_files(SYM, "ticks"))   # one file per year
    assert list(stored.columns) == ["ts", "price", "size", "bid", "ask"] and len(stored) == len(ticks)
    bars, health = data.load_bars(SYM, "5min", "research")
    assert bars.index.max() < h and health["bad_rows"] == 1 and health["quotes"]
    first = ticks[(ticks.index < bars.index[0] + pd.Timedelta("5min"))].drop(ticks.index[5])
    assert bars["high"].iloc[0] == first["price"].max() and bars["ticks"].iloc[0] == len(first)
    assert bars["spread"].iloc[0] == pytest.approx(0.5)
    full, _ = data.load_bars(SYM, "5min", "full")
    assert full.index.max() >= h and full["volume"].sum() == len(ticks) - 1
