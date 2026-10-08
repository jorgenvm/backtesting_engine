# backtesting_engine

A backtester built to tell you whether a strategy has an edge or whether you found one by
searching. Bring any data (ticks or bars, CSV or Parquet) and any strategy. You get a
walk-forward, cost-inclusive result, a set of overfitting checks, and a one-file HTML dashboard.

What it enforces, so you don't have to remember:

- **No look-ahead.** A decision on bar *t* fills at the open of bar *t+1*, and `delay=0` is
  rejected. Every run also recomputes your strategy on data cut off at random points. If an
  earlier signal changes when later data is removed, the run is rejected.
- **Costs always on.** Commission, slippage and half the measured bid/ask spread are charged
  on every position change, and contract rolls are charged as a round trip.
- **Walk-forward is the only reported result.** Parameters are picked on a rolling train
  window, with an embargo gap before each test window. The reported result is the test
  windows stitched together.
- **Every attempt is counted.** A trial registry logs every parameter set you ever ran, and
  the Deflated Sharpe ratio uses that count.
- **The holdout can be viewed once.** Research never reads data after `holdout_start`: the
  cutoff is in the SQL query. Running the holdout a second time needs `--force` and is
  flagged as contaminated.

## Install

```bash
pip install -r requirements.txt        # Python 3.10+
python -m pytest                       # known-answer tests, ~1 s
```

## Quick start

```bash
# 1. Data: any CSV/Parquet of ticks or bars (globs ok) …
python run.py ingest path/to/nq_1min_*.parquet --symbol NQ
# … or straight from Databento (needs DATABENTO_API_KEY in .env)
python run.py ingest --databento --symbol NQ --start-year 2015 --end-year 2025 --estimate

# 2. Research, as often as you like (every parameter set is logged as a trial)
python run.py research --strategy ma_cross
python run.py research --strategy orb_bracket

# 3. Look at the results
python run.py dashboard

# 4. Once, at the very end
python run.py holdout --strategy orb_bracket
```

Null test: `python tools/make_synthetic_ticks.py`, then `python run.py research --strategy <yours>
--symbol SYNTH`. The data is a random walk, so nothing has an edge on it. If your strategy looks
good on it, you have a bug or an overfit.

## Writing a strategy

Create `strategies/my_strategy.py` with a small parameter grid and **one** of two functions.
`bars` is a DataFrame indexed by bar open time (UTC) with columns `open high low close volume
ticks spread contract` (`spread` is NaN without quote data).

**Positions:** return target units per bar (contracts, shares or coins; NaN means flat).

```python
import numpy as np

BAR_SIZE = "1h"                                   # optional: SYMBOL, BAR_SIZE, DELAY
PARAMS = {"fast": [10, 20, 40], "slow": [100, 200, 400]}

def signal(bars, fast, slow):
    c = bars["close"]
    return np.sign(c.rolling(fast).mean() - c.rolling(slow).mean())
```

**Candidates:** return bracket orders, one row per possible entry. The engine fills each at
the next bar's open and walks the exit bars (`exit_bar_size`, e.g. 1-minute) to find the stop
or target.

| column | required | meaning |
|---|---|---|
| `time` | ✓ | the bar the decision was made on |
| `direction` | ✓ | `"long"` or `"short"` |
| `stop`, `tp` | ✓ | stop and target prices |
| `be` | | breakeven level: once reached, the stop moves to entry |
| `size` | | units (default: `account.risk_per_trade` ÷ risk in USD) |
| `signal_id` | | candidates sharing an id trade at most once |
| `session` | | used by `bracket.session_stop_r` |

Any other column is carried through to the trade list. See [strategies/orb_bracket.py](strategies/orb_bracket.py).

Rules of thumb:

- **Use only data up to the bar you decide on.** Anything else fails the causality check:
  `shift(-1)`, `rolling(center=True)`, full-sample means, or "today's high" before the day ends.
- **Keep grids small (9–27 points) and order each list so neighbours are similar.**
  Selection scores every point by the average of itself and its grid neighbours, so it picks
  a stable region rather than a lucky spike.
- **Editing a strategy file creates new trials.** The registry keys trials on a hash of the
  file, so a lower Deflated Sharpe after many edits is the honest number.
- **Session helpers** (`bt.sessions.label`, `minutes_since_open`) use the windows in `config.yaml`.

## Data

`ingest` recognises common column names on its own: `ts`/`timestamp`/`datetime`, separate
`Date` + `Time`, `open…close`, `price`/`last`, `size`/`volume`, `bid`/`ask`, `instrument_id`.
For anything else, map columns with SQL expressions:

```bash
python run.py ingest es_ticks.csv --symbol ES --tz America/Chicago \
       --map "ts=strptime(stamp, '%Y%m%d %H%M%S')" "price=last_px"
```

- **Timestamps** can be tz-aware, naive (pass `--tz`), text, or epoch s/ms/µs/ns. Everything is
  stored as UTC.
- **Storage:** `data/raw/{ticks|bars}/symbol=X/year=YYYY/data.parquet`, zstd-compressed.
  Re-ingesting replaces stored data over the new data's time span.
- **Bars** of any size are built from the raw store with DuckDB: open, high, low and close,
  plus volume, tick count and mean bid/ask spread. Built bars are cached under `data/bars/`,
  and the cache rebuilds when files change.
- **Health report:** bad rows dropped, gaps (weekends excluded), largest gap, and whether
  quotes are present. It's shown on the dashboard, or with `python run.py health --symbol NQ`.
- **Contracts:** for futures without `instrument_id`, set `roll: quarterly` in `config.yaml` and
  rolls are inferred from the expiry calendar. P&L never includes the price jump between
  contracts.
- **Other instruments:** add an entry under `instruments:` in `config.yaml` with `tick`,
  `point_value` and `commission`.

## Reading the dashboard

| check | pass | means |
|---|---|---|
| Deflated Sharpe | ≥ 0.95 | the out-of-sample Sharpe is unlikely to be the luck of N trials |
| OOS vs IS Sharpe | OOS > 0 and ≥ half of IS | performance survived leaving the training window |
| Grid positive | ≥ 60 % | the idea works across the grid, not at one point |
| Holdout | viewed exactly once | the final test is still clean |
| Causality | always (else the run is rejected) | no signal changed when future data was removed |
| Data | quotes present, no gaps | costs use the real spread; no holes in the data |

The panels are: key metrics against buy & hold; equity net, before costs and buy & hold, with
fold boundaries marked; drawdown; Sharpe per fold, in- and out-of-sample, with the parameters
each fold chose; the parameter heatmap (diverging around 0, with how often each cell was
chosen); yearly returns; rolling 1-year Sharpe; and data health. Pick a second run in
**Compare** to see it side by side, for example research next to holdout.
`python run.py chart --date 2023-06-14` draws one day of the newest run with its trades.

## Layout

```
config.yaml              every setting: data, instruments, costs, walk-forward, sessions
run.py                   CLI: ingest | health | research | holdout | dashboard | chart
dashboard.py             results/runs/* → results/dashboard.html
bt/data.py               ingest, raw store, DuckDB bar builder + cache, health report
bt/engine.py             positions + bracket engines, timing contract, costs, daily P&L
bt/metrics.py            Sharpe, Sortino, CAGR, drawdown, Deflated Sharpe, trade stats
bt/validate.py           causality check, plateau, walk-forward, trial registry, holdout ledger
bt/sessions.py, chart.py session windows; one-day trade chart
strategies/              your strategies (ma_cross, orb_bracket as examples)
tools/                   make_synthetic_ticks.py (null test)
tests/test_core.py       known-answer tests
results/                 trials.jsonl, holdout_ledger.jsonl, runs/<time>_<strategy>_<mode>/
```

Each run folder holds `meta.json` (settings, metrics, verdicts and the data health report)
plus `daily.parquet`, `folds.parquet`, `grid.parquet` and, for bracket strategies,
`trades.parquet`.

Out of scope on purpose: a tick-by-tick order simulator. Bar-level fills at the next open,
with spread-based costs, are accurate enough for anything that isn't latency-sensitive.
