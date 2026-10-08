"""
Data layer. Raw data of any kind is stored once, partitioned by symbol and year, and bars of
any size are built from it with DuckDB and cached.

  data/raw/ticks/symbol=NQ/year=2024/data.parquet   ts, price [, size, bid, ask, instrument_id]
  data/raw/bars/symbol=NQ/year=2024/data.parquet    ts, open, high, low, close [, volume, ticks,
                                                    bid, ask, spread, instrument_id]
  data/bars/NQ_5min_research_<fingerprint>.parquet  built bars, plus a .json health report

ts is the UTC event time (bars: the bar OPEN time), stored tz-naive, zstd-compressed.

  ingest(source, symbol)    CSV / Parquet file or glob, or a DataFrame → raw store
  ingest_databento(...)     Databento download → raw store
  load_bars(symbol, bar_size, mode) → (bars, health)

Bars: UTC index (bar open), columns open, high, low, close, volume, ticks, spread (mean ask − bid,
NaN without quotes) and contract (the underlying contract, so nothing is held across a roll).
In mode "research" the query stops before data.holdout_start, so later data is never read.
The cache key hashes every raw file's size and mtime plus the build settings: new data rebuilds it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from bt.config import CFG, DATA_DIR, holdout_start, instrument

RAW, BARS = DATA_DIR / "raw", DATA_DIR / "bars"
KINDS = ("ticks", "bars")                       # "auto" prefers the first one stored
COLUMNS = {"ticks": ["price", "size", "bid", "ask", "instrument_id"],
           "bars":  ["open", "high", "low", "close", "volume", "ticks", "bid", "ask", "spread",
                     "instrument_id"]}
ALIASES = {   # source column names recognised automatically (case-insensitive)
    "ts":     ["ts", "ts_event", "timestamp", "datetime", "date_time"],
    "open":   ["open", "o"], "high": ["high", "h"], "low": ["low", "l"], "close": ["close", "c"],
    "volume": ["volume", "vol", "v"],
    "price":  ["price", "last", "trade_price"],
    "size":   ["size", "qty", "quantity", "volume"],
    "bid":    ["bid", "bid_px_00", "bid_price"],
    "ask":    ["ask", "ask_px_00", "ask_price"],
    "spread": ["spread"],
    "ticks":  ["ticks", "tick_count", "trades"],
    "instrument_id": ["instrument_id"],
}


def _connect() -> duckdb.DuckDBPyConnection:
    tmp = DATA_DIR / ".duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{CFG['data']['duckdb_memory']}'")
    con.execute(f"SET temp_directory = '{tmp.as_posix()}'")      # spill instead of running out of RAM
    con.execute("SET TimeZone = 'UTC'")
    return con


# ── Ingest ────────────────────────────────────────────────────────────────────
def _ts_expr(con, rel: str, expr: str, tz: str) -> str:
    """SQL turning the source timestamp (tz-aware, naive in `tz`, text or epoch) into naive UTC."""
    typ, val = con.sql(f"SELECT typeof({expr}), {expr} FROM {rel} WHERE {expr} IS NOT NULL LIMIT 1").fetchone()
    if typ == "TIMESTAMP WITH TIME ZONE":
        return f"timezone('UTC', {expr})"
    if typ in ("BIGINT", "INTEGER", "UBIGINT", "HUGEINT", "DOUBLE"):          # epoch, UTC by definition
        v = abs(float(val))
        micros = (f"{expr} // 1000" if v > 1e17 else expr if v > 1e14
                  else f"{expr} * 1000" if v > 1e11 else f"{expr} * 1000000")
        return f"make_timestamp(CAST({micros} AS BIGINT))"
    if typ == "VARCHAR" and re.search(r"(Z|[+-]\d\d:?\d\d)$", str(val)):     # text with an offset
        return f"timezone('UTC', CAST({expr} AS TIMESTAMPTZ))"
    naive = f"CAST({expr} AS TIMESTAMP)"
    return naive if tz.upper() == "UTC" else f"timezone('UTC', timezone('{tz}', {naive}))"


def ingest(source, symbol: str, kind: str | None = None, mapping: dict | None = None,
           tz: str = "UTC") -> dict[int, int]:
    """
    Store a CSV / Parquet file (globs allowed) or a DataFrame in the raw store and return rows
    per year. Columns are recognised by ALIASES; `mapping` = {column: SQL expression over the
    source columns} covers anything else, e.g. {"ts": "Date || ' ' || Time"}. `tz` is the zone
    of tz-naive source timestamps. kind (ticks | bars) is inferred from the columns if omitted.
    New data replaces what was stored over its own time span, so re-ingesting is safe.
    """
    symbol = symbol.upper()
    instrument(symbol)                                       # fail early on unknown symbols
    con = _connect()
    if isinstance(source, pd.DataFrame):
        con.register("src_df", source.reset_index() if source.index.name else source)
        rel = "src_df"
    else:
        path = str(source).replace("\\", "/")
        rel = f"read_parquet('{path}')" if path.endswith(".parquet") else f"read_csv_auto('{path}')"
    cols = {c.lower(): c for c in con.sql(f"SELECT * FROM {rel} LIMIT 0").columns}
    exprs = {k: f'"{cols[n]}"' for k, names in ALIASES.items() for n in reversed(names) if n in cols}
    if "ts" not in exprs and {"date", "time"} <= set(cols):
        exprs["ts"] = f'CAST("{cols["date"]}" AS DATE) + CAST("{cols["time"]}" AS TIME)'
    elif "ts" not in exprs and ("date" in cols or "time" in cols):
        exprs["ts"] = f'"{cols.get("date") or cols["time"]}"'
    exprs.update(mapping or {})
    kind = kind or ("bars" if {"open", "high", "low", "close"} <= set(exprs) else "ticks")
    if kind == "ticks" and "price" not in exprs and {"bid", "ask"} <= set(exprs):
        exprs["price"] = f"({exprs['bid']} + {exprs['ask']}) / 2"            # quotes only: use the mid
    missing = [k for k in ["ts"] + (["open", "high", "low", "close"] if kind == "bars" else ["price"])
               if k not in exprs]
    if missing:
        raise ValueError(f"Cannot find {missing} in source columns {list(cols.values())}; "
                         f"pass them with --map column=expression.")

    select = ", ".join([f"{_ts_expr(con, rel, exprs['ts'], tz)} AS ts"] + [
        f"CAST({exprs[k]} AS {'BIGINT' if k == 'instrument_id' else 'DOUBLE'}) AS {k}"
        for k in COLUMNS[kind] if k in exprs])
    stage = DATA_DIR / ".staging"
    shutil.rmtree(stage, ignore_errors=True)
    con.execute(f"COPY (SELECT *, year(ts) AS year FROM (SELECT {select} FROM {rel}) WHERE ts IS NOT NULL) "
                f"TO '{stage.as_posix()}' (FORMAT parquet, PARTITION_BY (year), COMPRESSION zstd)")

    written = {}
    for ydir in sorted(stage.glob("year=*")):
        new = f"read_parquet('{ydir.as_posix()}/*.parquet', hive_partitioning = false)"
        lo, hi, n = con.sql(f"SELECT min(ts), max(ts), count(*) FROM {new}").fetchone()
        dest = RAW / kind / f"symbol={symbol}" / ydir.name
        dest.mkdir(parents=True, exist_ok=True)
        out, tmp = dest / "data.parquet", dest / "data.parquet.tmp"
        parts = [f"SELECT * FROM {new}"]
        if out.exists():                                     # keep stored rows outside the new span
            parts.insert(0, f"SELECT * FROM read_parquet('{out.as_posix()}', hive_partitioning = false) "
                            f"WHERE ts < TIMESTAMP '{lo}' OR ts > TIMESTAMP '{hi}'")
        con.execute(f"COPY (SELECT * FROM ({' UNION ALL BY NAME '.join(parts)}) ORDER BY ts) "
                    f"TO '{tmp.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
        os.replace(tmp, out)
        written[int(ydir.name[5:])] = n
    shutil.rmtree(stage)
    return written


def ingest_databento(symbol: str, start_year: int, end_year: int, schema: str = "ohlcv-1m",
                     estimate: bool = False, overwrite: bool = False) -> float:
    """
    Download from Databento (CME Globex, continuous front month) one year at a time and return
    the billed cost estimate. ohlcv-* schemas are stored as bars, trades / tbbo as ticks
    (tbbo carries bid/ask). Years already stored are skipped unless overwrite.
    Needs DATABENTO_API_KEY in the environment or .env.
    """
    import databento as db                                   # only this path needs the package

    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise EnvironmentError("DATABENTO_API_KEY is not set (environment or .env).")
    sym = instrument(symbol).get("databento")
    if not sym:
        raise KeyError(f"{symbol} has no databento symbol in config.yaml.")
    client = db.Historical(key)
    kind = "bars" if schema.startswith("ohlcv") else "ticks"
    total = 0.0
    for year in range(start_year, end_year + 1):
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        end = min(pd.Timestamp(f"{year + 1}-01-01", tz="UTC"),       # history ends ~24 h ago
                  (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).floor("D"))
        if start >= end:
            continue
        if not overwrite and (RAW / kind / f"symbol={symbol.upper()}" / f"year={year}").exists():
            print(f"  {year}  stored, skipping (--overwrite to fetch again)")
            continue
        kw = dict(dataset="GLBX.MDP3", symbols=[sym], schema=schema, stype_in="continuous",
                  start=start.isoformat(), end=end.isoformat())
        cost = float(client.metadata.get_cost(**kw))
        total += cost
        print(f"  {year}  ${cost:,.2f}", end="" if not estimate else "\n", flush=True)
        if estimate:
            continue
        df = client.timeseries.get_range(**kw).to_df()
        rows = ingest(df, symbol, kind) if len(df) else {}
        print(f"  {sum(rows.values()):,} rows")
    return total


# ── Bars ──────────────────────────────────────────────────────────────────────
def raw_files(symbol: str, kind: str) -> list[Path]:
    return sorted((RAW / kind / f"symbol={symbol.upper()}").glob("year=*/*.parquet"))


def _kind(symbol: str) -> str:
    have = [k for k in KINDS if raw_files(symbol, k)]
    want = CFG["data"]["source_kind"]
    if not have:
        raise FileNotFoundError(f"No raw data for {symbol} in {RAW}. Run: python run.py ingest ...")
    if want != "auto" and want not in have:
        raise FileNotFoundError(f"data.source_kind is {want} but only {have} are stored for {symbol}.")
    return have[0] if want == "auto" else want


def load_bars(symbol: str, bar_size: str, mode: str = "research") -> tuple[pd.DataFrame, dict]:
    """Bars of bar_size from the raw store, cached. mode: research (before holdout_start) | full."""
    symbol, d = symbol.upper(), CFG["data"]
    kind, end = _kind(symbol), (holdout_start() if mode == "research" else None)
    files = raw_files(symbol, kind)
    settings = dict(kind=kind, bar=bar_size, start=str(d["start"]), end=str(end),
                    roll=instrument(symbol)["roll"], spike=d["spike_pct"], gap=[d["gap_factor"], d["min_gap"]])
    h = hashlib.sha1(json.dumps(settings, sort_keys=True).encode())
    for f in files:
        st = f.stat()
        h.update(f"{f.relative_to(RAW).as_posix()}|{st.st_size}|{st.st_mtime_ns}".encode())
    stem = f"{symbol}_{bar_size}_{mode}"
    path = BARS / f"{stem}_{h.hexdigest()[:12]}.parquet"
    if path.exists():
        bars, health = pd.read_parquet(path), json.loads(path.with_suffix(".json").read_text())
    else:
        BARS.mkdir(parents=True, exist_ok=True)
        for old in BARS.glob(f"{stem}_*"):
            old.unlink()
        bars, health = _build(symbol, kind, bar_size, pd.Timestamp(d["start"], tz="UTC"), end)
        bars.to_parquet(path)
        path.with_suffix(".json").write_text(json.dumps(health, indent=1))
    bars.attrs.update(symbol=symbol, bar_size=bar_size)
    return bars, health


def _build(symbol: str, kind: str, bar_size: str, start: pd.Timestamp,
           end: pd.Timestamp | None) -> tuple[pd.DataFrame, dict]:
    con = _connect()
    src = (f"read_parquet('{(RAW / kind / f'symbol={symbol}').as_posix()}/*/*.parquet', "
           f"hive_partitioning = true, union_by_name = true)")
    have = set(con.sql(f"SELECT * FROM {src} LIMIT 0").columns)

    def col(c: str) -> str:
        return c if c in have else f"CAST(NULL AS {'BIGINT' if c == 'instrument_id' else 'DOUBLE'})"

    where = f"year >= {start.year} AND ts >= TIMESTAMP '{start:%Y-%m-%d %H:%M:%S}'"
    if end is not None:                                       # the holdout is never read in research
        where += f" AND year <= {end.year} AND ts < TIMESTAMP '{end:%Y-%m-%d %H:%M:%S}'"
    if kind == "ticks":
        spike = CFG["data"]["spike_pct"]
        con.execute(f"""CREATE TEMP VIEW clean AS
            WITH t AS (SELECT ts, price, {col('size')} AS size, {col('bid')} AS bid, {col('ask')} AS ask,
                              {col('instrument_id')} AS instrument_id, year FROM {src} WHERE {where}),
                 w AS (SELECT *, lag(price) OVER p AS prv, lead(price) OVER p AS nxt FROM t
                       WINDOW p AS (PARTITION BY year ORDER BY ts))
            SELECT *, (price IS NULL OR price <= 0 OR coalesce(size < 0, false) OR coalesce(bid > ask, false)
                       OR coalesce(abs(price / prv - 1) > {spike} AND abs(price / nxt - 1) > {spike}
                                   AND abs(prv / nxt - 1) <= {spike}, false)) AS bad FROM w""")
        agg = (f"arg_min(price, ts) AS open, max(price) AS high, min(price) AS low, "
               f"arg_max(price, ts) AS close, {'sum(size)' if 'size' in have else 'count(*)'} AS volume, "
               f"count(*) AS ticks, avg(ask - bid) AS spread")
    else:
        spread = ("spread" if "spread" in have else "ask - bid" if {"bid", "ask"} <= have
                  else "CAST(NULL AS DOUBLE)")
        con.execute(f"""CREATE TEMP VIEW clean AS
            SELECT ts, open, high, low, close, {col('volume')} AS volume, {col('ticks')} AS ticks,
                   {spread} AS spread, {col('instrument_id')} AS instrument_id,
                   (open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL OR low <= 0
                    OR high < greatest(open, close, low) OR low > least(open, close, high)) AS bad
            FROM {src} WHERE {where}""")
        agg = ("arg_min(open, ts) AS open, max(high) AS high, min(low) AS low, arg_max(close, ts) AS close, "
               "sum(volume) AS volume, sum(ticks) AS ticks, avg(spread) AS spread")
    secs = int(pd.Timedelta(bar_size).total_seconds())
    bars = con.sql(f"""SELECT time_bucket(INTERVAL '{secs} seconds', ts) AS ts, {agg},
                              arg_max(instrument_id, ts) AS instrument_id
                       FROM clean WHERE NOT bad GROUP BY 1 ORDER BY 1""").df()
    rows, bad = con.sql("SELECT count(*), coalesce(sum(bad::INT), 0) FROM clean").fetchone()
    con.close()

    bars["ts"] = pd.to_datetime(bars["ts"]).dt.tz_localize("UTC")
    bars = bars.set_index("ts")
    bars["volume"] = bars["volume"].fillna(0.0)
    if len(bars) and bars["instrument_id"].notna().all():
        bars["contract"] = bars["instrument_id"].astype(np.int64)
    elif instrument(symbol)["roll"] == "quarterly" and len(bars):
        bars["contract"] = _quarterly_contract(bars.index)
    else:
        bars["contract"] = 0
    bars = bars[["open", "high", "low", "close", "volume", "ticks", "spread", "contract"]].astype(
        {"ticks": float, "spread": float})
    return bars, _health(bars, symbol, kind, bar_size, int(rows), int(bad))


def _quarterly_contract(index: pd.DatetimeIndex) -> np.ndarray:
    """Contract number per bar: increments on the first UTC day after each 3rd-Friday expiry."""
    days = pd.date_range(f"{index[0].year - 1}-01-01", f"{index[-1].year + 1}-12-31", freq="D", tz="UTC")
    third_fridays = days[(days.weekday == 4) & (days.day >= 15) & (days.day <= 21)
                         & days.month.isin([3, 6, 9, 12])]
    return np.searchsorted((third_fridays + pd.Timedelta(days=1)).values, index.values, side="right")


def _health(bars: pd.DataFrame, symbol: str, kind: str, bar_size: str, rows: int, bad: int) -> dict:
    """Bad rows dropped, gaps (weekends excluded) and whether quotes are present."""
    d = CFG["data"]
    limit = max(d["gap_factor"] * pd.Timedelta(bar_size), pd.Timedelta(d["min_gap"]))
    idx = bars.index
    gaps = pd.DataFrame({"start": idx[:-1], "end": idx[1:]})
    gaps["hours"] = (gaps["end"] - gaps["start"]).dt.total_seconds() / 3600
    gaps = gaps[gaps["hours"] > limit.total_seconds() / 3600]
    weekend = [(pd.date_range(s.normalize(), e.normalize()).weekday == 5).any()
               for s, e in zip(gaps["start"], gaps["end"])]
    gaps = gaps[~np.array(weekend, dtype=bool)].sort_values("hours", ascending=False)
    return dict(
        symbol=symbol, source=kind, bar_size=bar_size, rows=rows, bad_rows=bad,
        bad_pct=round(100 * bad / rows, 4) if rows else 0.0, bars=len(bars),
        first=str(idx[0]) if len(idx) else None, last=str(idx[-1]) if len(idx) else None,
        quotes=bool(bars["spread"].notna().any()), gap_threshold_h=round(limit.total_seconds() / 3600, 2),
        gaps=len(gaps), largest_gap_h=round(float(gaps["hours"].iloc[0]), 2) if len(gaps) else 0.0,
        top_gaps=[[str(s), str(e), round(h, 2)] for s, e, h in gaps.head(10).itertuples(index=False)],
    )
