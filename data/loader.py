"""
M1 OHLCV sources behind a single DataSource interface.

  DatabentoSource : live pull from Databento (GLBX.MDP3, ohlcv-1m, continuous)
  ParquetSource   : local cache written by data/export.py — no API key needed

load_m1 returns a UTC-indexed DataFrame (index = bar OPEN time) with
columns open, high, low, close, volume, contract.

`contract` identifies the underlying contract of the continuous series so the
backtest never holds a trade across a roll. It is Databento's instrument_id when
the data has it, otherwise derived from the quarterly expiry calendar (.c.0 stays
on the expiring contract until settlement on the 3rd Friday, then resumes on the
next contract at the Sunday open). -1 = unknown.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

import config

CACHE_DIR = Path(__file__).parent / "cache"
_COLS = ["open", "high", "low", "close", "volume"]


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="first")].sort_index()
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "ts"
    keep = _COLS + (["instrument_id"] if "instrument_id" in df.columns else [])
    return df[keep].copy()


def _quarterly_contract(index: pd.DatetimeIndex) -> np.ndarray:
    """Contract number per bar: increments on the first UTC day after each 3rd-Friday expiry."""
    days = pd.date_range(f"{index[0].year - 1}-01-01", f"{index[-1].year + 1}-12-31", freq="D", tz="UTC")
    third_fridays = days[(days.weekday == 4) & (days.day >= 15) & (days.day <= 21)
                         & days.month.isin([3, 6, 9, 12])]
    boundaries = (third_fridays + pd.Timedelta(days=1)).values
    return np.searchsorted(boundaries, index.values, side="right").astype(np.int64)


def _with_contract(df: pd.DataFrame, code: str) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        df["contract"] = pd.Series(dtype=np.int64)
    elif "instrument_id" in df.columns and df["instrument_id"].notna().all():
        df["contract"] = df["instrument_id"].astype(np.int64)
    elif config.INSTRUMENTS[code.upper()].quarterly:
        df["contract"] = _quarterly_contract(df.index)
    else:
        df["contract"] = -1
    return df[_COLS + ["contract"]]


def cache_path(code: str, year: int) -> Path:
    return CACHE_DIR / f"m1_{code.lower()}_{year}.parquet"


class DataSource(ABC):
    @abstractmethod
    def get_m1_ohlcv(self, start, end) -> pd.DataFrame:
        """UTC-indexed M1 OHLCV for [start, end]. start/end: anything pd.Timestamp accepts."""


class DatabentoSource(DataSource):
    """
    Requires DATABENTO_API_KEY (config.DATABENTO_KEY_ENV) in the environment — never hardcoded.
    get_m1_ohlcv returns OHLCV + instrument_id (raw, as cached by data/export.py).
    """

    def __init__(self, code: str = "NQ", dataset: str = config.DATABENTO_DATASET):
        import databento as db  # deferred so the parquet path works without the package

        key = os.environ.get(config.DATABENTO_KEY_ENV)
        if not key:
            raise EnvironmentError(f"{config.DATABENTO_KEY_ENV} is not set.")
        self._client  = db.Historical(key)
        self._dataset = dataset
        self._symbol  = config.INSTRUMENTS[code.upper()].symbol

    def _range_kwargs(self, start, end) -> dict:
        return dict(
            dataset  = self._dataset,
            symbols  = [self._symbol],
            schema   = "ohlcv-1m",
            stype_in = "continuous",
            start    = _utc(start).isoformat(),
            end      = _utc(end).isoformat(),
        )

    def estimate_cost(self, start, end) -> float:
        """USD cost Databento will bill for this request."""
        return float(self._client.metadata.get_cost(**self._range_kwargs(start, end)))

    def get_m1_ohlcv(self, start, end) -> pd.DataFrame:
        df = self._client.timeseries.get_range(**self._range_kwargs(start, end)).to_df()
        if df.empty:
            return pd.DataFrame(columns=_COLS + ["instrument_id"])
        return _normalise(df)


class ParquetSource(DataSource):
    """Reads data/cache/m1_{code}_{year}.parquet files."""

    def __init__(self, code: str = "NQ", cache_dir: Path = CACHE_DIR):
        self._code = code.lower()
        self._dir  = Path(cache_dir)

    def get_m1_ohlcv(self, start, end) -> pd.DataFrame:
        s, e = _utc(start), _utc(end)
        parts = [pd.read_parquet(p) for yr in range(s.year, e.year + 1)
                 if (p := self._dir / f"m1_{self._code}_{yr}.parquet").exists()]
        if not parts:
            raise FileNotFoundError(
                f"No cache for {self._code.upper()} {s.year}-{e.year} in {self._dir}. "
                f"Run: python -m data.export --instrument {self._code.upper()} "
                f"--start-year {s.year} --end-year {e.year}"
            )
        return _normalise(pd.concat(parts)).loc[s:e]


def load_m1(code: str, start, end, source: str = "parquet") -> pd.DataFrame:
    """OHLCV + contract column. source = 'parquet' | 'databento'."""
    src = ParquetSource(code) if source == "parquet" else DatabentoSource(code)
    return _with_contract(src.get_m1_ohlcv(start, end), code)
