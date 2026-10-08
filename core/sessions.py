"""
Session windows. Defined as Europe/Berlin wall-clock times in config.SESSIONS and
evaluated per timestamp after converting UTC → Berlin, so DST is handled by the
timezone database, never by a fixed offset. Start inclusive, end exclusive.

  label(index)              session name per timestamp, "" outside every window
  minutes_since_open(index) minutes since the containing window opened, NaN outside
  windows(start, end)       UTC start/end of every window in a date range
  overlaps(starts, ends)    whether each [start, end) interval touches any window
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def _minute_of_day(t) -> int:
    return t.hour * 60 + t.minute


def _wall_minutes(index: pd.DatetimeIndex) -> np.ndarray:
    """Minutes (fractional) since Berlin midnight for each UTC timestamp."""
    local = pd.DatetimeIndex(index).tz_convert(config.SESSION_TZ)
    return (local.hour * 60 + local.minute + local.second / 60).to_numpy(dtype=np.float64)


def label(index: pd.DatetimeIndex, sessions: dict = config.SESSIONS) -> np.ndarray:
    m = _wall_minutes(index)
    out = np.full(len(m), "", dtype=object)
    for name, (start, end) in sessions.items():
        out[(m >= _minute_of_day(start)) & (m < _minute_of_day(end))] = name
    return out


def minutes_since_open(index: pd.DatetimeIndex, sessions: dict = config.SESSIONS) -> np.ndarray:
    m = _wall_minutes(index)
    out = np.full(len(m), np.nan)
    for start, end in sessions.values():
        s, e = _minute_of_day(start), _minute_of_day(end)
        inside = (m >= s) & (m < e)
        out[inside] = m[inside] - s
    return out


def windows(start, end, sessions: dict = config.SESSIONS) -> pd.DataFrame:
    """One row per session per Berlin calendar day in [start, end]: name, start, end (UTC)."""
    tz = config.SESSION_TZ
    days = pd.date_range(pd.Timestamp(start).tz_convert(tz).date(),
                         pd.Timestamp(end).tz_convert(tz).date(), freq="D")   # naive wall dates

    def utc(day, t):   # wall clock → UTC, so a DST change that day can't shift the time
        return (day + pd.Timedelta(minutes=_minute_of_day(t))).tz_localize(tz).tz_convert("UTC")

    rows = [(name, utc(d, s), utc(d, e)) for d in days for name, (s, e) in sessions.items()]
    return pd.DataFrame(rows, columns=["name", "start", "end"])


def overlaps(starts: pd.Series, ends: pd.Series, sessions: dict = config.SESSIONS) -> np.ndarray:
    """True where [start, end) intersects a session window, e.g. a shift line's live span."""
    s, e = pd.DatetimeIndex(starts), pd.DatetimeIndex(ends)
    out = np.zeros(len(s), dtype=bool)
    if len(s):
        for w in windows(s.min(), e.max(), sessions).itertuples():
            out |= (s < w.end) & (e > w.start)
    return out
