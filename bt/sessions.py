"""
Session windows for strategies, from config.yaml `sessions` (wall-clock in `timezone`).
Evaluated per timestamp after converting UTC → local, so DST comes from the timezone
database, never a fixed offset. Start inclusive, end exclusive.

  label(index)               session name per timestamp, "" outside every window
  minutes_since_open(index)  minutes since the containing window opened, NaN outside
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bt.config import CFG, TZ


def _windows(sessions: dict | None) -> list[tuple[str, int, int]]:
    out = []
    for name, (start, end) in (sessions or CFG["sessions"]).items():
        s, e = (int(t[:2]) * 60 + int(t[3:5]) for t in (start, end))
        out.append((name, s, e))
    return out


def _wall_minutes(index: pd.DatetimeIndex) -> np.ndarray:
    local = pd.DatetimeIndex(index).tz_convert(TZ).tz_localize(None)
    return (local.hour * 60 + local.minute + local.second / 60).to_numpy(dtype=np.float64)


def label(index: pd.DatetimeIndex, sessions: dict | None = None) -> np.ndarray:
    m = _wall_minutes(index)
    out = np.full(len(m), "", dtype=object)
    for name, s, e in _windows(sessions):
        out[(m >= s) & (m < e)] = name
    return out


def minutes_since_open(index: pd.DatetimeIndex, sessions: dict | None = None) -> np.ndarray:
    m = _wall_minutes(index)
    out = np.full(len(m), np.nan)
    for _, s, e in _windows(sessions):
        inside = (m >= s) & (m < e)
        out[inside] = m[inside] - s
    return out
