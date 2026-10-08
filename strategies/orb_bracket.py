"""
Opening-range breakout (candidates): the New York session's first `range_min` minutes set a
range; the first bar closing outside it goes with the break, stop at the other side of the
range, target `rr` times the risk. One trade per day.
"""
import numpy as np
import pandas as pd

from bt import sessions
from bt.engine import trading_day

BAR_SIZE = "5min"
PARAMS = {"range_min": [15, 30, 60], "rr": [1.0, 2.0, 3.0]}


def candidates(bars: pd.DataFrame, range_min: int, rr: float) -> pd.DataFrame:
    bar_min = pd.Timedelta(bars.attrs.get("bar_size", BAR_SIZE)).total_seconds() / 60
    ny = sessions.label(bars.index) == "New York"
    mins = sessions.minutes_since_open(bars.index)
    day = trading_day(bars.index)

    in_range = ny & (mins + bar_min <= range_min)            # bar closes inside the range window
    hi = bars["high"].where(in_range).groupby(day).cummax().groupby(day).ffill()
    lo = bars["low"].where(in_range).groupby(day).cummin().groupby(day).ffill()
    after = ny & (mins + bar_min > range_min)
    close = bars["close"]
    up, down = after & (close > hi), after & (close < lo)
    first = (up | down) & ((up | down).astype(int).groupby(day).cumsum() == 1)

    long = up[first]
    c, h, l = close[first], hi[first], lo[first]
    stop = np.where(long, l, h)
    return pd.DataFrame({
        "time": c.index, "direction": np.where(long, "long", "short"), "stop": stop,
        "tp": c + rr * (c - stop), "signal_id": day[first],
    })
