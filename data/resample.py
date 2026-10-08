import pandas as pd

M5 = pd.Timedelta(minutes=5)
_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
        "contract": "last"}


def to_m5(m1: pd.DataFrame) -> pd.DataFrame:
    """
    Resample M1 OHLCV to M5. Index = M5 bar OPEN time (left-labelled, left-closed),
    so the bar stamped 10:00 holds M1 bars 10:00–10:04 and is closed at 10:05.
    Never call on already-resampled or HA data.
    """
    agg = {k: v for k, v in _AGG.items() if k in m1.columns}
    return m1.resample(M5, label="left", closed="left").agg(agg).dropna(subset=["open"])
