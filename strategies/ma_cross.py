"""Moving-average crossover (positions): long one unit while the fast SMA is above the slow one, short below."""
import numpy as np
import pandas as pd

BAR_SIZE = "1h"
PARAMS = {"fast": [10, 20, 40], "slow": [100, 200, 400]}


def signal(bars: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    close = bars["close"]
    return np.sign(close.rolling(fast).mean() - close.rolling(slow).mean())
