"""
Strategies: one module each, run with `python run.py research --strategy <module>`.

A module defines PARAMS (a small grid, each list ordered so neighbours are similar) and ONE of
  signal(bars, **params)     -> pd.Series of target units per bar        (see ma_cross.py)
  candidates(bars, **params) -> pd.DataFrame of bracket orders           (see orb_bracket.py)
Optional: SYMBOL, BAR_SIZE, DELAY. Use only bars up to the one you decide on; the engine fills
on the next bar and the causality check rejects anything that peeks ahead.
"""
