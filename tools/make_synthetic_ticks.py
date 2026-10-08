"""
Null test data: random-walk ticks with bid/ask quotes. Price changes are independent by
construction, so no strategy has an edge on them. A strategy that looks good here has a bug
(look-ahead, cost errors) or is overfit.

  python tools/make_synthetic_ticks.py [--start-year 2014] [--end-year 2025] [--per-minute 2]
  python run.py research --strategy ma_cross --symbol SYNTH

Writes symbol SYNTH (see config.yaml instruments) through the normal ingest path,
24h Monday–Friday, one year at a time.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bt import data                                  # noqa: E402
from bt.config import instrument                     # noqa: E402


def year_of_ticks(year: int, per_minute: int, price: float, daily_vol: float,
                  tick: float, rng: np.random.Generator) -> pd.DataFrame:
    minutes = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="1min", inclusive="left", tz="UTC")
    minutes = minutes[minutes.weekday < 5]
    ts = np.repeat(minutes.asi8, per_minute) + np.sort(
        rng.integers(0, 60_000_000_000, (len(minutes), per_minute)), axis=1).ravel()
    n = len(ts)
    step = daily_vol / np.sqrt(1440 * per_minute)
    mid = price * np.exp(np.cumsum(rng.normal(0.0, step, n)))
    half = tick * rng.integers(1, 3, n) / 2                          # 1–2 tick spread
    bid = np.round((mid - half) / tick) * tick
    ask = np.maximum(np.round((mid + half) / tick) * tick, bid + tick)
    trade = np.where(rng.random(n) < 0.5, bid, ask)                  # trades hit bid or ask
    return pd.DataFrame({"ts": pd.to_datetime(ts, utc=True), "price": trade,
                         "size": rng.integers(1, 6, n).astype(float), "bid": bid, "ask": ask})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start-year", type=int, default=2014)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--per-minute", type=int, default=2, help="ticks per minute")
    ap.add_argument("--daily-vol", type=float, default=0.012)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng, price, tick = np.random.default_rng(args.seed), 10_000.0, instrument("SYNTH")["tick"]
    for year in range(args.start_year, args.end_year + 1):
        df = year_of_ticks(year, args.per_minute, price, args.daily_vol, tick, rng)
        price = float((df["bid"].iloc[-1] + df["ask"].iloc[-1]) / 2)
        rows = data.ingest(df.set_index("ts"), "SYNTH", "ticks")
        print(f"  SYNTH {year}  {sum(rows.values()):>12,} ticks  last {price:,.2f}")


if __name__ == "__main__":
    main()
