"""
Performance metrics for a closed-trade sequence.

Convention:
  CAGR    : calendar days, first entry → last entry
  Calmar  : CAGR / max drawdown (%)
  Sharpe  : daily-grouped PnL mean / std × √252
  Sortino : daily-grouped PnL mean / downside deviation × √252
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def compute(r_vals, entry_times, risk_usd: float, initial_balance: float) -> dict:
    r = np.asarray(r_vals, dtype=float)
    n = len(r)
    if n == 0:
        return {"n": 0}

    pnl = r * risk_usd
    eq  = np.concatenate([[initial_balance], initial_balance + np.cumsum(pnl)])

    peak       = np.maximum.accumulate(eq)
    max_dd_usd = float((peak - eq).max())
    max_dd_pct = float(np.where(peak > 0, (peak - eq) / peak * 100.0, 0.0).max())
    r_eq       = np.concatenate([[0.0], np.cumsum(r)])
    max_dd_r   = float((np.maximum.accumulate(r_eq) - r_eq).max())

    times  = pd.DatetimeIndex(pd.to_datetime(entry_times, utc=True))
    years  = max(int((times.max() - times.min()).days), 1) / 365.25
    growth = eq[-1] / eq[0]
    cagr   = float(growth ** (1.0 / years) - 1.0) if growth > 0 else -1.0
    calmar = cagr / (max_dd_pct / 100.0) if max_dd_pct > 0 else 0.0

    daily  = pd.Series(pnl, index=times.normalize()).groupby(level=0).sum()
    sharpe = (float(daily.mean() / daily.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
              if len(daily) > 1 and daily.std() > 0 else 0.0)
    dd_dev  = float(np.sqrt((np.minimum(daily.to_numpy(), 0.0) ** 2).mean()))
    sortino = float(daily.mean() / dd_dev * np.sqrt(TRADING_DAYS_PER_YEAR)) if dd_dev > 0 else 0.0

    wins, losses = r[r > 0], r[r <= 0]
    pf = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")

    # Longest losing streak
    streak = best = 0
    for x in r:
        streak = streak + 1 if x <= 0 else 0
        best = max(best, streak)

    return {
        "n":             n,
        "win_rate_pct":  float((r > 0).mean() * 100),
        "total_r":       float(r.sum()),
        "avg_r":         float(r.mean()),
        "avg_win_r":     float(wins.mean()) if wins.size else 0.0,
        "avg_loss_r":    float(losses.mean()) if losses.size else 0.0,
        "profit_factor": pf,
        "net_pnl":       float(pnl.sum()),
        "final_balance": float(eq[-1]),
        "total_ret_pct": float((growth - 1) * 100),
        "cagr_pct":      cagr * 100,
        "max_dd_usd":    max_dd_usd,
        "max_dd_pct":    max_dd_pct,
        "max_dd_r":      max_dd_r,
        "calmar":        calmar,
        "sharpe":        sharpe,
        "sortino":       sortino,
        "max_loss_streak": best,
        "years":         years,
    }
