"""
Performance metrics on DAILY returns (bar P&L is compounded to days first, so results are
comparable across bar sizes). Annualised with 252 trading days.

  Sharpe   mean / std x sqrt(252)          Sortino  mean / downside deviation x sqrt(252)
  CAGR     calendar years, first → last day  Calmar  CAGR / max drawdown
  Deflated Sharpe: probability the true Sharpe beats the best of N luck-only trials
  (Bailey & López de Prado, 2014), N and the trials' Sharpe variance from the trial registry.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

DAYS = 252
_ND = NormalDist()


def sharpe(r, annualise: bool = True) -> float:
    r = np.asarray(r, dtype=float)
    if len(r) < 2 or not np.std(r) > 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * (np.sqrt(DAYS) if annualise else 1.0))


def sortino(r) -> float:
    r = np.asarray(r, dtype=float)
    down = np.sqrt((np.minimum(r, 0.0) ** 2).mean()) if len(r) else 0.0
    return float(r.mean() / down * np.sqrt(DAYS)) if down > 0 else 0.0


def cagr(r: pd.Series) -> float:
    if len(r) < 2:
        return 0.0
    growth = float((1 + r).prod())
    years = max((r.index[-1] - r.index[0]).days, 1) / 365.25
    return growth ** (1 / years) - 1 if growth > 0 else -1.0


def drawdown(r: pd.Series) -> pd.Series:
    eq = (1 + r).cumprod()
    return eq / eq.cummax() - 1


def yearly(r: pd.Series) -> pd.Series:
    return (1 + r).groupby(r.index.year).prod() - 1


def rolling_sharpe(r: pd.Series, window: int = DAYS) -> pd.Series:
    return r.rolling(window).mean() / r.rolling(window).std() * np.sqrt(DAYS)


def deflated_sharpe(r, n_trials: int, sr_var: float) -> float:
    """r: daily returns of the selected strategy; sr_var: variance of the trials' DAILY
    (not annualised) Sharpe ratios. n_trials = 1 gives the probabilistic Sharpe vs 0."""
    r = np.asarray(r, dtype=float)
    if len(r) < 3 or not r.std() > 0:
        return 0.0
    sr = r.mean() / r.std(ddof=1)
    z = (r - r.mean()) / r.std()
    skew, kurt = float((z ** 3).mean()), float((z ** 4).mean())
    sr0 = 0.0
    if n_trials > 1 and sr_var > 0:
        g = 0.5772156649                                  # Euler–Mascheroni
        sr0 = np.sqrt(sr_var) * ((1 - g) * _ND.inv_cdf(1 - 1 / n_trials)
                                 + g * _ND.inv_cdf(1 - 1 / (n_trials * np.e)))
    den = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    return float(_ND.cdf((sr - sr0) * np.sqrt(len(r) - 1) / den))


def summary(d: pd.DataFrame, bench: pd.Series) -> dict:
    """d: engine.returns() frame; bench: benchmark daily returns over the same days."""
    years = max((d.index[-1] - d.index[0]).days, 1) / 365.25 if len(d) > 1 else 1.0
    net, gross = cagr(d["ret"]), cagr(d["ret_gross"])
    dd = float(drawdown(d["ret"]).min()) if len(d) else 0.0
    return {
        "cagr": net, "cagr_gross": gross, "sharpe": sharpe(d["ret"]), "sortino": sortino(d["ret"]),
        "vol": float(d["ret"].std() * np.sqrt(DAYS)), "max_dd": dd,
        "calmar": net / -dd if dd < 0 else 0.0, "cost_drag": gross - net,
        "turnover_per_year": float(d["turnover"].sum() / years),
        "net_pnl": float(d["pnl"].sum()), "costs": float(d["cost"].sum()),
        "bench_cagr": cagr(bench), "bench_sharpe": sharpe(bench),
        "days": len(d), "years": years,
    }


def trade_stats(trades: pd.DataFrame) -> dict:
    """Bracket trade statistics in R (net of costs)."""
    r = trades["r_net"].to_numpy(dtype=float)
    if not len(r):
        return {"trades": 0}
    wins, losses = r[r > 0], r[r <= 0]
    streak = best = 0
    for x in r:
        streak = streak + 1 if x <= 0 else 0
        best = max(best, streak)
    return {
        "trades": len(r), "win_rate": float((r > 0).mean()), "avg_r": float(r.mean()),
        "avg_win_r": float(wins.mean()) if wins.size else 0.0,
        "avg_loss_r": float(losses.mean()) if losses.size else 0.0,
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else None,
        "total_r": float(r.sum()), "max_loss_streak": best,
        "results": trades["result"].value_counts().to_dict(),
    }
