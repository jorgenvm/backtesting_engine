"""
Fill costs (hard rule: never clean fills). Per contract, in points:

  entry fill  = entry ± slippage          (market)
  exit fill   = exit ∓ slippage           (market: stop, breakeven, session end, roll)
  target fill = target                    (resting limit, no slippage)
  commission  = round-trip USD / point value

r_net = signed(exit_fill − entry_fill − commission) / planned risk

Expects trade columns: direction ("long"|"short"), entry_price, exit_price,
risk_pts, r_gross, result ("win" = target hit; anything else exited at market).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def apply(trades: pd.DataFrame, code: str,
          slippage_ticks: float = config.SLIPPAGE_TICKS,
          commission: float | None = config.COMMISSION,
          risk_usd: float = config.RISK_PER_TRADE) -> pd.DataFrame:
    """Return trades with entry_fill, exit_fill, r_net, cost_r, pnl_usd, contracts_at_risk."""
    inst = config.INSTRUMENTS[code.upper()]
    slip = slippage_ticks * inst.tick
    comm = (inst.commission if commission is None else commission) / inst.point_value
    t    = trades.copy()
    if t.empty:
        for col in ("entry_fill", "exit_fill", "r_net", "cost_r", "pnl_usd", "contracts_at_risk"):
            t[col] = pd.Series(dtype=float)
        return t

    sign        = np.where(t["direction"] == "long", 1.0, -1.0)
    market_exit = (t["result"] != "win").to_numpy()
    t["entry_fill"] = t["entry_price"] + sign * slip
    t["exit_fill"]  = t["exit_price"] - sign * slip * market_exit
    t["r_net"]      = (sign * (t["exit_fill"] - t["entry_fill"]) - comm) / t["risk_pts"]
    t["cost_r"]     = t["r_gross"] - t["r_net"]
    t["pnl_usd"]    = t["r_net"] * risk_usd
    # Whole contracts the risk budget buys; 0 means the stop is too wide to trade at this risk.
    t["contracts_at_risk"] = np.floor(risk_usd / (t["risk_pts"] * inst.point_value)).astype(int)
    return t
