"""
All tunable parameters of the backtesting engine. No strategy settings live here —
a strategy keeps its own and hands the engine finished trade candidates.

Importing this module loads .env (if python-dotenv is installed), so the Databento
key reaches the environment before the loader reads it. Keys are never stored here —
only the names of the environment variables that hold them.
"""
from datetime import time
from pathlib import Path
from typing import NamedTuple

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:                     # dotenv is optional; plain env vars work too
    pass


# ── Instruments ───────────────────────────────────────────────────────────────
class Instrument(NamedTuple):
    symbol:      str     # Databento continuous symbol (calendar roll, front month)
    tick:        float   # minimum price increment (points)
    point_value: float   # USD per 1.0 point per contract
    commission:  float   # USD round trip per contract (broker + exchange + NFA, all-in estimate)
    quarterly:   bool    # expires 3rd Friday of Mar/Jun/Sep/Dec (roll fallback without instrument_id)


INSTRUMENTS = {
    "NQ": Instrument("NQ.c.0", 0.25,   20.0, 4.50, quarterly=True),
    "CL": Instrument("CL.c.0", 0.01, 1000.0, 4.50, quarterly=False),
    "GC": Instrument("GC.c.0", 0.10,  100.0, 4.50, quarterly=False),
}

DATABENTO_DATASET = "GLBX.MDP3"
DATABENTO_KEY_ENV = "DATABENTO_API_KEY"

# ── Sessions (Europe/Berlin wall-clock, converted to UTC per day) ─────────────
SESSION_TZ = "Europe/Berlin"
SESSIONS = {
    "London":   (time(9, 0),   time(12, 0)),
    "New York": (time(15, 30), time(18, 0)),
    "Scalp":    (time(20, 0),  time(21, 30)),
}

# ── Position state (backtest/engine.py) ───────────────────────────────────────
MAX_OPEN_TRADES      = 1               # trades open at once; None = no limit
ONE_TRADE_PER_SIGNAL = True            # candidates sharing a signal_id trade once
SESSION_STOP_R       = None            # stop a session once its closed trades reach -this R (gross,
                                       # planned R: 2 = two full stops); None = off

# ── Account / sizing ──────────────────────────────────────────────────────────
INITIAL_BALANCE = 100_000.0            # USD
RISK_PER_TRADE  =     100.0            # USD per 1R

# ── Fill costs (hard rule: never clean fills) ─────────────────────────────────
SLIPPAGE_TICKS = 1.0                   # per market fill; resting target limits get no slip
COMMISSION     = None                  # USD round trip per contract; None = instrument default
