"""
config.yaml as a dict (CFG), plus paths and the instrument lookup.

Importing this loads .env (if python-dotenv is installed) so API keys reach the
environment. Keys are never stored in config.yaml.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:                     # dotenv is optional; plain env vars work too
    pass

CFG: dict = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
DATA_DIR    = ROOT / CFG["data"]["dir"]
RESULTS_DIR = ROOT / "results"
TZ: str     = CFG["timezone"]


def instrument(symbol: str) -> dict:
    try:
        return CFG["instruments"][symbol.upper()]
    except KeyError:
        raise KeyError(f"{symbol} is not in config.yaml instruments: add its tick, point_value, "
                       f"commission and roll.") from None


def holdout_start() -> pd.Timestamp:
    return pd.Timestamp(CFG["data"]["holdout_start"], tz="UTC")
