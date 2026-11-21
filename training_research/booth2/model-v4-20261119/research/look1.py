"""First look at the research frame: columns, per-cycle life, and what the end of life looks like."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V3 = Path(__file__).resolve().parents[2] / "model-v3-20261118"
sys.path.insert(0, str(V3 / "research"))
import data as D  # noqa: E402

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 400)
fr = D.build().reset_index(drop=True)
print("frame", fr.shape)
print("columns:", list(fr.columns))
raw = pd.read_parquet(V3 / "research/cache/raw28.parquet")
print("raw28 columns:", list(raw.columns))
g = fr.groupby("cycle_id", sort=False)
print(g.agg(n=("age_h", "size"), life_h=("life_h", "first"), event=("event", "first"), left_cens=("left_censored", "first"),
            pieces=("rul_pieces", "max"), first=("timestamp", "min"), last=("timestamp", "max"),
            v_first=("v60", lambda s: s.iloc[:360].median()), v_last=("v60", lambda s: s.iloc[-360:].median()),
            vsd_first=("vsd60", lambda s: s.iloc[:360].median()), vsd_last=("vsd60", lambda s: s.iloc[-360:].median())).round(3).to_string())
