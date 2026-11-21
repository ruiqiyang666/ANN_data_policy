"""What the last hours before each observed end look like (calendar minutes before T)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
V3 = Path(__file__).resolve().parents[2] / "model-v3-20261118"
sys.path.insert(0, str(V3 / "research"))
import data as D
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 400)
fr = D.build().reset_index(drop=True)
ev = pd.read_csv(D.REPO / "model_v2_GRU/data/events.csv", parse_dates=["time"])
ev = ev[ev.is_eol.eq(1)]
bins = [(-1e9, -600), (-600, -180), (-180, -120), (-120, -90), (-90, -60), (-60, -40), (-40, -20), (-20, -10), (-10, 0)]
rows = []
for e in ev.itertuples():
    q = fr[fr.cycle_id.eq(e.cycle_id)]
    m = (q.timestamp - e.time).dt.total_seconds() / 60
    for lo, hi in bins:
        s = q[(m >= lo) & (m < hi)]
        if not len(s):
            continue
        rows.append(dict(cycle=e.cycle_id, win=f"{int(max(lo,-9999))}..{hi}", n=len(s), v60=s.v60.median(), vq10=s.v60.quantile(.1), vmin=s.v_last.min(),
                         vsd=s.vsd60.median(), vsd90=s.vsd60.quantile(.9), low53=(s.v60 < 53).mean(), low52=(s.v60 < 52).mean(),
                         cur=s.current.median(), pwr=s.power.median(), eff=s.therm_eff.median(), dT=s.cool_dT.median(), gap=s.gap_s.max()))
r = pd.DataFrame(rows)
for c, g in r.groupby("cycle", sort=False):
    print(g.round(3).to_string(index=False)); print()
