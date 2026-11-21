"""Compare every fold output saved so far: life (remaining hours) and warning (two protocols)."""
import sys
import numpy as np, pandas as pd, v4lib as L
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300)
frame = L.load(); events, products = L.tables(); g = frame.cycle_id.to_numpy()
hours = L.hours_of(products, list(frame.cycle_id.unique()))
store = {}
for tag in sys.argv[1:] or ["r1", "r2"]:
    store.update(dict(np.load(L.OUT / f"bench_{tag}.npz")))
v3 = pd.read_csv(L.V3 / "model_v3/v3_predictions.csv.gz")
life = {}
for name, value in store.items():
    if value.ndim == 2:
        for a in (1.0, .1):
            life[f"{name}~{a}"] = L.life_summary(frame, L.quantiles(L.smooth_cdf(value, a, g)))
life["v3 (nested)"] = L.life_summary(frame, v3[["q10", "q50", "q90"]].to_numpy())
other = frame[frame.event.eq(1) & frame.rul_h.notna()]
const = np.zeros((len(frame), 3))
for c in frame.cycle_id.unique():
    const[frame.cycle_id.eq(c).to_numpy()] = np.quantile(other[other.cycle_id.ne(c)].rul_h, L.Q)
life["constant"] = L.life_summary(frame, const)
print("== remaining life, leave-one-cycle-out, 10 observed cycles weigh the same (hours)")
print(pd.DataFrame(life).T.sort_values("pinball").round(2).to_string())
rows = []
scores = {n: v for n, v in store.items() if v.ndim == 1 and not n.startswith("chart|")}
for n, v in store.items():
    if v.ndim == 2:
        for j, lab in ((2, "1h"), (4, "2h")):
            scores[f"{n}@{lab}"] = v[:, j]
scores["v3 warn (nested models)"] = v3.warn.to_numpy()
for name, s in scores.items():
    for a in (1.0, .3):
        sm = L.expit(L.ema_rows(L.logit(s), a, g)) if a < 1 else s
        fz, rm = L.frozen_curve(frame, sm, events, hours), L.reminder_curve(frame, sm, events)
        b1, b05 = L.best(fz, .1), L.best(fz, .05)
        r4, r2 = L.best(rm, 4.0, "fa_100h"), L.best(rm, 2.0, "fa_100h")
        rows.append(dict(score=f"{name}~{a}", hit_01=b1["hit"], fa_01=b1["fa"], th_01=b1["th"], hit_005=b05["hit"], fa_005=b05["fa"],
                         rem4_hit=r4["hit"], rem4_fa=r4["fa"], rem2_hit=r2["hit"], rem2_fa=r2["fa"]))
for name, s in store.items():  # control charts: thresholds on their own scale
    if name.startswith("chart|"):
        grid = np.unique(np.nanquantile(s, np.r_[np.linspace(.5, .99, 30), .995, .998, .999]))
        fz, rm = L.frozen_curve(frame, s, events, hours, grid), L.reminder_curve(frame, s, events, grid=grid)
        b1, b05 = L.best(fz, .1), L.best(fz, .05)
        r4, r2 = L.best(rm, 4.0, "fa_100h"), L.best(rm, 2.0, "fa_100h")
        rows.append(dict(score=name, hit_01=b1["hit"], fa_01=b1["fa"], th_01=b1["th"], hit_005=b05["hit"], fa_005=b05["fa"],
                         rem4_hit=r4["hit"], rem4_fa=r4["fa"], rem2_hit=r2["hit"], rem2_fa=r2["fa"]))
print("== warning. hit_01/hit_005: model_v2 protocol (first alarm 20-60 min early) at <=0.1 / <=0.05 false alarms per production hour;")
print("   rem4/rem2: alarm right when the end comes within 4 processing hours, >=20 min early, at <=4 / <=2 false alarms per 100 h. Best threshold per row (optimistic, same for all).")
print(pd.DataFrame(rows).sort_values(["hit_01", "hit_005", "rem4_hit", "rem2_hit"], ascending=False).head(60).to_string(index=False))
