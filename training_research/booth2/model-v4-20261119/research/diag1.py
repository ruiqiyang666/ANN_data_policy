"""Where round 1 is right and wrong: per-cycle life errors, every alarm of the best warning score, voltage vs context."""
import numpy as np, pandas as pd, v4lib as L
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
frame = L.load(); store = dict(np.load(L.OUT / "bench_r1.npz")); events, products = L.tables()
g = frame.cycle_id.to_numpy()
q = L.quantiles(L.smooth_cdf(store["cdf_logit|END"], 1.0, g))
r = L.life_scores(frame, q)
print("== life, cdf_logit|END per cycle"); print(r.round(2).to_string(index=False))
# trajectory of the median at fixed true remaining life, observed cycles
rows = []
for c, d in frame[frame.event.eq(1) & frame.rul_h.notna()].groupby("cycle_id", sort=False):
    row = dict(cycle=c)
    for t in (12, 8, 6, 4, 3, 2, 1.5, 1, .5, .25):
        m = (d.rul_h - t).abs() < .05
        row[f"@{t}h"] = np.median(q[d.index][m.to_numpy(), 1]) if m.any() else np.nan
    rows.append(row)
print("== median predicted remaining hours when the truth is ..."); print(pd.DataFrame(rows).round(1).to_string(index=False))
s = L.expit(L.ema_rows(L.logit(store["warn_hgb|END|mono"]), .3, g))
eol = events[events.is_eol.eq(1)].set_index("cycle_id").time
hours = L.hours_of(products, list(frame.cycle_id.unique()))
for th in (.9, .95):
    m, ev_rows, alarms = L.EV.evaluate(frame, s, th, events, hours)
    print(f"== alarms of warn_hgb|END|mono~0.3 at {th}: hit {m['hit20']} late {m['late']} missed {m['missed']} false {m['false_alarms']} unknown {m['unknown_alarms']}")
    out = []
    for a in alarms:
        i = a["index"]; x = frame.iloc[i]
        to_end = (eol[x.cycle_id] - x.timestamp).total_seconds() / 60 if x.cycle_id in eol.index else np.nan
        out.append(dict(cycle=x.cycle_id, time=str(x.timestamp)[5:16], kind="hit/late" if a["matched_event"] else "FALSE" if a["false_alarm"] else "unknown" if a["unknown"] else "post",
                        min_to_end=round(to_end, 0), rul_h=x.rul_h, age_h=x.age_h, since_restart_h=x.since_restart_h, v60=x.v60, vsd60=x.vsd60,
                        vsd30=x.vsd_mean_30m, d_peak_30m=x.d_peak_30m, margin=x.margin51_30m, q50=q[i, 1]))
    print(pd.DataFrame(out).round(2).to_string(index=False))
    print(pd.DataFrame(ev_rows)[["event_id", "cycle_id", "first_alarm", "lead_minutes", "hit20", "late", "missed", "earlier_alarm_count"]].round(1).to_string(index=False))
# context: how much of the within-cycle voltage variation in healthy operation do operating conditions explain?
h = frame[(frame.rul_h.gt(4) | frame.event.eq(0)) & frame.v60.notna()].copy()
ctx = ["current", "argon", "nitrogen", "inlet_T", "cool_flow", "since_restart_h", "hopper", "feed_act", "carrier_act", "o2"]
h["v_c"] = h.v60 - h.groupby("cycle_id").v60.transform("median")
print("== healthy origins", len(h), "within-cycle sd of v60", round(h.v_c.std(), 3))
print(h[ctx + ["v_c"]].corr(method="spearman").v_c.round(3).to_string())
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
x, y, grp = h[ctx].to_numpy(float), h.v_c.to_numpy(), h.cycle_id.to_numpy()
pred = np.zeros(len(h))
for a, b in GroupKFold(7).split(x, y, grp):
    pred[b] = HistGradientBoostingRegressor(max_depth=3, max_iter=150, learning_rate=.05, min_samples_leaf=200).fit(x[a], y[a]).predict(x[b])
print("grouped-CV R2 of context -> within-cycle voltage:", round(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(), 3))
print(h.groupby(pd.cut(h.since_restart_h, [0, .05, .1, .25, .5, 1, 2, 8], include_lowest=True)).agg(n=("v_c", "size"), v_c=("v_c", "median"), vsd=("vsd60", "median")).round(3).to_string())
print(h.groupby("hopper").agg(n=("v_c", "size"), v_c=("v_c", "median"), vsd=("vsd60", "median")).round(3).to_string())
