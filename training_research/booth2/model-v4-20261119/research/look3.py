"""Two checks before the design is frozen: (1) how much censoring biases the known-answer base rate,
(2) what each v4 model shows on the two recorded end-of-life mornings that the replay rows cover."""
import json, sys
import numpy as np, pandas as pd
import v4lib as L
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
frame = L.load().reset_index(drop=True)
# ---- (1) censoring weights: G = Kaplan-Meier of the censoring age over the 14 cycles
cyc = frame.groupby("cycle_id", sort=False).agg(life=("life_h", "first"), ev=("event", "first"))
ages = np.sort(cyc.life.to_numpy())
def G(t):  # P(censoring age > t)
    g = 1.0
    for a, e in sorted(zip(cyc.life, cyc.ev)):
        if a <= t and e == 0:
            g *= 1 - 1 / (cyc.life >= a).sum()
    return g
alive = frame.rul_h.notna().to_numpy()
rul, ev, age = frame.rul_h.to_numpy(), frame.event.to_numpy().astype(bool), frame.age_h.to_numpy()
y, known = L.cdf_targets(np.nan_to_num(rul, nan=-1.0), ev)
known = known * alive[:, None]
Gv = np.vectorize(G)
w = np.zeros_like(known)
Ga = Gv(age)
for j, h in enumerate(L.H):
    until = np.where(ev & (rul <= h), age + rul, age + h)  # the answer needed observation up to here
    w[:, j] = known[:, j] * Ga / np.maximum(Gv(until - 1e-6), 1e-6)
naive = (y * known).sum(0) / known.sum(0)
ipcw = (y * w).sum(0) / w.sum(0)
print("horizon h      ", np.round(L.H, 2).tolist())
print("known-answer   ", np.round(naive, 3).tolist())
print("censoring-wtd  ", np.round(ipcw, 3).tolist())
print("share of origins with unknown answer", np.round(1 - known.sum(0) / alive.sum(), 3).tolist())
print("max weight", round(float(w.max()), 2))
# ---- (2) the two mornings
v4 = pd.read_csv("../model_v4/v4_predictions.csv.gz", parse_dates=["timestamp"])
v3 = pd.read_csv(L.V3 / "model_v3/v3_predictions.csv.gz", parse_dates=["timestamp"])
for label, a, b in (("L008 10/28, end 08:25:39", "2026-10-28 07:30", "2026-10-28 08:32"), ("L009 11/01, end 13:33:02", "2026-11-01 12:30", "2026-11-01 13:38")):
    m = (v4.timestamp >= a) & (v4.timestamp < b)
    q = v4[m].iloc[::24]
    print("\n==", label, "origins", int(m.sum()))
    cols = ["timestamp", "rul_h"] + [f"{t}_{k}" for t in ("tree", "gru", "fusion") for k in ("q10", "q50", "q90", "p1h", "warn")]
    out = q[cols].copy(); out["v3_q50"] = v3[m].iloc[::24].q50.to_numpy(); out["v3_warn"] = v3[m].iloc[::24].warn.to_numpy()
    out["timestamp"] = out.timestamp.dt.strftime("%H:%M")
    print(out.round(2).to_string(index=False))
man = json.load(open("../model_v4/bundle/manifest.json", encoding="utf-8"))
for mid in ("life_tree", "life_gru", "life_fusion"):
    c = {x["cycle"]: x for x in man["evaluation"][mid]["cycles"]}
    print(mid, "thresholds L008/L009", c["L008"]["threshold"], c["L009"]["threshold"], "| L008 event", c["L008"]["event"]["first_alarm"], c["L008"]["event"]["lead_minutes"],
          "| L009 event", c["L009"]["event"]["first_alarm"], c["L009"]["event"]["lead_minutes"])
