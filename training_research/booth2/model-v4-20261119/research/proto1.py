"""Prototype post-processing on saved leave-one-cycle-out curves: shrink to the base rate per horizon,
countdown smoothing, cap by the fast estimate, conformal band. Not nested: used to choose a design, not to report."""
import numpy as np, pandas as pd, v4lib as L
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300)
frame = L.load(); g = frame.cycle_id.to_numpy(); cycles = list(pd.unique(g))
store = {}
for tag in ("r1", "r2"):
    store.update(dict(np.load(L.OUT / f"bench_{tag}.npz")))
y, known = L.cdf_targets(frame.rul_h.fillna(-1).to_numpy(), frame.event.to_numpy().astype(bool))
alive = frame.rul_h.notna().to_numpy()
known = known * alive[:, None]
wc = np.zeros(len(frame))  # every cycle weighs the same
for c in cycles:
    m = (g == c) & alive
    wc[m] = 1 / m.sum()


def base_rate(held):
    """Constant CDF from the other cycles (Kaplan-Meier style: known answers only)."""
    m = (g != held)
    return (y[m] * known[m]).sum(0) / np.maximum(known[m].sum(0), 1)


BASE = np.zeros((len(frame), len(L.H)))
for c in cycles:
    BASE[g == c] = base_rate(c)


def nll(F, mask=None):
    p = np.clip(F, 1e-4, 1 - 1e-4)
    l = -(y * np.log(p) + (1 - y) * np.log(1 - p)) * known * wc[:, None]
    if mask is not None:
        l = l[mask]
    return l.sum(0) / (known * wc[:, None])[mask if mask is not None else slice(None)].sum(0)


print("== per-horizon log-loss relative to the base rate (1.00 = no better than a constant); horizons in hours:", np.round(L.H, 2))
rows = {}
for name in ["cdf_logit|END", "cdf_logit|END|C0.01", "cdf_logit|END|C0.3", "cdf_hgb|END|mono", "cdf_hgb|PLUS|mono", "cdf_mlp8|END", "cdf_gru16|seq_raw",
             "cdf_tcn16|seq_raw", "cdf_gru16|seq_feat", "cox|END|pen0.3"]:
    rows[name] = nll(store[name]) / nll(BASE)
avg = lambda names: L.monotone(L.expit(np.mean([L.logit(store[n]) for n in names], 0)))
ENS = {"avg(logitC0.01,hgb)": ["cdf_logit|END|C0.01", "cdf_hgb|END|mono"],
       "avg(logitC0.01,hgb,gru_raw,tcn_raw)": ["cdf_logit|END|C0.01", "cdf_hgb|END|mono", "cdf_gru16|seq_raw", "cdf_tcn16|seq_raw"],
       "avg(hgb,gru_raw,tcn_raw)": ["cdf_hgb|END|mono", "cdf_gru16|seq_raw", "cdf_tcn16|seq_raw"]}
for k, names in ENS.items():
    store[k] = avg(names); rows[k] = nll(store[k]) / nll(BASE)
print(pd.DataFrame(rows, index=[f"{h:.2f}" for h in L.H]).T.round(2).to_string())


def countdown(q, alpha):
    out = np.empty_like(q)
    for c in cycles:
        ix = np.flatnonzero(g == c)
        s = np.log(np.maximum(q[ix[0]], .02))
        for k, i in enumerate(ix):
            if k:
                s = np.log(np.maximum(np.exp(s) - L.STEP_H, .02))
                s = s + alpha * (np.log(np.maximum(q[i], .02)) - s)
            out[i] = np.exp(s)
    return out


def conformal(q, nominal=.8, buckets=(1.0, 4.0)):
    """Leave-one-cycle-out band: for each cycle, log-ratio residuals of the other observed cycles, by predicted-median bucket."""
    out = q.copy()
    truth = frame.rul_h.to_numpy(); ev = frame.event.to_numpy().astype(bool)
    r = np.log(np.maximum(truth, 1e-3)) - np.log(np.maximum(q[:, 1], 1e-3))
    b = np.digitize(q[:, 1], buckets)
    lo, hi = (1 - nominal) / 2, 1 - (1 - nominal) / 2
    for c in cycles:
        for k in range(len(buckets) + 1):
            cal = (g != c) & ev & alive & (b == k)
            if cal.sum() < 50:
                cal = (g != c) & ev & alive
            w = wc[cal] / wc[cal].sum()
            order = np.argsort(r[cal]); cum = np.cumsum(w[order])
            a, z = r[cal][order][np.searchsorted(cum, lo)], r[cal][order][min(np.searchsorted(cum, hi), cal.sum() - 1)]
            m = (g == c) & (b == k)
            out[m, 0], out[m, 2] = q[m, 1] * np.exp(a), np.minimum(q[m, 1] * np.exp(z), L.CAP)
    return out


res = {}
for name in ["cdf_logit|END", "cdf_logit|END|C0.01", "cdf_hgb|END|mono", "cdf_gru16|seq_raw", "cdf_tcn16|seq_raw"] + list(ENS):
    F = store[name]
    for lam_name, lam in (("raw", np.ones(len(L.H))), ("shrinkfar", np.where(L.H <= 2, 1.0, np.where(L.H <= 6, .5, .15)))):
        Fs = L.monotone(L.expit(lam * L.logit(F) + (1 - lam) * L.logit(BASE)))
        q = L.quantiles(Fs)
        res[f"{name}|{lam_name}"] = L.life_summary(frame, q)
        res[f"{name}|{lam_name}|conf"] = L.life_summary(frame, conformal(q))
        for a in (.02, .005):
            slow = countdown(q, a)
            capped = np.minimum(slow, q)
            res[f"{name}|{lam_name}|cd{a}+cap|conf"] = L.life_summary(frame, conformal(capped))
res["base rate only"] = L.life_summary(frame, L.quantiles(BASE))
res["base rate only|conf"] = L.life_summary(frame, conformal(L.quantiles(BASE)))
v3 = pd.read_csv(L.V3 / "model_v3/v3_predictions.csv.gz")
res["v3 (nested)"] = L.life_summary(frame, v3[["q10", "q50", "q90"]].to_numpy())
print(pd.DataFrame(res).T.sort_values("pinball").round(2).to_string())
