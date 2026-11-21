"""Round 4: the small models the literature favours for ten events.

  logitK   discrete-time hazard / CDF with 3-5 hand-picked features (events-per-variable in mind)
  wiener   Brownian motion with drift on the margin to 51 V, drift prior from the other cycles and a
           conjugate update from the cycle's own path; first-hitting-time distribution of the 51 V line
  expo     the same with the drop from the cycle's early level as the indicator and the other cycles'
           end-of-life drops as a random threshold
usage: python bench4.py [tag]
"""
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm

import bench2 as B
import v4lib as L

SMALL = {
    "logit3": ["margin51_30m", "d_peak_30m", "log_vsd_30m"],
    "logit4": ["margin51_30m", "d_peak_30m", "log_vsd_30m", "v_slope_30m"],
    "logit5": ["margin51_30m", "d_peak_30m", "log_vsd_30m", "v_slope_30m", "vsd_ratio"],
    "logit2": ["margin51_q10", "log_vsd_30m"],
}
STEP = 60  # ten processing minutes, in origins


def fht_cdf(m, d, sigma2, t):
    """P(a Brownian motion starting m above the line, drifting down at d per hour, reaches it within t hours)."""
    s = np.sqrt(sigma2 * t)
    return norm.cdf((-m + d * t) / s) + np.exp(np.clip(2 * d * m / sigma2, -60, 60)) * norm.cdf((-m - d * t) / s)


def path_stats(q, col):
    """Per cycle: the indicator every ten processing minutes."""
    x = q[col].to_numpy(float)[::STEP]
    return x[np.isfinite(x)]


def degradation(frame, train, te, col, sign, threshold_values=None):
    """Drift prior from the training cycles, conjugate update inside the test cycle, hitting-time CDF at the horizons.

    col: indicator; sign=-1 when it falls towards a fixed line at 0 (margin), sign=+1 when it rises towards a random threshold.
    """
    dt = STEP * L.STEP_H
    drifts, incs = [], []
    for _, q in train.groupby("cycle_id", sort=False):
        x = path_stats(q, col)
        if len(x) > 6:
            drifts.append(sign * (x[-1] - x[0]) / (len(x) * dt))
            incs.append(np.diff(x))
    d0, s0 = float(np.mean(drifts)), float(np.std(drifts) + 1e-6)
    sigma2 = float(np.var(np.concatenate(incs)) / dt)
    x = te[col].to_numpy(float)
    ok = np.isfinite(x)
    first = np.flatnonzero(ok)[0] if ok.any() else 0
    t = np.maximum((np.arange(len(te)) - first) * L.STEP_H, 0)
    change = np.where(ok, sign * (x - x[first]), 0.0)
    prec = 1 / s0 ** 2 + t / sigma2
    mean = (d0 / s0 ** 2 + change / sigma2) / prec
    sd = np.sqrt(1 / prec)
    F = np.zeros((len(te), len(L.H)))
    zs = norm.ppf((np.arange(15) + .5) / 15)  # the posterior of the drift, integrated on 15 equal-probability points
    thresholds = [0.0] if threshold_values is None else list(threshold_values)
    for thr in thresholds:
        m = np.maximum(np.where(ok, (x if sign < 0 else thr - x), np.nan), .02)
        for z in zs:
            d = mean + z * sd
            F += np.column_stack([fht_cdf(m, d, sigma2, h) for h in L.H])
    F /= len(zs) * len(thresholds)
    return L.monotone(np.nan_to_num(F, nan=0.0))


def fold(frame, held):
    cycles = [c for c in frame.cycle_id.unique() if c != held]
    tr, w = B.train_rows(frame, cycles)
    te = frame[frame.cycle_id.eq(held)]
    train = frame[frame.cycle_id.isin(cycles)]
    y, known = L.cdf_targets(tr.rul_h.to_numpy(), tr.event.to_numpy().astype(bool))
    out = {}
    from sklearn.linear_model import LogisticRegression
    for name, cols in SMALL.items():
        sc = B.Scaler(tr[cols].to_numpy("float64"))
        ztr, zte = sc(tr[cols].to_numpy("float64")), sc(te[cols].to_numpy("float64"))
        units = []
        for j in range(len(L.H)):
            m = known[:, j] > 0
            units.append(LogisticRegression(C=1.0, max_iter=2000).fit(ztr[m], y[m, j], sample_weight=w[m]).predict_proba(zte)[:, 1])
        out[f"cdf_{name}"] = L.monotone(np.column_stack(units))
    out["wiener_bayes|margin"] = degradation(frame, train, te, "margin51_30m", -1)
    ends = []
    for _, q in train[train.event.eq(1) & train.rul_h.notna()].groupby("cycle_id", sort=False):
        v = q.d_first_30m.to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v):
            ends.append(max(float(v[-30:].mean()), .2))  # the drop this cycle showed in its last five minutes
    out["wiener_bayes|drop_random_threshold"] = degradation(frame, train, te, "d_first_30m", +1, ends)
    return held, te.index.to_numpy(), out


def main(tag="r4"):
    frame = L.load()
    frame["log_vsd_30m"] = np.log(frame.vsd_mean_30m.clip(lower=.02))
    results = Parallel(n_jobs=14)(delayed(fold)(frame, c) for c in frame.cycle_id.unique())
    store = {}
    for held, index, out in results:
        for name, value in out.items():
            store.setdefault(name, np.full((len(frame),) + value.shape[1:], np.nan))[index] = value
    np.savez_compressed(L.OUT / f"bench_{tag}.npz", **store)
    print("saved", sorted(store))


if __name__ == "__main__":
    main(*sys.argv[1:2])
