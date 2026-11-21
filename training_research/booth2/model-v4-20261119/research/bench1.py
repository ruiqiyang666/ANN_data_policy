"""Round 1: multi-horizon survival with boosted trees, leave-one-cycle-out.

Variants: feature set (END = v3's 23, PLUS = + damage/exposure/product features), monotone constraints on or off,
per-horizon linear baseline, and the v3-style warning classifier on both feature sets. Raw fold outputs are saved so
smoothing, ensembling and thresholds can be studied afterwards without refitting.
usage: python bench1.py [tag]
"""
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import v4lib as L

SEEDS = (0, 1, 2)


def train_rows(frame, cycles):
    """One origin per processing minute, every origin of the last three hours."""
    q = frame[frame.cycle_id.isin(cycles) & frame.rul_h.notna()]
    keep = (np.arange(len(q)) % 6 == 0) | (q.event.eq(1) & q.rul_h.le(3)).to_numpy()
    return q[keep]


def fit_cdf_hgb(x, rul, event, feats, mono, depth=3, iters=200, leaf=200):
    from sklearn.ensemble import HistGradientBoostingClassifier
    y, known = L.cdf_targets(rul, event)
    rows, cols = np.nonzero(known)
    X = np.column_stack([x[rows], L.H[cols]])
    cst = ([L.MONO.get(f, 0) for f in feats] + [1]) if mono else ([0] * len(feats) + [1])
    return [HistGradientBoostingClassifier(max_depth=depth, max_iter=iters, learning_rate=.05, min_samples_leaf=leaf, l2_regularization=2.0,
                                           max_features=.7, monotonic_cst=cst, early_stopping=False, random_state=s).fit(X, y[rows, cols])
            for s in SEEDS]


def predict_cdf(models, x):
    cols = [np.mean([m.predict_proba(np.column_stack([x, np.full(len(x), h)]))[:, 1] for m in models], 0) for h in L.H]
    return L.monotone(np.column_stack(cols))


def fit_cdf_logit(x, rul, event, c=.05):
    from sklearn.linear_model import LogisticRegression
    med = np.nanmedian(x, 0)
    q = np.nanpercentile(x, [25, 75], 0)
    spread = np.maximum(q[1] - q[0], 1e-3 + .05 * np.abs(med))
    z = np.clip((np.where(np.isfinite(x), x, med) - med) / spread, -8, 8)
    y, known = L.cdf_targets(rul, event)
    units = []
    for j in range(len(L.H)):
        m = known[:, j] > 0
        if y[m, j].min() == y[m, j].max():
            units.append(float(y[m, j].mean()))
        else:
            units.append(LogisticRegression(C=c, max_iter=2000).fit(z[m], y[m, j]))
    return med, spread, units


def predict_logit(model, x):
    med, spread, units = model
    z = np.clip((np.where(np.isfinite(x), x, med) - med) / spread, -8, 8)
    return L.monotone(np.column_stack([np.full(len(z), u) if isinstance(u, float) else u.predict_proba(z)[:, 1] for u in units]))


def fit_warn(x, y, feats, mono):
    from sklearn.ensemble import HistGradientBoostingClassifier
    cst = [L.MONO.get(f, 0) for f in feats] if mono else None
    return [HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=.05, min_samples_leaf=100, l2_regularization=2.0,
                                           max_features=.7, class_weight="balanced", monotonic_cst=cst, early_stopping=False,
                                           random_state=s).fit(x, y) for s in SEEDS]


def fold(frame, held):
    cycles = [c for c in frame.cycle_id.unique() if c != held]
    tr, te = train_rows(frame, cycles), frame[frame.cycle_id.eq(held)]
    rul, event = tr.rul_h.to_numpy(), tr.event.to_numpy().astype(bool)
    out = {}
    for sname, feats in (("END", L.END), ("PLUS", L.PLUS)):
        xtr, xte = tr[feats].to_numpy("float64"), te[feats].to_numpy("float64")
        for mono in (True, False):
            out[f"cdf_hgb|{sname}|{'mono' if mono else 'free'}"] = predict_cdf(fit_cdf_hgb(xtr, rul, event, feats, mono), xte)
        out[f"cdf_logit|{sname}"] = predict_logit(fit_cdf_logit(xtr, rul, event), xte)
        w = frame[frame.cycle_id.isin(cycles) & frame.y_eol_60m.notna()]
        for mono in (True, False):
            models = fit_warn(w[feats].to_numpy("float64"), w.y_eol_60m.to_numpy().astype(int), feats, mono)
            out[f"warn_hgb|{sname}|{'mono' if mono else 'free'}"] = np.mean([m.predict_proba(xte)[:, 1] for m in models], 0)
    return held, te.index.to_numpy(), out


def main(tag="r1"):
    frame = L.load()
    cycles = list(frame.cycle_id.unique())
    results = Parallel(n_jobs=14)(delayed(fold)(frame, c) for c in cycles)
    store = {}
    for held, index, out in results:
        for name, value in out.items():
            shape = (len(frame),) + value.shape[1:]
            store.setdefault(name, np.full(shape, np.nan))[index] = value
    L.OUT.mkdir(exist_ok=True)
    np.savez_compressed(L.OUT / f"bench_{tag}.npz", **store)
    report(frame, store)


def report(frame, store, alphas=(1.0, .1, .03)):
    events, products = L.tables()
    hours = L.hours_of(products, list(frame.cycle_id.unique()))
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    life = {}
    for name, value in store.items():
        if value.ndim != 2:
            continue
        for a in alphas:
            life[f"{name}~{a}"] = L.life_summary(frame, L.quantiles(L.smooth_cdf(value, a, frame.cycle_id.to_numpy())))
    v3 = pd.read_csv(L.V3 / "model_v3/v3_predictions.csv.gz")
    life["v3 (nested)"] = L.life_summary(frame, v3[["q10", "q50", "q90"]].to_numpy())
    other = frame[frame.event.eq(1) & frame.rul_h.notna()]
    const = np.zeros((len(frame), 3))
    for c in frame.cycle_id.unique():  # constant quantiles from the other cycles
        const[frame.cycle_id.eq(c).to_numpy()] = np.quantile(other[other.cycle_id.ne(c)].rul_h, L.Q)
    life["constant"] = L.life_summary(frame, const)
    print(pd.DataFrame(life).T.sort_values("mae_2h").round(2).to_string())
    rows = []
    scores = {n: v for n, v in store.items() if v.ndim == 1}
    scores.update({f"{n}@1h": v[:, 2] for n, v in store.items() if v.ndim == 2})
    scores.update({f"{n}@4h": v[:, 6] for n, v in store.items() if v.ndim == 2})
    scores["v3 warn (nested models)"] = v3.warn.to_numpy()
    for name, s in scores.items():
        for a in (1.0, .3):
            sm = L.expit(L.ema_rows(L.logit(s), a, frame.cycle_id.to_numpy())) if a < 1 else s
            fz = L.frozen_curve(frame, sm, events, hours)
            rm = L.reminder_curve(frame, sm, events)
            b1, b05 = L.best(fz, .1), L.best(fz, .05)
            r4, r2 = L.best(rm, 4.0, "fa_100h"), L.best(rm, 2.0, "fa_100h")
            rows.append(dict(score=f"{name}~{a}", hit_01=b1["hit"], fa_01=b1["fa"], th_01=b1["th"], hit_005=b05["hit"], fa_005=b05["fa"],
                             rem4_hit=r4["hit"], rem4_fa=r4["fa"], rem4_th=r4["th"], rem2_hit=r2["hit"], rem2_fa=r2["fa"]))
    print(pd.DataFrame(rows).sort_values(["hit_01", "hit_005", "rem4_hit"], ascending=False).to_string(index=False))


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["r1"]))
