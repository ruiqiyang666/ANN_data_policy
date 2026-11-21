"""Round 3: one model, consistent outputs.

Per held-out cycle (leave-one-cycle-out):
  life   : linear discrete-time hazard on end-stage features -> survival curve
  warn   : gradient-boosted trees on the same features -> P(end of life within 60 min)
  fusion : the warning probability (calibrated on inner out-of-fold scores of the
           13 training cycles) lifts the first-hour hazards when it is higher,
           so the life estimate cannot contradict an active warning
Only scikit-learn / NumPy models: they must run inside the panel container.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))
import data as D
import rul_bench as R
import rul_bench2 as R2
import alarm_bench as A

warnings.filterwarnings("ignore")
END = A.ENDSTAGE
FIRST_HOUR = 3  # hazard bins that end at one processing hour


def hgb(seed=0, depth=3):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_depth=depth, max_iter=150, learning_rate=.05, min_samples_leaf=100, l2_regularization=2.0,
                                          class_weight="balanced", random_state=seed)


def forest(kind, seed=0):
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    cls = RandomForestClassifier if kind == "rf" else ExtraTreesClassifier
    return cls(n_estimators=300, max_depth=6, min_samples_leaf=60, class_weight="balanced_subsample", n_jobs=1, random_state=seed)


def lift(hz, p_first, p_warn):
    """Raise the first-hour hazards so that P(end within 1 h) is at least the warning probability."""
    gamma = np.log(np.clip(1 - p_warn, 1e-6, 1)) / np.log(np.clip(1 - p_first, 1e-6, 1 - 1e-9))
    gamma = np.where(p_warn > p_first, np.maximum(gamma, 1), 1)
    out = hz.copy()
    out[:, :FIRST_HOUR] = 1 - (1 - hz[:, :FIRST_HOUR]) ** gamma[:, None]
    return out


def fold(frame, held, seed=0):
    import torch
    torch.set_num_threads(1)
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import GroupKFold
    tr_all = frame[frame.cycle_id != held]
    te = frame[frame.cycle_id == held]
    # --- warning classifiers on every supervised origin of the 13 training cycles
    w = tr_all[tr_all.y_eol_60m.notna()]
    xw, yw = w[END].to_numpy("float64"), w.y_eol_60m.to_numpy().astype(int)
    xte = te[END].to_numpy("float64")
    med = np.nanmedian(xw, 0)
    fill = lambda a: np.where(np.isfinite(a), a, med)
    scores = {}
    for depth in (2, 3):
        scores[f"hgb{depth}"] = hgb(seed, depth).fit(xw, yw).predict_proba(xte)[:, 1]
    scores["rf"] = forest("rf", seed).fit(fill(xw), yw).predict_proba(fill(xte))[:, 1]
    scores["et"] = forest("et", seed).fit(fill(xw), yw).predict_proba(fill(xte))[:, 1]
    scores["ens"] = (scores["hgb3"] + scores["rf"] + scores["et"]) / 3
    scores["ens2"] = (scores["hgb3"] + scores["rf"]) / 2
    # inner out-of-fold scores of the chosen warning model, for calibration to "end within one processing hour"
    inner = np.full(len(w), np.nan)
    for a, b in GroupKFold(5).split(xw, yw, w.cycle_id):
        inner[b] = (hgb(seed, 3).fit(xw[a], yw[a]).predict_proba(xw[b])[:, 1] + forest("rf", seed).fit(fill(xw[a]), yw[a]).predict_proba(fill(xw[b]))[:, 1]) / 2
    near = ((w.event == 1) & (w.rul_h <= 1)).to_numpy().astype(float)
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(inner, near)
    p_warn_all = iso.predict(scores["ens2"])
    # --- life: linear hazard on one origin per processing minute
    lt = tr_all[tr_all.rul_h.notna()].iloc[::6]
    sc = R.Scaler(lt[END].to_numpy("float64"))
    net = R.fit_hazard_mlp(sc(lt[END].to_numpy("float64")), lt.rul_h.to_numpy(), lt.event.to_numpy().astype(bool), hidden=(), epochs=40, seed=seed, wd=1e-2)
    hz_all = R.predict_hazard(net, sc(xte))
    p_first_all = 1 - np.prod(1 - hz_all[:, :FIRST_HOUR], 1)
    scores["hazard_1h"] = p_first_all
    scores["fused_1h"] = np.maximum(p_first_all, p_warn_all)
    scores["warn_cal"] = p_warn_all
    alive = te.rul_h.notna().to_numpy()
    truth, age = te.rul_h.to_numpy()[alive], te.age_h.to_numpy()[alive]
    res = {}
    base = R.survival_quantiles(hz_all[alive])
    fused = R.survival_quantiles(lift(hz_all, p_first_all, p_warn_all)[alive])
    res["hazard_lin"] = base
    res["fused"] = fused
    for a in (.01, .02):
        res[f"hazard_lin~s{a}"] = np.column_stack([R2.smooth(base[:, j], a) for j in range(3)])
        res[f"fused~s{a}"] = np.column_stack([R2.smooth(fused[:, j], a) for j in range(3)])
        # smooth the slow life estimate, then let the warning cap it: a warning is a fast signal
        s = np.column_stack([R2.smooth(base[:, j], a) for j in range(3)])
        res[f"smooth_then_cap~s{a}"] = np.minimum(s, fused)
    cyc = tr_all.groupby("cycle_id").agg(life=("life_h", "first"), ev=("event", "first"))
    res["mean_life"] = np.column_stack([np.maximum(cyc[cyc.ev == 1].life.mean() - age, .05)] * 3)
    return held, te.index.to_numpy(), scores, truth, age, res


def main(tag="v3", seed=0):
    frame = D.build().reset_index(drop=True)
    events = pd.read_csv(D.REPO / "model_v2_GRU/data/events.csv", parse_dates=["time"])
    prod = pd.read_csv(D.REPO / "model_v2_GRU/data/products.csv.gz", low_memory=False)
    cycles = list(frame.cycle_id.unique())
    out = Parallel(n_jobs=14)(delayed(fold)(frame, c, seed) for c in cycles)
    event = frame.groupby("cycle_id").event.first()
    names = list(out[0][2])
    score = {n: np.full(len(frame), np.nan) for n in names}
    rows, store = [], {}
    for held, ix, scores, truth, age, res in out:
        for n in names:
            score[n][ix] = scores[n]
        for name, p in res.items():
            rows.append(dict(model=name, cycle=held, event=int(event[held]), **R.score(truth, p, event[held])))
            store[f"{name}@{held}"] = np.column_stack([age, truth, p]).astype("float32")
    hrs = A.hours(prod, cycles)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    print(f"production hours {hrs:.1f}, events {int(events.is_eol.sum())}")
    grid = np.r_[np.arange(.05, 1, .05), .97, .98, .99, .995, .02, .03, .01]
    for n in names:
        c = []
        for th in np.unique(grid):
            m, ev_rows, alarms = A.EV.evaluate(frame, score[n], float(th), events, hrs)
            c.append(dict(model=n, th=round(float(th), 3), hit=m["hit20"], late=m["late"], missed=m["missed"], fa=m["false_alarms"],
                          fa_h=round(m["false_alarms_per_hour"], 3), post=m["post_event_alarms"], unknown=m["unknown_alarms"],
                          mean_excess=m["mean_excess_minutes"]))
        c = pd.DataFrame(c)
        ok = c[c.fa_h <= .1].sort_values(["hit", "fa"], ascending=[False, True]).head(3)
        ok05 = c[c.fa_h <= .05].sort_values(["hit", "fa"], ascending=[False, True]).head(2)
        print(pd.concat([ok, ok05]).to_string(index=False))
        c.to_csv(D.ROOT / "research" / "out" / f"alarm_{tag}_{n}.csv", index=False)
    r = pd.DataFrame(rows)
    ev = r[r.event == 1]
    s = ev.groupby("model")[["mae", "mae_last5h", "mae_last2h", "mae_last1h", "bias", "cover80", "rel_last5h"]].mean()
    s["mae_median_cycle"] = ev.groupby("model").mae.median()
    s["censored_under"] = r[r.event == 0].groupby("model").under_bound.mean()
    print(s.sort_values("mae").round(3).to_string())
    r.to_csv(D.ROOT / "research" / "out" / f"rul_{tag}_cycles.csv", index=False)
    np.savez_compressed(D.ROOT / "research" / "out" / f"round3_{tag}.npz", **{f"S:{n}": v.astype("float32") for n, v in score.items()}, **store)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["v3"]), seed=int(sys.argv[2]) if len(sys.argv) > 2 else 0)
