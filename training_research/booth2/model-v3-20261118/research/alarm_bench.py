"""Early-warning benchmark under the frozen model_v2 event protocol.

Scoring is the source project's own `evaluation.py` (alarm state machine, first
alarm >= 20 min before T, false alarms per production hour). Two protocols:

  frozen : train L000-L004, threshold picked on validation L005-L007 (budget
           0.1 false alarms / h), then L008-L013. Same as model_v2; that test
           period has been looked at before, so it is not a blind test.
  loco   : every cycle scored by a model that never saw it (13 others). One
           threshold over all out-of-fold scores gives an operating curve; the
           threshold is read off that curve, so the best point is optimistic,
           but every candidate is treated alike.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))
import data as D

sys.path.insert(0, str(D.REPO / "model_v2_GRU" / "src"))
import evaluation as EV  # noqa: E402  (source project's frozen scoring)

warnings.filterwarnings("ignore")
GRID = np.r_[np.arange(.05, 1, .05), .97, .98, .99, .995]
V9 = ["v60", "vsd60", "v_last", "gap_s", "since_restart_h"] + [c for c in D.feature_names(D.build()) if c.startswith(("v_", "vsd_", "d_", "margin51"))]
ENDSTAGE = ["vsd_mean_5m", "vsd_mean_30m", "vsd_mean_3h", "vsd60", "vsd_ratio", "v_std_5m", "v_std_30m", "d_peak_30m", "d_peak_3h",
            "d_5m_3h", "d_30m_base", "v_slope_30m", "v_slope_3h", "v_low53_30m", "margin51_30m", "margin51_q10", "v_min_30m",
            "cool_loss_kw_d_base", "cool_dT_d_base", "therm_eff_d_base", "power_d_base", "power_std_30m", "therm_eff_slope_3h"]


def hours(frame_products, cycles):
    return float(frame_products[frame_products.cycle_id.isin(cycles) & frame_products.supervised_product_eligible.eq(1)].spray_samples.sum() / 3600)


class Scaler:
    def __init__(self, x):
        self.med = np.nanmedian(x, 0)
        q = np.nanpercentile(x, [25, 75], 0)
        self.scale = np.maximum(q[1] - q[0], 1e-3 + 0.05 * np.abs(self.med))

    def __call__(self, x):
        return np.clip((np.where(np.isfinite(x), x, self.med) - self.med) / self.scale, -8, 8).astype("float32")


def fit_predict(name, xtr, ytr, xte, seed=0):
    """One candidate: returns a score in [0, 1] (or an arbitrary monotone score for rank models)."""
    kind = name.split(":")[0]
    if kind in ("logit", "mlp", "mlp_s", "svm"):
        sc = Scaler(xtr)
        ztr, zte = sc(xtr), sc(xte)
    if kind == "logit":
        from sklearn.linear_model import LogisticRegression
        c = float(name.split(":")[1])
        return LogisticRegression(C=c, max_iter=400, class_weight="balanced").fit(ztr, ytr).predict_proba(zte)[:, 1]
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        d = int(name.split(":")[1])
        return HistGradientBoostingClassifier(max_depth=d, max_iter=150, learning_rate=.05, min_samples_leaf=100, l2_regularization=2.0,
                                              class_weight="balanced", random_state=seed).fit(xtr, ytr).predict_proba(xte)[:, 1]
    if kind == "lgbm":
        import lightgbm as lgb
        leaves = int(name.split(":")[1])
        return lgb.LGBMClassifier(n_estimators=250, learning_rate=.03, num_leaves=leaves, min_child_samples=150, subsample=.7,
                                  subsample_freq=1, colsample_bytree=.6, reg_lambda=10, scale_pos_weight=5, verbose=-1,
                                  random_state=seed, n_jobs=1).fit(xtr, ytr).predict_proba(xte)[:, 1]
    if kind == "xgb":
        import xgboost as xgb
        d = int(name.split(":")[1])
        return xgb.XGBClassifier(n_estimators=250, learning_rate=.03, max_depth=d, min_child_weight=30, subsample=.7,
                                 colsample_bytree=.6, reg_lambda=10, scale_pos_weight=5, n_jobs=1, random_state=seed,
                                 tree_method="hist").fit(xtr, ytr).predict_proba(xte)[:, 1]
    if kind in ("rf", "et"):
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        cls = RandomForestClassifier if kind == "rf" else ExtraTreesClassifier
        med = np.nanmedian(xtr, 0)
        fill = lambda a: np.where(np.isfinite(a), a, med)
        return cls(n_estimators=300, max_depth=6, min_samples_leaf=60, class_weight="balanced_subsample", n_jobs=1,
                   random_state=seed).fit(fill(xtr), ytr).predict_proba(fill(xte))[:, 1]
    if kind in ("mlp", "mlp_s"):
        import torch
        from torch import nn
        torch.set_num_threads(1)
        hidden = (32, 16) if kind == "mlp" else (12,)
        outs = []
        for s in range(3):
            torch.manual_seed(seed + s)
            layers, d = [], ztr.shape[1]
            for h in hidden:
                layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(.25)]
                d = h
            net = nn.Sequential(*layers, nn.Linear(d, 1))
            opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
            xt, yt = torch.from_numpy(ztr), torch.from_numpy(ytr.astype("float32"))
            pos = torch.tensor(min(float((1 - ytr.mean()) / max(ytr.mean(), 1e-6)), 20.0))
            g = torch.Generator().manual_seed(seed + s)
            for _ in range(25):
                net.train()
                for ix in torch.randperm(len(xt), generator=g).split(512):
                    loss = nn.functional.binary_cross_entropy_with_logits(net(xt[ix])[:, 0], yt[ix], pos_weight=pos)
                    opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.inference_mode():
                outs.append(torch.sigmoid(net(torch.from_numpy(zte))[:, 0]).numpy())
        return np.mean(outs, 0)
    if kind == "iforest":  # healthy-only anomaly score, no failure labels
        from sklearn.ensemble import IsolationForest
        med = np.nanmedian(xtr, 0)
        fill = lambda a: np.where(np.isfinite(a), a, med)
        m = IsolationForest(n_estimators=300, random_state=seed, n_jobs=1).fit(fill(xtr[ytr == 0]))
        s = -m.score_samples(fill(xte))
        lo, hi = np.percentile(-m.score_samples(fill(xtr[ytr == 0])), [1, 99.9])
        return np.clip((s - lo) / (hi - lo), 0, 1)
    if kind == "maha":  # Mahalanobis / T^2 distance to healthy operation
        sc = Scaler(xtr[ytr == 0])
        z = sc(xtr[ytr == 0]).astype("float64")
        mu = z.mean(0)
        inv = np.linalg.pinv(np.cov(z - mu, rowvar=False) + 1e-2 * np.eye(z.shape[1]))
        dist = lambda a: np.sqrt(np.einsum("ij,jk,ik->i", a - mu, inv, a - mu))
        ref = np.percentile(dist(z), 99.9)
        return np.clip(dist(sc(xte).astype("float64")) / ref / 1.5, 0, 1)
    raise ValueError(name)


def run_fold(frame, train_cycles, test_cycles, cols, target, models, seed=0):
    tr = frame[frame.cycle_id.isin(train_cycles) & frame[target].notna()]
    te = frame[frame.cycle_id.isin(test_cycles)]
    xtr, ytr, xte = tr[cols].to_numpy("float64"), tr[target].to_numpy().astype(int), te[cols].to_numpy("float64")
    return te.index.to_numpy(), {m: fit_predict(m, xtr, ytr, xte, seed) for m in models}


def curve(frame, score, events, hrs):
    rows = []
    for th in GRID:
        m, _, _ = EV.evaluate(frame, score, float(th), events, hrs)
        rows.append(dict(th=th, hit=m["hit20"], late=m["late"], missed=m["missed"], fa=m["false_alarms"], fa_h=m["false_alarms_per_hour"],
                         post=m["post_event_alarms"], n=m["total_events"]))
    return pd.DataFrame(rows)


def best(c, budget):
    ok = c[c.fa_h <= budget]
    if not len(ok):
        return dict(hit=0, fa=0, th=np.nan, late=0)
    r = ok.sort_values(["hit", "fa", "late", "th"], ascending=[False, True, True, False]).iloc[0]
    return dict(hit=int(r.hit), fa=int(r.fa), th=float(r.th), late=int(r.late))


def main(tag="v1", models=None, sets=None, targets=("y_eol_60m", "y_eol_30m")):
    frame = D.build().reset_index(drop=True)
    events = pd.read_csv(D.REPO / "model_v2_GRU/data/events.csv", parse_dates=["time"])
    prod = pd.read_csv(D.REPO / "model_v2_GRU/data/products.csv.gz", low_memory=False)
    cyc = list(frame.cycle_id.unique())
    TR, VA, TE = cyc[:5], cyc[5:8], cyc[8:]
    allf = D.feature_names(frame, age=False)
    sets = sets or {"v9": [c for c in V9 if c in allf], "end": ENDSTAGE, "all28": allf}
    models = models or ["logit:0.03", "logit:0.3", "hgb:2", "hgb:3", "lgbm:4", "lgbm:8", "xgb:2", "xgb:3", "rf", "et", "mlp", "mlp_s",
                        "iforest", "maha"]
    jobs, keys = [], []
    for sname, cols in sets.items():
        for target in targets:
            jobs.append(delayed(run_fold)(frame, TR, cyc, cols, target, models)); keys.append(("frozen", sname, target, None))
            for held in cyc:
                jobs.append(delayed(run_fold)(frame, [c for c in cyc if c != held], [held], cols, target, models))
                keys.append(("loco", sname, target, held))
    out = Parallel(n_jobs=20)(jobs)
    scores = {}
    for (proto, sname, target, held), (ix, res) in zip(keys, out):
        for m, s in res.items():
            scores.setdefault((proto, f"{m}|{sname}|{target[2:]}"), np.full(len(frame), np.nan))[ix] = s
    v2 = pd.read_csv(D.REPO / "model_v2_GRU/artifacts/predictions.csv.gz")
    assert v2.anchor_id.equals(frame.anchor_id)
    for m in ("gru_16", "hgb_depth2", "tcn_16"):
        scores[("frozen", f"v2:{m}")] = v2[m].to_numpy()
    rows = []
    split = {"train": TR, "validation": VA, "test": TE}
    for (proto, name), s in scores.items():
        if proto == "frozen":
            va = frame[frame.cycle_id.isin(VA)]
            sel = EV.select_threshold(va.assign(split="validation"), s[va.index], events[events.cycle_id.isin(VA)].assign(split="validation"),
                                      hours(prod, VA), grid=GRID)
            row = dict(proto=proto, model=name, th=sel["threshold"])
            for sp, cs in split.items():
                q = frame[frame.cycle_id.isin(cs)]
                m, _, _ = EV.evaluate(q, s[q.index], sel["threshold"], events[events.cycle_id.isin(cs)], hours(prod, cs))
                row.update({f"{sp}_hit": f"{m['hit20']}/{m['total_events']}", f"{sp}_fa": m["false_alarms"], f"{sp}_late": m["late"]})
            row["test_fa_h"] = m["false_alarms_per_hour"]
            rows.append(row)
        else:
            c = curve(frame, s, events, hours(prod, cyc))
            b1, b05 = best(c, .1), best(c, .05)
            from sklearn.metrics import average_precision_score
            y = frame["y_eol_60m"]
            ok = y.notna() & np.isfinite(s)
            rows.append(dict(proto=proto, model=name, hit_at_0p1=b1["hit"], fa_at_0p1=b1["fa"], th_0p1=b1["th"], late_0p1=b1["late"],
                             hit_at_0p05=b05["hit"], fa_at_0p05=b05["fa"], ap60=average_precision_score(y[ok], s[ok])))
    r = pd.DataFrame(rows)
    out_dir = D.ROOT / "research" / "out"
    out_dir.mkdir(exist_ok=True)
    r.to_csv(out_dir / f"alarm_{tag}.csv", index=False)
    np.savez_compressed(out_dir / f"alarm_{tag}_scores.npz", **{f"{p}@{n}": s.astype("float32") for (p, n), s in scores.items()})
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
    lo = r[r.proto == "loco"].drop(columns=[c for c in r.columns if c.startswith(("train", "validation", "test")) or c == "th"])
    print(lo.sort_values(["hit_at_0p1", "hit_at_0p05", "ap60"], ascending=False).head(45).round(3).to_string(index=False))
    fz = r[r.proto == "frozen"][["model", "th", "train_hit", "train_fa", "validation_hit", "validation_fa", "test_hit", "test_fa", "test_late"]]
    print(fz.sort_values(["test_hit", "validation_hit", "test_fa"], ascending=[False, False, True]).head(45).round(3).to_string(index=False))


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["v1"]))
