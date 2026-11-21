"""Second remaining-life round: end-stage feature subset, sequence network,
neighbours, person-period boosting, prior blending and causal smoothing."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))
import data as D
import rul_bench as R

warnings.filterwarnings("ignore")
END = ["vsd_mean_5m", "vsd_mean_30m", "vsd_mean_3h", "vsd60", "vsd_ratio", "v_std_30m", "d_peak_30m", "d_peak_3h", "d_5m_3h",
       "d_30m_base", "v_slope_3h", "v_low53_30m", "margin51_30m", "margin51_q10", "cool_loss_kw_d_base", "therm_eff_d_base",
       "power_d_base", "power_std_30m"]
SEQ = ["v60", "vsd60", "vsd_mean_30m", "d_peak_30m", "d_30m_base", "margin51_30m", "cool_loss_kw_d_base", "therm_eff_d_base",
       "power_d_base", "v_slope_30m", "since_restart_h", "age_h"]
STEPS, STRIDE = 37, 60  # 6 h of processing history, one point per 10 processing minutes


def sequences(frame, cols, scaler):
    """Per origin: the last STEPS points STRIDE origins apart inside the same cycle (zero-padded with a mask)."""
    z = scaler(frame[cols].to_numpy("float64"))
    out = np.zeros((len(frame), STEPS, len(cols) + 1), "float32")
    start = 0
    for _, q in frame.groupby("cycle_id", sort=False):
        n = len(q)
        pos = np.arange(n)[:, None] - np.arange(STEPS - 1, -1, -1)[None, :] * STRIDE
        ok = pos >= 0
        out[start:start + n, :, :-1] = np.where(ok[:, :, None], z[start + np.maximum(pos, 0)], 0)
        out[start:start + n, :, -1] = ok
        start += n
    return out


def fit_gru_hazard(x, rul, event, hidden=16, epochs=30, seed=0):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    y, w = R.hazard_targets(rul, event)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.g = nn.GRU(x.shape[2], hidden, batch_first=True)
            self.d = nn.Dropout(.2)
            self.h = nn.Linear(hidden, y.shape[1])

        def forward(self, a):
            return self.h(self.d(self.g(a)[0][:, -1]))
    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
    xt, yt, wt = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(w)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        for ix in torch.randperm(len(xt), generator=g).split(256):
            loss = (nn.functional.binary_cross_entropy_with_logits(net(xt[ix]), yt[ix], reduction="none") * wt[ix]).sum() / wt[ix].sum()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


def lgbm_hazard(xtr, rul, event, xte, seed=0):
    """Person-period expansion: one row per (origin, bin still at risk), bin index as a feature."""
    import lightgbm as lgb
    y, w = R.hazard_targets(rul, event)
    rows, bins = np.nonzero(w)
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=.03, num_leaves=8, min_child_samples=200, subsample=.7, subsample_freq=1,
                           colsample_bytree=.7, reg_lambda=10, verbose=-1, random_state=seed, n_jobs=1)
    m.fit(np.column_stack([xtr[rows], bins]), y[rows, bins])
    k = y.shape[1]
    hz = np.column_stack([m.predict_proba(np.column_stack([xte, np.full(len(xte), b)]))[:, 1] for b in range(k)])
    return hz


def smooth(pred, alpha):
    """Causal countdown filter in log space: the estimate ticks down with processing time between updates."""
    out = np.empty_like(pred)
    state = np.log(np.maximum(pred[0], .02))
    for i in range(len(pred)):
        if i:
            state = np.log(np.maximum(np.exp(state) - D.STEP_H, .02))
            state = state + alpha * (np.log(np.maximum(pred[i], .02)) - state)
        out[i] = np.exp(state)
    return out


def fold(frame, held, seed=0):
    import torch
    torch.set_num_threads(1)
    full_tr = frame[(frame.cycle_id != held)]
    tr_mask = full_tr.rul_h.notna().to_numpy()
    te_full = frame[frame.cycle_id == held]
    te_mask = te_full.rul_h.notna().to_numpy()
    tr = full_tr[tr_mask].iloc[::6]
    te = te_full[te_mask]
    ev = tr.event.to_numpy().astype(bool)
    rul = tr.rul_h.to_numpy()
    cyc = full_tr.groupby("cycle_id").agg(life=("life_h", "first"), ev=("event", "first"))
    age = te.age_h.to_numpy()
    k, lam = R.weibull_fit(cyc.life, cyc.ev)
    prior = np.column_stack([R.weibull_residual(age, k, lam, q) for q in R.Q])
    res = {"weibull_age": prior, "mean_life": np.column_stack([np.maximum(cyc[cyc.ev == 1].life.mean() - age, .05)] * 3)}
    hz_store = {}
    for sname, cols in {"end": END, "end+age": END + ["age_h"]}.items():
        xtr, xte = tr[cols].to_numpy("float64"), te[cols].to_numpy("float64")
        sc = R.Scaler(xtr)
        ztr, zte = sc(xtr), sc(xte)
        hz = R.predict_hazard(R.fit_hazard_mlp(ztr, rul, ev, hidden=(), epochs=40, seed=seed, wd=1e-2), zte)
        res[f"hazard_lin|{sname}"] = R.survival_quantiles(hz); hz_store[f"hazard_lin|{sname}"] = hz
        hz = np.mean([R.predict_hazard(R.fit_hazard_mlp(ztr, rul, ev, hidden=(16,), epochs=40, seed=seed + s, wd=3e-2, drop=.3), zte) for s in range(3)], 0)
        res[f"hazard_mlp16|{sname}"] = R.survival_quantiles(hz); hz_store[f"hazard_mlp16|{sname}"] = hz
        hz = lgbm_hazard(xtr, rul, ev, xte, seed)
        res[f"lgbm_hazard|{sname}"] = R.survival_quantiles(hz); hz_store[f"lgbm_hazard|{sname}"] = hz
        from sklearn.ensemble import HistGradientBoostingRegressor as H
        ps = [np.exp(H(loss="quantile", quantile=q, max_depth=2, max_iter=120, learning_rate=.05, min_samples_leaf=150,
                       l2_regularization=2.0, random_state=seed).fit(xtr[ev], np.log(rul[ev])).predict(xte)) for q in R.Q]
        res[f"hgb_q|{sname}"] = np.sort(np.column_stack(ps), 1)
        from sklearn.neighbors import NearestNeighbors
        nn_ = NearestNeighbors(n_neighbors=150).fit(ztr[ev])
        _, ix = nn_.kneighbors(zte)
        res[f"knn|{sname}"] = np.quantile(rul[ev][ix], R.Q, axis=1).T
    # sequence network on a compact channel set
    sc = R.Scaler(tr[SEQ].to_numpy("float64"))
    xs_tr = sequences(full_tr, SEQ, sc)[tr_mask][::6]
    xs_te = sequences(te_full, SEQ, sc)[te_mask]
    import torch as T
    hs = []
    for s in range(3):
        net = fit_gru_hazard(xs_tr, rul, ev, seed=seed + s)
        with T.inference_mode():
            hs.append(T.sigmoid(net(T.from_numpy(xs_te))).numpy().astype("float64"))
    hz = np.mean(hs, 0)
    res["gru_hazard|seq"] = R.survival_quantiles(hz); hz_store["gru_hazard|seq"] = hz
    # blends with the age prior and causal smoothing of the better candidates
    for name in ["hazard_lin|end", "hazard_mlp16|end", "lgbm_hazard|end", "gru_hazard|seq", "hgb_q|end"]:
        p = res[name]
        res[f"{name}+prior"] = np.exp(.5 * np.log(np.maximum(p, .02)) + .5 * np.log(np.maximum(prior, .02)))
        for a in (.05, .01, .003):
            res[f"{name}~s{a}"] = np.column_stack([smooth(p[:, j], a) for j in range(3)])
    return held, te.rul_h.to_numpy(), age, res, hz_store


def main(tag="v2"):
    frame = D.build()
    cycles = list(frame.cycle_id.unique())
    out = Parallel(n_jobs=14)(delayed(fold)(frame, c) for c in cycles)
    event = frame.groupby("cycle_id").event.first()
    rows, store = [], {}
    for held, truth, age, res, hz in out:
        for name, p in res.items():
            rows.append(dict(model=name, cycle=held, event=int(event[held]), **R.score(truth, p, event[held])))
            store[f"{name}@{held}"] = np.column_stack([age, truth, p]).astype("float32")
        for name, h in hz.items():
            store[f"HZ:{name}@{held}"] = h.astype("float32")
    r = pd.DataFrame(rows)
    out_dir = D.ROOT / "research" / "out"
    r.to_csv(out_dir / f"rul_{tag}_cycles.csv", index=False)
    np.savez_compressed(out_dir / f"rul_{tag}_preds.npz", **store)
    ev = r[r.event == 1]
    s = ev.groupby("model")[["mae", "mae_last5h", "mae_last2h", "mae_last1h", "bias", "cover80", "rel_last5h"]].mean()
    s["mae_median_cycle"] = ev.groupby("model").mae.median()
    s["censored_under"] = r[r.event == 0].groupby("model").under_bound.mean()
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(s.sort_values("mae").round(3).to_string())
    s.to_csv(out_dir / f"rul_{tag}_summary.csv")


if __name__ == "__main__":
    main()
