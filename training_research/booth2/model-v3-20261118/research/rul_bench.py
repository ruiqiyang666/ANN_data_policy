"""Remaining-life benchmark, leave-one-electrode-cycle-out.

Every model is fitted on 13 cycles and scored on the origins of the held-out
cycle. Only the 10 cycles with an observed end of life can be scored; censored
cycles still contribute to the censoring-aware fits. Life is processing hours.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
import data as D

warnings.filterwarnings("ignore")
EDGES = np.array([0, 1 / 3, 2 / 3, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 40.0])  # hazard bins, processing hours
Q = (.1, .5, .9)
SENSOR_SMALL = ["v60", "vsd60", "v_med_5m", "v_med_30m", "v_med_3h", "v_q10_30m", "v_std_30m", "vsd_mean_30m", "vsd_mean_3h",
                "v_low53_30m", "v_low52_30m", "v_low53_3h", "v_slope_30m", "v_slope_3h", "d_5m_3h", "d_30m_base",
                "d_peak_30m", "d_peak_3h", "vsd_ratio", "margin51_30m", "margin51_q10", "therm_eff_med_30m",
                "therm_eff_d_base", "cool_loss_kw_med_30m", "power_med_30m", "power_d_base"]


# ---------- age-only Weibull ----------
def weibull_fit(life, event):
    life, event = np.asarray(life, float), np.asarray(event, bool)

    def nll(p):
        k, lam = np.exp(p)
        z = (life / lam) ** k
        return -(event * (np.log(k / lam) + (k - 1) * np.log(life / lam))).sum() + z.sum()
    r = minimize(nll, [np.log(1.5), np.log(life.mean())], method="Nelder-Mead")
    return np.exp(r.x)


def weibull_residual(age, k, lam, q):
    """Residual-life quantile given survival to `age`: S(age+t)/S(age) = 1-q."""
    return lam * ((age / lam) ** k - np.log(1 - q)) ** (1 / k) - age


# ---------- discrete-time hazard network (censoring-aware) ----------
def hazard_targets(rul, event):
    """Per bin: 1 at the bin the end falls in, 0 before it, masked afterwards / after censoring."""
    k = np.clip(np.searchsorted(EDGES, rul, side="right") - 1, 0, len(EDGES) - 2)
    n, m = len(rul), len(EDGES) - 1
    y, w = np.zeros((n, m), "float32"), np.zeros((n, m), "float32")
    for i in range(n):
        w[i, :k[i]] = 1
        if event[i]:
            y[i, k[i]], w[i, k[i]] = 1, 1
    return y, w


def survival_quantiles(hazard, qs=Q):
    """Hazards per bin -> residual-life quantiles by linear interpolation inside a bin."""
    s = np.cumprod(1 - hazard, axis=1)
    s0 = np.concatenate([np.ones((len(s), 1)), s], axis=1)
    out = np.zeros((len(s), len(qs)))
    for j, q in enumerate(qs):
        target = 1 - q
        k = np.minimum((s0[:, 1:] > target).sum(1), s.shape[1] - 1)
        hi, lo = s0[np.arange(len(s)), k], s0[np.arange(len(s)), k + 1]
        frac = np.clip((hi - target) / np.maximum(hi - lo, 1e-9), 0, 1)
        out[:, j] = EDGES[k] + frac * (EDGES[k + 1] - EDGES[k])
    return out


def fit_hazard_mlp(x, rul, event, hidden=(32, 16), epochs=60, seed=0, wd=1e-3, drop=.2):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    y, w = hazard_targets(rul, event)
    layers, d = [], x.shape[1]
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop)]
        d = h
    net = nn.Sequential(*layers, nn.Linear(d, y.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=wd)
    xt, yt, wt = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(w)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        for ix in torch.randperm(len(xt), generator=g).split(256):
            loss = (nn.functional.binary_cross_entropy_with_logits(net(xt[ix]), yt[ix], reduction="none") * wt[ix]).sum() / wt[ix].sum()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


def predict_hazard(net, x):
    import torch
    with torch.inference_mode():
        return torch.sigmoid(net(torch.from_numpy(x))).numpy().astype("float64")


class Scaler:
    def __init__(self, x):
        self.med = np.nanmedian(x, 0)
        q = np.nanpercentile(x, [25, 75], 0)
        self.scale = np.maximum(q[1] - q[0], 1e-3 + 0.05 * np.abs(self.med))

    def __call__(self, x):
        z = (np.where(np.isfinite(x), x, self.med) - self.med) / self.scale
        return np.clip(z, -8, 8).astype("float32")


# ---------- one fold ----------
def fold(frame, held, sets, models, seed=0):
    import torch
    torch.set_num_threads(1)
    tr = frame[(frame.cycle_id != held) & frame.rul_h.notna()]
    tr = tr.iloc[::6]  # one origin per processing minute
    te = frame[(frame.cycle_id == held) & frame.rul_h.notna()]
    cyc = frame[frame.cycle_id != held].groupby("cycle_id").agg(life=("life_h", "first"), ev=("event", "first"), lc=("left_censored", "first"))
    res = {}
    age = te.age_h.to_numpy()
    obs = cyc[cyc.ev == 1]
    if "mean_life" in models:
        m = obs.life.mean()
        res["mean_life"] = np.column_stack([np.maximum(m - age, .05)] * 3)
    if "weibull_age" in models:
        k, lam = weibull_fit(cyc.life, cyc.ev)
        res["weibull_age"] = np.column_stack([weibull_residual(age, k, lam, q) for q in Q])
    ev = tr.event.to_numpy().astype(bool)
    ylog = np.log(tr.rul_h.to_numpy())
    for sname, cols in sets.items():
        xtr, xte = tr[cols].to_numpy("float64"), te[cols].to_numpy("float64")
        sc = Scaler(xtr)
        ztr, zte = sc(xtr), sc(xte)
        if "ridge" in models:
            from sklearn.linear_model import Ridge
            m = Ridge(alpha=30).fit(ztr[ev], ylog[ev])
            p = np.exp(m.predict(zte))
            res[f"ridge|{sname}"] = np.column_stack([p, p, p])
        if "hgb" in models:
            from sklearn.ensemble import HistGradientBoostingRegressor as H
            ps = [np.exp(H(loss="quantile", quantile=q, max_depth=3, max_iter=150, learning_rate=.05, min_samples_leaf=80,
                           l2_regularization=1.0, random_state=seed).fit(xtr[ev], ylog[ev]).predict(xte)) for q in Q]
            res[f"hgb_q|{sname}"] = np.sort(np.column_stack(ps), 1)
        if "lgbm" in models:
            import lightgbm as lgb
            ps = [np.exp(lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=200, learning_rate=.04, num_leaves=8,
                                           min_child_samples=100, subsample=.8, subsample_freq=1, colsample_bytree=.7,
                                           reg_lambda=5, verbose=-1, random_state=seed, n_jobs=1).fit(xtr[ev], ylog[ev]).predict(xte)) for q in Q]
            res[f"lgbm_q|{sname}"] = np.sort(np.column_stack(ps), 1)
        if "xgb_aft" in models:
            import xgboost as xgb
            d = xgb.DMatrix(xtr)
            lo = tr.rul_h.to_numpy()
            d.set_float_info("label_lower_bound", lo)
            d.set_float_info("label_upper_bound", np.where(ev, lo, np.inf))
            sigma = 0.6
            b = xgb.train(dict(objective="survival:aft", aft_loss_distribution="normal", aft_loss_distribution_scale=sigma,
                               max_depth=3, eta=.05, subsample=.8, colsample_bytree=.7, min_child_weight=50, reg_lambda=5,
                               nthread=1, seed=seed), d, 200)
            p = b.predict(xgb.DMatrix(xte))
            res[f"xgb_aft|{sname}"] = np.column_stack([p * np.exp(sigma * z) for z in (-1.2816, 0, 1.2816)])
        if "hazard_mlp" in models:
            hs = [predict_hazard(fit_hazard_mlp(ztr, tr.rul_h.to_numpy(), ev, seed=seed + s), zte) for s in range(3)]
            res[f"hazard_mlp|{sname}"] = survival_quantiles(np.mean(hs, 0))
        if "hazard_lin" in models:
            hz = predict_hazard(fit_hazard_mlp(ztr, tr.rul_h.to_numpy(), ev, hidden=(), epochs=40, seed=seed, wd=1e-2), zte)
            res[f"hazard_lin|{sname}"] = survival_quantiles(hz)
    truth = te.rul_h.to_numpy()
    return held, truth, age, res


def score(truth, pred, event):
    """Per-cycle errors of the median; coverage of the 10–90 % band."""
    med = pred[:, 1]
    out = {}
    if event:
        err = np.abs(med - truth)
        out["mae"] = err.mean()
        for name, lim in (("mae_last5h", 5), ("mae_last2h", 2), ("mae_last1h", 1)):
            m = truth <= lim
            out[name] = err[m].mean() if m.any() else np.nan
        out["bias"] = (med - truth).mean()
        out["cover80"] = ((pred[:, 0] <= truth) & (truth <= pred[:, 2])).mean()
        out["rel_last5h"] = (err[truth <= 5] / truth[truth <= 5]).mean()
    else:  # censored: the truth is only a lower bound
        out["under_bound"] = (med < truth).mean()
    return out


def main(tag="v1", models=None, sets=None, seed=0):
    frame = D.build()
    allf = D.feature_names(frame)
    sets = sets or {"age": ["age_h"], "sensor": [c for c in SENSOR_SMALL], "sensor+age": SENSOR_SMALL + ["age_h", "restarts"],
                    "all": allf}
    models = models or ["mean_life", "weibull_age", "ridge", "hgb", "lgbm", "xgb_aft", "hazard_mlp", "hazard_lin"]
    cycles = list(frame.cycle_id.unique())
    out = Parallel(n_jobs=14)(delayed(fold)(frame, c, sets, models, seed) for c in cycles)
    event = frame.groupby("cycle_id").event.first()
    rows, preds = [], {}
    for held, truth, age, res in out:
        for name, p in res.items():
            rows.append(dict(model=name, cycle=held, event=int(event[held]), **score(truth, p, event[held])))
            preds[(name, held)] = np.column_stack([age, truth, p]).astype("float32")
    r = pd.DataFrame(rows)
    out_dir = D.ROOT / "research" / "out"
    out_dir.mkdir(exist_ok=True)
    r.to_csv(out_dir / f"rul_{tag}_cycles.csv", index=False)
    np.savez_compressed(out_dir / f"rul_{tag}_preds.npz", **{f"{k[0]}@{k[1]}": v for k, v in preds.items()})
    ev = r[r.event == 1]
    summary = ev.groupby("model")[["mae", "mae_last5h", "mae_last2h", "mae_last1h", "bias", "cover80", "rel_last5h"]].mean()
    summary["mae_median_cycle"] = ev.groupby("model").mae.median()
    summary["censored_under"] = r[r.event == 0].groupby("model").under_bound.mean()
    summary = summary.sort_values("mae")
    pd.set_option("display.width", 250)
    print(summary.round(3).to_string())
    summary.to_csv(out_dir / f"rul_{tag}_summary.csv")
    return r


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["v1"]))
