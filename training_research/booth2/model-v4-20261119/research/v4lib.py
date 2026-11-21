"""Booth2 v4 research library: causal features, multi-horizon survival targets,
leave-one-electrode-cycle-out evaluation.

Life is measured in processing hours (origins x 10 s), as in v3. A model here
predicts F(h) = P(end of life within h processing hours) at fixed horizons; the
remaining-life quantiles and the warning score are read off that curve.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
V3 = HERE.parents[1] / "model-v3-20261118"
sys.path.insert(0, str(V3 / "research"))
import data as D  # noqa: E402

_SRC = str(D.REPO / "model_v2_GRU" / "src")
sys.path.insert(0, _SRC)
import evaluation as EV  # noqa: E402  (model_v2's frozen alarm protocol)
sys.path.remove(_SRC)  # that folder also holds an inference.py, which must not shadow the panel's `inference` package

warnings.filterwarnings("ignore")
STEP_H = D.STEP_H
H = np.array([1 / 3, 2 / 3, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24.0])  # horizons, processing hours
CAP = 40.0
Q = (.1, .5, .9)
CACHE = HERE / "cache"
OUT = HERE / "out"

END = ["vsd_mean_5m", "vsd_mean_30m", "vsd_mean_3h", "vsd60", "vsd_ratio", "v_std_5m", "v_std_30m", "d_peak_30m", "d_peak_3h",
       "d_5m_3h", "d_30m_base", "v_slope_30m", "v_slope_3h", "v_low53_30m", "margin51_30m", "margin51_q10", "v_min_30m",
       "cool_loss_kw_d_base", "cool_dT_d_base", "therm_eff_d_base", "power_d_base", "power_std_30m", "therm_eff_slope_3h"]
DAMAGE = ["cum_low53_h", "cum_low52_h", "cum_vsd_hi_h", "cum_vsd_rel_h", "vsd_hi_30m", "vsd_hi_3h", "vsd_hi_10h", "low53_10h",
          "vsd_q90_30m", "vsd_q90_3h", "d_5m_base", "tte51_h", "since_restart_h"]
PRODUCT = ["pm_mean5", "pm_min5", "pm_min20", "pm_slope20", "pm_d_base", "p_low52_cnt20", "psd_mean5", "pmin_min5",
           "fails_cum", "fails_50"]
PLUS = END + DAMAGE + PRODUCT
# Direction in which the risk of an early end may only grow (+1) or only shrink (-1); 0 leaves the feature free.
MONO = {**{k: 1 for k in ["vsd_mean_5m", "vsd_mean_30m", "vsd_mean_3h", "vsd60", "vsd_ratio", "v_std_5m", "v_std_30m", "d_peak_30m",
                           "d_peak_3h", "d_5m_3h", "d_30m_base", "v_low53_30m", "cum_low53_h", "cum_low52_h", "cum_vsd_hi_h",
                           "cum_vsd_rel_h", "vsd_hi_30m", "vsd_hi_3h", "vsd_hi_10h", "low53_10h", "vsd_q90_30m", "vsd_q90_3h",
                           "d_5m_base", "pm_d_base", "p_low52_cnt20", "psd_mean5", "fails_cum", "fails_50"]},
        **{k: -1 for k in ["v_slope_30m", "v_slope_3h", "margin51_30m", "margin51_q10", "v_min_30m", "tte51_h", "pm_mean5",
                            "pm_min5", "pm_min20", "pm_slope20", "pmin_min5"]}}


def extra_features(frame, products):
    """Cumulative-damage, exposure and completed-product features; all causal inside one cycle."""
    parts = []
    products = products[products.supervised_product_eligible.eq(1)].copy()
    products["avail"] = pd.to_datetime(products.label_available_at)
    for cid, q in frame.groupby("cycle_id", sort=False):
        f = pd.DataFrame(index=q.index)
        v, sd = q.v60, q.vsd60
        base_sd = sd.expanding(min_periods=180).median()
        hi = (sd > 0.4).astype(float)
        f["cum_low53_h"] = (v < 53).cumsum() * STEP_H
        f["cum_low52_h"] = (v < 52).cumsum() * STEP_H
        f["cum_vsd_hi_h"] = hi.cumsum() * STEP_H
        f["cum_vsd_rel_h"] = (sd > 1.6 * base_sd).astype(float).cumsum() * STEP_H
        for name, k in (("30m", 180), ("3h", 1080), ("10h", 3600)):
            f[f"vsd_hi_{name}"] = hi.rolling(k, min_periods=max(3, k // 6)).mean()
        f["low53_10h"] = (v < 53).astype(float).rolling(3600, min_periods=600).mean()
        f["vsd_q90_30m"] = sd.rolling(180, min_periods=30).quantile(.9)
        f["vsd_q90_3h"] = sd.rolling(1080, min_periods=180).quantile(.9)
        f["d_5m_base"] = q.v_base - q.v_med_5m
        f["tte51_h"] = (q.margin51_30m / np.maximum(-q.v_slope_3h, 0.02)).clip(0, 40)
        p = products[products.cycle_id.eq(cid)].sort_values("avail")
        pm = p.voltage_mean.reset_index(drop=True)
        fail = p.voltage_fail.fillna(0).reset_index(drop=True)
        x = pd.Series(np.arange(len(pm), dtype="float64"))
        table = pd.DataFrame(dict(
            pm_mean5=pm.rolling(5, min_periods=2).mean(), pm_min5=pm.rolling(5, min_periods=2).min(),
            pm_min20=pm.rolling(20, min_periods=5).min(),
            pm_slope20=x.rolling(20, min_periods=8).cov(pm) / x.rolling(20, min_periods=8).var(),
            p_low52_cnt20=(pm < 52).astype(float).rolling(20, min_periods=1).sum(),
            psd_mean5=p.voltage_std.reset_index(drop=True).rolling(5, min_periods=2).mean(),
            pmin_min5=p.voltage_min.reset_index(drop=True).rolling(5, min_periods=2).min(),
            fails_cum=fail.cumsum(), fails_50=fail.rolling(50, min_periods=1).sum()))
        table["pm_d_base"] = pm.expanding(min_periods=30).median() - table.pm_mean5
        done = np.searchsorted(p.avail.to_numpy(), q.timestamp.to_numpy(), side="right")  # products whose label exists now
        rows = table.reindex(done - 1).to_numpy() if len(table) else np.full((len(q), table.shape[1]), np.nan)
        rows[done == 0] = np.nan
        f[list(table.columns)] = rows
        f.loc[done == 0, ["fails_cum", "fails_50", "p_low52_cnt20"]] = 0.0
        parts.append(f)
    return pd.concat(parts).loc[frame.index]


def load(force=False):
    path = CACHE / "frame_v4.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    CACHE.mkdir(parents=True, exist_ok=True)
    frame = D.build().reset_index(drop=True)
    products = pd.read_csv(D.REPO / "model_v2_GRU/data/products.csv.gz", low_memory=False)
    frame = pd.concat([frame, extra_features(frame, products)], axis=1)
    frame.to_parquet(path)
    return frame


def tables():
    events = pd.read_csv(D.REPO / "model_v2_GRU/data/events.csv", parse_dates=["time"])
    products = pd.read_csv(D.REPO / "model_v2_GRU/data/products.csv.gz", low_memory=False)
    return events, products


def hours_of(products, cycles):
    q = products[products.cycle_id.isin(cycles) & products.supervised_product_eligible.eq(1)]
    return float(q.spray_samples.sum() / 3600)


# ---------- targets and curves ----------
def cdf_targets(rul, event, horizons=H):
    """y[i, j] = 1 when the end falls within horizon j; known[i, j] = 0 where censoring hides the answer."""
    rul, event = np.asarray(rul, float), np.asarray(event, bool)
    within = rul[:, None] <= horizons[None, :]
    return (within & event[:, None]).astype("float32"), (event[:, None] | ~within).astype("float32")


def monotone(F):
    return np.clip(np.maximum.accumulate(F, axis=1), 0, 1)


def quantiles(F, qs=Q):
    """Remaining-life quantiles of a CDF given at the horizons; F(0) = 0 and F(CAP) = 1, linear in between."""
    edges = np.r_[0.0, H, CAP]
    full = np.column_stack([np.zeros(len(F)), monotone(F), np.ones(len(F))])
    rows = np.arange(len(F))
    out = np.zeros((len(F), len(qs)))
    for j, q in enumerate(qs):
        k = np.clip((full < q).sum(1) - 1, 0, len(edges) - 2)
        lo, hi = full[rows, k], full[rows, k + 1]
        out[:, j] = edges[k] + np.clip((q - lo) / np.maximum(hi - lo, 1e-9), 0, 1) * (edges[k + 1] - edges[k])
    return out


def ema_rows(x, alpha, groups):
    """Causal exponential smoothing along time inside each cycle, started at the first value."""
    out = np.array(x, dtype="float64", copy=True)
    for g in pd.unique(groups):
        ix = np.flatnonzero(groups == g)
        for i in range(1, len(ix)):
            out[ix[i]] = out[ix[i - 1]] + alpha * (out[ix[i]] - out[ix[i - 1]])
    return out


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


def smooth_cdf(F, alpha, groups):
    return monotone(expit(ema_rows(logit(F), alpha, np.asarray(groups)))) if alpha < 1 else monotone(F)


# ---------- scoring ----------
def life_scores(frame, quant):
    """Per observed cycle: error of the median and band coverage; censored cycles: median below the known lower bound."""
    rows = []
    for cycle, q in frame.groupby("cycle_id", sort=False):
        ok = q.rul_h.notna().to_numpy()
        truth, p = q.rul_h.to_numpy()[ok], quant[q.index][ok]
        row = dict(cycle=cycle, event=int(q.event.iloc[0]))
        if q.event.iloc[0]:
            err, d = np.abs(p[:, 1] - truth), truth[:, None] - p
            row.update(mae=err.mean(), **{f"mae_{n}h": (err[truth <= n].mean() if (truth <= n).any() else np.nan) for n in (8, 4, 2, 1)},
                       bias_2h=(p[:, 1] - truth)[truth <= 2].mean(), cover=((p[:, 0] <= truth) & (truth <= p[:, 2])).mean(),
                       width=(p[:, 2] - p[:, 0]).mean(), pinball=np.maximum(np.array(Q) * d, (np.array(Q) - 1) * d).mean(0).sum())
        else:
            row.update(under=(p[:, 1] < truth).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def life_summary(frame, quant):
    r = life_scores(frame, quant)
    s = r[r.event == 1][["mae", "mae_8h", "mae_4h", "mae_2h", "mae_1h", "bias_2h", "cover", "width", "pinball"]].mean()
    s["cens_under"] = r[r.event == 0].under.mean()
    return s


GRID = np.r_[np.arange(.05, 1, .05), .97, .98, .99, .995, .01, .02, .03]


def frozen_curve(frame, score, events, hours, grid=GRID):
    """model_v2's protocol: an alarm counts only 20-60 calendar minutes before the end."""
    rows = []
    for th in np.unique(grid):
        m, _, _ = EV.evaluate(frame, score, float(th), events, hours)
        rows.append(dict(th=round(float(th), 3), hit=m["hit20"], late=m["late"], missed=m["missed"], fa=m["false_alarms"],
                         fa_h=m["false_alarms_per_hour"], unknown=m["unknown_alarms"], post=m["post_event_alarms"]))
    return pd.DataFrame(rows)


def best(curve, budget, key="fa_h"):
    ok = curve[curve[key] <= budget]
    if not len(ok):
        return dict(hit=0, fa=0, th=np.nan)
    r = ok.sort_values(["hit", "fa", "th"], ascending=[False, True, False]).iloc[0]
    return {k: (float(r[k]) if k == "th" else int(r[k])) for k in ("hit", "fa", "th")}


def reminder_eval(frame, score, threshold, events, window_h=4.0, lead_min=20.0):
    """Change-the-gun reminder: an alarm is right when the end really comes within `window_h`
    processing hours, and counts for the event when it is at least `lead_min` calendar minutes early.
    Alarms come from the same state machine as model_v2 (2 confirmations, 10 min release, 30 min cooldown)."""
    alarms = EV.build_alarms(frame, score, threshold)
    eol = events[events.is_eol.eq(1)].set_index("cycle_id").time
    ts, rul, ev, cyc = frame.timestamp.to_numpy(), frame.rul_h.to_numpy(), frame.event.to_numpy(), frame.cycle_id.to_numpy()
    hit, fa, unknown, post, leads = {}, 0, 0, 0, {}
    for a in alarms:
        i = a["index"]
        if ev[i] and np.isfinite(rul[i]):
            if rul[i] <= window_h:
                lead = (eol[cyc[i]] - pd.Timestamp(ts[i])).total_seconds() / 60
                if lead >= lead_min and cyc[i] not in hit:
                    hit[cyc[i]], leads[cyc[i]] = True, rul[i]
            else:
                fa += 1
        elif ev[i]:
            post += 1
        elif np.isfinite(rul[i]) and rul[i] > window_h:
            fa += 1
        else:
            unknown += 1
    hours = float(frame.rul_h.notna().sum() * STEP_H)
    return dict(hit=len(hit), fa=fa, unknown=unknown, post=post, fa_100h=100 * fa / hours,
                lead_h=float(np.median(list(leads.values()))) if leads else np.nan, alarms=len(alarms))


def reminder_curve(frame, score, events, window_h=4.0, grid=GRID):
    return pd.DataFrame([dict(th=round(float(t), 3), **reminder_eval(frame, score, float(t), events, window_h)) for t in np.unique(grid)])
