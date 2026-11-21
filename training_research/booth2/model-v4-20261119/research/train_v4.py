"""Booth2 model v4: three survival models from five learners, nested leave-one-cycle-out.

usage: python train_v4.py BUNDLE_DIR [--refit]

Design is fixed before this script runs (see README): the 23 causal end-stage features of v3,
twelve horizons, five learners (three logistic layers with 3 / 5 / 23 inputs, monotone boosted
trees, a GRU + features network), a per-horizon pool with the historical base rate, light
smoothing, a warning that lifts the first hour, conformalised quantile band.

Every electrode cycle is scored by learners that never saw it. Everything a model takes from
data beyond the learners' fits - pool weights, warning calibration, interval widening, alarm
threshold - comes from inner out-of-fold predictions of its 13 training cycles, so the reported
results include the cost of choosing them. The learners' predictions are computed by the panel's
own NumPy code (inference/survival.py); scikit-learn and PyTorch only fit.
"""
import hashlib
import json
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v4lib as L  # noqa: E402
import bench2 as B  # noqa: E402
import nets  # noqa: E402

D, EV = L.D, L.EV
APP = D.ROOT.parent / "depuy-theme-preview" / "simulation" / "pipeline"
sys.path.insert(0, str(APP))
from inference import survival as S  # noqa: E402  (the panel's own inference code)

warnings.filterwarnings("ignore")
FEATURES = L.END
LOG_VSD = len(FEATURES)  # index of the derived 24th input
col = FEATURES.index
SMALL = {"logit3": ([col("margin51_30m"), col("d_peak_30m"), LOG_VSD], 1.0),
         "logit5": ([col("margin51_30m"), col("d_peak_30m"), LOG_VSD, col("v_slope_30m"), col("vsd_ratio")], 1.0),
         "logit23": (list(range(len(FEATURES))), 0.01)}
LEARNERS = ["logit3", "logit5", "logit23", "hgb", "net"]
MODELS = {
    "life_tree": dict(members=["hgb"], warn=["hgb"], family="trees", architecture=[24, 600, 12],
                      name="电极寿命 · 提升树 v4", name_en="Electrode life · boosted trees v4", short="提升树 v4", short_en="Trees v4",
                      describe="单调梯度提升树：23 项特征 → 12 个时间点的到寿概率", describe_en="Monotone boosted trees: 23 features → end-of-life probability at 12 horizons"),
    "life_gru": dict(members=["net"], warn=["net"], family="neural", architecture=[9, 8, 31, 12],
                     name="电极寿命 · GRU 网络 v4", name_en="Electrode life · GRU network v4", short="GRU 网络 v4", short_en="GRU net v4",
                     describe="循环神经网络：近 2 小时序列 + 23 项特征 → 12 个时间点的到寿概率", describe_en="Recurrent network: last 2 h sequence + 23 features → end-of-life probability at 12 horizons"),
    "life_fusion": dict(members=LEARNERS, warn=["hgb", "net"], family="ensemble", architecture=[5, 12],
                        name="电极寿命 · 融合 v4", name_en="Electrode life · fusion v4", short="融合 v4", short_en="Fusion v4",
                        describe="五个模型按时间点加权，并与历史基准混合", describe_en="Five learners pooled per horizon with the historical base rate"),
}
TARGET = "y_eol_60m"
TREE_SEEDS, NET_SEEDS = (0, 1, 2), (0, 1)
BUDGET = 0.1  # false alarms per production hour allowed when a threshold is chosen (model_v2's protocol)
NOMINAL = 0.8
INNER = 4
# The two ends of life inside the recorded replay rows (10-27 to 11-01); instrument clock read as UTC seconds.
SEGMENTS = [dict(name="electrode_eol_20261101", label="11月1日 12:30–13:38 电极末期", start=1793536200, stop=1793540280),
            dict(name="electrode_eol_20261028", label="10月28日 07:30–08:32 电极末期", start=1793172600, stop=1793176320)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------- fitting ----------
def stats(x):
    med = np.nanmedian(x, 0)
    q = np.nanpercentile(x, [25, 75], 0)
    return med, np.maximum(q[1] - q[0], 1e-3 + .05 * np.abs(med))


def export_trees(models):
    """scikit-learn predictor nodes -> padded arrays the panel walks in NumPy (depth-3 binary trees)."""
    width = 15
    shape = (len(models), models[0].n_iter_, width)
    out = dict(feature=np.zeros(shape, "int16"), threshold=np.zeros(shape, "float64"), left=np.zeros(shape, "int16"),
               right=np.zeros(shape, "int16"), value=np.zeros(shape, "float64"), leaf=np.ones(shape, bool), missing_left=np.zeros(shape, bool))
    for m, model in enumerate(models):
        assert model.n_iter_ == shape[1]
        for t, (predictor,) in enumerate(model._predictors):
            nodes = predictor.nodes
            assert len(nodes) <= width and not nodes["is_categorical"].any()
            n = len(nodes)
            out["feature"][m, t, :n] = np.where(nodes["is_leaf"], 0, nodes["feature_idx"])
            out["threshold"][m, t, :n] = nodes["num_threshold"]
            out["left"][m, t, :n], out["right"][m, t, :n] = nodes["left"], nodes["right"]
            out["value"][m, t, :n] = nodes["value"]
            out["leaf"][m, t, :n] = nodes["is_leaf"].astype(bool)
            out["missing_left"][m, t, :n] = nodes["missing_go_to_left"].astype(bool)
    out["baseline"] = np.array([float(np.ravel(m._baseline_prediction)[0]) for m in models])
    out["depth"] = np.array(3)
    return out


def fit_members(frame, raw, cycles):
    """The five learners on the given cycles, in the structure the panel loads."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    import torch
    torch.set_num_threads(1)
    tr, w = B.train_rows(frame, cycles)  # one origin per processing minute; the last three hours complete, each row a sixth
    x23 = tr[FEATURES].to_numpy("float64")
    x24 = S.design(x23)
    y, known = L.cdf_targets(tr.rul_h.to_numpy(), tr.event.to_numpy().astype(bool))
    model, check = {}, {}
    for name, (cols, c) in SMALL.items():
        med, spread = stats(x24[:, cols])
        z = S.scale(x24[:, cols], med, spread)
        weight, bias = np.zeros((len(L.H), len(cols))), np.zeros(len(L.H))
        for j in range(len(L.H)):
            m = known[:, j] > 0
            if y[m, j].min() == y[m, j].max():  # every known answer is the same here: a smoothed base rate
                rate = (float((y[m, j] * w[m]).sum()) + .5) / (float(w[m].sum()) + 1)
                bias[j] = np.log(rate / (1 - rate))
                continue
            unit = LogisticRegression(C=c, max_iter=3000).fit(z[m], y[m, j], sample_weight=w[m])
            weight[j], bias[j] = unit.coef_[0], unit.intercept_[0]
        model[name] = dict(cols=np.array(cols), median=med, spread=spread, weight=weight, bias=bias)
    # monotone boosted trees: one row per (origin, horizon whose answer is known), the horizon as the last input
    rows, cols_ = np.nonzero(known)
    cst = [L.MONO.get(f, 0) for f in FEATURES]
    cdf = [HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=.05, min_samples_leaf=200, l2_regularization=2.0,
                                          max_features=.7, monotonic_cst=cst + [1], early_stopping=False, random_state=s)
           .fit(np.column_stack([x23[rows], L.H[cols_]]), y[rows, cols_], sample_weight=w[rows]) for s in TREE_SEEDS]
    q = frame[frame.cycle_id.isin(cycles) & frame[TARGET].notna()]
    warn = [HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=.05, min_samples_leaf=100, l2_regularization=2.0,
                                           max_features=.7, class_weight="balanced", monotonic_cst=cst, early_stopping=False, random_state=s)
            .fit(q[FEATURES].to_numpy("float64"), q[TARGET].to_numpy().astype(int)) for s in TREE_SEEDS]
    model["hgb"] = dict(cdf=export_trees(cdf), warn=export_trees(warn))
    sample = x23[::41]
    check["hgb_cdf"] = np.mean([m.predict_proba(np.column_stack([sample, np.full(len(sample), L.H[2])]))[:, 1] for m in cdf], 0)
    check["hgb_warn"] = np.mean([m.predict_proba(sample)[:, 1] for m in warn], 0)
    # GRU + features network: twelve horizons and the warning
    med, spread = stats(x23)
    channels = raw.loc[tr.index].to_numpy("float64")
    smed, sspread = stats(channels)
    full = frame[frame.cycle_id.isin(cycles)]
    z_all = S.scale(raw.loc[full.index].to_numpy("float64"), smed, sspread)
    seq = S.sequences(z_all, first_of(full.cycle_id.to_numpy()), full.index.get_indexer(tr.index)).astype("float32")
    feat = S.scale(x23, med, spread).astype("float32")
    target = np.column_stack([y, np.nan_to_num(tr[TARGET].to_numpy())]).astype("float32")
    weight = np.column_stack([known, np.isfinite(tr[TARGET].to_numpy())]).astype("float32") * w[:, None]
    rate = float((target[:, -1] * weight[:, -1]).sum() / weight[:, -1].sum())
    weight[:, -1] *= np.where(target[:, -1] > 0, min((1 - rate) / max(rate, 1e-4), 20.0), 1.0)  # the warning is rare: weigh its positives up
    fitted = [nets.fit("hybrid", feat, seq, target, weight, hidden=8, seed=s) for s in NET_SEEDS]
    model["net"] = dict(median=med, spread=spread, seq_median=smed, seq_spread=sspread, seeds=[nets.export(n) for n in fitted])
    check["net"] = np.mean([nets.predict(n, feat[::41], seq[::41]) for n in fitted], 0)
    # the exported arrays, walked by the panel's code, must reproduce scikit-learn and PyTorch
    assert np.abs(S.tree_scores(model["hgb"]["cdf"], np.column_stack([sample, np.full(len(sample), L.H[2])])) - check["hgb_cdf"]).max() < 1e-9
    assert np.abs(S.tree_scores(model["hgb"]["warn"], sample) - check["hgb_warn"]).max() < 1e-9
    raw_net, _ = S.net_forward(model["net"], x23[::41], seq[::41].astype("float64"))
    assert np.abs(raw_net - check["net"]).max() < 1e-4, float(np.abs(raw_net - check["net"]).max())
    return model


def first_of(cycle):
    """Index of the first origin of each origin's cycle (cycles are contiguous)."""
    change = np.r_[True, cycle[1:] != cycle[:-1]]
    return np.maximum.accumulate(np.where(change, np.arange(len(cycle)), 0))


def predict(model, frame, raw, cycles):
    """Member outputs for every origin of the given cycles, by the panel's code."""
    q = frame[frame.cycle_id.isin(cycles)]
    z = S.scale(raw.loc[q.index].to_numpy("float64"), model["net"]["seq_median"], model["net"]["seq_spread"])
    out = S.member_outputs(model, LEARNERS, q[FEATURES].to_numpy("float64"), z, first_of(q.cycle_id.to_numpy()), np.arange(len(q)))
    out.pop("hidden_net")
    return q.index.to_numpy(), {k: v.astype("float32") for k, v in out.items()}


def inner_folds(cycles):
    from sklearn.model_selection import GroupKFold
    groups = np.array(cycles)
    return [list(groups[b]) for _, b in GroupKFold(INNER).split(groups, groups=groups)]


def task(frame, raw, held, split):
    """One fit. split 'fit': the model of `held` (None = every cycle). split j: inner fold j of the cycles other than `held`."""
    cycles = [c for c in frame.cycle_id.unique() if c != held]
    if split == "fit":
        model = fit_members(frame, raw, cycles)
        index, preds = predict(model, frame, raw, [held]) if held is not None else (np.array([], int), {})
        return held, split, index, preds, model
    val = inner_folds(cycles)[split]
    model = fit_members(frame, raw, [c for c in cycles if c not in val])
    index, preds = predict(model, frame, raw, val)
    return held, split, index, preds, None


# ---------- what a model takes from predictions on cycles it was not fitted on ----------
def hours_of(products, cycles):
    return L.hours_of(products, cycles)


GROUPS = [np.flatnonzero(L.H <= 1), np.flatnonzero((L.H > 1) & (L.H <= 4)), np.flatnonzero(L.H > 4)]  # near, middle, far horizons


def fit_pool(curves, base, y, weight, iterations=400):
    """Weights of [base rate, members...] on the simplex that minimise the log-loss of their mix.

    One weight vector per group of horizons (within an hour, 1.5-4 h, beyond): ten ends of life do not
    support a separate mix at each of the twelve horizons.
    """
    members = [np.broadcast_to(base, curves[0].shape)] + list(curves)
    out = np.zeros((len(L.H), len(members)))
    for group in GROUPS:
        like, v = [], []
        for j in group:
            m = weight[:, j] > 0
            p = np.clip(np.column_stack([c[m, j] for c in members]), 1e-4, 1 - 1e-4)
            like.append(np.where(y[m, j][:, None] > 0, p, 1 - p))
            v.append(weight[m, j] / weight[m, j].sum())  # every horizon of the group counts the same
        like, v = np.concatenate(like), np.concatenate(v)
        wgt = np.full(len(members), 1 / len(members))
        for _ in range(iterations):
            wgt = wgt * (v[:, None] * like / (like @ wgt)[:, None]).sum(0)
            wgt = wgt / wgt.sum()
        out[group] = wgt
    return out


def weighted_quantile(values, weights, level):
    order = np.argsort(values)
    cum = np.cumsum(weights[order]) / weights.sum()
    return float(values[order][min(np.searchsorted(cum, level), len(values) - 1)])


def calibrate(frame, events, products, cycles, P):
    """Pool weights, warning calibration, interval widening and alarm threshold of every model, from predictions P on `cycles`."""
    from sklearn.isotonic import IsotonicRegression
    q = frame[frame.cycle_id.isin(cycles)]
    ix = q.index.to_numpy()
    cyc = q.cycle_id.to_numpy()
    alive = q.rul_h.notna().to_numpy()
    event = q.event.to_numpy().astype(bool)
    rul = q.rul_h.to_numpy()
    y, known = L.cdf_targets(np.nan_to_num(rul, nan=-1.0), event)
    # Probabilities are per unit of processing time: every origin counts the same for the base rate, the pool,
    # the warning calibration and the band (the learners are fitted that way too). A short-lived electrode
    # therefore weighs little here; the evaluation reports the per-electrode view beside the per-origin one.
    weight = known * alive[:, None]
    base = (y * weight).sum(0) / weight.sum(0)
    near_known = alive & (event | (rul > 1))  # the outcome within the next processing hour is known
    near = (event & (rul <= 1)).astype(float)
    params = {}
    for model_id, spec in MODELS.items():
        curves = [P[name][ix].astype("float64") for name in spec["members"]]
        warn = S.warning({f"warn_{n}": P[f"warn_{n}"][ix].astype("float64") for n in spec["warn"]}, spec["warn"])
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(warn[near_known], near[near_known])
        p = dict(pool=fit_pool(curves, base, y, weight), base=base, cal_x=iso.X_thresholds_.astype("float64"),
                 cal_y=iso.y_thresholds_.astype("float64"), widen=np.zeros((len(S.BUCKETS) + 1, 2)))
        # conformalised quantile band: per size of the median, the shift of each end (in log hours) that leaves
        # (1 - NOMINAL) / 2 of the out-of-fold origins of the observed cycles outside on that side
        quant = np.zeros((len(q), 3))
        for c in cycles:
            m = cyc == c
            quant[m] = S.finish([curve[m] for curve in curves], warn[m], p, raw=True)["quantiles"]
        ok = alive & event
        truth = np.log(np.maximum(rul, 1e-3))
        below = np.log(np.maximum(quant[:, 0], 1e-3)) - truth  # positive when the truth lies under the lower end
        above = truth - np.log(np.maximum(quant[:, 2], 1e-3))
        bucket = np.digitize(quant[:, 1], S.BUCKETS)
        side = 1 - (1 - NOMINAL) / 2
        for b in range(len(S.BUCKETS) + 1):
            m = ok & (bucket == b)
            if m.sum() < 60:  # too few origins of this size: use all of them
                m = ok
            p["widen"][b] = np.clip([np.quantile(below[m], side), np.quantile(above[m], side)], 0.0, 3.0)  # never narrower than the raw band
        picked = EV.select_threshold(q.assign(split="validation"), warn, events[events.cycle_id.isin(cycles)].assign(split="validation"),
                                     hours_of(products, cycles), false_budget=BUDGET)
        p["threshold"] = np.array(float(picked["threshold"]))
        params[model_id] = p
    return params


def apply(frame, cycle, params, P):
    """Outputs of every model for one cycle from member predictions P and parameters."""
    q = frame[frame.cycle_id.eq(cycle)]
    ix = q.index.to_numpy()
    out = {}
    for model_id, spec in MODELS.items():
        warn = S.warning({f"warn_{n}": P[f"warn_{n}"][ix].astype("float64") for n in spec["warn"]}, spec["warn"])
        res = S.finish([P[name][ix].astype("float64") for name in spec["members"]], warn, params[model_id])
        out[model_id] = dict(res, warn=warn)
    return ix, out


# ---------- evaluation ----------
def event_scores(frame, events, products, warn, thresholds):
    """Per cycle: its own model's threshold. Totals, event rows and alarm rows (model_v2's protocol)."""
    totals = dict(total_events=0, hit20=0, late=0, missed=0, false_alarms=0, unknown_alarms=0, post_event_alarms=0, alarms=0)
    event_rows, alarm_rows = [], []
    for cycle, q in frame.groupby("cycle_id", sort=False):
        m, rows, alarms = EV.evaluate(q, warn[q.index], float(thresholds[cycle]), events[events.cycle_id.eq(cycle)], hours_of(products, [cycle]))
        for key in totals:
            totals[key] += int(m[key])
        event_rows += [dict(r, threshold=float(thresholds[cycle])) for r in rows]
        for a in alarms:
            i = q.index[a["index"]]
            alarm_rows.append(dict(a, threshold=float(thresholds[cycle]), rul_h=None if pd.isna(frame.rul_h[i]) else float(frame.rul_h[i]),
                                   cycle_event=int(frame.event[i]), age_h=float(frame.age_h[i])))
    hours = hours_of(products, list(frame.cycle_id.unique()))
    totals.update(operating_hours=hours, false_alarms_per_hour=totals["false_alarms"] / hours)
    return totals, event_rows, alarm_rows


def processing_time_reading(alarm_rows, event_rows, frame):
    """The same alarms read in processing time: how many were followed by the end within 1 / 2 / 4 processing hours."""
    known = [a for a in alarm_rows if a["rul_h"] is not None]
    hours = float(frame.rul_h.notna().sum() * L.STEP_H)
    out = dict(alarms=len(alarm_rows), processing_hours=hours,
               after_end_or_unknown=len(alarm_rows) - len(known))
    for h in (1, 2, 4):
        right = [a for a in known if a["cycle_event"] and a["rul_h"] <= h]
        wrong = [a for a in known if a["rul_h"] > h]
        out[f"end_within_{h}h"] = len(right)
        out[f"no_end_within_{h}h"] = len(wrong)
        out[f"cycles_warned_within_{h}h"] = len({a["cycle_id"] for a in right})
    return out


def life_table(frame, quant):
    """Per-electrode view (each observed cycle weighs the same) with the per-origin view under `origin_`."""
    r = L.life_scores(frame, quant)
    obs = r[r.event == 1]
    s = obs[["mae", "mae_8h", "mae_4h", "mae_2h", "mae_1h", "bias_2h", "cover", "width", "pinball"]].mean()
    s["censored_median_below_bound"] = r[r.event == 0].under.mean()
    s["cover_min_cycle"] = obs.cover.min()
    m = (frame.event.eq(1) & frame.rul_h.notna()).to_numpy()
    truth, p = frame.rul_h.to_numpy()[m], quant[m]
    err, d = np.abs(p[:, 1] - truth), truth[:, None] - p
    qs = np.array(S.QUANTILES)
    s["origin_mae"], s["origin_mae_2h"], s["origin_mae_1h"] = err.mean(), err[truth <= 2].mean(), err[truth <= 1].mean()
    s["origin_cover"] = ((p[:, 0] <= truth) & (truth <= p[:, 2])).mean()
    s["origin_width"] = (p[:, 2] - p[:, 0]).mean()
    s["origin_pinball"] = np.maximum(qs * d, (qs - 1) * d).mean(0).sum()
    return {k: float(v) for k, v in s.items()}, r


def horizon_table(frame, horizons):
    rows = []
    for j, h in enumerate(S.HORIZONS):
        known = (frame.rul_h.notna() & (frame.event.eq(1) | frame.rul_h.gt(h))).to_numpy()
        y = (frame.event.eq(1) & frame.rul_h.le(h)).to_numpy()[known].astype(float)
        p = horizons[known, j]
        row = dict(hours=h, prevalence=float(y.mean()), mean_predicted=float(p.mean()), brier=float(((p - y) ** 2).mean()),
                   brier_constant=float(((y.mean() - y) ** 2).mean()))
        row["skill"] = 1 - row["brier"] / row["brier_constant"]
        rows.append(row)
    return rows


def baselines(frame):
    """Constant quantiles and mean-life-minus-age, each from the other cycles."""
    cyc = frame.groupby("cycle_id").agg(life=("life_h", "first"), ev=("event", "first"))
    const, mean = np.zeros((len(frame), 3)), np.zeros((len(frame), 3))
    for cycle, q in frame.groupby("cycle_id", sort=False):
        other = frame[frame.cycle_id.ne(cycle) & frame.event.eq(1) & frame.rul_h.notna()].rul_h
        const[q.index] = np.quantile(other, S.QUANTILES)
        others = cyc.drop(cycle)
        mean[q.index] = np.maximum(others[others.ev == 1].life.mean() - q.age_h.to_numpy(), .05)[:, None]
    return dict(constant_quantiles=life_table(frame, const)[0], mean_life_minus_age=life_table(frame, mean)[0])


def main(out_dir, refit=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fits = out_dir.parent / "fits.pkl"
    frame = L.load().reset_index(drop=True)
    frame[FEATURES] = frame[FEATURES].astype("float32")  # the bundle stores float32: fit and score on exactly those values
    raw = B.raw_channels(frame).astype("float32")
    slim = frame[["cycle_id", "rul_h", "event", TARGET] + FEATURES]  # all a fit needs
    events, products = L.tables()
    cycles = list(frame.cycle_id.unique())
    if fits.exists() and not refit:
        import joblib
        results = joblib.load(fits)
    else:
        jobs = [delayed(task)(slim, raw, c, s) for c in cycles for s in ["fit"] + list(range(INNER))] + [delayed(task)(slim, raw, None, "fit")]
        results = Parallel(n_jobs=22, verbose=5)(jobs)
        import joblib
        joblib.dump(results, fits, compress=3)
    n = len(frame)
    shape = lambda name: (n, len(L.H)) if not name.startswith("warn_") else (n,)  # noqa: E731
    names = LEARNERS + ["warn_hgb", "warn_net"]
    outer = {name: np.full(shape(name), np.nan, "float32") for name in names}
    inner = {c: {name: np.full(shape(name), np.nan, "float32") for name in names} for c in cycles}
    fold_models, final_members = {}, None
    for held, split, index, preds, model in results:
        if split == "fit" and held is None:
            final_members = model
        elif split == "fit":
            fold_models[held] = model
            for name in names:
                outer[name][index] = preds[name]
        else:
            for name in names:
                inner[held][name][index] = preds[name]
    # every model's parameters for a cycle come from the inner predictions of the 13 other cycles
    lean = frame[["anchor_id", "timestamp", "cycle_id", "observation_horizon_seconds", "rul_h", "event"]]  # all the calibration reads
    done = Parallel(n_jobs=len(cycles) + 1)(
        [delayed(calibrate)(lean, events, products, [x for x in cycles if x != c], inner[c]) for c in cycles]
        + [delayed(calibrate)(lean, events, products, cycles, outer)])  # the last: the model for new data, fitted on every cycle
    params, final_params = dict(zip(cycles, done[:-1])), done[-1]
    res = {m: dict(warn=np.zeros(n), quantiles=np.zeros((n, 3)), curve=np.zeros((n, len(L.H))), horizons=np.zeros((n, 3))) for m in MODELS}
    for c in cycles:
        ix, out = apply(frame, c, params[c], outer)
        for m in MODELS:
            for key in res[m]:
                res[m][key][ix] = out[m][key]

    # ---- evaluation
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    cyc = pd.read_csv(D.REPO / "model_v2_GRU/data/cycles.csv")
    evaluation, workbench_models = {}, {}
    for m, spec in MODELS.items():
        thresholds = {c: float(params[c][m]["threshold"]) for c in cycles}
        nested, event_rows, alarm_rows = event_scores(frame, events, products, res[m]["warn"], thresholds)
        single, _, _ = event_scores(frame, events, products, res[m]["warn"], {c: float(final_params[m]["threshold"]) for c in cycles})
        life, per = life_table(frame, res[m]["quantiles"])
        per_cycle = []
        for c in cycles:
            row = cyc[cyc.cycle_id.eq(c)].iloc[0]
            ev = next((r for r in event_rows if r["cycle_id"] == c), None)
            lf = per[per.cycle == c].iloc[0]
            observed = bool(row.eol_strict_observed)
            per_cycle.append(dict(
                cycle=c, end_type=row.end_type, observed=int(observed), life_hours=float(frame[frame.cycle_id.eq(c)].life_h.iloc[0]), threshold=thresholds[c],
                event=None if ev is None else dict(id=ev["event_id"], time=ev["time"], first_alarm=ev["first_alarm"], lead_minutes=ev["lead_minutes"],
                                                   hit20=ev["hit20"], late=ev["late"], missed=ev["missed"]),
                false_alarms=int(sum(a["false_alarm"] for a in alarm_rows if a["cycle_id"] == c)),
                life_mae=float(lf.mae) if observed else None, life_mae_last2h=float(lf.mae_2h) if observed else None,
                cover=float(lf.cover) if observed else None))
        evaluation[m] = dict(
            protocol="nested leave-one-cycle-out: 14 held-out cycles; pool weights, warning calibration, interval widening and threshold from inner out-of-fold predictions",
            events=nested, single_threshold_reference=dict(single, threshold=float(final_params[m]["threshold"])),
            processing_time=processing_time_reading(alarm_rows, event_rows, frame),
            life=dict(model=life, mae=life["mae"], mae_last2h=life["mae_2h"]), cycles=per_cycle, nominal_interval=NOMINAL,
            horizons=horizon_table(frame, res[m]["horizons"]),
            pool_final={name: [float(v) for v in final_params[m]["pool"][:, k + 1]] for k, name in enumerate(spec["members"])},
            pool_final_base=[float(v) for v in final_params[m]["pool"][:, 0]],
            widen_final=np.asarray(final_params[m]["widen"], dtype=float).tolist())
        workbench_models[m] = dict(
            alarms=[{k: a[k] for k in ("time", "cycle_id", "probability", "matched_event", "false_alarm", "unknown", "post_event", "threshold", "rul_h")} for a in alarm_rows],
            events=[{k: r[k] for k in ("event_id", "cycle_id", "time", "first_alarm", "lead_minutes", "hit20", "late", "missed", "earlier_alarm_count")} for r in event_rows])
        print(f"\n===== {m}")
        print("nested:", json.dumps(nested))
        print("single threshold (optimistic):", json.dumps({k: single[k] for k in ("hit20", "late", "missed", "false_alarms", "unknown_alarms")}), float(final_params[m]["threshold"]))
        print("processing time:", json.dumps(evaluation[m]["processing_time"]))
        print(pd.DataFrame(event_rows)[["event_id", "cycle_id", "first_alarm", "lead_minutes", "hit20", "late", "missed", "threshold", "earlier_alarm_count"]].round(1).to_string(index=False))
        print("life:", {k: round(v, 2) for k, v in life.items()})
        print(per[per.event == 1][["cycle", "mae", "mae_2h", "mae_1h", "cover", "width"]].round(2).to_string(index=False))
        print(pd.DataFrame(evaluation[m]["horizons"]).round(4).to_string(index=False))
        print("pool (final), rows = horizons", [round(float(h), 2) for h in L.H])
        print(pd.DataFrame(final_params[m]["pool"], columns=["base"] + spec["members"]).round(2).T.to_string())
        print("band shift [lower, upper] per size of median, final:", np.round(final_params[m]["widen"], 2).tolist())
    base = baselines(frame)
    v3 = pd.read_csv(L.V3 / "model_v3/v3_predictions.csv.gz")
    base["v3"] = life_table(frame, v3[["q10", "q50", "q90"]].to_numpy())[0]
    members_life = {name: life_table(frame, S.quantiles(outer[name].astype("float64")))[0] for name in LEARNERS}
    evaluation["baselines"] = dict(life=base, members_life=members_life)
    print("\nbaselines", json.dumps({k: {a: round(b, 2) for a, b in v.items()} for k, v in base.items()}))
    print("members (raw curves, no post-processing)", json.dumps({k: {a: round(b, 2) for a, b in v.items()} for k, v in members_life.items()}))

    # ---- bundle
    rate = float(frame.groupby("cycle_id").rul_pieces.max().sum() / frame.groupby("cycle_id").life_h.first().sum())
    arrays = {"models": np.array(len(cycles) + 1)}
    for k, (members, post) in enumerate([(fold_models[c], params[c]) for c in cycles] + [(final_members, final_params)]):
        for name in SMALL:
            for key, value in members[name].items():
                arrays[f"m{k}_{name}_{key}"] = np.asarray(value)
        for part in ("cdf", "warn"):
            for key, value in members["hgb"][part].items():
                arrays[f"m{k}_hgb_{part}_{key}"] = np.asarray(value)
        net = members["net"]
        for key in ("median", "spread", "seq_median", "seq_spread"):
            arrays[f"m{k}_net_{key}"] = np.asarray(net[key])
        arrays[f"m{k}_net_seeds"] = np.array(len(net["seeds"]))
        for s, seed in enumerate(net["seeds"]):
            for key, value in seed.items():
                if key != "kind":
                    arrays[f"m{k}_net{s}_{key}"] = np.asarray(value)
        for model_id, p in post.items():
            for key, value in p.items():
                arrays[f"m{k}_post_{model_id}__{key}"] = np.asarray(value)
    np.savez_compressed(out_dir / "weights.npz", **arrays)
    index = {c: i for i, c in enumerate(cycles)}
    np.savez_compressed(
        out_dir / "origins.npz", t=(frame.timestamp.astype("int64") // 10**9).to_numpy(), cycle=frame.cycle_id.map(index).to_numpy("int16"),
        features=frame[FEATURES].to_numpy("float32"), channels=raw.to_numpy("float32"), age_h=frame.age_h.to_numpy("float32"),
        **{f"ledger_{name}": outer[name] for name in names}, **{f"ledger_q_{m}": res[m]["quantiles"] for m in MODELS})
    # predicted against actual remaining life along each observed cycle, 90 points evenly spaced in processing time
    trajectories = []
    for c in cycles:
        q = frame[frame.cycle_id.eq(c) & frame.rul_h.notna()]
        if not len(q) or not q.event.iloc[0]:
            continue
        pick = q.index[np.unique(np.linspace(0, len(q) - 1, 90).round().astype(int))]
        trajectories.append(dict(cycle=c, life_hours=float(q.life_h.iloc[0]), actual=[round(float(v), 3) for v in frame.rul_h[pick]],
                                 models={m: dict(q10=[round(float(v), 3) for v in res[m]["quantiles"][pick, 0]],
                                                 q50=[round(float(v), 3) for v in res[m]["quantiles"][pick, 1]],
                                                 q90=[round(float(v), 3) for v in res[m]["quantiles"][pick, 2]],
                                                 warn=[round(float(v), 3) for v in res[m]["warn"][pick]]) for m in MODELS}))
    workbench = dict(evaluation=evaluation, features=FEATURES, channels=list(raw.columns), models=workbench_models, trajectories=trajectories,
                     learners={name: dict(inputs=[(FEATURES + ["log_vsd_30m"])[i] for i in SMALL[name][0]]) for name in SMALL})
    (out_dir / "workbench.json").write_text(json.dumps(workbench, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    files = {name: sha(out_dir / name) for name in ("weights.npz", "origins.npz", "workbench.json")}
    manifest = dict(
        kind="survival_v4", version="booth2-v4-survival-" + files["weights.npz"][:12], created_utc=datetime.now(timezone.utc).isoformat(),
        source=dict(repository="ruiqiyang666/ANN_data_policy",
                    commit=subprocess.check_output(["git", "-C", str(D.REPO), "rev-parse", "HEAD"], text=True).strip(),
                    tables=["training_data/features_10s.csv.gz", "model_v2_GRU/data/anchors.csv.gz", "model_v2_GRU/data/targets.csv.gz",
                            "model_v2_GRU/data/cycles.csv", "model_v2_GRU/data/events.csv", "model_v2_GRU/data/products.csv.gz"]),
        files=files, features=FEATURES, channels=list(raw.columns), sensors=28, members=LEARNERS, horizons_hours=[float(h) for h in L.H],
        models=[dict(id=m, **{k: v for k, v in spec.items()}) for m, spec in MODELS.items()], cycle_ids=cycles,
        smoothing=dict(rise_alpha=S.RISE_ALPHA, fade_alpha=S.FADE_ALPHA),
        alarm_policy=dict(confirm_origins=EV.POLICY.confirm_origins, max_confirm_gap_seconds=EV.POLICY.max_confirm_gap_seconds,
                          release_seconds=EV.POLICY.release_seconds, cooldown_seconds=EV.POLICY.cooldown_seconds),
        products_per_processing_hour=rate, showcase_segments=SEGMENTS, evaluation=evaluation,
        limits=["实验模型：只有 10 次观察到的电极末期。剩余寿命在最后约 2 个加工小时内才有分辨力，更早时接近历史平均；它是范围估计，不是倒计时，不能作为自动换枪依据。",
                "预警评分未校准，不是概率；提前量按“连续失败第一件开始加工”计。",
                "到寿概率按答案已知的原点估计，未做删失加权：12 小时以上的时间点可能偏高约 0.02–0.04。",
                "回放中每个电极周期由未见过该周期的留一模型评分；这些历史周期此前已被查看，不是新增盲测。",
                "加工原点沿用离线确认的产品资格，不等同于现场异步数据全链路。"])
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    bundle = S.SurvivalBundle(out_dir)
    assert bundle.verify()
    print("\nbundle", bundle.version, "member parity", bundle.member_error)
    for segment in SEGMENTS:
        seg = (bundle.times >= segment["start"]) & (bundle.times < segment["stop"])
        for m, model in bundle.models.items():
            print(" ", segment["name"], int(seg.sum()), "origins", m, "alarms", [str(pd.Timestamp(int(t), unit="s"))[11:] for t in bundle.times[seg & model.emitted]])
    cols = {}
    for m in MODELS:
        tag = m.split("_")[1]
        cols.update({f"{tag}_warn": res[m]["warn"], f"{tag}_q10": res[m]["quantiles"][:, 0], f"{tag}_q50": res[m]["quantiles"][:, 1],
                     f"{tag}_q90": res[m]["quantiles"][:, 2], f"{tag}_p1h": res[m]["horizons"][:, 0], f"{tag}_p4h": res[m]["horizons"][:, 1]})
    frame.assign(**cols)[["anchor_id", "timestamp", "cycle_id", "rul_h", "event"] + list(cols)].to_csv(out_dir.parent / "v4_predictions.csv.gz", index=False)


if __name__ == "__main__":
    main(sys.argv[1], refit="--refit" in sys.argv)
