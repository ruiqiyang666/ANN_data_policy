"""Booth2 model v3: electrode warning + remaining life, nested leave-one-cycle-out.

usage: python train_v3.py OUTPUT_DIR

Design is fixed before this script runs (see README): 23 causal end-stage
features from the 28 sensors; five gradient-boosted tree models averaged for the
warning; a logistic hazard layer for life; the warning lifts first-hour hazards.

Every electrode cycle is scored by a model that never saw it. The alarm
threshold, the warning calibration and the interval widening of that model come
from inner out-of-fold predictions of its 13 training cycles, so the reported
event results include the cost of choosing a threshold.
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
import data as D  # noqa: E402
import rul_bench as R  # noqa: E402

APP = D.ROOT.parent / "depuy-theme-preview" / "simulation" / "pipeline"
# model_v2/src has its own inference.py; the panel's `inference` package must win the name.
sys.path.insert(0, str(D.REPO / "model_v2_GRU" / "src"))
sys.path.insert(0, str(APP))
import evaluation as EV  # noqa: E402  (model_v2's frozen alarm protocol)
from inference import life as L  # noqa: E402  (the panel's own inference code)

warnings.filterwarnings("ignore")
FEATURES = ["vsd_mean_5m", "vsd_mean_30m", "vsd_mean_3h", "vsd60", "vsd_ratio", "v_std_5m", "v_std_30m", "d_peak_30m", "d_peak_3h",
            "d_5m_3h", "d_30m_base", "v_slope_30m", "v_slope_3h", "v_low53_30m", "margin51_30m", "margin51_q10", "v_min_30m",
            "cool_loss_kw_d_base", "cool_dT_d_base", "therm_eff_d_base", "power_d_base", "power_std_30m", "therm_eff_slope_3h"]
TARGET = "y_eol_60m"
SEEDS = (0, 1, 2, 3, 4)
ALPHA = 0.003  # hazard smoothing per origin: a time constant of about 55 processing minutes
BUDGET = 0.1  # false alarms per production hour allowed when a threshold is chosen
PENALTIES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1)  # inverse L2 strength of the hazard layer, chosen inside each fold
NOMINAL = 0.8
SEGMENT = dict(name="electrode_eol_20261028", label="10月28日 07:30–08:32 电极末期", start=1793172600, stop=1793176320)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------- fitting ----------
def fit_warn(x, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return [HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=.05, min_samples_leaf=100, l2_regularization=2.0,
                                           max_features=.7, class_weight="balanced", early_stopping=False, random_state=s).fit(x, y)
            for s in SEEDS]


def warn_predict(models, x):
    return np.mean([m.predict_proba(x)[:, 1] for m in models], axis=0)


def export_trees(models):
    """scikit-learn predictor nodes -> padded arrays the panel walks in NumPy."""
    width = 15  # a depth-3 binary tree
    shape = (len(models), models[0].n_iter_, width)
    out = dict(feature=np.zeros(shape, "int16"), threshold=np.zeros(shape, "float64"), left=np.zeros(shape, "int16"),
               right=np.zeros(shape, "int16"), value=np.zeros(shape, "float64"), leaf=np.ones(shape, bool),
               missing_left=np.zeros(shape, bool))
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


def fit_hazard(x, rul, event, c):
    """Logistic hazard layer: one sigmoid unit per time bin, fitted to convergence.

    Unit k sees the origins still at risk at the start of bin k (masked after the
    end / after censoring) and learns whether the end falls inside the bin. The
    units share no parameters, so each is an L2-penalised logistic regression; the
    intercept is not penalised, which keeps the base rate of every bin right.
    """
    from sklearn.linear_model import LogisticRegression
    sc = R.Scaler(x)
    z = sc(x).astype("float64")
    y, w = R.hazard_targets(rul, event)
    weight, bias = np.zeros((y.shape[1], z.shape[1])), np.zeros(y.shape[1])
    for k in range(y.shape[1]):
        risk = w[:, k] > 0
        ends = float(y[risk, k].sum())
        if ends == 0 or ends == risk.sum():  # no information in this bin: a smoothed base rate
            rate = (ends + .5) / (risk.sum() + 1)
            bias[k] = np.log(rate / (1 - rate))
            continue
        unit = LogisticRegression(C=c, max_iter=3000).fit(z[risk], y[risk, k])
        weight[k], bias[k] = unit.coef_[0], unit.intercept_[0]
    return dict(median=sc.med.astype("float64"), spread=sc.scale.astype("float64"), weight=weight, bias=bias)


def hazard_nll(logit, rul, event):
    """Masked discrete-time negative log-likelihood per origin."""
    y, w = R.hazard_targets(rul, event)
    loss = np.logaddexp(0, logit) - y * logit
    return float((loss * w).sum() / len(rul))


def hazard_logit(h, x):
    return L.scale(x, h["median"], h["spread"]) @ h["weight"].T + h["bias"]


def fit_pair(frame, cycles, penalties):
    """Warning trees and one hazard layer per candidate penalty on the given cycles."""
    w = frame[frame.cycle_id.isin(cycles) & frame[TARGET].notna()]
    warn = fit_warn(w[FEATURES].to_numpy("float32"), w[TARGET].to_numpy().astype(int))
    lt = frame[frame.cycle_id.isin(cycles) & frame.rul_h.notna()].iloc[::6]  # one origin per processing minute
    x, rul, event = lt[FEATURES].to_numpy("float32").astype("float64"), lt.rul_h.to_numpy(), lt.event.to_numpy().astype(bool)
    return warn, {c: fit_hazard(x, rul, event, c) for c in penalties}


# ---------- operating point, calibration and interval width from out-of-fold predictions ----------
def hours_of(products, cycles):
    q = products[products.cycle_id.isin(cycles) & products.supervised_product_eligible.eq(1)]
    return float(q.spray_samples.sum() / 3600)


def choose(frame, cycles, warn, logit, events, products):
    """Everything a model takes from predictions on cycles it did not train on."""
    from sklearn.isotonic import IsotonicRegression
    q = frame[frame.cycle_id.isin(cycles)]
    s = warn[q.index]
    picked = EV.select_threshold(q.assign(split="validation"), s, events[events.cycle_id.isin(cycles)].assign(split="validation"),
                                 hours_of(products, cycles), false_budget=BUDGET)
    alive = q.rul_h.notna() & (q.event.eq(1) | q.rul_h.gt(1))  # outcome within the next hour is known
    near = (q.event.eq(1) & q.rul_h.le(1)).to_numpy().astype(float)
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(s[alive.to_numpy()], near[alive.to_numpy()])
    model = dict(calibration_x=iso.X_thresholds_.astype("float64"), calibration_y=iso.y_thresholds_.astype("float64"),
                 threshold=np.array(float(picked["threshold"])), widen=np.array(0.0), mode=np.array(0))
    observed = [c for c in cycles if q[q.cycle_id.eq(c)].event.iloc[0]]

    def loss(mode):  # mean pinball loss of the three quantiles, every observed cycle weighing the same
        total = 0.0
        for cycle in observed:
            c = q[q.cycle_id.eq(cycle)]
            ok = c.rul_h.notna().to_numpy()
            total += pinball(c.rul_h.to_numpy()[ok], staged(dict(model, mode=mode), warn[c.index], logit[c.index])[ok]).sum()
        return total / len(observed)
    model["mode"] = np.array(min(range(len(L.MODES)), key=loss))
    # Interval widening: the factor that makes the 10-90 % band hold NOMINAL of the out-of-fold origins
    # (each observed cycle weighs the same; censored cycles have no known remaining life).
    errors, weights = [], []
    for cycle in cycles:
        c = q[q.cycle_id.eq(cycle)]
        if not c.event.iloc[0]:
            continue
        out = staged(model, warn[c.index], logit[c.index])
        ok = c.rul_h.notna().to_numpy()
        truth = np.log(c.rul_h.to_numpy()[ok])
        e = np.maximum(np.log(np.maximum(out[ok, 0], 1e-3)) - truth, truth - np.log(np.maximum(out[ok, 2], 1e-3)))
        errors.append(e); weights.append(np.full(len(e), 1 / len(e)))
    e, w = np.concatenate(errors), np.concatenate(weights)
    order = np.argsort(e)
    cum = np.cumsum(w[order]) / w.sum()
    model["widen"] = np.array(max(0.0, float(e[order][np.searchsorted(cum, NOMINAL)])))
    return model, picked


def staged(model, warn, logit):
    """Quantiles of one cycle from given warning scores and hazard logits (no widening)."""
    return L.quantiles(L.fuse(logit, np.interp(warn, model["calibration_x"], model["calibration_y"]), ALPHA, model["mode"]))


# ---------- one outer fold ----------
def outer(frame, events, products, held):
    from sklearn.model_selection import GroupKFold
    frame = frame.copy()
    cycles = [c for c in frame.cycle_id.unique() if c != held]
    inner_warn = np.full(len(frame), np.nan)
    inner_logit = {c: np.full((len(frame), len(L.EDGES) - 1), np.nan) for c in PENALTIES}
    groups = np.array(cycles)
    for a, b in GroupKFold(5).split(groups, groups=groups):
        warn, hazards = fit_pair(frame, groups[a], PENALTIES)
        q = frame[frame.cycle_id.isin(groups[b])]
        x = q[FEATURES].to_numpy("float32")
        inner_warn[q.index] = warn_predict(warn, x)
        for c in PENALTIES:
            inner_logit[c][q.index] = hazard_logit(hazards[c], x)
    known = frame[frame.cycle_id.isin(cycles) & frame.rul_h.notna()]
    nll = {c: hazard_nll(inner_logit[c][known.index], known.rul_h.to_numpy(), known.event.to_numpy().astype(bool)) for c in PENALTIES}
    penalty = min(nll, key=nll.get)
    chosen, picked = choose(frame, cycles, inner_warn, inner_logit[penalty], events, products)
    warn, hazards = fit_pair(frame, cycles, [penalty])
    q = frame[frame.cycle_id.eq(held)]
    x = q[FEATURES].to_numpy("float32")
    model = dict(trees=export_trees(warn), **hazards[penalty], **chosen)
    info = dict(picked["metrics"], penalty=penalty, inner_nll=nll, mode=L.MODES[int(chosen["mode"])])
    return held, model, q.index.to_numpy(), warn_predict(warn, x), hazard_logit(hazards[penalty], x), info


# ---------- evaluation ----------
def pinball(truth, pred):
    q = np.array(L.QUANTILES)
    d = truth[:, None] - pred
    return np.maximum(q * d, (q - 1) * d).mean(0)


def life_scores(frame, quant):
    rows = []
    cyc = frame.groupby("cycle_id").agg(life=("life_h", "first"), ev=("event", "first"))
    for cycle, q in frame.groupby("cycle_id", sort=False):
        ok = q.rul_h.notna().to_numpy()
        truth, age = q.rul_h.to_numpy()[ok], q.age_h.to_numpy()[ok]
        others = cyc.drop(cycle)
        other_rul = frame[frame.cycle_id.ne(cycle) & frame.event.eq(1) & frame.rul_h.notna()].rul_h
        k, lam = R.weibull_fit(others.life, others.ev)
        cands = {
            "v3": quant[q.index][ok],
            "constant_quantiles": np.tile(np.quantile(other_rul, L.QUANTILES), (len(truth), 1)),
            "mean_life_minus_age": np.column_stack([np.maximum(others[others.ev == 1].life.mean() - age, .05)] * 3),
            "weibull_age": np.column_stack([R.weibull_residual(age, k, lam, p) for p in L.QUANTILES]),
        }
        for name, p in cands.items():
            row = dict(model=name, cycle=cycle, event=int(cyc.ev[cycle]), life_h=float(cyc.life[cycle]))
            if cyc.ev[cycle]:
                err = np.abs(p[:, 1] - truth)
                pb = pinball(truth, p)
                row.update(mae=err.mean(), mae_last5h=err[truth <= 5].mean(), mae_last2h=err[truth <= 2].mean(),
                           mae_last1h=err[truth <= 1].mean(), bias=(p[:, 1] - truth).mean(), pinball=float(pb.sum()),
                           cover=float(((p[:, 0] <= truth) & (truth <= p[:, 2])).mean()), width=float((p[:, 2] - p[:, 0]).mean()))
            else:
                row.update(median_below_lower_bound=float((p[:, 1] < truth).mean()))
            rows.append(row)
    return pd.DataFrame(rows)


def event_scores(frame, events, products, warn, thresholds):
    """Per cycle: its own model's threshold. Returns totals, event rows and alarm rows."""
    totals = dict(total_events=0, hit20=0, late=0, missed=0, false_alarms=0, unknown_alarms=0, post_event_alarms=0, alarms=0)
    event_rows, alarm_rows = [], []
    for cycle, q in frame.groupby("cycle_id", sort=False):
        m, rows, alarms = EV.evaluate(q, warn[q.index], float(thresholds[cycle]), events[events.cycle_id.eq(cycle)], hours_of(products, [cycle]))
        for key in totals:
            totals[key] += int(m[key])
        event_rows += [dict(r, threshold=float(thresholds[cycle])) for r in rows]
        alarm_rows += [dict(a, threshold=float(thresholds[cycle])) for a in alarms]
    hours = hours_of(products, list(frame.cycle_id.unique()))
    totals.update(operating_hours=hours, false_alarms_per_hour=totals["false_alarms"] / hours)
    return totals, event_rows, alarm_rows


def main(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = D.build().reset_index(drop=True)
    frame[FEATURES] = frame[FEATURES].astype("float32")
    events = pd.read_csv(D.REPO / "model_v2_GRU/data/events.csv", parse_dates=["time"])
    products = pd.read_csv(D.REPO / "model_v2_GRU/data/products.csv.gz", low_memory=False)
    cycles = list(frame.cycle_id.unique())
    results = Parallel(n_jobs=14)(delayed(outer)(frame, events, products, c) for c in cycles)
    warn, logit = np.full(len(frame), np.nan), np.full((len(frame), len(L.EDGES) - 1), np.nan)
    models, inner_metrics = {}, {}
    for held, model, index, w, lg, metrics in results:
        models[held], inner_metrics[held] = model, metrics
        warn[index], logit[index] = w, lg
    # final model: fitted on every cycle, operating point from the 14 held-out predictions
    chosen, picked = choose(frame, cycles, warn, logit, events, products)
    votes = [inner_metrics[c]["penalty"] for c in cycles]
    penalty = max(sorted(set(votes)), key=votes.count)  # the penalty most held-out models chose
    final_warn, final_hazards = fit_pair(frame, cycles, [penalty])
    final = dict(trees=export_trees(final_warn), **final_hazards[penalty], **chosen)
    # outputs of every origin through the panel's code; trees walked in NumPy
    n = len(frame)
    quant, horizons, ledger_warn = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n)
    for cycle, q in frame.groupby("cycle_id", sort=False):
        out = L.run_cycle(models[cycle], q[FEATURES].to_numpy("float32"), ALPHA)
        assert np.abs(out["warn"] - warn[q.index]).max() < 1e-9, "NumPy trees differ from scikit-learn"
        assert np.abs(L.sigmoid(logit[q.index]) - out["hazard"]).max() < 1e-9
        quant[q.index], horizons[q.index], ledger_warn[q.index] = out["quantiles"], out["horizons"], out["warn"]
    check = frame.iloc[::37]
    assert np.abs(L.tree_scores(final["trees"], check[FEATURES].to_numpy("float32")) - warn_predict(final_warn, check[FEATURES].to_numpy("float32"))).max() < 1e-9

    # ---- evaluation
    thresholds = {c: float(models[c]["threshold"]) for c in cycles}
    nested, event_rows, alarm_rows = event_scores(frame, events, products, ledger_warn, thresholds)
    single, _, _ = event_scores(frame, events, products, ledger_warn, {c: float(final["threshold"]) for c in cycles})
    life = life_scores(frame, quant)
    obs = life[life.event == 1]
    life_summary = obs.groupby("model")[["mae", "mae_last5h", "mae_last2h", "mae_last1h", "bias", "pinball", "cover", "width"]].mean()
    life_summary["censored_median_below_bound"] = life[life.event == 0].groupby("model").median_below_lower_bound.mean()
    # the same design under model_v2's frozen split, for a like-for-like line beside GRU-16
    TR, VA, TE = cycles[:5], cycles[5:8], cycles[8:]
    fw, _ = fit_pair(frame, TR, [])
    fs = warn_predict(fw, frame[FEATURES].to_numpy("float32"))
    va = frame[frame.cycle_id.isin(VA)]
    fpick = EV.select_threshold(va.assign(split="validation"), fs[va.index], events[events.cycle_id.isin(VA)].assign(split="validation"),
                                hours_of(products, VA), false_budget=BUDGET)
    frozen = {}
    for name, cs in (("train", TR), ("validation", VA), ("test", TE)):
        q = frame[frame.cycle_id.isin(cs)]
        m, _, _ = EV.evaluate(q, fs[q.index], fpick["threshold"], events[events.cycle_id.isin(cs)], hours_of(products, cs))
        frozen[name] = {k: m[k] for k in ("total_events", "hit20", "late", "missed", "false_alarms", "post_event_alarms", "operating_hours", "false_alarms_per_hour")}
    frozen["threshold"] = float(fpick["threshold"])

    pd.set_option("display.width", 250)
    print("nested leave-one-cycle-out:", json.dumps(nested, ensure_ascii=False))
    print("one threshold on held-out scores (optimistic):", json.dumps(single), "threshold", float(final["threshold"]))
    print("frozen v2 split:", json.dumps(frozen))
    print(pd.DataFrame(event_rows)[["event_id", "cycle_id", "time", "first_alarm", "lead_minutes", "hit20", "late", "missed", "threshold", "earlier_alarm_count"]].to_string())
    print(life_summary.round(3).to_string())
    print(obs.pivot(index="cycle", columns="model", values="mae").round(2).to_string())
    print("widen", {c: round(float(models[c]["widen"]), 3) for c in cycles}, "final", round(float(final["widen"]), 3))
    print("penalty", {c: inner_metrics[c]["penalty"] for c in cycles}, "final", penalty)
    print("mode", {c: inner_metrics[c]["mode"] for c in cycles}, "final", L.MODES[int(final["mode"])])
    horizon_rows = []
    for j, h in enumerate(L.HORIZONS):
        known = (frame.rul_h.notna() & (frame.event.eq(1) | frame.rul_h.gt(h))).to_numpy()
        y = (frame.event.eq(1) & frame.rul_h.le(h)).to_numpy()[known].astype(float)
        p = horizons[known, j]
        row = dict(hours=h, prevalence=float(y.mean()), mean_predicted=float(p.mean()), brier=float(((p - y) ** 2).mean()),
                   brier_constant=float(((y.mean() - y) ** 2).mean()))
        row["skill"] = 1 - row["brier"] / row["brier_constant"]
        horizon_rows.append(row)
    print(pd.DataFrame(horizon_rows).round(4).to_string(index=False))

    # ---- bundle
    rate = float(frame.groupby("cycle_id").rul_pieces.max().sum() / frame.groupby("cycle_id").life_h.first().sum())
    arrays = {"models": np.array(len(cycles) + 1)}
    for k, model in enumerate([models[c] for c in cycles] + [final]):
        for key, value in model.items():
            if key == "trees":
                arrays.update({f"m{k}_tree_{name}": v for name, v in value.items()})
            else:
                arrays[f"m{k}_{key}"] = np.asarray(value)
    np.savez_compressed(out_dir / "weights.npz", **arrays)
    index = {c: i for i, c in enumerate(cycles)}
    np.savez_compressed(out_dir / "origins.npz", t=(frame.timestamp.astype("int64") // 10**9).to_numpy(),
                        cycle=frame.cycle_id.map(index).to_numpy("int16"), features=frame[FEATURES].to_numpy("float32"),
                        age_h=frame.age_h.to_numpy("float32"), ledger_warn=ledger_warn, ledger_quantiles=quant)
    cyc = pd.read_csv(D.REPO / "model_v2_GRU/data/cycles.csv")
    per_cycle = []
    for c in cycles:
        row = cyc[cyc.cycle_id.eq(c)].iloc[0]
        ev = next((r for r in event_rows if r["cycle_id"] == c), None)
        lf = life[(life.cycle == c) & (life.model == "v3")].iloc[0]
        per_cycle.append(dict(
            cycle=c, end_type=row.end_type, observed=int(row.eol_strict_observed), life_hours=float(lf.life_h), threshold=thresholds[c],
            event=None if ev is None else dict(id=ev["event_id"], time=ev["time"], first_alarm=ev["first_alarm"], lead_minutes=ev["lead_minutes"],
                                               hit20=ev["hit20"], late=ev["late"], missed=ev["missed"]),
            false_alarms=int(sum(a["false_alarm"] for a in alarm_rows if a["cycle_id"] == c)),
            life_mae=None if not row.eol_strict_observed else float(lf.mae),
            life_mae_last2h=None if not row.eol_strict_observed else float(lf.mae_last2h),
            cover=None if not row.eol_strict_observed else float(lf.cover)))
    evaluation = dict(
        protocol="nested leave-one-cycle-out: 14 held-out cycles, threshold / calibration / interval width from inner out-of-fold predictions",
        events=nested, single_threshold_reference=dict(single, threshold=float(final["threshold"])), frozen_split=frozen,
        life={name: {k: (None if pd.isna(v) else float(v)) for k, v in r.items()} for name, r in life_summary.iterrows()},
        cycles=per_cycle, nominal_interval=NOMINAL, horizons=horizon_rows, hazard_penalty=penalty, life_mode=L.MODES[int(final["mode"])])
    workbench = dict(evaluation=evaluation, features=FEATURES, alarms=[
        {k: a[k] for k in ("time", "cycle_id", "probability", "matched_event", "false_alarm", "unknown", "post_event", "threshold")} for a in alarm_rows],
        inner_selection={c: {k: inner_metrics[c][k] for k in ("hit20", "false_alarms", "false_alarms_per_hour", "threshold", "penalty", "mode")} for c in cycles})
    (out_dir / "workbench.json").write_text(json.dumps(workbench, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    files = {name: sha(out_dir / name) for name in ("weights.npz", "origins.npz", "workbench.json")}
    manifest = dict(
        version="booth2-v3-life-" + files["weights.npz"][:12], created_utc=datetime.now(timezone.utc).isoformat(),
        source=dict(repository="ruiqiyang666/ANN_data_policy",
                    commit=subprocess.check_output(["git", "-C", str(D.REPO), "rev-parse", "HEAD"], text=True).strip(),
                    tables=["training_data/features_10s.csv.gz", "model_v2_GRU/data/anchors.csv.gz", "model_v2_GRU/data/targets.csv.gz",
                            "model_v2_GRU/data/cycles.csv", "model_v2_GRU/data/events.csv", "model_v2_GRU/data/products.csv.gz"]),
        files=files, features=FEATURES, sensors=28, cycle_ids=cycles, smoothing_alpha=ALPHA,
        alarm_policy=dict(confirm_origins=EV.POLICY.confirm_origins, max_confirm_gap_seconds=EV.POLICY.max_confirm_gap_seconds,
                          release_seconds=EV.POLICY.release_seconds, cooldown_seconds=EV.POLICY.cooldown_seconds),
        products_per_processing_hour=rate, showcase_segment=SEGMENT, evaluation=evaluation,
        limits=["实验模型：只有 10 次观察到的电极末期，剩余寿命是宽范围估计，不是倒计时，不能作为自动换枪依据。",
                "预警评分未校准，不是概率；提前量按“连续失败第一件开始加工”计。",
                "回放中每个电极周期由未见过该周期的留一模型评分；这些历史周期此前已被查看，不是新增盲测。",
                "加工原点沿用离线确认的产品资格，不等同于现场异步数据全链路。"])
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    bundle = L.LifeBundle(out_dir)
    assert bundle.verify()
    seg = (bundle.times >= SEGMENT["start"]) & (bundle.times < SEGMENT["stop"])
    print("bundle", bundle.version, "origins in showcase", int(seg.sum()), "alarms there",
          [str(pd.Timestamp(int(t), unit="s")) for t in bundle.times[seg & bundle.emitted]])
    frame.assign(warn=ledger_warn, q10=quant[:, 0], q50=quant[:, 1], q90=quant[:, 2], p1h=horizons[:, 0], p4h=horizons[:, 1], p8h=horizons[:, 2])[
        ["anchor_id", "timestamp", "cycle_id", "warn", "q10", "q50", "q90", "p1h", "p4h", "p8h", "rul_h", "event"]].to_csv(out_dir.parent / "v3_predictions.csv.gz", index=False)


if __name__ == "__main__":
    main(sys.argv[1])
