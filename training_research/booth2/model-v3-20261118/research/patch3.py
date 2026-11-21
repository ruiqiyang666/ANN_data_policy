from pathlib import Path
import sys


def sub(path, pairs):
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert s.count(old) == 1, (str(path), old[:70], s.count(old))
        s = s.replace(old, new)
    p.write_bytes(s.encode("utf-8"))


sub(Path(sys.argv[1]) / "simulation/pipeline/inference/life.py", [
("""def fuse(logit, p_warn, alpha):
    \"\"\"Survival curve per origin (N x 14, at the bin edges) from hazard logits and the warning.

    Two causal readings of the same hazard layer are combined by taking the lower
    survival at every edge: a slow one (heavily smoothed, for the hours ahead) and
    a fast one (lightly smoothed and lifted by the warning, for the last hour).
    The lower envelope of two survival curves is itself a survival curve.
    \"\"\"
    slow = survival(sigmoid(ema(logit, alpha)))
    fast = survival(lift(sigmoid(ema(logit, FAST_ALPHA)), p_warn))
    return np.minimum(slow, fast)""", """MODES = ("fast", "slow", "envelope")


def fuse(logit, p_warn, alpha, mode=2):
    \"\"\"Survival curve per origin (N x 14, at the bin edges) from hazard logits and the warning.

    Three causal readings of the same hazard layer, one of which a model uses:
      fast      lightly smoothed hazards, lifted by the warning
      slow      heavily smoothed hazards, lifted by the warning
      envelope  the lower of the slow curve and the fast curve at every edge (the
                lower envelope of two survival curves is itself a survival curve)
    The reading is chosen per model from predictions on cycles it was not fitted on.
    \"\"\"
    fast = survival(lift(sigmoid(ema(logit, FAST_ALPHA)), p_warn))
    if MODES[int(mode)] == "fast":
        return fast
    slow = sigmoid(ema(logit, alpha))
    if MODES[int(mode)] == "slow":
        return survival(lift(slow, p_warn))
    return np.minimum(survival(slow), fast)"""),
("""    s = fuse(logit, p_warn, alpha)
    q = quantiles(s)""", """    s = fuse(logit, p_warn, alpha, model["mode"])
    q = quantiles(s)"""),
("""        plain = ("median", "spread", "weight", "bias", "calibration_x", "calibration_y", "widen", "threshold")""",
 """        plain = ("median", "spread", "weight", "bias", "calibration_x", "calibration_y", "widen", "threshold", "mode")"""),
])

sub(Path(sys.argv[2]) / "research/train_v3.py", [
("""def fit_hazard(x, rul, event, seed=0):
    \"\"\"Logistic hazard layer: one sigmoid unit per time bin, masked after the end / after censoring.\"\"\"
    import torch
    torch.set_num_threads(1)
    sc = R.Scaler(x)
    net = R.fit_hazard_mlp(sc(x), rul, event, hidden=(), epochs=40, seed=seed, wd=1e-2)
    layer = net[0]
    return dict(median=sc.med.astype("float64"), spread=sc.scale.astype("float64"),
                weight=layer.weight.detach().numpy().astype("float64"), bias=layer.bias.detach().numpy().astype("float64"))
""", """def fit_hazard(x, rul, event, c):
    \"\"\"Logistic hazard layer: one sigmoid unit per time bin, fitted to convergence.

    Unit k sees the origins still at risk at the start of bin k (masked after the
    end / after censoring) and learns whether the end falls inside the bin. The
    units share no parameters, so each is an L2-penalised logistic regression; the
    intercept is not penalised, which keeps the base rate of every bin right.
    \"\"\"
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
    \"\"\"Masked discrete-time negative log-likelihood per origin.\"\"\"
    y, w = R.hazard_targets(rul, event)
    loss = np.logaddexp(0, logit) - y * logit
    return float((loss * w).sum() / len(rul))
"""),
("""def fit_pair(frame, cycles):
    \"\"\"Warning trees and hazard layer on the given cycles.\"\"\"
    w = frame[frame.cycle_id.isin(cycles) & frame[TARGET].notna()]
    warn = fit_warn(w[FEATURES].to_numpy("float32"), w[TARGET].to_numpy().astype(int))
    lt = frame[frame.cycle_id.isin(cycles) & frame.rul_h.notna()].iloc[::6]  # one origin per processing minute
    hazard = fit_hazard(lt[FEATURES].to_numpy("float32").astype("float64"), lt.rul_h.to_numpy(), lt.event.to_numpy().astype(bool))
    return warn, hazard
""", """def fit_pair(frame, cycles, penalties):
    \"\"\"Warning trees and one hazard layer per candidate penalty on the given cycles.\"\"\"
    w = frame[frame.cycle_id.isin(cycles) & frame[TARGET].notna()]
    warn = fit_warn(w[FEATURES].to_numpy("float32"), w[TARGET].to_numpy().astype(int))
    lt = frame[frame.cycle_id.isin(cycles) & frame.rul_h.notna()].iloc[::6]  # one origin per processing minute
    x, rul, event = lt[FEATURES].to_numpy("float32").astype("float64"), lt.rul_h.to_numpy(), lt.event.to_numpy().astype(bool)
    return warn, {c: fit_hazard(x, rul, event, c) for c in penalties}
"""),
("""    model = dict(calibration_x=iso.X_thresholds_.astype("float64"), calibration_y=iso.y_thresholds_.astype("float64"),
                 threshold=np.array(float(picked["threshold"])), widen=np.array(0.0))
""", """    model = dict(calibration_x=iso.X_thresholds_.astype("float64"), calibration_y=iso.y_thresholds_.astype("float64"),
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
"""),
("""    return L.quantiles(L.fuse(logit, np.interp(warn, model["calibration_x"], model["calibration_y"]), ALPHA))""",
 """    return L.quantiles(L.fuse(logit, np.interp(warn, model["calibration_x"], model["calibration_y"]), ALPHA, model["mode"]))"""),
("""    inner_warn, inner_logit = np.full(len(frame), np.nan), np.full((len(frame), len(L.EDGES) - 1), np.nan)
    groups = np.array(cycles)
    for a, b in GroupKFold(5).split(groups, groups=groups):
        warn, hazard = fit_pair(frame, groups[a])
        q = frame[frame.cycle_id.isin(groups[b])]
        x = q[FEATURES].to_numpy("float32")
        inner_warn[q.index], inner_logit[q.index] = warn_predict(warn, x), hazard_logit(hazard, x)
    chosen, picked = choose(frame, cycles, inner_warn, inner_logit, events, products)
    warn, hazard = fit_pair(frame, cycles)
    q = frame[frame.cycle_id.eq(held)]
    x = q[FEATURES].to_numpy("float32")
    model = dict(trees=export_trees(warn), **hazard, **chosen)
    return held, model, q.index.to_numpy(), warn_predict(warn, x), hazard_logit(hazard, x), picked["metrics"]
""", """    inner_warn = np.full(len(frame), np.nan)
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
"""),
("""    chosen, picked = choose(frame, cycles, warn, logit, events, products)
    final_warn, final_hazard = fit_pair(frame, cycles)
    final = dict(trees=export_trees(final_warn), **final_hazard, **chosen)""",
 """    chosen, picked = choose(frame, cycles, warn, logit, events, products)
    votes = [inner_metrics[c]["penalty"] for c in cycles]
    penalty = max(sorted(set(votes)), key=votes.count)  # the penalty most held-out models chose
    final_warn, final_hazards = fit_pair(frame, cycles, [penalty])
    final = dict(trees=export_trees(final_warn), **final_hazards[penalty], **chosen)"""),
("""    fw, _ = fit_pair(frame, TR)
""", """    fw, _ = fit_pair(frame, TR, [])
"""),
("""    print("widen", {c: round(float(models[c]["widen"]), 3) for c in cycles}, "final", round(float(final["widen"]), 3))""",
 """    print("widen", {c: round(float(models[c]["widen"]), 3) for c in cycles}, "final", round(float(final["widen"]), 3))
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
    print(pd.DataFrame(horizon_rows).round(4).to_string(index=False))"""),
("""        cycles=per_cycle, nominal_interval=NOMINAL)""",
 """        cycles=per_cycle, nominal_interval=NOMINAL, horizons=horizon_rows, hazard_penalty=penalty, life_mode=L.MODES[int(final["mode"])])"""),
("""        inner_selection={c: {k: inner_metrics[c][k] for k in ("hit20", "false_alarms", "false_alarms_per_hour", "threshold")} for c in cycles})""",
 """        inner_selection={c: {k: inner_metrics[c][k] for k in ("hit20", "false_alarms", "false_alarms_per_hour", "threshold", "penalty", "mode")} for c in cycles})"""),
("""BUDGET = 0.1  # false alarms per production hour allowed when a threshold is chosen""",
 """BUDGET = 0.1  # false alarms per production hour allowed when a threshold is chosen
PENALTIES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1)  # inverse L2 strength of the hazard layer, chosen inside each fold"""),
])
print("patched")
