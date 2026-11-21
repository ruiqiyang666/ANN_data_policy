"""Booth2 electrode survival models (v4): three selectable models from one bundle.

Every model answers the same question at each confirmed processing origin: how likely is
the electrode to reach its end of life within h processing hours, for twelve horizons from
20 minutes to 24 hours. From that one curve follow the remaining processing hours
(10 / 50 / 90 %), the risk within 1 / 4 / 8 hours, and with a separate warning score the
change-the-gun alarm.

  life_tree    gradient-boosted trees (monotone in the wear features)
  life_gru     a small recurrent network: GRU over two processing hours + the 23 features
  life_fusion  per-horizon weighted pool of five learners and the historical base rate

Members are plain NumPy here: exported tree nodes, logistic layers and GRU weights. The
pooling weights, the warning calibration, the interval widening and the alarm threshold of
a model come from predictions on electrode cycles it was not fitted on.

Each cycle of the recorded history is scored by the fold model that never saw it, so a
replay shows out-of-sample behaviour; the bundle also carries the model fitted on every
cycle, for data that is new. Nothing here claims a validated countdown: ten observed ends
of life support a wide range, and origins inherit the offline product eligibility.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from inference.electrode import replay_alarms, sha256
from inference.life import tree_scores

H = np.array([1 / 3, 2 / 3, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24.0])  # horizons, processing hours
CAP = 40.0  # the curve is closed at this many processing hours
QUANTILES = (0.1, 0.5, 0.9)
HORIZONS = (1.0, 4.0, 8.0)
HORIZON_INDEX = tuple(int(np.flatnonzero(np.isclose(H, h))[0]) for h in HORIZONS)
ONE_HOUR = HORIZON_INDEX[0]
BUCKETS = (1.0, 4.0)  # the band correction depends on whether the median is under 1 h, 1-4 h or longer
RISE_ALPHA, FADE_ALPHA = 0.3, 0.005  # risk follows the evidence within about 30 s and fades with a half-life of about 23 processing minutes
HOLD_SECONDS = 60  # the latest origin stays the current prediction between products
SEQ_STEPS, SEQ_STRIDE = 120, 6  # the recurrent member sees two processing hours, one point per minute
LOGIT_MEMBERS = ("logit3", "logit5", "logit23")
HORIZON_LABEL = ["20 分", "40 分", "1 时", "1.5 时", "2 时", "3 时", "4 时", "6 时", "8 时", "12 时", "16 时", "24 时"]
TREE_KEYS = ("feature", "threshold", "left", "right", "value", "leaf", "missing_left", "baseline", "depth")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def monotone(curve):
    return np.clip(np.maximum.accumulate(curve, axis=1), 0, 1)


def design(features):
    """The stored 23 features plus the logarithm of the 30-minute voltage ripple (index 23)."""
    x = np.asarray(features, dtype="float64")
    return np.column_stack([x, np.log(np.clip(x[:, 1], 0.02, None))])


def scale(x, median, spread):
    """Training-set median and interquartile scaling; a missing value sits at the median."""
    x = np.asarray(x, dtype="float64")
    return np.clip((np.where(np.isfinite(x), x, median) - median) / spread, -8, 8)


# ---------- members ----------
def logit_curve(member, x24):
    z = scale(x24[:, member["cols"]], member["median"], member["spread"])
    return monotone(sigmoid(z @ member["weight"].T + member["bias"]))


def tree_curve(trees, x23):
    """Boosted trees with the horizon as their last input: one pass per horizon."""
    return monotone(np.column_stack([tree_scores(trees, np.column_stack([x23, np.full(len(x23), h)])) for h in H]))


def sequences(z, first, positions):
    """Per origin: the last SEQ_STEPS points SEQ_STRIDE origins apart inside its cycle, zero-padded, plus a presence channel.

    z: scaled channels of every origin; first[i]: index of the first origin of the cycle of origin i.
    """
    positions = np.asarray(positions, dtype="int64")
    at = positions[:, None] - np.arange(SEQ_STEPS - 1, -1, -1)[None, :] * SEQ_STRIDE
    ok = at >= first[positions][:, None]
    out = np.zeros((len(positions), SEQ_STEPS, z.shape[1] + 1), dtype="float64")
    out[:, :, :-1] = np.where(ok[:, :, None], z[np.maximum(at, 0)], 0)
    out[:, :, -1] = ok
    return out


def gru_state(w, seq):
    """Last hidden state of a one-layer GRU, PyTorch gate order (reset, update, new). seq: N x T x C."""
    wi, wh, bi, bh = w["gru.weight_ih_l0"], w["gru.weight_hh_l0"], w["gru.bias_ih_l0"], w["gru.bias_hh_l0"]
    size = wh.shape[1]
    h = np.zeros((seq.shape[0], size))
    for step in range(seq.shape[1]):
        gi, gh = seq[:, step] @ wi.T + bi, h @ wh.T + bh
        r = sigmoid(gi[:, :size] + gh[:, :size])
        u = sigmoid(gi[:, size:2 * size] + gh[:, size:2 * size])
        n = np.tanh(gi[:, 2 * size:] + r * gh[:, 2 * size:])
        h = (1 - u) * n + u * h
    return h


def net_forward(net, x23, seq):
    """Mean logits of the seed networks (12 horizons + the warning) and the hidden layer of the first seed.

    Each network: [GRU last state, scaled features] -> one linear layer.
    """
    feat = scale(x23, net["median"], net["spread"])
    total, shown = 0.0, None
    for w in net["seeds"]:
        hidden = np.concatenate([gru_state(w, seq), feat], axis=1)
        total = total + hidden @ w["head.weight"].T + w["head.bias"]
        if shown is None:
            shown = hidden
    return total / len(net["seeds"]), shown


def member_outputs(model, names, x23, z_channels=None, first=None, positions=None):
    """Curves (and warning scores) of the named members for origins `positions` of the full arrays."""
    positions = np.arange(len(x23)) if positions is None else np.asarray(positions)
    rows = np.asarray(x23, dtype="float64")[positions]
    x24 = design(rows)
    out = {}
    for name in names:
        if name in LOGIT_MEMBERS:
            out[name] = logit_curve(model[name], x24)
        elif name == "hgb":
            out["hgb"] = tree_curve(model["hgb"]["cdf"], rows)
            out["warn_hgb"] = tree_scores(model["hgb"]["warn"], rows)
        elif name == "net":
            raw, hidden = net_forward(model["net"], rows, sequences(z_channels, first, positions))
            out["net"], out["warn_net"], out["hidden_net"] = monotone(sigmoid(raw[:, :-1])), sigmoid(raw[:, -1]), hidden
    return out


# ---------- one curve -> every output ----------
def hold(curve, rise=RISE_ALPHA, fade=FADE_ALPHA):
    """Causal smoothing of the curve along time: risk rises quickly and fades slowly.

    Electrode wear does not heal, so evidence that the end is near is not dropped the moment the voltage
    recovers; it also keeps the remaining-life reading from flipping when the curve is flat around 50 %.
    """
    x = logit(curve)
    out = np.empty_like(x)
    state = x[0] if len(x) else None
    for i in range(len(x)):
        state = state + np.where(x[i] > state, rise, fade) * (x[i] - state)
        out[i] = state
    return monotone(sigmoid(out))


def pool(curves, weights, base):
    """Per-horizon weighted mix of the member curves and the historical base rate (weights sum to 1 per horizon)."""
    out = weights[:, 0] * base
    for k, curve in enumerate(curves):
        out = out + weights[:, k + 1] * curve
    return monotone(out)


def lift(curve, p_near):
    """Make P(end within one hour) at least the calibrated warning.

    When the warning is the higher of the two, the first hour is scaled up to it; beyond the first hour
    the curve keeps the model's own conditional shape: the chance of ending by h, given that the hour
    was survived, is unchanged.
    """
    one = curve[:, ONE_HOUR]
    target = np.maximum(p_near, one)
    out = curve.copy()
    early = curve[:, :ONE_HOUR + 1] * (target / np.maximum(one, 1e-6))[:, None]
    out[:, :ONE_HOUR + 1] = np.minimum(early, target[:, None])
    out[:, ONE_HOUR] = target
    out[:, ONE_HOUR + 1:] = 1 - (1 - curve[:, ONE_HOUR + 1:]) * ((1 - target) / np.maximum(1 - one, 1e-6))[:, None]
    return monotone(out)


def quantiles(curve, qs=QUANTILES):
    """Remaining-life quantiles of a curve given at the horizons; 0 at zero hours, 1 at CAP, linear in between."""
    edges = np.r_[0.0, H, CAP]
    full = np.column_stack([np.zeros(len(curve)), monotone(curve), np.ones(len(curve))])
    rows = np.arange(len(curve))
    out = np.zeros((len(curve), len(qs)))
    for j, q in enumerate(qs):
        k = np.clip((full < q).sum(1) - 1, 0, len(edges) - 2)
        lo, hi = full[rows, k], full[rows, k + 1]
        out[:, j] = edges[k] + np.clip((q - lo) / np.maximum(hi - lo, 1e-9), 0, 1) * (edges[k + 1] - edges[k])
    return out


def widen(q, factors):
    """Move each end of the 10-90 % band outwards in log space (conformalised quantiles).

    factors: one [lower, upper] pair per size of the median, never negative: the band is only ever widened.
    """
    f = np.asarray(factors, dtype="float64").reshape(-1, 2)[np.digitize(q[:, 1], BUCKETS)]
    out = q.copy()
    out[:, 0] = np.minimum(q[:, 0] * np.exp(-f[:, 0]), q[:, 1])
    out[:, 2] = np.maximum(np.minimum(q[:, 2] * np.exp(f[:, 1]), CAP), q[:, 1])
    return out


def finish(curves, warn, params, raw=False):
    """All outputs for the origins of one electrode cycle, in time order.

    curves: member curves in the model's member order; warn: the model's warning score.
    `raw` stops before the widening, for the code that estimates it.
    """
    curve = pool(curves, params["pool"], params["base"])
    p_near = np.interp(warn, params["cal_x"], params["cal_y"])
    curve = hold(lift(curve, p_near))
    q = quantiles(curve)
    if not raw:
        q = widen(q, params["widen"])
    return dict(curve=curve, quantiles=q, horizons=curve[:, list(HORIZON_INDEX)], p_near=p_near)


def warning(outputs, names):
    """A model's warning score: the mean of its members' warning scores, in logit space."""
    return sigmoid(np.mean([logit(outputs[f"warn_{name}"]) for name in names], axis=0))


# ---------- bundle ----------
def load_models(path, members):
    """weights.npz -> list of fold models; the last one is fitted on every cycle."""
    with np.load(path, allow_pickle=False) as stored:
        files = set(stored.files)
        models = []
        for k in range(int(stored["models"])):
            get = lambda key, k=k: np.asarray(stored[f"m{k}_{key}"], dtype="float64")  # noqa: E731
            model = {}
            for name in members:
                if name in LOGIT_MEMBERS:
                    model[name] = dict(cols=np.asarray(stored[f"m{k}_{name}_cols"], dtype="int64"), median=get(f"{name}_median"),
                                       spread=get(f"{name}_spread"), weight=get(f"{name}_weight"), bias=get(f"{name}_bias"))
                elif name == "hgb":
                    model["hgb"] = {part: {key: stored[f"m{k}_hgb_{part}_{key}"] for key in TREE_KEYS} for part in ("cdf", "warn")}
                elif name == "net":
                    seeds = []
                    for s in range(int(stored[f"m{k}_net_seeds"])):
                        prefix = f"m{k}_net{s}_"
                        seeds.append({key[len(prefix):]: np.asarray(stored[key], dtype="float64") for key in files if key.startswith(prefix)})
                    model["net"] = dict(median=get("net_median"), spread=get("net_spread"), seq_median=get("net_seq_median"),
                                        seq_spread=get("net_seq_spread"), seeds=seeds)
            model["params"] = {}
            for key in files:
                prefix = f"m{k}_post_"
                if key.startswith(prefix):
                    model_id, name = key[len(prefix):].rsplit("__", 1)
                    model["params"].setdefault(model_id, {})[name] = np.asarray(stored[key], dtype="float64")
            models.append(model)
    return models


class SurvivalBundle:
    """Validated, read-only access to one exported v4 bundle; `models` holds the selectable models."""

    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.manifest = json.loads((self.folder / "manifest.json").read_text(encoding="utf-8"))
        m = self.manifest
        if m.get("kind") != "survival_v4" or list(m["horizons_hours"]) != [float(h) for h in H]:
            raise ValueError("Not a v4 survival bundle with the horizons this panel reads")
        self.version = m["version"]
        self.features, self.members = list(m["features"]), list(m["members"])
        self.fold_models = load_models(self.folder / "weights.npz", self.members)
        with np.load(self.folder / "origins.npz") as stored:
            self.times = np.asarray(stored["t"], dtype="int64")  # epoch seconds of the instrument clock
            self.cycles = np.asarray(stored["cycle"], dtype="int16")
            self.x = np.asarray(stored["features"], dtype="float32")
            self.channels = np.asarray(stored["channels"], dtype="float32")
            self.age = np.asarray(stored["age_h"], dtype="float64")
            self.ledger = {key: np.asarray(stored[key], dtype="float64") for key in stored.files if key.startswith("ledger_")}
        if not (np.all(np.diff(self.times) > 0) and np.all(np.diff(self.cycles) >= 0)):
            raise ValueError("Origins must be chronological and grouped by electrode cycle")
        self.cycle_ids = list(m["cycle_ids"])
        if len(self.fold_models) != len(self.cycle_ids) + 1 or self.x.shape[1] != len(self.features):
            raise ValueError("One held-out model per cycle and one final model are required")
        self.first = np.searchsorted(self.cycles, self.cycles, side="left")  # first origin of each origin's cycle
        self.policy = dict(m["alarm_policy"])
        self.rate = float(m["products_per_processing_hour"])
        self.segments = [dict(segment) for segment in m["showcase_segments"]]  # recorded end-of-life mornings, replayed in turn
        self.segment = self.segments[0]
        self.lock = threading.Lock()
        self.cache = {}
        self.check_members()
        self.models = {spec["id"]: SurvivalModel(self, spec) for spec in m["models"]}

    def scaled_channels(self, model, positions):
        """Scaled recurrent inputs for the given origins and everything they look back on."""
        net = model["net"]
        return scale(self.channels, net["seq_median"], net["seq_spread"])

    def members_at(self, k, positions):
        """Member outputs of fold model k for the given origins, recomputed from the exported weights."""
        model = self.fold_models[k]
        z = self.scaled_channels(model, positions) if "net" in self.members else None
        return member_outputs(model, self.members, self.x, z, self.first, positions)

    def origin(self, i):
        """Member outputs of origin i by its held-out model, recomputed now and checked against the ledger."""
        with self.lock:
            hit = self.cache.get(i)
        if hit is not None:
            return hit
        out = self.members_at(int(self.cycles[i]), [i])
        for name, value in out.items():
            if not name.startswith("hidden_") and float(np.abs(value[0] - self.ledger[f"ledger_{name}"][i]).max()) > 2e-4:
                raise RuntimeError(f"Survival member {name} differs from the training ledger at origin {i}")
        with self.lock:
            if len(self.cache) > 256:
                self.cache.clear()
            self.cache[i] = out
        return out

    def check_members(self):
        """Recompute a spread of origins at start-up; they must equal the training ledger.

        Every origin that is shown is recomputed again when it is requested, see `origin`.
        """
        sample = np.arange(0, len(self.times), 199)
        worst = 0.0
        for k in range(len(self.cycle_ids)):
            at = sample[self.cycles[sample] == k]
            if not len(at):
                continue
            out = self.members_at(k, at)
            for name, value in out.items():
                if name.startswith("hidden_"):
                    continue
                worst = max(worst, float(np.abs(value - self.ledger[f"ledger_{name}"][at]).max()))
        if worst > 2e-4:
            raise RuntimeError(f"Survival members differ from the training ledger (max {worst:.2e})")
        self.member_error = worst

    def verify(self):
        for name, expected in self.manifest["files"].items():
            if sha256(self.folder / name) != expected:
                raise RuntimeError(f"Bundle artifact changed: {name}")
        return True

    def workbench(self):
        return json.loads((self.folder / "workbench.json").read_text(encoding="utf-8"))


class SurvivalModel:
    """One selectable model of a v4 bundle; the interface matches the other replay models."""

    def __init__(self, bundle, spec):
        self.bundle, self.spec = bundle, dict(spec)
        self.id, self.version = spec["id"], f"{bundle.version}:{spec['id']}"
        self.members, self.warn_members = list(spec["members"]), list(spec["warn"])
        self.segment, self.segments = bundle.segment, bundle.segments
        b, n = bundle, len(bundle.times)
        self.warn, self.q, self.curve = np.zeros(n), np.zeros((n, 3)), np.zeros((n, len(H)))
        self.horizons, self.threshold = np.zeros((n, 3)), np.zeros(n)
        self.emitted, self.active = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        self.confirmations, self.last_alarm = np.zeros(n, dtype="int16"), np.full(n, -1, dtype="int64")
        for k in range(len(b.cycle_ids)):
            at = np.flatnonzero(b.cycles == k)
            if not len(at):
                continue
            params = b.fold_models[k]["params"][self.id]
            ledger = {f"warn_{name}": b.ledger[f"ledger_warn_{name}"][at] for name in self.warn_members}
            warn = warning(ledger, self.warn_members)
            out = finish([b.ledger[f"ledger_{name}"][at] for name in self.members], warn, params)
            self.warn[at], self.q[at], self.curve[at], self.horizons[at] = warn, out["quantiles"], out["curve"], out["horizons"]
            self.threshold[at] = float(params["threshold"])
            e, a, c, last = replay_alarms(b.times[at], warn, b.cycles[at], float(params["threshold"]), b.policy)
            self.emitted[at], self.active[at], self.confirmations[at], self.last_alarm[at] = e, a, c, last
        if float(np.abs(self.q - b.ledger[f"ledger_q_{self.id}"]).max()) > 1e-6:
            raise RuntimeError(f"{self.id}: remaining-life outputs differ from the training ledger")

    def verify(self):
        return self.bundle.verify()

    def workbench(self):
        return self.bundle.workbench()

    def summary(self):
        m, spec = self.bundle.manifest, self.spec
        return {
            "id": self.id, "kind": "survival", "version": self.version, "bundle": self.bundle.version,
            "name": spec["name"], "name_en": spec["name_en"], "short": spec["short"], "short_en": spec["short_en"],
            "describe": spec["describe"], "describe_en": spec["describe_en"], "family": spec["family"],
            "source": m["source"], "features": len(self.bundle.features), "sensors": int(m["sensors"]),
            "members": self.members, "architecture": spec.get("architecture"),
            "final_threshold": float(self.bundle.fold_models[-1]["params"][self.id]["threshold"]),
            "alarm_policy": self.bundle.policy, "origin_seconds": 10, "hold_seconds": HOLD_SECONDS,
            "horizons": list(HORIZONS), "curve_hours": [float(h) for h in H], "cap_hours": CAP,
            "interval": [QUANTILES[0], QUANTILES[2]], "products_per_processing_hour": self.bundle.rate,
            "segment": dict(name=self.segment["name"], label=self.segment["label"]),
            "segments": [dict(name=segment["name"], label=segment["label"]) for segment in self.segments],
            "limits": m["limits"], "evaluation": m["evaluation"][self.id], "baselines": m["evaluation"]["baselines"],
        }

    def position(self, stamp):
        """Latest processing origin at or before `stamp`, at most HOLD_SECONDS old."""
        i = int(np.searchsorted(self.bundle.times, stamp, side="right")) - 1
        if i < 0 or stamp - int(self.bundle.times[i]) > HOLD_SECONDS:
            return None
        return i

    def outputs(self, stamp):
        """Prediction block for one replayed second, or the reason there is none."""
        b, i = self.bundle, self.position(stamp)
        if i is None:
            return dict(status="idle", reason="当前不在加工原点：模型只在确认加工的 10 秒原点更新",
                        model_version=self.version, model_id=self.id)
        k = int(b.cycles[i])
        origin, last, active = int(b.times[i]), int(self.last_alarm[i]), bool(self.active[i])
        q, threshold, score = self.q[i], float(self.threshold[i]), float(self.warn[i])
        params = b.fold_models[k]["params"][self.id]
        reference = b.fold_models[k].get("logit23") or b.fold_models[k].get("net")
        z = scale(b.x[i], reference["median"][:len(b.features)], reference["spread"][:len(b.features)]) if reference is not None else np.zeros(len(b.features))
        order = np.argsort(-np.abs(z))[:6]
        live = b.origin(i)  # recomputed from the exported weights for this request
        members = [dict(name=name, curve=[float(v) for v in live[name][0]],
                        weight=[float(v) for v in params["pool"][:, 1 + j]]) for j, name in enumerate(self.members)]
        return dict(
            status="ok", reason="", model_version=self.version, model_id=self.id, model_name=self.spec["name"],
            score=score, threshold=threshold, above=bool(score >= threshold),
            alarm_active=active, alarm_emitted=bool(self.emitted[i]),
            confirmations=int(self.confirmations[i]), confirm_needed=int(b.policy["confirm_origins"]),
            alarm_at=pd.Timestamp(last, unit="s").isoformat() if active and last >= 0 else None,
            alarm_age_s=int(stamp - last) if active and last >= 0 else None,
            rul_hours=float(q[1]), rul_interval_hours=[float(q[0]), float(q[2])],
            rul_pieces=int(q[1] * b.rate), rul_interval_pieces=[int(q[0] * b.rate), int(q[2] * b.rate)],
            horizons=[dict(hours=h, probability=float(p)) for h, p in zip(HORIZONS, self.horizons[i])],
            curve=[float(v) for v in self.curve[i]], base=[float(v) for v in params["base"]],
            base_weight=[float(v) for v in params["pool"][:, 0]], members=members,
            age_hours=float(b.age[i]), age_pieces=int(b.age[i] * b.rate),
            origin=pd.Timestamp(origin, unit="s").isoformat(), origin_age_s=int(stamp - origin),
            cycle_id=b.cycle_ids[k], held_out=True,
            inputs=[dict(key=b.features[j], value=float(b.x[i, j]) if np.isfinite(b.x[i, j]) else None, z=float(z[j])) for j in order],
        )

    def neural_payload(self, stamp, seq):
        """The recurrent model as it is: [GRU state, scaled features] -> twelve horizon units.

        Left layer and weights are those of the first of the seed networks, exactly; the right layer is the
        model's curve before pooling, the mean of the seed networks. The other models are not neural
        networks and return None; the page draws their curve instead.
        """
        if self.members != ["net"]:
            return None
        b, i = self.bundle, self.position(stamp)
        if i is None:
            return None
        model = b.fold_models[int(b.cycles[i])]
        out = b.origin(i)
        hidden = out["hidden_net"][0]
        w = model["net"]["seeds"][0]
        size = w["gru.weight_hh_l0"].shape[1]
        names = ["状态与特征", "分段风险"]
        head = w["head.weight"][:-1]  # the twelve horizon units; the last row is the warning
        width = len(hidden)
        layers = [
            dict(name=names[0], units=width, total_units=width, unit_indices=list(range(width)),
                 labels=[f"GRU {j + 1}" for j in range(size)] + list(b.features)),
            dict(name=names[1], units=len(H), total_units=len(H), unit_indices=list(range(len(H))), labels=list(HORIZON_LABEL)),
        ]
        return dict(model_type="neural_network", real_neuron_activations_available=True, recurrent=True,
                    input_parameter_count=int(b.manifest["sensors"]), input_feature_count=width,
                    tasks=["life"], topology="recurrent_plus_features", heads=[], sample_id=seq,
                    prediction_id=f"{self.version}:{int(b.times[i])}",
                    graph=dict(layers=layers, connections=[{"from": 0, "to": 1, "weights": head.tolist()}]),
                    activation_layers=[dict(name=names[0], values=hidden.tolist()),
                                       dict(name=names[1], values=[float(v) for v in out["net"][0]])])
