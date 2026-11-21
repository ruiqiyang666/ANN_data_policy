"""Booth2 electrode life and warning model (v3).

A third, independent model on the same recorded replay. One set of causal
features per confirmed processing origin feeds two small heads:

  warning  gradient-boosted trees -> score that the electrode ends within an hour
  life     a logistic hazard layer over 13 time bins -> survival curve, from which
           the remaining processing hours (10 / 50 / 90 %) and horizon risks follow

The warning lifts the first-hour hazards when it is the higher of the two, so the
life estimate cannot contradict an active warning. Everything is NumPy: the trees
are exported node arrays, not pickles.

Each electrode cycle of the recorded history is scored by the fold model that never
saw that cycle (leave-one-cycle-out), so a replay shows out-of-sample behaviour.
The bundle also carries the model fitted on every cycle, for data that is new.

Nothing here claims a live interface or a validated countdown: with ten observed
ends of life the remaining-life range is wide, and origins inherit the offline
product eligibility.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from inference.electrode import replay_alarms, sha256

EDGES = np.array([0, 1 / 3, 2 / 3, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 40.0])  # hazard bins, processing hours
FIRST_HOUR = 3  # bins that end at one processing hour
QUANTILES = (0.1, 0.5, 0.9)
HORIZONS = (1.0, 4.0, 8.0)
HOLD_SECONDS = 60  # the latest origin stays the current prediction between products
BIN_LABEL = ["0–20 分", "20–40 分", "40–60 分", "1–1.5 时", "1.5–2 时", "2–3 时", "3–4 时", "4–6 时", "6–8 时",
             "8–12 时", "12–16 时", "16–24 时", "24–40 时"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def tree_scores(trees, x):
    """Mean probability of the boosted-tree ensemble. trees: arrays of shape M x T x K nodes.

    The split rule is scikit-learn's: left when the value is at most the threshold,
    a missing value follows `missing_left`.
    """
    x = np.asarray(x, dtype="float64")
    feature, threshold = trees["feature"], trees["threshold"]
    left, right, value = trees["left"], trees["right"], trees["value"]
    leaf, missing_left = trees["leaf"], trees["missing_left"]
    models, count, _ = feature.shape
    rows = np.arange(len(x))
    total = np.zeros(len(x))
    for m in range(models):
        raw = np.full(len(x), float(trees["baseline"][m]))
        for t in range(count):
            node = np.zeros(len(x), dtype="int64")
            for _ in range(int(trees["depth"])):
                v = x[rows, feature[m, t, node]]
                go_left = np.where(np.isnan(v), missing_left[m, t, node], v <= threshold[m, t, node])
                node = np.where(leaf[m, t, node], node, np.where(go_left, left[m, t, node], right[m, t, node]))
            raw += value[m, t, node]
        total += sigmoid(raw)
    return total / models


def scale(x, median, spread):
    """Training-set median and interquartile scaling; a missing value sits at the median."""
    x = np.asarray(x, dtype="float64")
    return np.clip((np.where(np.isfinite(x), x, median) - median) / spread, -8, 8)


def ema(x, alpha):
    """Causal exponential smoothing along the first axis, started at the first value."""
    x = np.asarray(x, dtype="float64")
    if len(x) == 0:
        return x
    y, _ = lfilter([alpha], [1, alpha - 1], x, axis=0, zi=((1 - alpha) * x[:1]))
    return y


FAST_ALPHA = 0.05  # the fast path follows the hazard layer with a time constant of about three processing minutes


def lift(hazard, p_warn):
    """Raise the first-hour hazards so P(end within one hour) is at least the warning's."""
    p_first = 1 - np.prod(1 - hazard[:, :FIRST_HOUR], axis=1)
    gamma = np.log(np.clip(1 - p_warn, 1e-6, 1)) / np.log(np.clip(1 - p_first, 1e-6, 1 - 1e-9))
    gamma = np.where(p_warn > p_first, np.maximum(gamma, 1), 1)
    out = hazard.copy()
    out[:, :FIRST_HOUR] = 1 - (1 - hazard[:, :FIRST_HOUR]) ** gamma[:, None]
    return out


def survival(hazard):
    return np.concatenate([np.ones((len(hazard), 1)), np.cumprod(1 - hazard, axis=1)], axis=1)


MODES = ("fast", "slow", "envelope")


def fuse(logit, p_warn, alpha, mode=2):
    """Survival curve per origin (N x 14, at the bin edges) from hazard logits and the warning.

    Three causal readings of the same hazard layer, one of which a model uses:
      fast      lightly smoothed hazards, lifted by the warning
      slow      heavily smoothed hazards, lifted by the warning
      envelope  the lower of the slow curve and the fast curve at every edge (the
                lower envelope of two survival curves is itself a survival curve)
    The reading is chosen per model from predictions on cycles it was not fitted on.
    """
    fast = survival(lift(sigmoid(ema(logit, FAST_ALPHA)), p_warn))
    if MODES[int(mode)] == "fast":
        return fast
    slow = sigmoid(ema(logit, alpha))
    if MODES[int(mode)] == "slow":
        return survival(lift(slow, p_warn))
    return np.minimum(survival(slow), fast)


def quantiles(s, qs=QUANTILES):
    """Residual-life quantiles of survival curves; survival is taken as linear inside a bin."""
    rows = np.arange(len(s))
    out = np.zeros((len(s), len(qs)))
    for j, q in enumerate(qs):
        k = np.minimum((s[:, 1:] > 1 - q).sum(1), s.shape[1] - 2)
        hi, lo = s[rows, k], s[rows, k + 1]
        part = np.clip((hi - (1 - q)) / np.maximum(hi - lo, 1e-9), 0, 1)
        out[:, j] = EDGES[k] + part * (EDGES[k + 1] - EDGES[k])
    return out


def horizon_risk(s, hours=HORIZONS):
    """P(end of life within each horizon); the horizons are bin edges."""
    return np.column_stack([1 - s[:, int(np.searchsorted(EDGES, h))] for h in hours])


def run_cycle(model, x, alpha, warn=None, logit=None):
    """All outputs for the origins of one electrode cycle, in time order.

    `warn` and `logit` may be supplied by the training code, which scores the same
    models through scikit-learn and PyTorch; the panel always computes both here.
    """
    warn = tree_scores(model["trees"], x) if warn is None else np.asarray(warn, dtype="float64")
    z = scale(x, model["median"], model["spread"])
    logit = z @ model["weight"].T + model["bias"] if logit is None else np.asarray(logit, dtype="float64")
    p_warn = np.interp(warn, model["calibration_x"], model["calibration_y"])
    s = fuse(logit, p_warn, alpha, model["mode"])
    q = quantiles(s)
    widen = float(model["widen"])
    q[:, 0], q[:, 2] = q[:, 0] * np.exp(-widen), np.minimum(q[:, 2] * np.exp(widen), EDGES[-1])
    fused = 1 - s[:, 1:] / np.maximum(s[:, :-1], 1e-12)  # the hazards of the combined curve
    return dict(warn=warn, hazard=sigmoid(logit), hazard_fused=fused, p_warn=p_warn,
                quantiles=q, horizons=horizon_risk(s), z=z)


def load_models(path):
    """weights.npz -> list of fold models (the last one is fitted on every cycle)."""
    with np.load(path) as stored:
        count = int(stored["models"])
        tree_keys = ("feature", "threshold", "left", "right", "value", "leaf", "missing_left", "baseline", "depth")
        plain = ("median", "spread", "weight", "bias", "calibration_x", "calibration_y", "widen", "threshold", "mode")
        return [dict(trees={key: stored[f"m{k}_tree_{key}"] for key in tree_keys},
                     **{key: stored[f"m{k}_{key}"] for key in plain}) for k in range(count)]


class LifeBundle:
    """Validated, read-only access to one exported life-and-warning bundle."""

    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.manifest = json.loads((self.folder / "manifest.json").read_text(encoding="utf-8"))
        self.version = self.manifest["version"]
        self.features = list(self.manifest["features"])
        self.models = load_models(self.folder / "weights.npz")
        with np.load(self.folder / "origins.npz") as stored:
            self.times = np.asarray(stored["t"], dtype="int64")  # epoch seconds of the instrument clock
            self.cycles = np.asarray(stored["cycle"], dtype="int16")
            self.x = np.asarray(stored["features"], dtype="float32")
            self.age = np.asarray(stored["age_h"], dtype="float64")
            ledger = {key: np.asarray(stored[key], dtype="float64") for key in ("ledger_warn", "ledger_quantiles")}
        if not (np.all(np.diff(self.times) > 0) and np.all(np.diff(self.cycles) >= 0)):
            raise ValueError("Origins must be chronological and grouped by electrode cycle")
        if self.x.shape[1] != len(self.features) or self.models[0]["weight"].shape != (len(EDGES) - 1, len(self.features)):
            raise ValueError("Feature matrix does not match the exported model")
        self.cycle_ids = list(self.manifest["cycle_ids"])
        if len(self.models) != len(self.cycle_ids) + 1:
            raise ValueError("One held-out model per cycle and one final model are required")
        self.alpha = float(self.manifest["smoothing_alpha"])
        self.policy = dict(self.manifest["alarm_policy"])
        self.rate = float(self.manifest["products_per_processing_hour"])
        self.segment = dict(self.manifest["showcase_segment"])
        # Recomputed here from the exported weights; the stored ledger only proves the two agree.
        n = len(self.times)
        self.warn, self.q = np.zeros(n), np.zeros((n, 3))
        self.horizons, self.hazard, self.z = np.zeros((n, 3)), np.zeros((n, len(EDGES) - 1)), np.zeros(self.x.shape)
        self.threshold = np.zeros(n)
        self.emitted, self.active = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        self.confirmations, self.last_alarm = np.zeros(n, dtype="int16"), np.full(n, -1, dtype="int64")
        for k in range(len(self.cycle_ids)):
            at = np.flatnonzero(self.cycles == k)
            if not len(at):
                continue
            model = self.models[k]
            out = run_cycle(model, self.x[at], self.alpha)
            self.warn[at], self.q[at], self.horizons[at] = out["warn"], out["quantiles"], out["horizons"]
            self.hazard[at], self.z[at] = out["hazard_fused"], out["z"]
            self.threshold[at] = float(model["threshold"])
            e, a, c, last = replay_alarms(self.times[at], out["warn"], self.cycles[at], float(model["threshold"]), self.policy)
            self.emitted[at], self.active[at], self.confirmations[at], self.last_alarm[at] = e, a, c, last
        if max(float(np.abs(self.warn - ledger["ledger_warn"]).max()),
               float(np.abs(self.q - ledger["ledger_quantiles"]).max())) > 1e-6:
            raise RuntimeError("Life model outputs differ from the training ledger")
        self.lock = threading.Lock()

    def verify(self):
        for name, expected in self.manifest["files"].items():
            if sha256(self.folder / name) != expected:
                raise RuntimeError(f"Bundle artifact changed: {name}")
        return True

    def workbench(self):
        return json.loads((self.folder / "workbench.json").read_text(encoding="utf-8"))

    def summary(self):
        m = self.manifest
        return {
            "id": "life", "kind": "survival", "version": self.version, "bundle": self.version,
            "name": "电极寿命 · 风险层 v3", "name_en": "Electrode life · hazard layer v3", "short": "风险层 v3", "short_en": "Hazard v3",
            "describe": "逻辑风险层 + 提升树预警：23 项特征 → 13 个时间段的到寿风险",
            "describe_en": "Logistic hazard layer + boosted-tree warning: 23 features → hazard in 13 time bins",
            "family": "neural", "members": ["hazard"], "curve_hours": [float(v) for v in EDGES[1:]], "cap_hours": float(EDGES[-1]),
            "source": m["source"], "features": len(self.features), "sensors": int(m["sensors"]),
            "architecture": [len(self.features), len(EDGES) - 1],
            "trees": int(self.models[-1]["trees"]["feature"].shape[0] * self.models[-1]["trees"]["feature"].shape[1]),
            "final_threshold": float(self.models[-1]["threshold"]), "alarm_policy": self.policy,
            "origin_seconds": 10, "hold_seconds": HOLD_SECONDS, "horizons": list(HORIZONS),
            "interval": [QUANTILES[0], QUANTILES[2]], "products_per_processing_hour": self.rate,
            "segment": dict(name=self.segment["name"], label=self.segment["label"]),
            "limits": m["limits"], "evaluation": dict(m["evaluation"], life=dict(
                m["evaluation"]["life"], mae=m["evaluation"]["life"]["v3"]["mae"], mae_last2h=m["evaluation"]["life"]["v3"]["mae_last2h"])),
        }

    def position(self, stamp):
        """Latest processing origin at or before `stamp`, at most HOLD_SECONDS old."""
        i = int(np.searchsorted(self.times, stamp, side="right")) - 1
        if i < 0 or stamp - int(self.times[i]) > HOLD_SECONDS:
            return None
        return i

    def outputs(self, stamp):
        """Prediction block for one replayed second, or the reason there is none."""
        i = self.position(stamp)
        if i is None:
            return dict(status="idle", reason="当前不在加工原点：模型只在确认加工的 10 秒原点更新",
                        model_version=self.version, model_id="life")
        origin, last, active = int(self.times[i]), int(self.last_alarm[i]), bool(self.active[i])
        q, threshold, score = self.q[i], float(self.threshold[i]), float(self.warn[i])
        order = np.argsort(-np.abs(self.z[i]))[:6]
        return dict(
            status="ok", reason="", model_version=self.version, model_id="life", model_name="电极寿命 · 风险层 v3",
            curve=[float(v) for v in 1 - np.cumprod(1 - self.hazard[i])],  # P(end within each bin edge) of the combined curve
            score=score, threshold=threshold, above=bool(score >= threshold),
            alarm_active=active, alarm_emitted=bool(self.emitted[i]),
            confirmations=int(self.confirmations[i]), confirm_needed=int(self.policy["confirm_origins"]),
            alarm_at=pd.Timestamp(last, unit="s").isoformat() if active and last >= 0 else None,
            alarm_age_s=int(stamp - last) if active and last >= 0 else None,
            rul_hours=float(q[1]), rul_interval_hours=[float(q[0]), float(q[2])],
            rul_pieces=int(q[1] * self.rate), rul_interval_pieces=[int(q[0] * self.rate), int(q[2] * self.rate)],
            horizons=[dict(hours=h, probability=float(p)) for h, p in zip(HORIZONS, self.horizons[i])],
            age_hours=float(self.age[i]), age_pieces=int(self.age[i] * self.rate),
            origin=pd.Timestamp(origin, unit="s").isoformat(), origin_age_s=int(stamp - origin),
            cycle_id=self.cycle_ids[int(self.cycles[i])], held_out=True,
            inputs=[dict(key=self.features[k], value=float(self.x[i, k]) if np.isfinite(self.x[i, k]) else None,
                         z=float(self.z[i, k])) for k in order],
        )

    def neural_payload(self, stamp, seq):
        """The hazard layer as it is: scaled features -> 13 sigmoid units, one per time bin.

        Weights and inputs are exact. The output activations are the hazards of the
        combined survival curve, so they also carry the smoothing and the tree warning.
        """
        i = self.position(stamp)
        if i is None:
            return None
        model = self.models[int(self.cycles[i])]
        names = ["特征输入", "分段风险"]
        size, bins = len(self.features), len(EDGES) - 1
        layers = [
            dict(name=names[0], units=size, total_units=size, unit_indices=list(range(size)), labels=list(self.features)),
            dict(name=names[1], units=bins, total_units=bins, unit_indices=list(range(bins)), labels=list(BIN_LABEL)),
        ]
        return dict(model_type="neural_network", real_neuron_activations_available=True,
                    input_parameter_count=int(self.manifest["sensors"]), input_feature_count=size,
                    tasks=["life"], topology="hazard_layer", heads=[], sample_id=seq,
                    prediction_id=f"{self.version}:{int(self.times[i])}",
                    graph=dict(layers=layers, connections=[{"from": 0, "to": 1, "weights": model["weight"].astype("float64").tolist()}]),
                    activation_layers=[dict(name=names[0], values=self.z[i].tolist()),
                                       dict(name=names[1], values=self.hazard[i].tolist())])
