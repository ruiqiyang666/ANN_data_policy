"""Booth2 electrode end-of-life warning (ANN_data_policy model_v2, frozen GRU-16).

A second, independent model shown beside the 28-input risk networks on the same
recorded replay. The bundle carries the frozen weights, the training-set
preprocessor and the prepared processing origins ("anchors": nine sensor readings
plus the causal 60-second voltage mean / deviation) that the source project
exported. The score is computed here, at request time, from the 61-point history
of each origin; the stored ledger score is only used to assert that this
re-implementation still equals the frozen evaluation.

Nothing here claims a live interface, a calibrated probability or a countdown:
the score is an uncalibrated end-of-life risk, origins inherit the offline
product eligibility, and the alarm state is the frozen policy replayed over the
whole electrode cycle (so a replay that starts mid-cycle shows the same alarms
as the frozen evaluation).
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ["Voltage", "Current", "Power", "Station_1", "Station_2",
          "Feedrate_1", "Feedrate_2", "Press._Hopper_1", "Press._Hopper_2"]
VALUE_FIELDS = FIELDS + ["active_feedrate", "active_pressure", "Voltage_mean60s", "Voltage_std60s"]
TARGETS = ["y_eol_30m", "y_eol_45m", "y_eol_60m", "y_pre20_40m"]
TARGET_LABEL = {"y_eol_30m": "30 分钟内末期", "y_eol_45m": "45 分钟内末期",
                "y_eol_60m": "60 分钟内末期", "y_pre20_40m": "20–40 分钟后末期"}
VALUE_LABEL = {"Voltage": "喷枪电压", "Current": "喷枪电流", "Power": "喷枪功率",
               "Station_1": "转台 1 转速", "Station_2": "转台 2 转速",
               "Feedrate_1": "送料量 1", "Feedrate_2": "送料量 2",
               "Press._Hopper_1": "料斗压力 1", "Press._Hopper_2": "料斗压力 2",
               "active_feedrate": "活动送料量", "active_pressure": "活动料斗压力",
               "Voltage_mean60s": "电压 60 秒均值", "Voltage_std60s": "电压 60 秒波动"}
VALUE_UNIT = {"Voltage": "V", "Current": "A", "Power": "kW", "Station_1": "rpm", "Station_2": "rpm",
              "Feedrate_1": "g/min", "Feedrate_2": "g/min", "Press._Hopper_1": "mbar",
              "Press._Hopper_2": "mbar", "active_feedrate": "g/min", "active_pressure": "mbar",
              "Voltage_mean60s": "V", "Voltage_std60s": "V"}
HOLD_SECONDS = 60  # the latest origin stays the current prediction between products


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_values(values):
    """N x 11 prepared inputs -> N x 13 model channels (adds the active hopper pair)."""
    a = np.asarray(values, dtype="float32")
    on1, on2 = a[:, 5] > 0, a[:, 6] > 0
    single = on1 ^ on2
    feed = np.where(single, np.where(on1, a[:, 5], a[:, 6]), np.nan)
    pressure = np.where(single, np.where(on1, a[:, 7], a[:, 8]), np.nan)
    return np.column_stack([a[:, :9], feed, pressure, a[:, 9:11]]).astype("float32")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def gru_forward(weights, x):
    """torch.nn.GRU(batch_first) + Linear head on the last step, in float32 NumPy.

    x: B x T x C. Returns (B x 4 sigmoid scores, B x H final hidden state).
    """
    w_ih, w_hh = weights["weight_ih"], weights["weight_hh"]
    b_ih, b_hh = weights["bias_ih"], weights["bias_hh"]
    hidden = w_hh.shape[1]
    h = np.zeros((x.shape[0], hidden), dtype="float32")
    for step in range(x.shape[1]):
        gi = x[:, step] @ w_ih.T + b_ih
        gh = h @ w_hh.T + b_hh
        r = sigmoid(gi[:, :hidden] + gh[:, :hidden])
        z = sigmoid(gi[:, hidden:2 * hidden] + gh[:, hidden:2 * hidden])
        n = np.tanh(gi[:, 2 * hidden:] + r * gh[:, 2 * hidden:])
        h = ((1 - z) * n + z * h).astype("float32")
    return sigmoid(h @ weights["head_weight"].T + weights["head_bias"]), h


def encode(times, values13, cycles, positions, pre):
    """Causal 61-point window per origin: scaled values, observed mask and data age.

    The same construction as the source project's `encode`: one grid point per
    minute looking back, each filled by the latest origin of the same electrode
    cycle that is at most `max_age_seconds` old.
    """
    med = np.asarray(pre["median"], dtype="float64")
    scale = np.asarray(pre["scale"], dtype="float64")
    minutes, grid_s, max_age = int(pre["history_minutes"]), int(pre["grid_seconds"]), int(pre["max_age_seconds"])
    offsets = np.arange(minutes, -1, -1) * grid_s
    out = np.zeros((len(positions), minutes + 1, 2 * len(VALUE_FIELDS) + 1), dtype="float32")
    for row, i in enumerate(positions):
        cycle = cycles[i]
        first = int(np.searchsorted(cycles, cycle, side="left"))
        local = times[first:i + 1]
        grid = times[i] - offsets
        j = np.searchsorted(local, grid, side="right") - 1
        safe = np.maximum(j, 0)
        age = grid - local[safe]
        good = (j >= 0) & (age >= 0) & (age <= max_age)
        a = values13[first + safe]
        seen = np.isfinite(a)
        z = np.clip((np.where(seen, a, med) - med) / scale, -12, 12)
        mask = seen & good[:, None]
        out[row, :, :13] = np.where(mask, z, 0)
        out[row, :, 13:26] = mask
        out[row, :, 26] = np.where(good, age / max_age, 1)
    return out


def replay_alarms(times, scores, cycles, threshold, policy):
    """The frozen incremental alarm policy over every origin of every cycle.

    Returns per-origin arrays: alarm emitted here, episode active after this
    origin, consecutive confirmations, and the time of the latest emitted alarm.
    """
    n = len(times)
    emitted = np.zeros(n, dtype=bool)
    active_after = np.zeros(n, dtype=bool)
    confirmations = np.zeros(n, dtype="int16")
    last_alarm_at = np.full(n, -1, dtype="int64")
    cycle = None
    for i in range(n):
        at, score = float(times[i]), float(scores[i])
        if cycles[i] != cycle:
            cycle = cycles[i]
            last_origin = last_signal = last_alarm = None
            consecutive, active = 0, False
        if last_origin is not None and at - last_origin > policy["max_confirm_gap_seconds"]:
            consecutive = 0
        last_origin = at
        if active and at - last_signal >= policy["release_seconds"]:
            active = False
        if not (np.isfinite(score) and score >= threshold):
            consecutive = 0
        else:
            if last_signal is None or at - last_signal > policy["max_confirm_gap_seconds"]:
                consecutive = 0
            consecutive += 1
            last_signal = at
            cooled = last_alarm is None or at - last_alarm >= policy["cooldown_seconds"]
            if consecutive >= policy["confirm_origins"] and not active and cooled:
                last_alarm, active, emitted[i] = at, True, True
        active_after[i] = active
        confirmations[i] = consecutive
        last_alarm_at[i] = -1 if last_alarm is None else int(last_alarm)
    return emitted, active_after, confirmations, last_alarm_at


class ElectrodeBundle:
    """Validated, read-only access to one exported electrode-warning bundle."""

    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.manifest = json.loads((self.folder / "manifest.json").read_text(encoding="utf-8"))
        self.version = self.manifest["version"]
        self.pre = json.loads((self.folder / "preprocessor.json").read_text(encoding="utf-8"))
        if self.pre["fields"] != VALUE_FIELDS:
            raise ValueError("Preprocessor input order differs from the model")
        with np.load(self.folder / "weights.npz") as stored:
            self.weights = {name: np.asarray(stored[name], dtype="float32") for name in stored.files}
        if self.weights["weight_ih"].shape[1] != 2 * len(VALUE_FIELDS) + 1:
            raise ValueError("Model input width differs from the 13 value channels, masks and age")
        with np.load(self.folder / "anchors.npz") as stored:
            self.times = np.asarray(stored["t"], dtype="int64")  # epoch seconds of the instrument clock
            self.cycles = np.asarray(stored["cycle"], dtype="int16")  # ascending cycle index
            self.values = raw_values(stored["values"])
            self.ledger = np.asarray(stored["ledger_score"], dtype="float64")
        if not (np.all(np.diff(self.times) > 0) and np.all(np.diff(self.cycles) >= 0)):
            raise ValueError("Origins must be chronological and grouped by electrode cycle")
        self.threshold = float(self.manifest["threshold"])
        self.head_index = int(self.manifest["head_index"])
        self.policy = dict(self.manifest["alarm_policy"])
        self.emitted, self.active, self.confirmations, self.last_alarm = replay_alarms(
            self.times, self.ledger, self.cycles, self.threshold, self.policy)
        self.cycle_ids = list(self.manifest["cycle_ids"])
        self.segment = dict(self.manifest["showcase_segment"])
        self.cache = {}
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
            "id": "electrode", "version": self.version, "name": "电极末期预警 GRU-16",
            "source": m["source"], "model_version": m["model_version"],
            "architecture": [len(VALUE_FIELDS) * 2 + 1, self.weights["weight_hh"].shape[1], len(TARGETS)],
            "history_minutes": int(self.pre["history_minutes"]), "sensors": len(FIELDS),
            "threshold": self.threshold, "head": TARGETS[self.head_index], "alarm_policy": self.policy,
            "origin_seconds": 10, "hold_seconds": HOLD_SECONDS,
            "segment": dict(name=self.segment["name"], label=self.segment["label"]),
            "limits": m["limits"], "evaluation": m["evaluation"],
        }

    def position(self, stamp):
        """Latest processing origin at or before `stamp`, at most HOLD_SECONDS old."""
        i = int(np.searchsorted(self.times, stamp, side="right")) - 1
        if i < 0 or stamp - int(self.times[i]) > HOLD_SECONDS:
            return None
        return i

    def score(self, i):
        """All four heads, final hidden state and the encoded window of origin i."""
        with self.lock:
            hit = self.cache.get(i)
        if hit is not None:
            return hit
        x = encode(self.times, self.values, self.cycles, [i], self.pre)
        heads, hidden = gru_forward(self.weights, x)
        if abs(float(heads[0, self.head_index]) - self.ledger[i]) > 1e-4:
            raise RuntimeError("Electrode score differs from the frozen evaluation ledger")
        result = (heads[0].astype("float64"), hidden[0].astype("float64"), x[0])
        with self.lock:
            if len(self.cache) > 512:
                self.cache.clear()
            self.cache[i] = result
        return result

    def outputs(self, stamp):
        """Prediction block for one replayed second, or the reason there is none."""
        i = self.position(stamp)
        if i is None:
            return dict(status="idle", reason="当前不在加工原点：电极模型只在确认加工的 10 秒原点评分",
                        model_version=self.version, threshold=self.threshold)
        heads, _, x = self.score(i)
        origin = int(self.times[i])
        last = int(self.last_alarm[i])
        active = bool(self.active[i])
        score = float(heads[self.head_index])
        values = self.values[i]
        return dict(
            status="ok", reason="", model_version=self.version, model_name="电极末期预警 GRU-16",
            score=score, threshold=self.threshold, above=bool(score >= self.threshold),
            head=TARGETS[self.head_index],
            heads=[dict(key=key, label=TARGET_LABEL[key], score=float(v)) for key, v in zip(TARGETS, heads)],
            alarm_active=active, alarm_emitted=bool(self.emitted[i]),
            confirmations=int(self.confirmations[i]), confirm_needed=int(self.policy["confirm_origins"]),
            alarm_at=pd.Timestamp(last, unit="s").isoformat() if active and last >= 0 else None,
            alarm_age_s=int(stamp - last) if active and last >= 0 else None,
            origin=pd.Timestamp(origin, unit="s").isoformat(), origin_age_s=int(stamp - origin),
            cycle_id=self.cycle_ids[int(self.cycles[i])],
            history_coverage=float(x[:, 13].mean()),  # share of the 61 grid points with a voltage reading
            inputs=[dict(key=key, label=VALUE_LABEL[key], unit=VALUE_UNIT[key],
                         value=float(v) if np.isfinite(v) else None, z=float(x[-1, k]),
                         coverage=float(x[:, 13 + k].mean()))
                    for k, (key, v) in enumerate(zip(VALUE_FIELDS, values))],
        )

    def neural_payload(self, stamp, seq):
        """One recurrent cell drawn as input -> hidden state -> four heads.

        Hidden activations and head weights are exact. The input -> hidden lines are
        the candidate-gate weights applied to the latest of the 61 steps: a faithful
        view of one step of a recurrent network, not of the whole hour of history.
        """
        i = self.position(stamp)
        if i is None:
            return None
        heads, hidden, x = self.score(i)
        size = len(hidden)
        names = ["序列输入", "GRU 隐状态", "末期风险"]
        layers = [
            dict(name=names[0], units=13, total_units=27, unit_indices=list(range(13)),
                 labels=[VALUE_LABEL[k] for k in VALUE_FIELDS]),
            dict(name=names[1], units=size, total_units=size, unit_indices=list(range(size)),
                 labels=[str(k + 1) for k in range(size)]),
            dict(name=names[2], units=len(TARGETS), total_units=len(TARGETS),
                 unit_indices=list(range(len(TARGETS))), labels=[TARGET_LABEL[k] for k in TARGETS]),
        ]
        candidate = self.weights["weight_ih"][2 * size:, :13].astype("float64")
        connections = [
            {"from": 0, "to": 1, "weights": candidate.tolist()},
            {"from": 1, "to": 2, "weights": self.weights["head_weight"].astype("float64").tolist()},
        ]
        activations = [
            dict(name=names[0], values=x[-1, :13].astype("float64").tolist()),
            dict(name=names[1], values=hidden.tolist()),
            dict(name=names[2], values=heads.tolist()),
        ]
        short = {"y_eol_30m": "30 分钟", "y_eol_45m": "45 分钟", "y_eol_60m": "60 分钟", "y_pre20_40m": "20–40 分钟"}
        shown = [dict(task=key, label=short[key], raw=float(v), score=float(v), unit="score",
                      threshold=self.threshold if k == self.head_index else None,
                      alarm=bool(k == self.head_index and self.active[i]))
                 for k, (key, v) in enumerate(zip(TARGETS, heads))]
        return dict(model_type="neural_network", real_neuron_activations_available=True,
                    input_parameter_count=len(FIELDS), input_feature_count=27, recurrent=True,
                    tasks=list(TARGETS), topology="recurrent_last_step", heads=shown, sample_id=seq,
                    prediction_id=f"{self.version}:{int(self.times[i])}",
                    graph=dict(layers=layers, connections=connections), activation_layers=activations)
