"""Real 28-input model replay provider (run 20261112_v4).

Input mode: recorded instrument rows of a period the model never saw. The bundle
carries the 1 Hz readings, the observed-value mask and the 292 causal features the
training pipeline computed for every 10-second landmark; predictions are computed
here, at request time, by the exported sklearn / AFT artifacts. The panel never
re-derives features, so there is no train/serve feature skew by construction.

Nothing here claims live inference, a calibrated probability or a validated
lifetime: the contract file in the bundle states the training cut-off, the weak
label basis and the outputs that do not exist.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

TIME_STRIDE = 10  # seconds between model landmarks
WINDOW_SECONDS = 60  # rows shown by the instrument rail
SESSION_SECONDS = 1800
TYPE_ORDER = ["voltage", "feedrate", "hopper_pressure"]
TYPE_LABEL = {"voltage": "电压偏低", "feedrate": "送料超限", "hopper_pressure": "料斗压力超限"}
RULE_FEATURES = {"voltage": "product_voltage_fail", "feedrate": "product_feedrate_fail",
                 "hopper_pressure": "product_hopper_pressure_fail"}
QUANTILES = (-1.2815515655, 0.0, 1.2815515655)  # 10 %, 50 %, 90 % of a lognormal


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transform(values, center, mean, scale):
    """Exactly the training-time Transform: non-finite -> train centre, then standardise.

    Kept in float32 like training: a tree split compares against float32 thresholds,
    so a float64 detour could send a borderline row down another branch.
    """
    x = np.asarray(values, dtype="float32")
    x = np.where(np.isfinite(x), x, center)
    return np.clip((x - mean) / scale, -12, 12).astype("float32")


def stage_features(raw):
    """The four constrained lifetime inputs, identical to train_stage_life.stage_features."""
    raw = np.asarray(raw, dtype="float32")
    return np.column_stack([
        np.log1p(np.maximum(raw[:, 0] - 51, 0)),
        np.log1p(np.maximum(raw[:, 1], 0)),
        np.log1p(np.maximum(-raw[:, 2], 0)),
        np.log1p(np.maximum(raw[:, 3], 0)),
    ]).astype("float32")


class Real28Bundle:
    """Validated, read-only access to one exported bundle directory."""

    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.manifest = json.loads((self.folder / "manifest.json").read_text(encoding="utf-8"))
        self.contract = json.loads((self.folder / "contract.json").read_text(encoding="utf-8"))
        self.version = self.manifest["version"]
        self.field_keys = list(self.manifest["field_keys"])
        self.feature_names = list(self.manifest["feature_names"])
        replay = self.folder / "replay"
        self.landmarks = np.load(replay / "landmarks.npy")  # epoch seconds, ascending
        self.landmark_block = np.load(replay / "landmark_block.npy")
        self.features = np.load(replay / "features.npy", mmap_mode="r")
        self.raw_t = np.load(replay / "raw_t.npy")  # epoch seconds, ascending
        self.raw_block = np.load(replay / "raw_block.npy")
        self.raw = np.load(replay / "raw.npy", mmap_mode="r")  # T x 28, forward-filled inside a block
        self.raw_seen = np.load(replay / "raw_seen.npy", mmap_mode="r")  # T x 28, value was actually read
        if self.features.shape != (len(self.landmarks), len(self.feature_names)):
            raise ValueError("Feature matrix does not match the recorded landmarks and feature names")
        if self.raw.shape != (len(self.raw_t), len(self.field_keys)):
            raise ValueError("Instrument rows do not match the recorded field list")
        self.models = {task: joblib.load(self.folder / "models" / f"{task}.joblib") for task in TYPE_ORDER}
        self.neural = all(b.get("algorithm") == "mlp_64_32" for b in self.models.values())
        self.life = {task: joblib.load(self.folder / "models" / f"life_{task}.joblib") for task in ("hours", "pieces")}
        for bundle in self.models.values():
            # One landmark at a time: a thread pool would cost more than the prediction.
            if "n_jobs" in bundle["model"].get_params():
                bundle["model"].set_params(n_jobs=1)
        self.rule_columns = {task: self.feature_names.index(name) for task, name in RULE_FEATURES.items()}
        self.product_available = self.feature_names.index("product_available")
        self.segments = list(self.manifest["segments"])

    def verify(self):
        """Re-hash every listed artifact; the export step and the contract test call this."""
        for name, expected in self.manifest["files"].items():
            if sha256(self.folder / name) != expected:
                raise RuntimeError(f"Bundle artifact changed: {name}")
        return True

    def summary(self):
        return {
            "version": self.version,
            "input_mode": self.contract["input_mode"],
            "period": self.contract["replay_period"],
            "model_trained_through": self.contract["model_trained_through"],
            "prediction_interval_seconds": TIME_STRIDE,
            "landmarks": int(len(self.landmarks)),
            "features": len(self.feature_names),
            "fields": len(self.field_keys),
            "algorithms": {task: self.models[task]["algorithm"] for task in TYPE_ORDER},
            "neural": self.neural,
            "segments": [dict(name=s["name"], label=s["label"]) for s in self.segments],
        }

    def index_for(self, timestamp, block):
        """Landmark at or before `timestamp` in the same continuous block, at most one stride old."""
        position = int(np.searchsorted(self.landmarks, timestamp, side="right")) - 1
        if position < 0 or timestamp - int(self.landmarks[position]) >= TIME_STRIDE:
            return None
        return position if self.landmark_block[position] == block else None

    def risk(self, task, rows):
        """Calibrated score, alarm flag and conditional ETA for feature rows."""
        bundle = self.models[task]
        z = transform(rows, bundle["center"], bundle["mean"], bundle["scale"])
        raw = bundle["model"].predict_proba(z)[:, 1].astype("float64")
        score = self.calibrate(bundle["calibration"], raw)
        head = bundle["eta_head"]
        if head["kind"] == "constant":
            eta = np.full(len(z), head["bin"] * 30 + 15, dtype="float64")
        else:
            time_probability = np.zeros((len(z), 10))
            time_probability[:, head["classes"]] = head["model"].predict_proba(z)
            total = time_probability.sum(1, keepdims=True)
            time_probability = np.divide(time_probability, total, out=np.full_like(time_probability, 0.1), where=total > 0)
            eta = (time_probability * np.arange(15, 300, 30)).sum(1)
        return raw, score, score >= bundle["threshold"], eta

    @staticmethod
    def calibrate(calibration, probability):
        if calibration["kind"] != "sigmoid":
            return np.asarray(probability, dtype="float64")
        logit = np.log(np.clip(probability, 1e-6, 1 - 1e-6) / np.clip(1 - probability, 1e-6, 1))[:, None]
        return calibration["model"].predict_proba(logit)[:, 1]

    def lifetime(self, name, rows):
        """10 % / 50 % / 90 % quantiles of the weak-label AFT; pieces are whole products."""
        bundle = self.life[name]
        stage = stage_features(np.asarray(rows)[:, bundle["feature_indices"]])
        z = transform(stage, bundle["center"], bundle["mean"], bundle["scale"])
        beta = np.asarray(bundle["beta"], dtype="float64")
        mu = beta[0] + z @ beta[1:]
        quantiles = np.exp(np.clip(mu[:, None] + float(bundle["parameter"]) * np.asarray(QUANTILES), -25, 25))
        return np.floor(quantiles) if name == "pieces" else quantiles

    def neural_payload(self, position, seq):
        """Actual voltage-risk MLP activations and matching learned weights.

        The input drawing samples 28 of the 292 derived features, not 28 fake
        raw-input neurons. Hidden layers and output are computed in full first.
        """
        if not self.neural or position is None:
            return None
        b = self.models["voltage"]
        model = b["model"]
        z = transform(np.asarray(self.features[position])[None, :], b["center"], b["mean"], b["scale"])
        values = [z[0]]
        for i, (weight, bias) in enumerate(zip(model.coefs_, model.intercepts_)):
            z = z @ weight + bias
            z = np.maximum(z, 0) if i < len(model.coefs_) - 1 else 1 / (1 + np.exp(-np.clip(z, -80, 80)))
            values.append(z[0])
        expected = model.predict_proba(transform(np.asarray(self.features[position])[None, :], b["center"], b["mean"], b["scale"]))[0, 1]
        if abs(float(values[-1][0]) - float(expected)) > 1e-5:
            raise RuntimeError("Neural display differs from the prediction")
        indices = [np.linspace(0, len(values[0]) - 1, 28, dtype=int)] + [np.arange(len(v)) for v in values[1:]]
        names = ["特征输入", "隐藏层 1", "隐藏层 2", "电压风险"]
        layers = [dict(name=name, units=len(ix), total_units=len(v), unit_indices=ix.tolist(),
                       labels=[self.feature_names[j] for j in ix] if i == 0 else [str(j + 1) for j in ix])
                  for i, (name, v, ix) in enumerate(zip(names, values, indices))]
        return dict(model_type="neural_network", real_neuron_activations_available=True,
                    input_parameter_count=28, input_feature_count=len(self.feature_names), task="voltage", sample_id=seq,
                    graph=dict(layers=layers, connections=[dict(**{"from": i, "to": i + 1},
                        weights=model.coefs_[i][np.ix_(indices[i], indices[i + 1])].T.tolist()) for i in range(len(model.coefs_))]),
                    activation_layers=[dict(name=name, values=v[ix].astype(float).tolist()) for name, v, ix in zip(names, values, indices)])

    def rows(self, start, stop):
        first = int(np.searchsorted(self.raw_t, start, side="left"))
        last = int(np.searchsorted(self.raw_t, stop, side="left"))
        return (self.raw_t[first:last], np.asarray(self.raw[first:last], dtype="float64"),
                np.asarray(self.raw_seen[first:last], dtype=bool), self.raw_block[first:last])


class ReplayStream:
    """One presentation session replaying recorded rows, one recorded row per step."""

    def __init__(self, bundle, segment):
        self.bundle = bundle
        self.segment = segment
        first = int(np.searchsorted(bundle.raw_t, int(segment["start"]), side="left"))
        last = int(np.searchsorted(bundle.raw_t, int(segment["stop"]), side="left"))
        if last - first < WINDOW_SECONDS:
            raise ValueError("Replay segment is shorter than the display window")
        self.first, self.last, self.cursor, self.seq = first, last, first - 1, -1
        self.buffer = []

    def append(self, count=1):
        for _ in range(count):
            self.cursor += 1
            if self.cursor >= self.last:
                # A presentation loop over the same recorded rows; never new data.
                self.cursor = self.first
                self.buffer = []
            self.seq += 1
            self.buffer.append((self.cursor, self.seq))
            self.buffer = self.buffer[-WINDOW_SECONDS:]

    def current(self):
        index, seq = self.buffer[-1]
        return index, seq

    def record(self, index):
        values = np.asarray(self.bundle.raw[index], dtype="float64")
        seen = np.asarray(self.bundle.raw_seen[index], dtype=bool)
        # A value that was not read at this second is missing, not the previous reading.
        return {key: (float(v) if ok and np.isfinite(v) else None)
                for key, v, ok in zip(self.bundle.field_keys, values, seen)}

    def instrument_snapshot(self, session_id):
        index, seq = self.current()
        rows = []
        for i, _ in self.buffer:
            row = self.record(i)
            row["timestamp"] = pd.Timestamp(int(self.bundle.raw_t[i]), unit="s").isoformat()
            rows.append(row)
        times = np.array([int(self.bundle.raw_t[i]) for i, _ in self.buffer])
        fields = [dict(key=key, name=self.bundle.manifest["field_names"][key], unit=self.bundle.manifest["field_units"][key])
                  for key in self.bundle.field_keys]
        return dict(
            data_source="recorded_replay", simulation_basis="real_28_recorded",
            source=dict(id=session_id, name=self.segment["name"], label=self.segment["label"]),
            timestamp=rows[-1]["timestamp"], seq=seq, fields=fields, inputs=self.record(index),
            window=dict(rows=rows, count=len(rows), continuous=bool(len(times) < 2 or np.all(np.diff(times) == 1))),
        )


class Real28Service:
    """Same session surface as the simulated stream, backed by recorded rows.

    A row without a usable landmark (unpowered, or less than the model's causal
    history since a data gap) returns no prediction instead of an invented one.
    """

    def __init__(self, bundle):
        self.bundle = bundle
        self.sessions = {}
        self.created = 0
        self.lock = threading.RLock()

    def config(self):
        contract = self.bundle.contract
        return {
            "model_kind": "real28", "prediction_ready": True,
            "neural": {"name": "28 参数神经网络", "architecture": [292, 64, 32, 1], "input_parameter_count": 28} if self.bundle.neural else None,
            "real28": self.bundle.summary(), "input_mode": contract["input_mode"],
            "prediction_limits": contract["limits"], "unavailable_outputs": contract["unavailable_outputs"],
            "display": {"window_seconds": WINDOW_SECONDS, "sampling_seconds": 1, "prediction_seconds": TIME_STRIDE},
        }

    def create(self, model_id="active"):
        with self.lock:
            now = time.monotonic()
            self.sessions = {k: v for k, v in self.sessions.items() if now - v[1] < SESSION_SECONDS}
            if len(self.sessions) >= 16:
                raise OverflowError("回放会话数量已达上限，请稍后重试")
            segment = self.bundle.segments[self.created % len(self.bundle.segments)]
            self.created += 1
            stream = ReplayStream(self.bundle, segment)
            stream.append(WINDOW_SECONDS)
            session_id = uuid.uuid4().hex
            self.sessions[session_id] = [stream, now, threading.Lock()]
        return self.snapshot(session_id, stream)

    def _session(self, session_id):
        with self.lock:
            session = self.sessions.get(session_id)
        if session is None or time.monotonic() - session[1] >= SESSION_SECONDS:
            raise KeyError("回放会话已过期，请重新开始")
        return session

    def advance(self, session_id, expected_seq, steps=1, model_id="active"):
        session = self._session(session_id)
        with session[2]:
            if expected_seq != session[0].seq:
                raise ValueError("采样序号已变化，请重新开始")
            session[1] = time.monotonic()
            session[0].append(steps)
            return self.snapshot(session_id, session[0])

    def current(self, session_id, model_id="active"):
        session = self._session(session_id)
        with session[2]:
            session[1] = time.monotonic()
            return self.snapshot(session_id, session[0])

    def remove(self, session_id):
        with self.lock:
            self.sessions.pop(session_id, None)

    def snapshot(self, session_id, stream):
        index, seq = stream.current()
        instrument = stream.instrument_snapshot(session_id)
        inputs = instrument["inputs"]
        stamp = int(self.bundle.raw_t[index])
        current, power = inputs.get("Current"), inputs.get("Power")
        powered = current is not None and power is not None and current > 100 and power > 5
        position = self.bundle.index_for(stamp, self.bundle.raw_block[index]) if powered else None
        if not powered:
            status, outputs, reason = "unpowered", None, "设备未带电或电流/功率读数缺失：不生成预测"
        elif position is None:
            status, outputs, reason = "warming_up", None, "数据断档后不足 120 秒连续带电历史：暂无可用预测点"
        else:
            status, outputs, reason = "ok", self.outputs(position, seq, stamp, inputs), ""
        return {
            "session_id": session_id, "seq": seq, "status": status, "prediction_ready": True, "reason": reason,
            "needed": None, "data_source": "recorded_replay", "simulation_basis": "real_28_recorded",
            "model_kind": "real28", "timestamp": instrument["timestamp"],
            "inputs": inputs, "window": [],
            "features": [dict(key=k, value=float(v) if np.isfinite(v) else None,
                z=float(z)) for k, v, z in zip(self.bundle.feature_names, self.bundle.features[position],
                    transform(np.asarray(self.bundle.features[position])[None, :], self.bundle.models['voltage']['center'],
                        self.bundle.models['voltage']['mean'], self.bundle.models['voltage']['scale'])[0])] if position is not None else [],
            "neural": self.bundle.neural_payload(position, seq),
            "instrument": instrument, "outputs": outputs,
        }

    def outputs(self, position, seq, stamp, inputs):
        started = time.perf_counter()
        row = np.asarray(self.bundle.features[position], dtype="float32")[None, :]
        scores, alarms, etas = {}, {}, {}
        for task in TYPE_ORDER:
            _, score, alarm, eta = self.bundle.risk(task, row)
            scores[task], alarms[task], etas[task] = float(score[0]), bool(alarm[0]), float(eta[0])
        hours = self.bundle.lifetime("hours", row)[0]
        pieces = self.bundle.lifetime("pieces", row)[0]
        top = max(TYPE_ORDER, key=lambda task: scores[task] / max(self.bundle.models[task]["threshold"], 1e-9))
        rule = self.rule_state(row[0])
        reasons = [f"{TYPE_LABEL[task]}风险分数 {scores[task]:.3f} 达到实验阈值 {self.bundle.models[task]['threshold']:.3f}"
                   for task in TYPE_ORDER if alarms[task]]
        if rule not in ("正常", "暂无已完成产品"):
            reasons.append(f"最近完成产品按工艺规则判定：{rule}")
        landmark = int(self.bundle.landmarks[position])
        pressure = next((inputs[key] / 1000.0 for key, feed in (("Press._Hopper_1", "Feedrate_1"), ("Press._Hopper_2", "Feedrate_2"))
                         if inputs.get(key) is not None and (inputs.get(feed) or 0) > 0), None)
        return {
            "rul_hours": float(hours[1]), "rul_pieces": int(pieces[1]),
            "rul_interval_hours": [float(hours[0]), float(hours[2])],
            "rul_interval_pieces": [int(pieces[0]), int(pieces[2])],
            "interval_nominal": None,
            "pressure_now": pressure, "pressure_60s": None, "pressure_trend": None,
            "gun_health": None, "tank_health": None,
            "anomaly_type": rule, "anomaly_source": "last_completed_product_rule", "anomaly_probability": None,
            "next_anomaly_type": TYPE_LABEL[top] if alarms[top] else "无预警",
            "anomaly_300s_probability": scores[top], "anomaly_300s_type": top,
            "risk_by_type": scores, "alarm_by_type": alarms,
            "alarm_thresholds": {task: float(self.bundle.models[task]["threshold"]) for task in TYPE_ORDER},
            "predicted_anomaly_eta_s": etas[top] if alarms[top] else None,
            "warning_level": "experimental", "warning_reasons": reasons,
            "sigma_deviations": [], "root_candidates": [], "root_status": "not_available",
            "model_kind": "real28", "model_name": self.bundle.contract["model_name"],
            "model_version": self.bundle.version, "model_trained_through": self.bundle.contract["model_trained_through"],
            "prediction_landmark": pd.Timestamp(landmark, unit="s").isoformat(), "prediction_age_s": int(stamp - landmark),
            "input_parameters": inputs, "timestamp": pd.Timestamp(stamp, unit="s").isoformat(),
            "gun_id": 1, "seq": seq, "latency_ms": (time.perf_counter() - started) * 1000,
        }

    def rule_state(self, row):
        """Verdict of the most recent completed product by the plant rules; not a model output."""
        available = row[self.bundle.product_available]
        if not np.isfinite(available) or available < 0.5:
            return "暂无已完成产品"
        failed = [TYPE_LABEL[task] for task in TYPE_ORDER if row[self.bundle.rule_columns[task]] == 1]
        return "、".join(failed) if failed else "正常"
