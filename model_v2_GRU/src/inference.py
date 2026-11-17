"""Frozen batch inference and bounded-memory, sensor-only online replay.

The online adapter must supply real, causal 60-second voltage statistics.
cycle_id is only a reset key, never a numeric predictor. Historical cycle keys
are available in this benchmark; a live deployment requires a real reset event.
"""
from collections import deque
from functools import lru_cache
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import torch

from evaluation import AlarmMachine, AlarmPolicy
from modeling import ROOT, FIELDS, VALUE_FIELDS, TARGETS, RiskNet, margin_score, predict_net, summaries


INPUT_FIELDS = FIELDS + ['Voltage_mean60s', 'Voltage_std60s']


def file_digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def verify_frozen_files(root, config, include_data=False):
    """Fail before inference when recorded weights, preprocessing or code changed."""
    root = Path(root)
    art = root / 'artifacts'
    checked = []
    groups = [(art, config.get('fitted_file_sha256', {})),
              (root, config.get('source_file_sha256', {})),
              (root / 'src', config.get('source_sha256', {}))]
    if include_data:
        groups.append((root / 'data', config.get('data_sha256', {})))
    for base, entries in groups:
        for name, expected in entries.items():
            path = base / name
            if not path.is_file() or file_digest(path) != expected:
                raise RuntimeError(f'Frozen file mismatch: {path}')
            checked.append(str(path.relative_to(root)))
    expected = config.get('preprocessor_sha256')
    if expected:
        path = art / 'preprocessor.json'
        if file_digest(path) != expected:
            raise RuntimeError('Frozen preprocessor hash mismatch')
        checked.append('artifacts/preprocessor.json')
    expected = config.get('protocol_sha256')
    if expected:
        path = art / 'experiment_plan.json'
        if file_digest(path) != expected:
            raise RuntimeError('Frozen experiment protocol hash mismatch')
        checked.append('artifacts/experiment_plan.json')
    expected = config.get('validation_extension_sha256')
    if expected:
        path = art / 'validation_extension_plan.json'
        if file_digest(path) != expected:
            raise RuntimeError('Frozen validation extension protocol hash mismatch')
        checked.append('artifacts/validation_extension_plan.json')
    return checked


def voltage_summary_indices(width):
    """Voltage, causal mean/std from each 13-channel block, plus four slopes."""
    n = len(VALUE_FIELDS)
    groups, remainder = divmod(width - 4, n)
    if remainder or groups != 18:
        raise ValueError('Unexpected summary layout; save explicit summary_indices in the model spec')
    return [g * n + j for g in range(groups) for j in [0, 11, 12]] + list(range(width - 4, width))


@lru_cache(maxsize=32)
def _load_model(path_string, kind, hidden, modified_ns, size):
    # File metadata participates in the cache key for validation-stage workflows;
    # final replay additionally validates the immutable SHA-256 manifest.
    path = Path(path_string)
    if kind in ['tcn', 'gru']:
        model = RiskNet(kind, int(hidden))
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model
    return joblib.load(path)


class ModelRunner:
    """Loads one model once and evaluates its saved heads without target access."""

    def __init__(self, spec, root=ROOT):
        self.spec = dict(spec)
        self.kind = self.spec['kind']
        self.root = Path(root)
        self.model = None
        if self.kind != 'margin':
            suffix = 'best.pt' if self.kind in ['tcn', 'gru'] else 'model.joblib'
            path = self.root / 'artifacts' / 'models' / self.spec['name'] / suffix
            stat = path.stat()
            self.model = _load_model(str(path.resolve()), self.kind,
                                     int(self.spec.get('hidden', 0)), stat.st_mtime_ns, stat.st_size)

    def predict(self, x, frame):
        if len(frame) != len(x):
            raise ValueError('One sensor frame row is required per sequence')
        if self.kind == 'margin':
            return margin_score(frame)[:, None]
        if self.kind in ['tcn', 'gru']:
            return predict_net(self.model, np.asarray(x, dtype=np.float32))
        z = summaries(x)
        indices = self.spec.get('summary_indices')
        if indices is None and self.kind == 'logistic_voltage':
            indices = voltage_summary_indices(z.shape[1])
        if indices is not None:
            z = z[:, np.asarray(indices, dtype=int)]
        models = self.model
        if isinstance(models, dict):
            models = models.get('models', models.get('heads'))
        if not isinstance(models, (list, tuple)):
            raise ValueError('Saved sklearn model must contain one estimator per target head')
        columns = []
        for model in models:
            if np.isscalar(model):
                columns.append(np.full(len(z), float(model)))
            else:
                classes = np.asarray(model.classes_)
                positive = np.flatnonzero(classes == 1)
                columns.append(model.predict_proba(z)[:, positive[0]] if len(positive) else np.zeros(len(z)))
        result = np.column_stack(columns)
        if result.shape != (len(x), len(TARGETS)):
            raise ValueError('Expected all four saved prediction heads')
        return result


def predict_model(spec, x, frame, root=ROOT):
    """Return four trained heads, or one engineering-margin head."""
    return ModelRunner(spec, root).predict(x, frame)


class CausalBuffer:
    """Encode only the latest 61-grid-point window from <=61 minutes of history."""

    def __init__(self, preprocessor):
        self.pre = dict(preprocessor)
        self.med = np.asarray(self.pre['median'], dtype=float)
        self.scale = np.asarray(self.pre['scale'], dtype=float)
        if self.pre['fields'] != VALUE_FIELDS:
            raise ValueError('Preprocessor input order differs from the model')
        self.history_minutes = int(self.pre['history_minutes'])
        self.max_age = int(self.pre.get('max_age_seconds', 60))
        self.grid_seconds = int(self.pre.get('grid_seconds', 60))
        if self.grid_seconds != 60 or self.max_age != 60:
            raise ValueError('The frozen V2 representation requires a 60-second grid and maximum age')
        self.offsets = np.arange(self.history_minutes, -1, -1) * self.grid_seconds
        self.records = deque()
        self.seen_cycles = set()
        self.cycle = None
        self.last_time = None

    @staticmethod
    def _value(value):
        return np.nan if value is None or pd.isna(value) else float(value)

    def push(self, row):
        """Ignore all non-input metadata; return x[1,61,27] for this origin only."""
        at = pd.Timestamp(row['timestamp'])
        if pd.isna(at):
            raise ValueError('timestamp must be present')
        epoch = int(at.value // 10**9)
        if self.last_time is not None and epoch <= self.last_time:
            raise ValueError('Online origins must arrive in strictly increasing time order')
        if pd.isna(row['cycle_id']):
            raise ValueError('A real cycle reset key is required')
        cycle = str(row['cycle_id'])
        if cycle != self.cycle:
            if cycle in self.seen_cycles:
                raise ValueError('An old cycle reset key may not reappear')
            self.seen_cycles.add(cycle)
            self.records.clear()
            self.cycle = cycle
        self.last_time = epoch
        a = np.array([self._value(row.get(k)) for k in FIELDS], dtype=np.float32)
        on1, on2 = a[5] > 0, a[6] > 0
        single = bool(on1 ^ on2)
        feed = (a[5] if on1 else a[6]) if single else np.nan
        pressure = (a[7] if on1 else a[8]) if single else np.nan
        values = np.r_[a, feed, pressure, self._value(row.get('Voltage_mean60s')),
                       self._value(row.get('Voltage_std60s'))].astype(np.float32)
        mask = np.isfinite(values)
        scaled = np.clip((np.where(mask, values, self.med) - self.med) / self.scale, -12, 12).astype(np.float32)
        self.records.append((epoch, scaled, mask))
        earliest = epoch - self.history_minutes * self.grid_seconds - self.max_age
        while self.records and self.records[0][0] < earliest:
            self.records.popleft()
        times = np.fromiter((r[0] for r in self.records), dtype=np.int64, count=len(self.records))
        scaled_history = np.stack([r[1] for r in self.records])
        masks = np.stack([r[2] for r in self.records])
        grid = epoch - self.offsets
        indices = np.searchsorted(times, grid, side='right') - 1
        safe = np.maximum(indices, 0)
        age_seconds = grid - times[safe]
        valid = (indices >= 0) & (age_seconds >= 0) & (age_seconds <= self.max_age)
        selected_mask = masks[safe] & valid[:, None]
        selected = np.where(selected_mask, scaled_history[safe], 0)
        age = np.where(valid, age_seconds / self.max_age, 1).astype(np.float32)
        return np.concatenate([selected, selected_mask.astype(np.float32), age[:, None]], axis=1)[None].astype(np.float32)


class StreamingPredictor:
    """Selected frozen model plus the same incremental alarm policy as scoring."""

    def __init__(self, root=ROOT, verify=True):
        self.root = Path(root)
        self.config = json.loads((self.root / 'artifacts/frozen_config.json').read_text(encoding='utf-8'))
        if verify:
            verify_frozen_files(self.root, self.config)
        self.pre = json.loads((self.root / 'artifacts/preprocessor.json').read_text(encoding='utf-8'))
        self.name = self.config['selected_model']
        self.spec = next(s for s in self.config['model_specs'] if s['name'] == self.name)
        self.policy = self.config['policies'][self.name]
        alarm_policy = self.config.get('alarm_policy', {})
        self.alarm = AlarmMachine(self.policy['threshold'], AlarmPolicy(**alarm_policy))
        self.buffer = CausalBuffer(self.pre)
        self.runner = ModelRunner(self.spec, self.root)

    def push(self, row):
        missing = [name for name in INPUT_FIELDS if name not in row]
        if missing:
            raise ValueError(f'Missing online input fields: {missing}; pass NaN explicitly for unavailable measurements')
        x = self.buffer.push(row)
        # Explicit whitelist: final product labels, future maintenance and
        # observation horizons cannot reach the predictor.
        current = pd.DataFrame([{k: row.get(k, np.nan) for k in INPUT_FIELDS}])
        scores = self.runner.predict(x, current)[0]
        score = float(scores[self.policy['head_index']])
        at = pd.Timestamp(row['timestamp'])
        emitted = self.alarm.step(at.value / 1e9, score, str(row['cycle_id']))
        return dict(timestamp=str(at), model=self.name, model_version=self.config['model_version'],
                    head=self.policy['head'], score=score, alarm=bool(emitted),
                    threshold=float(self.policy['threshold']))
