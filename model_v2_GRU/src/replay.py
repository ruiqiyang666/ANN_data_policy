"""Verify final selected model on each held-out origin after frozen evaluation.

This command requires artifacts/evaluation.json and never tunes parameters.
It compares independent bounded-memory inference against the saved ledger.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import platform
import time

import numpy as np
import pandas as pd
import torch

from inference import StreamingPredictor, file_digest, verify_frozen_files
from modeling import ROOT


def ledger_for_model(path, model):
    ledger = pd.read_csv(path)
    if 'split' in ledger:
        ledger = ledger[ledger.split.eq('test')]
    if 'model' in ledger:
        ledger = ledger[ledger.model.eq(model)]
    if 'score' not in ledger:
        candidates = [f'score_{model}', model]
        name = next((k for k in candidates if k in ledger), None)
        if name is None:
            raise ValueError('Prediction ledger requires score, score_MODEL, or MODEL column')
        ledger = ledger.rename(columns={name: 'score'})
    if not ledger.anchor_id.is_unique:
        raise ValueError('Selected-model prediction ledger has duplicate anchor IDs')
    return ledger.reset_index(drop=True)


def verify(root=ROOT):
    root = Path(root)
    art = root / 'artifacts'
    if not (art / 'evaluation.json').is_file():
        raise RuntimeError('Run final frozen evaluation before any held-out replay')
    started = time.perf_counter()
    torch.set_num_threads(1)
    predictor = StreamingPredictor(root)
    if not predictor.config.get('selection_final'):
        raise RuntimeError('Final selection must be sealed before held-out replay')
    checked = verify_frozen_files(root, predictor.config, include_data=True)
    evaluation = json.loads((art / 'evaluation.json').read_text(encoding='utf-8'))
    if file_digest(art / 'frozen_config.json') != evaluation['frozen_config_sha256']:
        raise RuntimeError('Frozen configuration differs from evaluated configuration')
    for name, expected_hash in evaluation.get('result_sha256', {}).items():
        if file_digest(art / name) != expected_hash:
            raise RuntimeError(f'Frozen result ledger changed: {name}')
    anchors = pd.read_csv(root / 'data/anchors.csv.gz', parse_dates=['timestamp'])
    frame = anchors[anchors.split.eq('test')].reset_index(drop=True)
    ledger = ledger_for_model(art / 'predictions.csv.gz', predictor.name)
    if frame.anchor_id.tolist() != ledger.anchor_id.tolist():
        raise AssertionError('Test anchor IDs/order differ between stream and frozen prediction ledger')
    stream_scores = np.empty(len(frame), dtype=float)
    emitted = []
    for i, row in enumerate(frame.to_dict('records')):
        output = predictor.push(row)
        stream_scores[i] = output['score']
        if output['alarm']:
            emitted.append((str(row['cycle_id']), pd.Timestamp(row['timestamp']).isoformat()))
        if (i + 1) % 5000 == 0:
            print(f'Replayed {i + 1}/{len(frame)} origins', flush=True)
    expected = ledger.score.to_numpy(float)
    errors = np.abs(stream_scores - expected)
    scores_match = bool(np.allclose(stream_scores, expected, rtol=1e-5, atol=1e-5, equal_nan=False))
    alarms = pd.read_csv(art / 'alarm_results.csv')
    if 'split' in alarms:
        alarms = alarms[alarms.split.eq('test')]
    if 'model' in alarms:
        alarms = alarms[alarms.model.eq(predictor.name)]
    expected_alarms = [(str(r.cycle_id), pd.Timestamp(r.time).isoformat()) for r in alarms.itertuples()]
    alarm_match = emitted == expected_alarms
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), host=platform.node(),
                  model=predictor.name, model_version=predictor.config['model_version'],
                  split='test', origins=len(frame), exact_anchor_order=True,
                  rtol=1e-5, atol=1e-5, max_absolute_score_error=float(errors.max()) if len(errors) else 0.,
                  score_match=scores_match, emitted_alarms=len(emitted), expected_alarms=len(expected_alarms),
                  alarm_times_match=alarm_match, passed=bool(scores_match and alarm_match),
                  elapsed_seconds=time.perf_counter() - started, checked_frozen_files=checked,
                  method='One origin at a time; independent bounded deque, per-origin searchsorted and single-window forward pass; no whole-history encode.',
                  limitation='Historical eligible processing anchors with supplied causal 60s statistics and offline cycle reset keys. Not an end-to-end raw PLC stream or a field connection.',
                  server_execution=False, site_connected=False)
    (art / 'replay_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    if not result['passed']:
        raise AssertionError(f'Streaming replay mismatch: {result}')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    arguments = parser.parse_args()
    verify(arguments.root)
