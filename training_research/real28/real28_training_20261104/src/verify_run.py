"""Verify frozen data boundaries and, when present, trained artifact integrity."""
import sys
sys.dont_write_bytecode = True
import torch
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import DATA, MODEL, EVAL, ROOT, FIELDS, dump, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    s = pd.read_parquet(DATA / 'samples.parquet')
    r = pd.read_parquet(DATA / 'rul_samples.parquet')
    t = np.load(DATA / 'grid_t.npy', mmap_mode='r')
    part = np.load(DATA / 'grid_part.npy', mmap_mode='r')
    block = np.load(DATA / 'grid_block_start.npy', mmap_mode='r')
    ends = s.end_index.to_numpy()
    target = np.load(DATA / 'targets.npz')
    normalization = json.loads((DATA / 'normalization.json').read_text('utf-8'))
    checks = []
    assert normalization['fields'] == FIELDS and len(FIELDS) == 28
    assert normalization['fit_partition'] == 'train'
    checks.append('28-field order and train-only normalization contract')
    np.testing.assert_array_equal(part[ends], s.partition)
    assert np.all(ends - 119 >= block[ends])
    assert np.all(t[ends] - t[ends - 119] == 119)
    assert np.all(part[ends] == part[ends - 119])
    assert s.groupby('possible_cycle').partition.nunique().max() == 1
    assert s[s.event_in_300s].groupby('event_id').partition.nunique().max() == 1
    checks.append('No context, possible-cycle or supervised event crosses partitions')
    products = pd.read_parquet(DATA / 'products.parquet')
    available = products[products.label_available_at.notna()].sort_values('label_available_at')
    at_product = available.label_available_at.to_numpy(dtype='datetime64[s]').astype('int64')
    previous_index = np.searchsorted(at_product, t[ends], side='right') - 1
    previous = available.iloc[np.maximum(previous_index, 0)]
    used = (previous_index >= 0) & (at_product[np.maximum(previous_index, 0)] >= t[block[ends]]) & previous.label_valid.eq(1).to_numpy()
    previous_start = previous.start.to_numpy(dtype='datetime64[s]').astype('int64')
    assert np.all(previous_start[used] >= t[block[ends]][used])
    assert np.all(at_product[np.maximum(previous_index, 0)][used] <= t[ends][used])
    checks.append('Completed-product inputs are available by prediction time and stay within context partition')
    np.testing.assert_array_equal(target['hazard_y'].sum(1) > 0, s.event_in_300s)
    assert np.all(target['hazard_y'] <= target['hazard_mask'])
    assert np.all(s.loc[s.event_in_300s, 'eta_seconds'].between(1, 300))
    assert np.all(s.loc[s.event_in_300s, 'eta_seconds'] <= s.loc[s.event_in_300s, 'observed_seconds'])
    checks.append('300-second event targets and censor masks are consistent')
    np.testing.assert_array_equal(s.end_index, r.end_index)
    valid = r.rul_valid.to_numpy()
    event = r.rul_event.to_numpy()
    assert np.all(r.loc[valid & event, 'rul_lower_hours'] > 0)
    assert np.all(r.loc[valid & event, 'rul_upper_hours'] >= r.loc[valid & event, 'rul_lower_hours'])
    assert np.isinf(r.loc[valid & ~event, 'rul_upper_hours']).all()
    assert np.isinf(r.loc[valid & ~event, 'rul_upper_pieces']).all()
    assert r[valid & event].groupby('rul_endpoint_id').partition.nunique().max() == 1
    checks.append('Lifetime intervals, right censoring and endpoint partition isolation')
    c = pd.read_parquet(DATA / 'replacement_candidates.parquet')
    at = c[c.candidate_kind.ne('after_voltage_failure')].start.to_numpy(dtype='datetime64[s]').astype('int64')
    now = r.timestamp.to_numpy(dtype='datetime64[s]').astype('int64')
    next_intervention = np.r_[at, np.iinfo('int64').max // 2][np.searchsorted(at, now, side='right')]
    observation_end = r.rul_observation_end.to_numpy(dtype='datetime64[s]').astype('int64')
    assert np.all(observation_end[valid] <= next_intervention[valid])
    checks.append('Ambiguous maintenance truncates lifetime follow-up')
    source_manifest = pd.read_parquet(DATA / 'sources.parquet')
    assert not source_manifest.astype(str).apply(lambda x: x.str.contains('Voltage Sample', case=False)).any().any()
    checks.append('Other-device Voltage Sample absent from source manifest')
    if args.final:
        frozen = json.loads((MODEL / 'pretest_freeze.json').read_text('utf-8'))
        for filename, digest in frozen['artifact_sha256'].items():
            assert sha(MODEL / filename) == digest, filename
        for filename, digest in frozen['data_sha256'].items():
            assert sha(DATA / filename) == digest, filename
        final = json.loads((EVAL / 'final_metrics.json').read_text('utf-8'))
        assert final['pretest_freeze_sha256'] == sha(MODEL / 'pretest_freeze.json')
        assert final['primary_risk_model'] == frozen['primary_risk_model']
        assert frozen['test_used_for_selection'] is False
        checks.append('Test used frozen checkpoints, targets, normalization and selection')
        for name in frozen['evaluated_models']:
            pred = np.load(EVAL / f'{name}_test.npz')
            np.testing.assert_array_equal(pred['ids'], np.flatnonzero(s.partition.eq(3)))
            assert np.isfinite(pred['risk']).all()
            assert ((pred['risk'] >= 0) & (pred['risk'] <= 1)).all()
        for path in EVAL.glob('lifetime_*_test.npz'):
            pred = np.load(path)['predictions']
            assert np.isfinite(pred).all() and (pred >= 0).all()
            assert (np.diff(pred, axis=1) >= 0).all()
        checks.append('All saved held-out predictions are finite and in valid ranges')
        contract = json.loads((MODEL / 'output_contract.json').read_text('utf-8'))
        payload = json.loads((EVAL / 'dashboard_payload_example.json').read_text('utf-8'))
        assert contract['pressure_forecast'] is False and contract['automatic_gun_replacement'] is False
        assert 'pressure_60s' not in payload and 'pressure_trend' not in payload
        assert payload['gun_health'] is None and payload['tank_health'] is None
        assert payload['interval_nominal'] is None
        checks.append('Untrained health scores and uncalibrated coverage are not published as valid')
    result = {'mode': 'final' if args.final else 'data', 'passed': True, 'checks': checks,
              'window_counts': s.groupby('partition').size().to_dict(),
              'effective_lifetime_counts': {str(k): {'windows': int((valid & r.partition.eq(k)).sum()),
                 'weak_endpoints': int(r.loc[valid & event & r.partition.eq(k), 'rul_endpoint_id'].nunique())} for k in range(4)}}
    dump(result, EVAL / ('verification_final.json' if args.final else 'verification_data.json'))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
