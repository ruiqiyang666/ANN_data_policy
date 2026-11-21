"""Read-only audit of V1 plus one train-only regularization ablation.

No old models, targets, thresholds or reports are changed. No test predictions
are used to fit or choose the diagnostic candidate.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from common import DATA, MODEL, EVAL, dump, sha, TYPES
from survival import fit_aft, predict_aft, interval_distance
from modeling import CachedData, masked_mean, weighted_stratified_ids
from train_neural import load_model


def summary(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return dict(zip(['min', 'p10', 'median', 'p90', 'max'], np.quantile(a, [0, .1, .5, .9, 1]).tolist())) if len(a) else {}


def gradient_audit(cache, train_ids):
    model = load_model(MODEL / 'b1_ssl_tcn.pt')
    model.eval()
    rng = np.random.default_rng(42)
    ids, weights = weighted_stratified_ids(train_ids, cache.samples.event_in_300s.to_numpy(), 1024, rng)
    x, stats, targets = cache.batch(ids)
    out = model(x, stats)
    event = targets['hazard_y'].sum(1) > 0
    importance = torch.from_numpy(weights).cuda()
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    components = {
        'future_hazard': masked_mean(bce(out['hazard'], targets['hazard_y'], reduction='none'), targets['hazard_mask'], importance),
        'future_type_weighted': .25 * bce(out['type'][event], targets['type_y'][event]),
        'last_product_weighted': .25 * masked_mean(bce(out['current'], targets['current_y'], reduction='none'), targets['current_mask'], importance),
    }
    parameters = list(model.input_projection.parameters()) + list(model.blocks.parameters())
    gradients = {}
    result = {'sample_count': len(ids), 'sampled_event_windows': int(event.sum()), 'checkpoint': 'b1_ssl_tcn',
              'scope': 'single deterministic training-only stratified batch; gradient imbalance is a diagnostic, not causal proof of generalization error',
              'components': {}}
    for name, loss in components.items():
        g = torch.autograd.grad(loss, parameters, retain_graph=True)
        gradients[name] = torch.cat([p.reshape(-1) for p in g])
        result['components'][name] = {'weighted_loss': float(loss.detach()), 'encoder_gradient_norm': float(gradients[name].norm())}
    result['gradient_cosines'] = {}
    for name in ['future_type_weighted', 'last_product_weighted']:
        result['gradient_cosines'][name] = float(torch.nn.functional.cosine_similarity(gradients['future_hazard'], gradients[name], dim=0))
    return result


def main():
    OUT.mkdir(exist_ok=True)
    before = {p: sha(p) for p in [MODEL / 'pretest_freeze.json', MODEL / 'lifetime_models.joblib', EVAL / 'final_metrics.json']}
    torch.set_num_threads(4)
    cache = CachedData(DATA)
    samples = cache.samples
    tr = np.flatnonzero(samples.partition.eq(0))
    r = pd.read_parquet(DATA / 'rul_samples.parquet')
    valid = r.rul_valid.to_numpy()
    ev = r.rul_event.to_numpy()
    ids = np.flatnonzero(r.partition.eq(0).to_numpy() & valid)
    norm = json.loads((DATA / 'normalization.json').read_text('utf-8'))
    name_to_id = {name: i for i, name in enumerate(norm['stats_names'])}
    ix = [name_to_id['Voltage_mean_300s'], name_to_id['Current_mean_300s']]
    x = np.asarray(cache.stats[ids][:, ix], dtype=float)
    rows = r.iloc[ids]
    lo = rows.rul_lower_hours.to_numpy()
    hi = rows.rul_upper_hours.to_numpy()
    event = rows.rul_event.to_numpy()
    baseline = json.loads((MODEL / 'lifetime_fit.json').read_text('utf-8'))['variants']['physical']['hours']['fit']
    result = {'scope': 'V1 diagnosis; no new deployment model selected', 'life_feature_count': 2,
              'life_features': [norm['stats_names'][i] for i in ix],
              'life_features_missing_from_v1': ['powered-time voltage slope', 'recipe-adjusted voltage margin', 'observed usage history'],
              'original_regularization': baseline['penalty'], 'original_coefficients': baseline['coefficients'],
              'original_train_prediction_hours': summary(predict_aft(baseline, x)[:, 1]),
              'train_event_target_lower_hours': summary(lo[event]), 'train_cycles': [], 'event_type_distribution': {}}
    for cycle, group in rows.groupby('possible_cycle'):
        e = group.rul_event
        result['train_cycles'].append({'cycle': int(cycle), 'start': group.timestamp.min(), 'end': group.timestamp.max(),
                                      'windows': len(group), 'event_windows': int(e.sum()),
                                      'event_windows_last_5_minutes': int((e & group.rul_lower_hours.le(5/60)).sum()),
                                      'weak_endpoints': group.loc[e, 'rul_endpoint_id'].unique().tolist()})
    events = pd.read_parquet(DATA / 'events.parquet')
    for partition, name in [(0, 'train'), (1, 'selection')]:
        event_ids = samples.loc[samples.partition.eq(partition) & samples.event_in_300s, 'event_id'].unique()
        result['event_type_distribution'][name] = events[events.event_id.isin(event_ids)].first_types.value_counts().to_dict()
    # Quantify how much the raw rolling voltage is affected by past unpowered state.
    grid_t = np.load(DATA / 'grid_t.npy', mmap_mode='r')
    block = np.load(DATA / 'grid_block_start.npy', mmap_mode='r')
    ends = cache.ends[tr]
    left = np.maximum(block[ends], np.searchsorted(grid_t, grid_t[ends] - 299))
    powered = (cache.x[:, 4] > 100) & (cache.x[:, 5] > 5)
    count = np.r_[0, np.cumsum(powered, dtype=np.int64)]
    fraction = (count[ends+1] - count[left]) / (ends-left+1)
    voltage_means = cache.stats[tr, ix[0]] * norm['stats_scale'][ix[0]] + norm['stats_center'][ix[0]]
    result['past_300s_operating_state'] = {'fraction_powered': summary(fraction),
        'windows_with_unpowered_history_fraction': float(np.mean(fraction < .999)),
        'rolling_voltage_vs_powered_fraction_correlation': float(np.corrcoef(voltage_means, fraction)[0, 1]),
        'interpretation': 'The V1 rolling mean mixes powered and unpowered rows. This is an observed feature property, not proof that all predictive failures have this cause.'}
    # One-factor ablation. Same labels/features/group weights; only penalty changes.
    with threadpool_limits(limits=2):
        ablation = fit_aft(x, lo, hi, event, rows.possible_cycle.to_numpy(), penalty=.1)
    pred = predict_aft(ablation, x)[:, 1]
    old = predict_aft(baseline, x)[:, 1]
    near = event & (lo <= 5/60)
    result['regularization_ablation_train_only'] = {
        'penalty': .1, 'optimizer_success': ablation['optimizer_success'], 'coefficients': ablation['coefficients'],
        'prediction_hours': summary(pred),
        'baseline_train_event_mean_interval_distance': float(interval_distance(old[event], lo[event], hi[event]).mean()),
        'ablation_train_event_mean_interval_distance': float(interval_distance(pred[event], lo[event], hi[event]).mean()),
        'near_failure_windows': int(near.sum()), 'near_failure_baseline_hours': summary(old[near]),
        'near_failure_ablation_hours': summary(pred[near]),
        'validation_or_test_used': False, 'deployment_claim': False}
    dump(ablation, OUT / 'diagnostic_penalty_0p1_aft.json')
    result['multitask_gradient_audit'] = gradient_audit(cache, tr)
    for path, digest in before.items():
        assert sha(path) == digest, f'Original artifact changed: {path}'
    result['v1_frozen_artifacts_unchanged'] = True
    dump(result, OUT / 'diagnosis.json')
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == '__main__':
    main()
