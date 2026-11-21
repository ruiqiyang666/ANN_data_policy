"""Freeze selection and exploratory alarm rules BEFORE a single held-out test pass."""
import sys
sys.dont_write_bytecode = True
import torch
import json
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import average_precision_score, brier_score_loss
from common import DATA, MODEL, EVAL, ROOT, TYPES, FIELDS, dump, sha, status
from modeling import CachedData
from train_neural import load_model, predict, score
from train_lifetime import features_for, evaluate_lifetime


def alert_indices(samples, risk, threshold):
    """Rising-edge alerts, 300 s minimum separation; gaps reset high-state."""
    times = samples.timestamp.to_numpy(dtype='datetime64[s]').astype('int64')
    observable = samples.event_in_300s.to_numpy() | (samples.observed_seconds.to_numpy() >= 300)
    last_alert = -10**18
    previous_high = False
    last_time = -10**18
    alerts = []
    for i, (t, probability) in enumerate(zip(times, risk)):
        high = bool(observable[i] and probability >= threshold)
        if t - last_time > 10:
            previous_high = False
        if high and not previous_high and t - last_alert >= 300:
            alerts.append(i)
            last_alert = t
        previous_high = high
        last_time = t
    return np.asarray(alerts, dtype=int)


def event_evaluation(samples, risk, threshold, events):
    alerts = alert_indices(samples, risk, threshold)
    a = samples.iloc[alerts]
    correct = a.event_in_300s.to_numpy()
    detected = set(a.loc[correct, 'event_id'])
    observable_ids = set(samples.loc[samples.event_in_300s, 'event_id'])
    known_hours = float(((samples.observed_seconds >= 300) | samples.event_in_300s).sum() * 10 / 3600)
    records = []
    for event in events.itertuples():
        hits = a[(a.event_id == event.event_id) & a.event_in_300s]
        first = hits.timestamp.min() if len(hits) else pd.NaT
        records.append({'event_id': event.event_id, 'types': event.first_types,
                        'product_start': event.first_product_start, 'label_available_at': event.event_at,
                        'has_observable_prediction_window': event.event_id in observable_ids,
                        'alerted': event.event_id in detected, 'first_alert': first if pd.notna(first) else None,
                        'lead_seconds_before_result': float((event.event_at - first).total_seconds()) if pd.notna(first) else None,
                        'lead_seconds_before_product_start': float((event.first_product_start - first).total_seconds()) if pd.notna(first) else None})
    before_start = sum(r['alerted'] and r['lead_seconds_before_product_start'] > 0 for r in records)
    summary = {'threshold': threshold, 'threshold_status': 'exploratory; selection-negative 99.5 percentile, no independent calibration',
               'alert_policy': 'rising edge; minimum 300 natural seconds between emitted alerts; >10 s landmark gap resets high state',
               'total_events_in_retained_time_partition': len(events), 'events_with_observable_prediction_windows': len(observable_ids),
               'events_alerted': len(detected), 'events_alerted_before_product_start': before_start,
               'event_recall_observable': len(detected) / len(observable_ids) if observable_ids else None,
               'event_recall_all': len(detected) / len(events) if len(events) else None,
               'alert_episodes': len(alerts), 'false_alert_episodes': int((~correct).sum()),
               'observable_powered_landmark_hours': known_hours,
               'false_alerts_per_100_observable_powered_hours': float((~correct).sum() / known_hours * 100) if known_hours else None,
               'exposure_method': '10 seconds per eligible powered landmark; excludes warm-up, gaps and censored horizons'}
    return summary, pd.DataFrame(records), a.assign(predicted_risk=np.asarray(risk)[alerts])


def task_metrics(cache, ids, prediction):
    samples = cache.samples.iloc[ids]
    positive = samples.event_in_300s.to_numpy()
    type_result = {}
    current_result = {}
    for j, name in enumerate(TYPES):
        y = cache.targets['type_y'][ids, j][positive].astype(bool)
        p = prediction['type_probability'][positive, j]
        type_result[name] = {'positive_windows': int(y.sum()), 'evaluated_event_windows': len(y),
                             'average_precision': float(average_precision_score(y, p)) if y.any() else None,
                             'brier': float(brier_score_loss(y, p)) if len(y) else None,
                             'independent_events': int(samples.loc[positive & cache.targets['type_y'][ids, j].astype(bool), 'event_id'].nunique())}
        mask = cache.targets['current_mask'][ids, j].astype(bool)
        cy = cache.targets['current_y'][ids, j][mask].astype(bool)
        cp = prediction['current_probability'][mask, j]
        current_result[name] = {'positive_windows': int(cy.sum()), 'evaluated_windows': len(cy),
                                'average_precision': float(average_precision_score(cy, cp)) if cy.any() else None,
                                'label_meaning': 'most recent available completed-product result, not independently annotated instantaneous physical state',
                                'result_available_in_input': True}
    eta_error = np.abs(prediction['eta'][positive] - samples.eta_seconds.to_numpy()[positive])
    by_event = []
    for eid in samples.loc[positive, 'event_id'].unique():
        mask = samples.event_id.eq(eid).to_numpy() & positive
        by_event.append({'event_id': eid, 'windows': int(mask.sum()),
                         'conditional_eta_mae_seconds': float(np.abs(prediction['eta'][mask] - samples.eta_seconds.to_numpy()[mask]).mean())})
    return {'future_type': type_result, 'current_last_completed_product': current_result,
            'conditional_eta_all_positive_windows_mae_seconds': float(eta_error.mean()) if len(eta_error) else None,
            'conditional_eta_event_macro_mae_seconds': float(np.mean([v['conditional_eta_mae_seconds'] for v in by_event])) if by_event else None,
            'eta_evaluation_scope': 'all true event windows including unalerted events; does not imply useful advance warning',
            'eta_by_event': by_event}


def freeze(cache):
    path = MODEL / 'pretest_freeze.json'
    if path.exists():
        frozen = json.loads(path.read_text('utf-8'))
        for filename, digest in frozen['artifact_sha256'].items():
            if sha(MODEL / filename) != digest:
                raise RuntimeError(f'Frozen artifact changed: {filename}')
        return frozen
    required = ['b0_tcn', 'b1_ssl_tcn', 'b1_ssl_tcn_seed43', 'b1_ssl_tcn_seed44']
    for name in required:
        if not (MODEL / f'{name}_metrics.json').exists():
            raise RuntimeError(f'Training still incomplete: {name}')
    if not (MODEL / 'lifetime_fit.json').exists():
        raise RuntimeError('Lifetime fitting still incomplete')
    sel = np.flatnonzero(cache.samples.partition.eq(1))
    selection = cache.samples.iloc[sel]
    negative = (~selection.event_in_300s.to_numpy()) & (selection.observed_seconds.to_numpy() >= 300)
    scores = {}
    thresholds = {}
    names = ['logistic', 'hist_tree'] + required
    for name in names:
        pred = np.load(EVAL / f'{name}_selection.npz')
        np.testing.assert_array_equal(pred['ids'], sel)
        scores[name] = score(selection, pred['risk'])
        thresholds[name] = float(np.quantile(pred['risk'][negative], .995))
    candidates = ['logistic', 'hist_tree', 'b0_tcn', 'b1_ssl_tcn']
    selected = max(candidates, key=lambda name: scores[name]['average_precision'])
    files = ['logistic.joblib', 'hist_tree.joblib', 'lifetime_models.joblib'] + [f'{n}.pt' for n in required]
    frozen = {'frozen_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'selection_scores': scores,
              'primary_risk_model': selected, 'primary_survival_model': 'physical',
              'primary_basis': 'highest seed42 selection average precision across four candidates; repeat seeds are sensitivity only',
              'neural_reference': json.loads((MODEL / 'selected_model.json').read_text('utf-8'))['selected'],
              'evaluated_models': names, 'thresholds': thresholds,
              'threshold_method': '99.5 percentile of fully observed selection negatives; fixed exploratory rule, not independent probability calibration',
              'independent_calibration_windows': 0, 'automated_replacement_enabled': False,
              'test_used_for_selection': False, 'artifact_sha256': {f: sha(MODEL / f) for f in files},
              'data_sha256': {f: sha(DATA / f) for f in ['samples.parquet', 'rul_samples.parquet', 'normalization.json', 'targets.npz']},
              'release_decision_predeclared': 'research/retrospective use only; no confirmed natural-failure labels and no independent calibration',
              'additional_architectures': 'PatchTST, Chronos and 60 s ablation deferred; two selection events cannot support further model search in this first run'}
    dump(frozen, path)
    return frozen


def main():
    if (EVAL / 'final_metrics.json').exists():
        print('Held-out test already evaluated; retaining frozen report.', flush=True)
        return
    torch.set_num_threads(8)
    cache = CachedData(DATA)
    frozen = freeze(cache)
    status('heldout_evaluation_started', selected=frozen['primary_risk_model'], freeze_sha256=sha(MODEL / 'pretest_freeze.json'))
    ids = np.flatnonzero(cache.samples.partition.eq(3))
    s = cache.samples.iloc[ids]
    events = pd.read_parquet(DATA / 'events.parquet')
    gt = np.load(DATA / 'grid_t.npy', mmap_mode='r')
    gp = np.load(DATA / 'grid_part.npy', mmap_mode='r')
    et = events.event_at.to_numpy(dtype='datetime64[s]').astype('int64')
    ix = np.minimum(np.searchsorted(gt, et), len(gt) - 1)
    events = events[(gt[ix] == et) & (gp[ix] == 3)]
    report = {'test_windows': len(ids), 'pretest_freeze_sha256': sha(MODEL / 'pretest_freeze.json'),
              'primary_risk_model': frozen['primary_risk_model'], 'risk_models': {}, 'neural_multitask': {},
              'lifetime': {}, 'independent_calibration': False, 'production_ready': False,
              'true_natural_failures_confirmed': 0}
    predictions = {}
    for name in frozen['evaluated_models']:
        if name in ['logistic', 'hist_tree']:
            pred = {'risk': joblib.load(MODEL / f'{name}.joblib').predict_proba(cache.stats[ids])[:, 1]}
        else:
            model = load_model(MODEL / f'{name}.pt')
            pred = predict(model, cache, ids)
            report['neural_multitask'][name] = task_metrics(cache, ids, pred)
            del model
        predictions[name] = pred
        window_score = score(s, pred['risk'])
        alert_score, per_event, alerts = event_evaluation(s, pred['risk'], frozen['thresholds'][name], events)
        report['risk_models'][name] = {'windows': window_score, 'alarms': alert_score}
        np.savez(EVAL / f'{name}_test.npz', ids=ids, **pred)
        per_event.to_csv(EVAL / f'{name}_test_events.csv', index=False, encoding='utf-8-sig')
        alerts.to_csv(EVAL / f'{name}_test_alerts.csv', index=False, encoding='utf-8-sig')
        print(json.dumps({'model': name, 'window_score': window_score, 'alarm_score': alert_score}, ensure_ascii=False), flush=True)
    bundle = joblib.load(MODEL / 'lifetime_models.joblib')
    lifetime_samples = pd.read_parquet(DATA / 'rul_samples.parquet').iloc[ids]
    valid = lifetime_samples.rul_valid.to_numpy()
    features = features_for(bundle, cache.stats[ids], predictions[bundle['encoder']]['embedding'])
    life_predictions = {}
    for name, tasks in bundle['models'].items():
        report['lifetime'][name] = {}
        for task, model in tasks.items():
            metrics, pred = evaluate_lifetime(model, features[name][valid], lifetime_samples.loc[valid], task)
            report['lifetime'][name][task] = metrics
            np.savez(EVAL / f'lifetime_{name}_{task}_test.npz', ids=ids[valid], predictions=pred)
            life_predictions[(name, task)] = pred
    current_zero = {t: int(cache.targets['current_y'][ids, j].sum()) for j, t in enumerate(TYPES)}
    report['current_positive_window_counts'] = current_zero
    dump(report, EVAL / 'final_metrics.json')
    contract = {'version': 'real28_20261104_v1', 'raw_input_fields': FIELDS, 'sequence_channels': 56,
                'statistics_features': 108, 'history_seconds': 120, 'forecast_horizon_seconds': 300,
                'operating_scope': 'observed Current > 100 A and Power > 5 kW; 120 s continuous historical context; gaps >2 s reset context',
                'timestamp_basis': 'instrument local time Asia/Shanghai', 'rul_basis': 'remaining powered hours, not wall-clock hours',
                'risk_primary_model': frozen['primary_risk_model'], 'lifetime_primary_model': 'physical AFT',
                'pressure_forecast': False, 'pressure_now_conversion': 'mbar / 1000 = bar; preserve physical hopper identity',
                'deployment_status': 'experimental_offline_only', 'automatic_gun_replacement': False,
                'tasks': {'rul_hours': 'fitted_weak_proxy_not_natural_failure_validated',
                          'rul_pieces': 'fitted_independent_discrete_count_weak_proxy',
                          'rul_interval_hours': 'model_10_90_quantiles_not_calibrated_coverage',
                          'anomaly_300s_probability': 'fitted_uncalibrated_experimental_probability',
                          'predicted_anomaly_eta_s': 'TCN conditional estimate only; unavailable for primary tree/logistic risk',
                          'next_anomaly_type': 'TCN conditional multi-label estimate; not paired with a separate tree/logistic probability',
                          'current_anomaly': 'last_completed_product_proxy_head; online instantaneous detection not validated',
                          'root_candidates': 'not_implemented_no_verified_physical_root_causes',
                          'gun_health': 'not_trained_no_independent_score_labels', 'tank_health': 'not_trained_no_independent_score_labels'},
                'compatibility': 'Requires adapter and null/status-aware dashboard; not compatible with legacy 21-parameter checkpoint loader'}
    dump(contract, MODEL / 'output_contract.json')
    # One actual held-out landmark is exported for review, with unvalidated numbers
    # explicitly separated from the operational fields. No dashboard files change.
    selected = frozen['primary_risk_model']
    at = int(np.flatnonzero(valid)[0])
    rid = int(ids[at])
    row = int(cache.ends[rid])
    physical_hours = life_predictions[('physical', 'hours')][0]
    physical_pieces = life_predictions[('physical', 'pieces')][0]
    example = {'timestamp': s.iloc[at].timestamp, 'model_version': contract['version'], 'model_name': selected,
               'model_type': 'real28_experimental', 'seq': rid, 'gun_id': None,
               'history_context': {'seconds': 120, 'statistics_max_seconds': 3600, 'statistics_coverage_is_masked': True},
               'deployment_status': 'experimental_offline_only', 'rul_hours': None, 'rul_pieces': None,
               'rul_interval_hours': None, 'rul_interval_pieces': None, 'interval_nominal': None,
               'anomaly_300s_probability': None, 'predicted_anomaly_eta_s': None, 'next_anomaly_type': None,
               'anomaly_type': None, 'anomaly_probability': None, 'gun_health': None, 'tank_health': None,
               'warning_level': None, 'root_candidates': [], 'root_status': 'not_implemented', 'sigma_deviations': None,
               'pressure_now': {'hopper_1_bar': float(cache.x[row, 15]) / 1000,
                                'hopper_2_bar': float(cache.x[row, 22]) / 1000},
               'input_parameters': dict(zip(FIELDS, cache.x[row])), 'task_status': contract['tasks'],
               'experimental_estimates': {'rul_hours': float(physical_hours[1]), 'rul_pieces': float(physical_pieces[1]),
                                          'rul_model_quantiles_hours': physical_hours[[0, 2]],
                                          'rul_model_quantiles_pieces': physical_pieces[[0, 2]],
                                          'anomaly_300s_probability': float(predictions[selected]['risk'][at]),
                                          'exploratory_alarm_threshold': frozen['thresholds'][selected]},
               'warning_reasons': ['首轮实验模型，未校准；自然失效寿命仅弱证据拟合，暂不据此自动安排换枪。']}
    dump(example, EVAL / 'dashboard_payload_example.json')
    status('first_training_run_complete', selected=selected, test_evaluated=True, production_ready=False,
           results=str(EVAL / 'final_metrics.json'))


if __name__ == '__main__':
    main()
