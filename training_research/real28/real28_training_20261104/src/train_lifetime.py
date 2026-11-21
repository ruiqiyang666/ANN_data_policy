"""Fit low-dimensional, group-weighted survival experiments; never use test to fit.

All endpoints remain weak operational proxies. These fits are not proof of
natural electrode lifetime, and do not provide an automatic replacement policy.
"""
import sys
sys.dont_write_bytecode = True
import torch
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from common import DATA, MODEL, EVAL, dump, sha
from modeling import CachedData
from train_neural import load_model, predict
from survival import fit_aft, predict_aft, lognormal_nll, interval_distance


def features_for(bundle, stats, embedding=None):
    physical = np.asarray(stats)[:, bundle['physical_indices']]
    result = {'physical': physical.astype('float64')}
    if embedding is not None:
        pc = bundle['pca_scaler'].transform(bundle['pca'].transform(embedding))
        result['frozen_tcn'] = np.column_stack([physical, pc])
    return result


def bounds(samples, task):
    lo = samples[f'rul_lower_{task}'].to_numpy(dtype=float)
    hi = samples[f'rul_upper_{task}'].to_numpy(dtype=float)
    event = samples.rul_event.to_numpy(dtype=bool)
    # N=floor(T): observed integer interval [L,U] has continuous mass [L,U+1).
    if task == 'pieces':
        hi = np.where(event, hi + 1, hi)
    return lo, hi, event


def evaluate_lifetime(model, features, samples, task):
    lo, hi, event = bounds(samples, task)
    pred = predict_aft(model, features, pieces=(task == 'pieces'))
    beta = np.asarray(model['coefficients'])
    nll = lognormal_nll(beta[0] + features @ beta[1:], model['sigma'], lo, hi, event)
    groups = samples.possible_cycle.to_numpy()
    grouped_nll = [float(nll[groups == g].mean()) for g in np.unique(groups)]
    result = {'windows': len(samples), 'possible_cycles': len(np.unique(groups)),
              'event_windows': int(event.sum()), 'weak_endpoints': int(samples.loc[event, 'rul_endpoint_id'].nunique()),
              'group_mean_interval_censor_nll': float(np.mean(grouped_nll)) if grouped_nll else None,
              'confirmed_natural_failure_mae': None,
              'independent_interval_coverage': None,
              'prediction_median_min': float(pred[:, 1].min()) if len(pred) else None,
              'prediction_median_max': float(pred[:, 1].max()) if len(pred) else None}
    per_event = []
    for endpoint, rows in samples.loc[event].groupby('rul_endpoint_id'):
        ids = np.flatnonzero(samples.rul_endpoint_id.eq(endpoint).to_numpy() & event)
        upper = rows[f'rul_upper_{task}'].to_numpy()
        lower = rows[f'rul_lower_{task}'].to_numpy()
        per_event.append({'endpoint': endpoint, 'windows': len(ids),
                          'median_distance_to_weak_interval': float(np.median(interval_distance(pred[ids, 1], lower, upper))),
                          'mean_distance_to_weak_interval': float(np.mean(interval_distance(pred[ids, 1], lower, upper))),
                          'model_interval_overlap_fraction': float(np.mean((pred[ids, 0] <= upper) & (pred[ids, 2] >= lower))),
                          'mean_model_interval_width': float(np.mean(pred[ids, 2] - pred[ids, 0])),
                          'mean_true_lower': float(lower.mean()), 'mean_true_upper': float(upper.mean()),
                          'mean_prediction': float(pred[ids, 1].mean())})
    result['per_weak_endpoint'] = per_event
    return result, pred


def main():
    output = MODEL / 'lifetime_models.joblib'
    if output.exists() and (MODEL / 'lifetime_fit.json').exists():
        print('Existing immutable lifetime fit retained.', flush=True)
        return
    if not (MODEL / 'selected_model.json').exists():
        raise RuntimeError('Wait for seed42 neural model selection before lifetime fitting')
    torch.set_num_threads(8)
    cache = CachedData(DATA)
    samples = pd.read_parquet(DATA / 'rul_samples.parquet')
    np.testing.assert_array_equal(cache.samples.end_index, samples.end_index)
    tr = np.flatnonzero(samples.partition.eq(0).to_numpy() & samples.rul_valid.to_numpy())
    selection = np.flatnonzero(samples.partition.eq(1).to_numpy() & samples.rul_valid.to_numpy())
    encoder = json.loads((MODEL / 'selected_model.json').read_text('utf-8'))['selected']
    neural = load_model(MODEL / f'{encoder}.pt')
    tr_embedding = predict(neural, cache, tr)['embedding']
    selection_embedding = predict(neural, cache, selection)['embedding']
    names = json.loads((DATA / 'normalization.json').read_text('utf-8'))['stats_names']
    indices = [names.index('Voltage_mean_300s'), names.index('Current_mean_300s')]
    pca = PCA(n_components=1, svd_solver='full').fit(tr_embedding)
    pca_scaler = StandardScaler().fit(pca.transform(tr_embedding))
    bundle = {'physical_indices': indices, 'physical_features': [names[i] for i in indices],
              'pca': pca, 'pca_scaler': pca_scaler, 'encoder': encoder,
              'encoder_sha256': sha(MODEL / f'{encoder}.pt'), 'models': {}}
    fit_features = features_for(bundle, cache.stats[tr], tr_embedding)
    selection_features = features_for(bundle, cache.stats[selection], selection_embedding)
    audit = {'train_windows': len(tr), 'selection_windows': len(selection),
             'fit_partition': 'train', 'test_accessed': False, 'encoder': encoder,
             'true_natural_failures_confirmed': 0, 'penalty': 10.0,
             'primary': 'physical',
             'primary_basis': 'predeclared simple two-feature model; one weak selection endpoint is insufficient to select complex survival models',
             'piece_distribution': 'floor of independent lognormal latent count; interval upper edge U+1',
             'weighting': 'each possible cycle has equal total likelihood weight; windows within each cycle share that weight',
             'censor_assumption': 'weak normal-product observation prefix, interrupted by gaps, unknown labels, low voltage or ambiguous maintenance; not confirmed electrode identity',
             'label_sha256': sha(DATA / 'rul_samples.parquet'), 'variants': {}}
    for name, x in fit_features.items():
        bundle['models'][name] = {}
        audit['variants'][name] = {}
        for task in ['hours', 'pieces']:
            lo, hi, event = bounds(samples.iloc[tr], task)
            fitted = fit_aft(x, lo, hi, event, samples.iloc[tr].possible_cycle.to_numpy(), penalty=10.0)
            if not fitted['optimizer_success']:
                raise RuntimeError(f'{name}/{task}: {fitted["optimizer_message"]}')
            bundle['models'][name][task] = fitted
            scores, predictions = evaluate_lifetime(fitted, selection_features[name], samples.iloc[selection], task)
            audit['variants'][name][task] = {'fit': fitted, 'selection': scores}
            np.savez(EVAL / f'lifetime_{name}_{task}_selection.npz', ids=selection, predictions=predictions)
            print(json.dumps({'variant': name, 'task': task, 'objective': fitted['objective'],
                              'selection': scores}, ensure_ascii=False), flush=True)
    joblib.dump(bundle, output)
    audit['artifact_sha256'] = sha(output)
    dump(audit, MODEL / 'lifetime_fit.json')
    print('Lifetime experiments fitted. Natural-failure reliability remains unverified.', flush=True)


if __name__ == '__main__':
    main()
