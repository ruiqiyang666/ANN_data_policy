"""Explicit, isolated synthetic training for the unchanged supplied Phase2 engine.

Generated labels are scenario truth. Fits and holdout metrics are real computations,
not factory evidence. No files or trained artifacts from other projects are loaded.
"""
from __future__ import annotations
import hashlib
import argparse
import sys
import tempfile
import platform
import json
import math
import shutil
import time
from pathlib import Path
sys.dont_write_bytecode = True
COPY_ROOT = Path(__file__).resolve().parents[1]
SOURCE = COPY_ROOT / 'source/ANN-feature-ai-agent/simulation/pipeline'
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
# Load the original architecture and feature implementation without modifying them.
import torch
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
import joblib
from domain.contract import NUMERIC, STATES, ENERGIZED, FIELDS, REG_TARGETS
from domain.features import feature_rows
from networks.multitask import MultiTaskNet, normalize_inputs, decode, SCALE

SEED = 20261110
ROWS_PER_GUN = 7200
GUN_COUNT = 10
STRIDE = 60
VERSION = 'SYNTHETIC-EXTRATREES-20261110'
LIMITATIONS = [
    'Entire dataset and lifecycle/anomaly labels are generated fictional scenarios.',
    'Metrics are held-out synthetic lifecycle results, not real-equipment accuracy.',
    'Short two-hour fictional lifecycles fall inside the unchanged original warning bands.',
    'Classification probabilities and intervals have not been validated on factory data.',
    'Intervals are empirical calibration-window residual ranges; temporal dependence prevents a nominal coverage guarantee.',
    'Only two independent synthetic test cycles and two calibration cycles; windows are repeated observations, not independent equipment.',
    'Current original warning thresholds are retained; these short scenarios are not maintenance recommendations.',
    'The existing 28-channel display is derived from the original 21-input model contract.',
]
BASE = np.array([480, 80, 68, 32.64, 42, 8, 12, 5, 18, 12, 2, 2.4, 1.8, 4.5, 1.45, 1.2, 35, 40, 28], dtype=float)
SPREAD = np.array([8, 1.5, .9, .8, 1.5, .35, .5, .3, .5, .3, .08, .08, .07, .12, .035, .04, 1.2, 1.2, 1.2])


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    temporary.replace(path)


def split_for(gun):
    return 'train' if gun <= 6 else 'calibration' if gun <= 8 else 'test'


def make_gun(gun, rows=ROWS_PER_GUN):
    rng = np.random.default_rng(SEED + gun)
    t = np.arange(rows)
    period = int(rng.integers(293, 330))
    phase = int(rng.integers(0, period))
    cycle = np.floor(((t + phase) % period) * 300 / period).astype(int)
    state_index = np.select([cycle < 10, cycle < 25, cycle < 40, cycle < 220, cycle < 235,
                             cycle < 255, cycle < 265, cycle < 280, cycle < 295], range(9), default=9)
    state = np.array(STATES)[state_index]
    energized = np.isin(state, ENERGIZED)
    spraying = state == 'spraying'
    wear = np.clip(t / max(rows - 1, 1) + rng.normal(0, .015), 0, 1)
    common = np.sin(t / 83 + gun) * .35
    values = BASE + SPREAD * (rng.normal(size=(rows, 19)) * .35 + common[:, None])
    values[:, 2] += 5 * wear
    values[:, 9] += 2 * wear
    values[:, 14] -= .18 * wear
    values[:, 0] *= energized
    values[:, 1:3] *= energized[:, None]
    values[:, 3] = values[:, 0] * values[:, 2] / 1000
    values[:, 4:8] *= np.where(energized, 1, .08)[:, None]
    values[:, [13, 16, 17, 18]] *= np.where(spraying, 1, .03)[:, None]
    anomaly = np.zeros(rows, dtype=int)
    events = []
    for event_index, onset in enumerate(range(700, rows - 400, 1100)):
        kind = 1 + (event_index + gun) % 5
        end = min(onset + 110, rows - 301)
        active = (t >= onset) & (t < end) & energized
        anomaly[active] = kind
        field, multiplier = {1:(8,.52),2:(13,.40),3:(18,.28),4:(4,.48),5:(14,.45)}[kind]
        values[active, field] *= multiplier
        events.append(dict(kind=kind, onset=int(t[active][0]) if active.any() else onset, end=int(t[active][-1]) if active.any() else end))
    values = np.maximum(values, 0)
    remaining_energy = energized[::-1].cumsum()[::-1] - energized
    completed = np.cumsum((state == 'powder_clear') & np.r_[False, spraying[:-1]])
    future = np.zeros(rows, dtype=int)
    eta = np.full(rows, 300., dtype=float)
    next_fault = rows + 301
    for i in range(rows - 1, -1, -1):
        if anomaly[i]: next_fault = i
        if next_fault - i <= 300:
            future[i] = anomaly[next_fault]
            eta[i] = next_fault - i
    frame = pd.DataFrame(values, columns=NUMERIC)
    frame['timestamp'] = pd.date_range(pd.Timestamp('2026-11-01T00:00:00+08:00') + pd.Timedelta(hours=3*(gun-1)), periods=rows, freq='s')
    frame['seq'] = (gun - 1) * rows + t
    frame['gun_id'] = gun
    frame['operation_state'] = state
    frame['machine_message'] = ['SIM_STATE_' + s.upper() for s in state]
    frame['workpiece_id'] = gun * 1000 + completed
    frame['completed_pieces'] = completed
    frame['tank_service'] = False
    frame['is_censored'] = False
    frame['y_rul_valid'] = True
    frame['y_rul_hours'] = remaining_energy / 3600
    frame['y_rul_pieces'] = completed[-1] - completed
    frame['y_gun_health'] = 100 * (1 - wear)
    frame['y_tank_health'] = np.clip(96 - 35*wear - (anomaly == 5)*20, 0, 100)
    frame['y_pressure_60s'] = frame.powder_tank_pressure.shift(-60)
    frame['y_anomaly'] = anomaly
    frame['y_future_anomaly'] = future
    frame['y_anomaly_eta_s'] = eta
    metadata = dict(gun_id=gun, row_group=gun-1, start_seq=int(frame.seq.iloc[0]), terminal_seq=int(frame.seq.iloc[-1]),
                    rows=rows, life_hours=float(energized.sum()/3600), wall_hours=rows/3600,
                    pieces=int(completed[-1]), censored=False, split=split_for(gun), events=events, seed=SEED+gun, operation_period_seconds=period, operation_phase_seconds=phase)
    return frame, metadata


def metric_set(truth, prediction):
    return {key:dict(mae=float(mean_absolute_error(truth[:, i],prediction[:, i])),
                     rmse=float(np.sqrt(mean_squared_error(truth[:, i],prediction[:, i]))),
                     r2=float(r2_score(truth[:, i],prediction[:, i])), rows=len(truth))
            for i,key in enumerate(REG_TARGETS)}


def classification(y, pred):
    normal = y == 0
    return dict(confusion=confusion_matrix(y,pred,labels=np.arange(6)).tolist(), accuracy=float(accuracy_score(y,pred)),
                macro_f1=float(f1_score(y,pred,labels=np.arange(6),average='macro',zero_division=0)),
                false_positive_rate=float(np.mean(pred[normal]!=0)) if normal.any() else None)


def evaluation_frame(labels, prediction):
    result = labels[['gun_id','seq','y_rul_valid',*REG_TARGETS]].copy()
    for i,key in enumerate(REG_TARGETS): result['pred_'+key[2:]] = prediction[:, i]
    return result


def _build_artifacts(destination, include_neural, progress):
    started = time.perf_counter()
    destination.mkdir(parents=True,exist_ok=False)
    frames, guns = [], []
    for gun in range(1,GUN_COUNT+1):
        progress('Generating fictional lifecycles',gun/GUN_COUNT*.15)
        frame, meta = make_gun(gun)
        frames.append(frame); guns.append(meta)
    path = destination/'模拟数据.parquet'
    with pq.ParquetWriter(path,pa.Table.from_pandas(frames[0],preserve_index=False).schema,compression='zstd') as writer:
        for frame in frames: writer.write_table(pa.Table.from_pandas(frame,preserve_index=False))
    dataset = pd.concat(frames,ignore_index=True)
    manifest = dict(rows=len(dataset),bytes=path.stat().st_size,complete_guns=GUN_COUNT,sample_seconds=1,
                    process_channels=21,guns=guns,source_kind='synthetic_demo',seed=SEED,limitations=LIMITATIONS,
                    independent_unit='synthetic equipment lifecycle', cadence_seconds=1, training_stride_seconds=STRIDE,
                    generated_scenarios=True, generator_version='synthetic-training-v1',
                    contract_sha256=hashlib.sha256((SOURCE/'contract.json').read_bytes()).hexdigest())
    write_json(destination/'数据清单.json',manifest)
    parts = {}
    for part in ['train','calibration','test']:
        xx, seq, labels = [], [], []
        for frame,gun in zip(frames,guns):
            if gun['split'] != part: continue
            indices=np.arange(59,len(frame)-300,STRIDE)
            xx.append(feature_rows(frame,indices))
            numeric=frame[NUMERIC].to_numpy(np.float32)
            states=np.stack([(frame.operation_state==s).to_numpy(np.float32) for s in STATES],1)
            rawseq=np.column_stack([numeric,states,np.ones(len(frame),dtype=np.float32)])
            seq.append(rawseq[indices[:,None]-np.arange(59,-1,-1)])
            labels.append(frame.iloc[indices].copy())
        parts[part]=(np.concatenate(xx),np.concatenate(seq),pd.concat(labels,ignore_index=True))
        if not np.isfinite(parts[part][0]).all() or not np.isfinite(parts[part][2][REG_TARGETS].to_numpy()).all():
            raise ValueError('Nonfinite training features or labels')
    X, seq, labels=parts['train']; y=labels[REG_TARGETS].to_numpy(float)
    means, stds = [], []
    normal=pd.concat(frames[:6],ignore_index=True)
    normal=normal[normal.y_anomaly==0]
    for state in STATES:
        a=normal.loc[normal.operation_state==state,NUMERIC].to_numpy(float)
        means.append(a.mean(0)); stds.append(np.maximum(a.std(0),SPREAD*.2))
    scale=np.maximum(y.std(0),1e-4)
    params=dict(n_estimators=48,max_depth=12,min_samples_leaf=2,n_jobs=4,random_state=SEED)
    progress('Fitting ExtraTrees regression',.22)
    reg=ExtraTreesRegressor(**params).fit(X,y/scale)
    progress('Fitting current-anomaly classifier',.30)
    clf=ExtraTreesClassifier(**params,class_weight='balanced').fit(X,labels.y_anomaly)
    progress('Fitting future-anomaly classifier',.38)
    forecast=ExtraTreesClassifier(**params,class_weight='balanced').fit(X,labels.y_future_anomaly)
    progress('Fitting anomaly-time regressor',.44)
    eta=ExtraTreesRegressor(**params).fit(X,labels.y_anomaly_eta_s)
    CX,_,CY=parts['calibration']; TX,_,TY=parts['test']
    cal=reg.predict(CX)*scale; prediction=reg.predict(TX)*scale
    quantile=min(1,math.ceil((len(cal)+1)*.9)/len(cal))
    conformal=np.quantile(np.abs(cal-CY[REG_TARGETS].to_numpy()),quantile,axis=0,method='higher')
    bundle=dict(reg=reg,clf=clf,forecast=forecast,eta=eta,scale=scale,normal_mean=np.array(means),normal_std=np.array(stds),conformal=conformal,version=VERSION)
    joblib.dump(bundle,destination/'模型.joblib')
    test_truth=TY[REG_TARGETS].to_numpy()
    report=dict(version=VERSION,features=88,seconds=time.perf_counter()-started,metrics=metric_set(test_truth,prediction),
                splits={part:dict(rows=len(val[0]),guns=[g['gun_id'] for g in guns if g['split']==part]) for part,val in parts.items()},
                anomaly=classification(TY.y_anomaly.to_numpy(),clf.predict(TX)),
                forecast=classification(TY.y_future_anomaly.to_numpy(),forecast.predict(TX)),
                baselines=dict(rul_train_median_mae=float(np.mean(np.abs(TY.y_rul_hours-np.median(labels.y_rul_hours)))),
                               pressure_persistence_mae=float(np.mean(np.abs(TY.y_pressure_60s-TY.powder_tank_pressure)))),
                conformal_test_coverage=np.mean(np.abs(test_truth-prediction)<=conformal,axis=0).tolist(),
                root_top3_hit=None,acceptance={'synthetic_only':True,'whole_lifecycle_split':True,'causal_features':True},
                limitations=LIMITATIONS,latency_acceptance=None,seed=SEED,
                sample_unit='60-second window nested within independent generated lifecycle',
                interval_method='Empirical window residual quantile on held-out calibration cycles; no coverage guarantee',
                interval_nominal_compatibility=.9, interval_calibration_independent_cycles=2,
                feature_state_counts={part: val[2].operation_state.value_counts().to_dict() for part,val in parts.items()},
                versions={'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
                          'sklearn':__import__('sklearn').__version__},
                per_test_cycle={str(gun): metric_set(TY.loc[TY.gun_id.eq(gun),REG_TARGETS].to_numpy(),
                              prediction[TY.gun_id.eq(gun).to_numpy()]) for gun in sorted(TY.gun_id.unique())})
    evaluation_frame(TY,prediction).to_parquet(destination/'评估预测.parquet',index=False)
    write_json(destination/'评估报告.json',report)
    validation=dict(rows_checked=len(dataset),missing_process_values=int(dataset[NUMERIC].isna().sum().sum()),
                    timestamp_gaps=sum(int((frame.timestamp.diff().iloc[1:].dt.total_seconds()!=1).sum()) for frame in frames),
                    duplicate_rows=int(dataset.duplicated(['gun_id','seq']).sum()),
                    parameter_stats=[dict(key=k,min=float(dataset[k].min()),max=float(dataset[k].max())) for k in NUMERIC],
                    synthetic=True,life_group_splits={g['gun_id']:g['split'] for g in guns})
    write_json(destination/'验证结果.json',validation)
    (destination/'运行报告.md').write_text('# Synthetic model demonstration\n\n'+ '\n'.join('- '+x for x in LIMITATIONS)+'\n\nMeasured holdout metrics are in 评估报告.json. Original model and feature functions are unchanged.\n',encoding='utf8')
    progress('Baseline evaluation complete',.52,metrics=report['metrics'])
    if include_neural:
        build_neural(destination,parts,report,progress)
    files=[dict(path=str(p.relative_to(destination)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
           for p in destination.rglob('*') if p.is_file()]
    write_json(destination/'SYNTHETIC_MODEL_BUILD.json',dict(source='synthetic_demo',seed=SEED,rows=len(dataset),training_seconds=time.perf_counter()-started,
                 created_at=pd.Timestamp.now(tz='UTC').isoformat(),files=files,limits=LIMITATIONS,
                 generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 native_functions_unchanged=True, neural_active=include_neural))
    progress('Synthetic simulation ready',1.0,metrics=report['metrics'])
    return report


def build_neural(destination,parts,baseline_report,progress):
    began=time.perf_counter();torch.set_num_threads(4);torch.manual_seed(SEED)
    X,rawseq,labels=parts['train']
    stats=dict(mean=X.mean(0).tolist(),std=np.maximum(X.std(0),1e-3).tolist())
    seq,flat=normalize_inputs(rawseq,X,stats)
    y=labels[REG_TARGETS].to_numpy(np.float32); y[:,4]-=labels.powder_tank_pressure.to_numpy(np.float32);y/=SCALE
    xt=torch.from_numpy(flat);st=torch.from_numpy(seq);yt=torch.from_numpy(y)
    ct=torch.tensor(labels.y_anomaly.to_numpy(),dtype=torch.long);ft=torch.tensor(labels.y_future_anomaly.to_numpy(),dtype=torch.long)
    et=torch.tensor(labels.y_anomaly_eta_s.to_numpy(np.float32)/300)
    net=MultiTaskNet('patch');optimizer=torch.optim.Adam(net.parameters(),lr=.003)
    history=[]
    for epoch in range(24):
        net.train();order=torch.randperm(len(X));total=0.
        for indices in order.split(128):
            raw=net(st[indices],xt[indices])
            loss=torch.nn.functional.smooth_l1_loss(raw[:,:5],yt[indices])+torch.nn.functional.cross_entropy(raw[:,5:11],ct[indices])+torch.nn.functional.cross_entropy(raw[:,11:17],ft[indices])+torch.nn.functional.mse_loss(raw[:,17].sigmoid(),et[indices])
            optimizer.zero_grad();loss.backward();optimizer.step();total+=float(loss.detach())*len(indices)
        history.append(dict(epoch=epoch+1,loss=total/len(X)))
        progress('Training Patch-Attention',.55+.35*(epoch+1)/24,loss_history=history.copy())
    net.eval()
    def evaluate(part):
        a,b,c=parts[part];ss,ff=normalize_inputs(b,a,stats)
        with torch.inference_mode(): raw=net(torch.from_numpy(ss),torch.from_numpy(ff)).numpy()
        pred,prob,future,eta=decode(raw,c.powder_tank_pressure.to_numpy())
        return c,pred,prob
    CY,cal,_=evaluate('calibration');TY,prediction,prob=evaluate('test')
    quantile=min(1,math.ceil((len(cal)+1)*.9)/len(cal))
    half=np.quantile(np.abs(cal-CY[REG_TARGETS].to_numpy()),quantile,axis=0,method='higher')
    candidate=destination/'neural';candidate.mkdir()
    checkpoint=dict(arch='patch',state_dict=net.state_dict(),stats=stats,conformal=half.tolist(),version='SYNTHETIC-PATCH-20261110',model_name='Patch-Attention · synthetic training')
    torch.save(checkpoint,candidate/'active.pt')
    metrics=metric_set(TY[REG_TARGETS].to_numpy(),prediction)
    model=dict(id='patch',name='Patch-Attention · synthetic training',seconds=time.perf_counter()-began,test=dict(regression=metrics,current=classification(TY.y_anomaly.to_numpy(),prob.argmax(1)),coverage=np.mean(np.abs(TY[REG_TARGETS].to_numpy()-prediction)<=half,axis=0).tolist()),epochs=24,loss_history=history)
    summary=dict(active='patch',name='Patch-Attention · synthetic training',models=[model],version=checkpoint['version'],source_kind='synthetic_demo',limitations=LIMITATIONS)
    write_json(candidate/'summary.json',summary)
    evaluation_frame(TY,prediction).to_parquet(candidate/'evaluation.parquet',index=False)
    write_json(candidate/'training-history.json',history)



def build_artifacts(destination, include_neural=True, progress=None):
    """Build once into a NEW directory inside this copy, publishing only on success.

    Callback signature: progress(phase: str, fraction: float, **details).
    A sibling <directory>.progress.json is updated atomically for callers without
    callbacks. Original source, prior artifacts and model files are never replaced.
    Fixed seed/epochs and group splits are chosen before viewing test metrics.
    """
    destination = Path(destination).expanduser().absolute()
    if destination.is_symlink() or destination.exists():
        raise FileExistsError('Output must be a new unique directory: ' + str(destination))
    resolved = destination.resolve()
    if not resolved.is_relative_to(COPY_ROOT.resolve()) or resolved.is_relative_to(SOURCE.resolve()):
        raise ValueError('Training output must be inside this copy and outside original source')
    destination.parent.mkdir(parents=True, exist_ok=True)
    progress_path = destination.with_name(destination.name + '.progress.json')
    if progress_path.exists() or progress_path.is_symlink():
        raise FileExistsError('Existing progress is preserved: ' + str(progress_path))
    staging = Path(tempfile.mkdtemp(prefix='.synthetic-training-', dir=destination.parent))
    staging.rmdir()  # Builder creates this fresh directory itself.
    started = time.perf_counter()
    def notify(phase, fraction, **details):
        record = dict(status='building', phase=phase, stage=phase, progress=fraction, percent=100*fraction, source='synthetic',
                      epoch=len(details.get('loss_history', [])), epochs=24 if include_neural else 0,
                      elapsed_seconds=time.perf_counter()-started, **details)
        write_json(progress_path, record)
        if progress is not None:
            progress(phase, fraction, **details)
    try:
        torch.set_num_threads(4)
        report = _build_artifacts(staging, bool(include_neural), notify)
        metadata = json.loads((staging/'SYNTHETIC_MODEL_BUILD.json').read_text())
        metadata['provenance'] = 'synthetic'
        metadata['active_model'] = 'Patch-Attention' if include_neural else 'ExtraTrees'
        metadata['activation_basis'] = 'Fixed demonstration choice, not selected using test performance'
        metadata['neural_epochs'] = 24 if include_neural else 0
        if include_neural:
            candidate = json.loads((staging/'neural/summary.json').read_text())['models'][0]
            metadata['comparison'] = {
                'baseline_rul_test_mae': report['metrics']['y_rul_hours']['mae'],
                'patch_rul_test_mae': candidate['test']['regression']['y_rul_hours']['mae'],
                'scope': 'Observed synthetic holdout only; active neural demo is not a quality recommendation',
            }
        write_json(staging/'training-metadata.json', metadata)
        write_json(staging/'progress.json', dict(status='ready', stage='Complete', progress=1., percent=100., epochs=24 if include_neural else 0))
        # Never overwrite an output created by another task while fitting.
        if destination.exists() or destination.is_symlink():
            raise FileExistsError('Output appeared during training; preserved')
        staging.rename(destination)
        write_json(progress_path, dict(status='ready', phase='Synthetic training complete',
                   stage='Complete', percent=100., epoch=24 if include_neural else 0, epochs=24 if include_neural else 0,
                   progress=1., source='synthetic', elapsed_seconds=time.perf_counter()-started,
                   output=str(destination), model='Patch-Attention' if include_neural else 'ExtraTrees',
                   metrics=report['metrics']))
        return report
    except Exception as error:
        write_json(progress_path, dict(status='error', phase='Training failed', progress=0.,
                   source='synthetic', elapsed_seconds=time.perf_counter()-started,
                   error=f'{type(error).__name__}: {error}'))
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--baseline-only', action='store_true')
    args = parser.parse_args()
    report = build_artifacts(args.output, include_neural=not args.baseline_only)
    print(json.dumps(dict(output=str(args.output), source='synthetic', metrics=report['metrics']), indent=2))


if __name__ == '__main__':
    main()
