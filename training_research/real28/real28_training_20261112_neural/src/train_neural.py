"""Fixed MLP risk comparison using the v4 chronological folds, read-only data.

No holdout selection: architecture/epochs are fixed before this run. October's
holdout was already inspected in v4; these are historical comparisons, not a new
blind test. Lifetime and conditional ETA remain the separately fitted v4 heads.
"""
import sys
sys.dont_write_bytecode = True
import os, json, time, hashlib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

BASE = Path(os.environ.get('V4_ROOT', '/old/v4'))
OUT = Path(os.environ.get('NEURAL_OUT', '/new'))
sys.path.insert(0, str(BASE/'src'))
from common_v2 import FOLDS, plain
from risk_models import Transform, row_weights
from metrics_v2 import eligible, ranking, alarms, threshold_from_oof
from train_v4 import fold_ids, calibrate, apply_calibration

TASKS = ['voltage', 'feedrate', 'hopper_pressure']
CHOICES = {'voltage':'extra_trees','feedrate':'histgb','hopper_pressure':'extra_trees'}
PROTOCOL = dict(architecture=[292,64,32,1], activation='relu', output='sigmoid',
    epochs=30, batch_size=512, seed=42, alpha=.01, learning_rate=.001,
    early_stopping=False, folds=FOLDS, holdout='historical week already inspected in v4; no new blind claim',
    selection='User requests neural display; no algorithm chosen by holdout performance',
    heads='Three independent MLP risks; existing conditional ETA heads and weak-label AFT lifetime retained',
    input='28 measured channels represented by 292 causal features; only train rows fit imputation/scaling')

def save(d,p):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(plain(d),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def status(**d):
    d.update(time=time.strftime('%Y-%m-%d %H:%M:%S %z'))
    save(d,OUT/'status.json');print(json.dumps(plain(d)),flush=True)

def main():
    warnings.filterwarnings('ignore',category=ConvergenceWarning)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==PROTOCOL
    else:save(PROTOCOL,OUT/'protocol.json')
    features=np.load(BASE/'data/features.npy',mmap_mode='r')
    names=json.loads((BASE/'data/features.json').read_text())['names']
    reports={}
    for task in TASKS:
        samples=pd.read_parquet(BASE/'data'/task/'samples.parquet')
        events=pd.read_parquet(BASE/'data'/task/'events.parquet')
        preds={}; taskreport={'folds':{},'production_ready':False,'algorithm':'mlp_64_32'}
        for k,fold in enumerate(FOLDS):
            name=fold['name'];dest=OUT/'models'/task/name;dest.mkdir(parents=True,exist_ok=True)
            tr,va=fold_ids(samples,fold);tr=tr[eligible(samples.iloc[tr])]
            rows=samples.iloc[tr];y=rows.event_in_300s.to_numpy()
            if not y.any() or y.all() or not len(va):continue
            status(stage='training',task=task,fold=name,train_rows=len(tr))
            if not (dest/'bundle.joblib').exists():
                started=time.perf_counter();trans=Transform().fit(np.asarray(features[tr]));x=trans.transform(np.asarray(features[tr]))
                model=MLPClassifier(hidden_layer_sizes=(64,32),activation='relu',solver='adam',alpha=.01,
                    batch_size=512,learning_rate_init=.001,max_iter=30,tol=0,n_iter_no_change=31,
                    early_stopping=False,random_state=42)
                with threadpool_limits(limits=4):model.fit(x,y,sample_weight=row_weights(rows))
                old=joblib.load(BASE/'models'/task/'risk'/CHOICES[task]/name/'bundle.joblib')
                head=old['aux']['time']
                eta={'kind':'constant','bin':int(head['constant'])} if isinstance(head,dict) else {'kind':'logistic','model':head,'classes':head.classes_.astype(int)}
                bundle=dict(task=task,algorithm='mlp_64_32',center=trans.center,mean=trans.mean,scale=trans.scale,
                    model=model,eta_head=eta,feature_names=names,trained_through=str(rows.timestamp.max()),
                    train_windows=len(tr),train_events=int(rows.loc[y,'event_id'].nunique()),
                    fit_seconds=time.perf_counter()-started,epochs=model.n_iter_,loss=model.loss_)
                joblib.dump(bundle,dest/'bundle.joblib',compress=3)
            b=joblib.load(dest/'bundle.joblib')
            z=np.clip((np.where(np.isfinite(features[va]),features[va],b['center'])-b['mean'])/b['scale'],-12,12).astype('float32')
            with threadpool_limits(limits=4):raw=b['model'].predict_proba(z)[:,1].astype('float64')
            previous=list(preds.values());preds[name]=dict(ids=va,risk=raw)
            if previous:
                oi=np.concatenate([q['ids'] for q in previous]);op=np.concatenate([q['risk'] for q in previous]);orows=samples.iloc[oi];ok=eligible(orows)
                cal=calibrate(op[ok],orows.event_in_300s.to_numpy()[ok]);best,_=threshold_from_oof(orows,apply_calibration(cal,op))
                threshold=best['threshold']
            else:cal={'kind':'identity'};threshold=1.000001
            b.update(calibration=cal,threshold=float(threshold))
            joblib.dump(b,dest/'export.joblib',compress=3)
            prob=apply_calibration(cal,raw);vrows=samples.iloc[va]
            result=dict(ranking=ranking(vrows,prob),alarms=alarms(vrows,prob,threshold,events)[0],
                fit_seconds=b['fit_seconds'],epochs=b['epochs'],loss=b['loss'],train_windows=b['train_windows'],
                train_events=b['train_events'],threshold_from_folds=list(preds)[:-1])
            taskreport['folds'][name]=result
            np.savez(OUT/f'{task}_{name}.npz',ids=va,risk=raw,score=prob)
            save(taskreport,OUT/f'{task}_report.json');status(stage='fold_complete',task=task,fold=name,metrics=result)
        tree=json.loads((BASE/'evaluation'/task/'v4_report.json').read_text())
        taskreport['tree_holdout']=tree['folds']['f6']['scores'][CHOICES[task]]
        taskreport['neural_holdout']=taskreport['folds']['f6']
        reports[task]=taskreport;save(taskreport,OUT/f'{task}_report.json')
    save(reports,OUT/'comparison.json');status(stage='complete',tasks=TASKS)

if __name__=='__main__':main()
