"""Separate all-five-folder neural candidates. No validation claim, not deployed."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,time
import numpy as np,pandas as pd,joblib
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
sys.path.insert(0,'/old/v4/src')
from risk_models import Transform,row_weights
from metrics_v2 import eligible,threshold_from_oof
from train_v4 import calibrate,apply_calibration
from common_v2 import plain
root=Path('/new');base=Path('/old/v4');x=np.load(base/'data/features.npy',mmap_mode='r')
report={}
for task in ['voltage','feedrate','hopper_pressure']:
 dest=root/'models'/task/'all_dates';dest.mkdir(exist_ok=True)
 assert not (dest/'bundle.joblib').exists(),'Do not overwrite a trained candidate'
 s=pd.read_parquet(base/'data'/task/'samples.parquet');ids=np.flatnonzero(eligible(s));rows=s.iloc[ids]
 t=Transform().fit(np.asarray(x[ids]));z=t.transform(np.asarray(x[ids]))
 model=MLPClassifier(hidden_layer_sizes=(64,32),alpha=.01,batch_size=512,learning_rate_init=.001,
                     max_iter=30,tol=0,n_iter_no_change=31,early_stopping=False,random_state=42)
 with threadpool_limits(limits=4):model.fit(z,rows.event_in_300s.to_numpy(),sample_weight=row_weights(rows))
 earlier=[dict(np.load(root/f'{task}_f{i}.npz')) for i in range(1,7)]
 oi=np.concatenate([p['ids'] for p in earlier]);op=np.concatenate([p['risk'] for p in earlier]);o=s.iloc[oi];ok=eligible(o)
 cal=calibrate(op[ok],o.event_in_300s.to_numpy()[ok]);best,_=threshold_from_oof(o,apply_calibration(cal,op))
 b=joblib.load(root/'models'/task/'f6/export.joblib')
 b.update(model=model,center=t.center,mean=t.mean,scale=t.scale,calibration=cal,threshold=float(best['threshold']),
          trained_through='2026-11-01',validation_metrics=None,train_windows=len(ids),train_events=int(rows.loc[rows.event_in_300s,'event_id'].nunique()))
 targets=np.load(base/'data'/task/'targets.npz')
 event=rows.event_in_300s.to_numpy();bins=np.argmax(targets['hazard_y'][ids[event]],axis=1)
 if len(np.unique(bins))<2:b['eta_head']={'kind':'constant','bin':int(bins[0])}
 else:
  with threadpool_limits(limits=4):head=LogisticRegression(C=.05,max_iter=350).fit(z[event],bins)
  b['eta_head']={'kind':'logistic','model':head,'classes':head.classes_.astype(int)}
 joblib.dump(b,dest/'bundle.joblib',compress=3)
 report[task]=dict(train_windows=len(ids),train_events=b['train_events'],trained_through='2026-11-01',
                   validation_metrics=None,deployed=False,epochs=model.n_iter_)
 print(json.dumps(report[task]),flush=True)
(root/'all_dates_refit.json').write_text(json.dumps(report,indent=2))
