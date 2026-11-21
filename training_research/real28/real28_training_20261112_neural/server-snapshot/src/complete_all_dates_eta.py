"""Fit all-date candidate ETA heads with those candidates' own normalization."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,joblib,numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
root=Path('/new');base=Path('/old/v4');features=np.load(base/'data/features.npy',mmap_mode='r')
report={}
for task in ['voltage','feedrate','hopper_pressure']:
 p=root/'models'/task/'all_dates/bundle.joblib';b=joblib.load(p)
 s=pd.read_parquet(base/'data'/task/'samples.parquet');ids=np.flatnonzero(s.event_in_300s.to_numpy())
 a=np.asarray(features[ids]);z=np.clip((np.where(np.isfinite(a),a,b['center'])-b['mean'])/b['scale'],-12,12).astype('float32')
 bins=np.argmax(np.load(base/'data'/task/'targets.npz')['hazard_y'][ids],axis=1)
 if len(np.unique(bins))<2:b['eta_head']={'kind':'constant','bin':int(bins[0])}
 else:
  with threadpool_limits(limits=4):head=LogisticRegression(C=.05,max_iter=350).fit(z,bins)
  b['eta_head']={'kind':'logistic','model':head,'classes':head.classes_.astype(int)}
 joblib.dump(b,p,compress=3);report[task]={'positive_windows':len(ids),'eta_normalization':'same all-date candidate transform','deployed':False}
(root/'all_dates_eta.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
