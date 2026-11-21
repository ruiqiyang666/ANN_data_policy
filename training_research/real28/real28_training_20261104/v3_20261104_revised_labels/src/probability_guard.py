"""Guard conditional-time normalization without changing frozen risks/weights.

Original GPU inference divided subnormal event mass by float32.tiny. Its
conditional vector could consequently sum to <1. Normalize in float64 at the
public prediction boundary; retain original training source for reproducibility.
"""
import sys
sys.dont_write_bytecode=True
import time,shutil
import numpy as np
import pandas as pd
from common_v2 import ROOT,SHARED,load,dump,sha

def normalize_time(value):
    a=np.asarray(value,dtype='float64')
    if not np.isfinite(a).all() or (a<0).any():raise ValueError('Nonfinite/negative conditional mass')
    total=a.sum(1,keepdims=True)
    return np.divide(a,total,out=np.full_like(a,1/a.shape[1]),where=total>0).astype('float32')

def main():
    np.testing.assert_allclose(normalize_time(np.array([[.2,.3,0],[0,0,0],[1e-39,2e-39,3e-39]])),[[.4,.6,0],[1/3]*3,[1/6,2/6,3/6]],atol=1e-7)
    auditfile=ROOT/'evaluation/conditional_probability_repair.json'
    existing=load(auditfile) if auditfile.exists() else {'repairs':[]}
    for path in (ROOT/'evaluation').rglob('*.npz'):
        if 'numerical_originals' in path.parts:continue
        a=dict(np.load(path))
        if 'time' not in a:continue
        bad=np.abs(a['time'].sum(1)-1)>2e-6
        if not bad.any():continue
        relative=path.relative_to(ROOT/'evaluation');backup=ROOT/'evaluation/numerical_originals'/relative;backup.parent.mkdir(parents=True,exist_ok=True)
        if not backup.exists():shutil.copy2(path,backup)
        risk=a['risk'].copy();originalsha=sha(path);a['time']=normalize_time(a['time'])
        if 'eta' in a:a['eta']=(a['time']*np.arange(15,300,30)).sum(1)
        np.savez(path,**a);np.testing.assert_array_equal(np.load(path)['risk'],risk)
        existing['repairs'].append({'archive':str(relative),'rows':int(bad.sum()),'min_raw_risk':float(risk[bad].min()),'max_raw_risk':float(risk[bad].max()),'original_sha256':originalsha,'normalized_sha256':sha(path),'risk_unchanged':True})
    # Only ETA metrics depend on conditional time; alarms/ranking are untouched.
    for task in ['voltage','feedrate','hopper_pressure']:
        cp=ROOT/'evaluation'/task/'comparison.json'
        if not cp.exists():continue
        report=load(cp);samples=pd.read_parquet(SHARED/task/'samples.parquet')
        for name,parts in report['algorithms'].items():
            for part,record in parts.items():
                a=np.load(ROOT/'evaluation'/task/f'{name}_{part}.npz');rows=samples.iloc[a['ids']];pos=rows.event_in_300s.to_numpy()
                record['eta_mae_s_all_positive_windows']=float(np.abs(a['eta'][pos]-rows.eta_seconds.to_numpy()[pos]).mean()) if pos.any() else None
        dump(report,cp)
    existing.update(regression_passed=True,model_weights_changed=False,risk_values_changed=False,method_selection_changed=False,freeze_changed=False,
                    cause='subnormal probability mass divided by float32.tiny; normalized at inference boundary',updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    dump(existing,auditfile);print(existing,flush=True)
if __name__=='__main__':main()
