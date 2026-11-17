"""One bounded validation-only extension, before opening test predictions.

The initial eight candidates could not reach both validation events under the
fixed false-alert budget. Add four low-dimensional tabular candidates, keeping
targets, splits, input opportunities and alarm policy unchanged.
"""
import json,time,joblib
from datetime import datetime,timezone
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from modeling import ROOT,load_frames,encode,summaries,TARGETS
from evaluation import select_threshold
from train import dump,digest,operating_hours

def main():
    art=ROOT/'artifacts'
    if (art/'evaluation.json').exists():raise RuntimeError('Test already opened. Extension prohibited.')
    cfg=json.loads((art/'frozen_config.json').read_text('utf-8'))
    if (art/'validation_extension_plan.json').exists():raise RuntimeError('Bounded extension already run.')
    dump(art/'initial_selection.json',cfg)
    indices=[g*13+c for g in range(18) for c in [0,11,12]]+list(range(234,238))
    specs=[dict(name='voltage_logistic_c'+str(c).replace('.','p'),kind='logistic_voltage',C=c,summary_indices=indices) for c in [.03,.3]]
    specs += [dict(name=f'hgb_depth{d}',kind='hgb',max_depth=d,summary_indices=indices) for d in [2,3]]
    dump(art/'validation_extension_plan.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),reason='Initial validation hit20 at most 1/2; voltage precursor differs by event. Add exactly four regularized voltage-only summary candidates; no test predictions seen.',specs=specs,unchanged=['targets','splits','60min historical representation','all online opportunities','0.1 false alarms per production hour budget','alarm rule','selection order'],stop='No further search after these four candidates. Freeze before test.'))
    f,y,e,p=load_frames();keep=f.split.isin(['train','validation']);f=f[keep].reset_index(drop=True);y=y[keep].reset_index(drop=True)
    pre=json.loads((art/'preprocessor.json').read_text('utf-8'));x=encode(f,pre);z=summaries(x)[:,indices]
    train=f.split.eq('train').to_numpy();val=f.split.eq('validation').to_numpy()
    sample=~pd_duplicated(f);ti=np.flatnonzero(train&sample);vi=np.flatnonzero(val)
    yy=y[TARGETS].to_numpy(np.float32);vf=f.iloc[vi].reset_index(drop=True)
    ev=e[e.split.eq('validation')&e.is_eol.eq(1)];hours=operating_hours(p,'validation')
    search=json.loads((art/'validation_search.json').read_text('utf-8'))
    for spec in specs:
        started=time.perf_counter();name=spec['name'];out=art/'models'/name;out.mkdir(exist_ok=True)
        models=[]
        for j in range(4):
            ix=ti[np.isfinite(yy[ti,j])]
            if spec['kind']=='logistic_voltage':
                m=make_pipeline(StandardScaler(),LogisticRegression(C=spec['C'],class_weight='balanced',max_iter=1500,random_state=20261117))
            else:
                m=HistGradientBoostingClassifier(max_iter=120,learning_rate=.04,max_depth=spec['max_depth'],max_leaf_nodes=2**spec['max_depth'],min_samples_leaf=40,l2_regularization=10,class_weight='balanced',early_stopping=False,random_state=20261117)
            m.fit(z[ix],yy[ix,j]);models.append(m)
        joblib.dump(models,out/'model.joblib');pred=np.column_stack([m.predict_proba(z[vi])[:,1] for m in models]);best=None
        for j in [0,1,3]:
            chosen=select_threshold(vf,pred[:,j],ev,hours)
            row=dict(model=name,head=TARGETS[j],head_index=j,threshold=float(chosen['threshold']),metrics=chosen['metrics'],selection_key=list(chosen['selection_key']))
            search.append(row)
            if best is None or tuple(row['selection_key'])>tuple(best['selection_key']):best=row
        spec['fit_seconds']=time.perf_counter()-started;cfg['model_specs'].append(spec);cfg['validation_results'][name]=best
        cfg['thresholds'][name]=best['threshold'];cfg['policies'][name]={j:best[j] for j in ['head','head_index','threshold']}
        dump(out/'validation_policy.json',best);print(name,json.dumps(best,ensure_ascii=False),flush=True)
    cfg['selected_model']=max(cfg['model_specs'],key=lambda s:(tuple(cfg['validation_results'][s['name']]['selection_key']),-cfg['model_specs'].index(s)))['name']
    cfg['frozen_at_utc']=datetime.now(timezone.utc).isoformat();cfg['model_version']='booth2-v2-'+datetime.now().strftime('%Y%m%dT%H%M%S')
    cfg['validation_extension_sha256']=digest(art/'validation_extension_plan.json')
    cfg['fitted_file_sha256']={str(p.relative_to(art)):digest(p) for p in (art/'models').rglob('*') if p.suffix in ['.pt','.joblib']}
    cfg['source_sha256']={p.name:digest(p) for p in (ROOT/'src').glob('*.py') if p.name!='report.py'}
    dump(art/'validation_search.json',search);dump(art/'frozen_config.json',cfg)
    log=json.loads((art/'run_log.json').read_text('utf-8'));log['initial_selected_model']=log['selected_model'];log['selected_model']=cfg['selected_model'];log['extension_fit_seconds']=sum(s['fit_seconds'] for s in specs);log['frozen_utc']=cfg['frozen_at_utc'];dump(art/'run_log.json',log)
    print('FINAL SELECTION',cfg['selected_model'],flush=True)

def pd_duplicated(f):
    import pandas as pd
    return pd.DataFrame({'cycle':f.cycle_id,'minute':f.timestamp.dt.floor('min')}).duplicated().to_numpy()

if __name__=='__main__':main()
