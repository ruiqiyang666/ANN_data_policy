"""Audit correction only: preserve every learned model, fix unsupported KM-age CDF.

Quantiles/selection losses must remain unchanged. Archive old results before rewriting.
"""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
import train as T
import v5 as V

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
args=p.parse_args();frame=T.prep(pd.read_parquet(args.data/'frame.parquet'))
archive=args.run/'before-km-support-correction';archive.mkdir(exist_ok=True)
for name in ('report.json','predictions.npz','provenance.json'):
    if not (archive/name).exists():shutil.copy2(args.run/name,archive/name)
report=json.loads((archive/'report.json').read_text(encoding='utf-8'));changed=[]
for fold in report['runs']:
    directory=args.run/'folds'/fold['tag'];old=archive/fold['tag'];old.mkdir(exist_ok=True)
    for name in ('result.json','predictions.npz','km_age.json'):
        if not (old/name).exists():shutil.copy2(directory/name,old/name)
    with np.load(old/'predictions.npz',allow_pickle=False) as a:arrays={k:a[k] for k in a.files}
    meta=json.loads((directory/'km_age.json').read_text(encoding='utf-8'));assert V.digest(directory/meta['artifact'])==meta['sha256']
    c,q,details=T.predict_model(load(directory/meta['artifact']),frame.loc[arrays['test_index']])
    np.testing.assert_allclose(q,arrays['km_age_quant'],rtol=1e-6,atol=1e-5)
    delta=int(np.any(np.abs(c-arrays['km_age_curve'])>1e-6,axis=1).sum())
    rows=[]
    for val in V.inner_groups(fold['training']):
        fit=[cid for cid in fold['training'] if cid not in val];vi=frame.index[frame.cycle_id.isin(val)]
        ci,qi,_=T.fit_predict(frame,fit,vi,'km_age');rows+=V.metrics(frame,vi,ci,qi,fit)
    inner=V.summarize(rows)
    assert abs(inner['mae']+.5*inner['last2h']-fold['models']['km_age']['selection_loss'])<1e-6
    fold['models']['km_age'].update(inner=inner,test=V.metrics(frame,arrays['test_index'],c,q,fold['training']),details=details)
    meta['details']=details;V.save_json(directory/'km_age.json',meta)
    arrays['km_age_curve']=c.astype('float32')
    np.savez_compressed(directory/'predictions.npz',**arrays);V.save_json(directory/'result.json',fold)
    changed.append(dict(fold=fold['tag'],affected_origins=delta,details=details))
for mode in ('loco','forward'):
    for name in ('km_age','nested_selected'):
        rows=[]
        for fold in report['runs']:
            if not fold['tag'].startswith(mode):continue
            chosen=fold['selected'] if name=='nested_selected' else name
            rows+=fold['models'][chosen]['test']
        report['comparison'][mode][name]=dict(life=V.summarize(rows),cycles=rows)
with np.load(archive/'predictions.npz',allow_pickle=False) as a:ledger={k:a[k] for k in a.files}
for fold in report['runs']:
    if not fold['tag'].startswith('loco'):continue
    with np.load(args.run/'folds'/fold['tag']/'predictions.npz',allow_pickle=False) as a:
        ledger['km_age_curve'][a['test_index']]=a['km_age_curve']
        if fold['selected']=='km_age':ledger['nested_selected_curve'][a['test_index']]=a['km_age_curve']
report['audit_correction']='Unsupported KM-age conditional CDF corrected to match declared sentinel point-mass convention. All quantiles, selection losses and learned models unchanged.'
V.save_json(args.run/'report.json',report);np.savez_compressed(args.run/'predictions.npz',**ledger)
provenance=json.loads((archive/'provenance.json').read_text(encoding='utf-8'))
provenance['original_sources']=provenance['sources'].copy();provenance['sources']['train.py']=V.digest(Path(__file__).with_name('train.py'))
provenance['audit_correction']=dict(script_sha256=V.digest(__file__),changes=changed,original_artifacts='before-km-support-correction')
V.save_json(args.run/'provenance.json',provenance)
V.save_json(args.run/'km-support-audit.json',changed)
print('CORRECTED',sum(c['affected_origins'] for c in changed),'unsupported-age origins; no quantile/selection changes')
