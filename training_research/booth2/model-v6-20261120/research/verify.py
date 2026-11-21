"""Audit complete folds and persisted models, then fit full-data research candidates.

Only reads joblib files produced in this run, after verifying their SHA-256 metadata.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
import train as T
import v5 as V


def paired_ci(deltas):
    rng=np.random.default_rng(20261120)
    a=np.asarray(deltas,float)
    samples=a[rng.integers(0,len(a),size=(10000,len(a)))].mean(1)
    return dict(mean=float(a.mean()),lo=float(np.quantile(samples,.025)),hi=float(np.quantile(samples,.975)),
                cycles=len(a),note='paired cycle bootstrap, exploratory, no multiple-comparison correction')


def verify(args):
    frame=T.prep(pd.read_parquet(args.data/'frame.parquet'))
    report=json.loads((args.run/'report.json').read_text(encoding='utf-8'))
    provenance=json.loads((args.run/'provenance.json').read_text(encoding='utf-8'))
    for name,digest in provenance['inputs'].items():assert V.digest(args.data/name)==digest,name
    for name in ('train.py','v5.py'):assert V.digest(Path(__file__).with_name(name))==provenance['sources'][name],name
    assert V.digest(Path(__file__).with_name('protocol.json'))==provenance['protocol_sha256']
    seen=np.zeros(len(frame),int);parity={};quant_errors=[];curve_errors=[]
    with np.load(args.run/'predictions.npz',allow_pickle=False) as a:
        ledger={k:a[k] for k in a.files}
    for fold in report['runs']:
        assert not set(fold['training'])&set(fold['held_out'])
        if fold['tag'].startswith('forward'):assert max(fold['training'])<min(fold['held_out'])
        else:seen[frame.cycle_id.isin(fold['held_out'])]+=1
        for val in V.inner_groups(fold['training']):assert not set(val)&set(fold['held_out'])
        directory=args.run/'folds'/fold['tag']
        with np.load(directory/'predictions.npz',allow_pickle=False) as a:
            ix=a['test_index'];sample=np.arange(0,len(ix),137)
            for name in T.NAMES:
                meta=json.loads((directory/f'{name}.json').read_text(encoding='utf-8'))
                artifact=directory/meta['artifact'];assert V.digest(artifact)==meta['sha256']
                model=load(artifact)
                assert set(model['training_cycles'])==set(fold['training'])
                c,q,_=T.predict_model(model,frame.loc[ix[sample]])
                dc=float(np.abs(c-a[name+'_curve'][sample]).max());dq=float(np.abs(q-a[name+'_quant'][sample]).max())
                curve_errors.append(dc);quant_errors.append(dq)
                assert dc<3e-6 and dq<.002,(fold['tag'],name,dc,dq)
                # Prediction row is independent of all future rows in the call.
                c0,q0,_=T.predict_model(model,frame.loc[ix[sample[:1]]])
                np.testing.assert_allclose(c0,c[:1],atol=3e-6)
                np.testing.assert_allclose(q0,q[:1],atol=.002)
        print('verified',fold['tag'],flush=True)
    assert np.all(seen==1)
    for name in T.NAMES+['nested_selected']:
        c,q=ledger[name+'_curve'],ledger[name+'_quant']
        assert np.isfinite(c).all() and np.isfinite(q).all()
        assert np.all((c>=0)&(c<=1)) and np.all(np.diff(c,axis=1)>=-1e-6)
        assert np.all(q>0) and np.all(np.diff(q,axis=1)>=-1e-6)
        errors=[];tail=[]
        for _,g in frame[frame.event.eq(1)&frame.rul_h.notna()].groupby('cycle_id'):
            e=np.abs(q[g.index,1]-g.rul_h.to_numpy());errors.append(e.mean());tail.append(e[g.rul_h.to_numpy()<=2].mean())
        expected=report['comparison']['loco'][name]['life']
        assert abs(float(np.mean(errors))-expected['mae'])<1e-6
        assert abs(float(np.mean(tail))-expected['last2h'])<1e-6
    comparison={}
    if args.v5:
        previous=json.loads((args.v5/'report.json').read_text(encoding='utf-8'))
        for mode in ('loco','forward'):
            comparison[mode]={}
            # Fixed v5 comparator: XGB23, plus v5 nested strategy. No selection using v6 test outcomes.
            for base in ('aft_xgb23','nested_selected'):
                old={r['cycle']:r for r in previous['comparison'][mode][base]['cycles'] if r['observed']}
                comparison[mode][base]={}
                for name,v in report['comparison'][mode].items():
                    rows=[r for r in v['cycles'] if r['observed']]
                    comparison[mode][base][name]={metric:paired_ci([r[metric]-old[r['cycle']][metric] for r in rows]) for metric in ('mae','last2h')}
    final=args.run/'final';final.mkdir(exist_ok=True);cycles=sorted(frame.cycle_id.unique());ix=frame.index[::83]
    final_parity={}
    if not args.audit_only:
        for name in T.NAMES:
            c,q,detail=T.fit_predict(frame,cycles,ix,name,final/f'{name}.json')
            meta=json.loads((final/f'{name}.json').read_text(encoding='utf-8'));assert V.digest(final/meta['artifact'])==meta['sha256']
            rc,rq,_=T.predict_model(load(final/meta['artifact']),frame.loc[ix])
            dc=float(np.abs(c-rc).max());dq=float(np.abs(q-rq).max())
            assert dc<3e-6 and dq<.002
            final_parity[name]=dict(curve_abs=dc,quantile_hours_abs=dq,details=detail)
            print('final',name,flush=True)
        for name in ('warn_logit6','warn_hgb32'):T.warning_fit(frame,cycles,ix,name,final/f'{name}.joblib')
        V.save_json(final/'manifest.json',dict(schema_version=1,status='research_only_not_approved',auto_publish=False,
                    units='processing_hours',warning_units='wall_clock_60min',models=T.NAMES,warnings=['warn_logit6','warn_hgb32'],
                    runtime='train.predict_model; trusted local joblib only after digest verification',
                    files={p.name:V.digest(p) for p in final.iterdir() if p.is_file() and p.name!='manifest.json'}))
    verification=dict(outer_folds=14,forward_blocks=3,origins=len(frame),all_input_hashes=True,whole_cycle_splits=True,
        cdf_max_abs=max(curve_errors),quantile_hours_max_abs=max(quant_errors),final_parity=final_parity,paired_comparison=comparison,
        verifier_sha256=V.digest(__file__))
    V.save_json(args.run/'verification.json',verification)
    print('VERIFIED',len(frame),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--v5',type=Path);p.add_argument('--audit-only',action='store_true');verify(p.parse_args())
