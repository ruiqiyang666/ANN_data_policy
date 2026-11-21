"""Checks tied to the revised labels, availability, history and isolation."""
import sys
sys.dont_write_bytecode=True
import time,os
import torch
import numpy as np
import pandas as pd
from common_v2 import ROOT,SHARED,V1,GRID,FOLDS,TYPES,load,dump,sha,time_seconds as sec
from prepare_v3 import targets,window_stats,build_episodes
from risk_models import RiskNet,distribution,Transform
from train_risk import fold_ids

def main():
    checks=[];torch.set_num_threads(2)
    y,m,e,_,_=targets(np.zeros(5,dtype=int),np.array([30,31,300,301,299]),np.array([300,300,300,300,298]))
    assert y[0,0]==1 and y[1,1]==1 and y[2,9]==1 and not e[3] and not e[4] and m[4].sum()==9
    checks.append('300-second horizon, 30-second bins, partial followup censoring')
    v=np.r_[np.zeros(150),np.full(150,10.),np.full(100,999.)];valid=np.r_[np.zeros(150,dtype=bool),np.ones(250,dtype=bool)]
    result=window_stats(v,valid,np.array([299]),np.array([0]),np.arange(400)/3600);v[300:]=-999
    np.testing.assert_allclose(result,window_stats(v,valid,np.array([299]),np.array([0]),np.arange(400)/3600))
    assert result[0][0]==10 and result[3][0]==150;checks.append('Future-value poisoning leaves causal rolling history unchanged; unpowered zeros excluded')
    t=np.load(GRID/'grid_t.npy',mmap_mode='r');block=np.load(GRID/'grid_block.npy',mmap_mode='r');s=pd.read_parquet(SHARED/'life/samples.parquet');now=sec(s.timestamp);end=s.end_index.to_numpy()
    assert np.all(end-119>=block[end]) and np.all(t[end]-t[end-119]==119)
    assert s.loc[s.rul_valid].groupby('possible_cycle').partition.nunique().max()==1
    assert s.loc[s.rul_valid&s.rul_event].groupby('rul_endpoint_id').partition.nunique().max()==1
    term=pd.read_parquet(SHARED/'natural_endpoint_evidence.parquet')
    assert len(term)==6 and term.eligible.all() and not term.physical_cycle_identity_confirmed.any()
    assert (s.rul_valid&s.rul_event&s.rul_endpoint_id.eq('R011')).sum()>100
    assert (s.rul_valid&s.rul_event&s.rul_endpoint_id.eq('R007')).sum()>2
    assert not (s.rul_observation_end==pd.Timestamp('2026-10-14 13:40:45')).any()
    checks.append('R009 ordinary wait removed as censor; R011 weak sustained endpoint retained; R007 onset interval expanded; zero confirmed natural failures')
    p=pd.read_parquet(V1/'products.parquet');assert (p.end-p.start).dt.total_seconds().max()<3600
    folds={}
    for task in TYPES:
        rows=pd.read_parquet(SHARED/task/'samples.parquet');target=np.load(SHARED/task/'targets.npz');events=pd.read_parquet(SHARED/task/'events.parquet')
        assert (events.event_at==events.first_product_start).all()
        np.testing.assert_array_equal(target['hazard_y'].sum(1)>0,rows.event_in_300s)
        assert np.all(target['hazard_y']<=target['hazard_mask'])
        for r in events.itertuples():
            mask=rows.event_id.eq(r.event_id)&rows.event_in_300s
            assert (rows.loc[mask,'timestamp']<r.first_product_start).all()
        for f in FOLDS:
            tr,va=fold_ids(rows,f);a=rows.iloc[tr];b=rows.iloc[va]
            assert len(tr)>0 and len(va)>0 and a.event_in_300s.any()
            assert (sec(a.timestamp)+3900<sec(f['train_end'])).all()
            assert not set(a.loc[a.event_in_300s,'event_id'])&set(b.loc[b.event_in_300s,'event_id'])
            folds[f'{task}/{f["name"]}']={'train_windows':len(tr),'validation_windows':len(va),'train_events':a.loc[a.event_in_300s,'event_id'].nunique(),'validation_events':b.loc[b.event_in_300s,'event_id'].nunique()}
        # A new event cannot be created by an unpowered pause alone.
        if task=='voltage':
            assert not ((events.event_at>=pd.Timestamp('2026-10-15'))&(events.event_at<pd.Timestamp('2026-10-15 08:00'))).any()
        checks.append(task+': before-product-start positives, episode isolation and outcome-availability purge')
    names=load(SHARED/'features.json')['names'];x=np.load(SHARED/'features.npy',mmap_mode='r');assert x.shape==(len(s),len(names))
    transform=Transform().fit(x[s.partition.eq(0)]);assert np.isfinite(transform.transform(x[::101])).all()
    # Spot-check new 2h statistics against direct past-only slices.
    gx=np.load(GRID/'grid_x.npy',mmap_mode='r');seen=np.load(GRID/'grid_seen.npy',mmap_mode='r')
    for k in np.linspace(0,len(s)-1,17,dtype=int):
        stop=end[k];start=max(int(block[stop]),int(np.searchsorted(t,now[k]-7199)))
        a=gx[start:stop+1];m=(a[:,4]>100)&(a[:,5]>5)&seen[start:stop+1,4]&seen[start:stop+1,5]&seen[start:stop+1,6]
        expected=float(a[m,6].mean()) if m.any() else np.nan
        np.testing.assert_allclose(x[k,names.index('history_Voltage_mean_7200s')],expected,atol=2e-5,equal_nan=True)
    checks.append('2h history matches direct causal raw slices; all 28 inputs retained; train-only preprocessing')
    for name in ['tcn','gru','patch_transformer']:
        model=RiskNet(name,len(names)).cuda();out=model(torch.randn(4,58,120,device='cuda'),torch.randn(4,len(names),device='cuda'))
        a,b=distribution(out);torch.testing.assert_close(b.sum(1),torch.ones(4,device='cuda'));out.square().mean().backward()
        assert torch.isfinite(out).all() and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        del model
    torch.cuda.empty_cache();checks.append('TCN/GRU/patch Transformer finite CUDA forward and backward')
    # Hash existing experiments and raw-cache provenance before training.
    manifest={}
    for prior in [ROOT.parent/'models',ROOT.parent/'src',ROOT.parent/'v2_20261104_multimodel']:
        for path in prior.rglob('*'):
            if path.is_file() and path.suffix in ['.py','.md','.pt','.joblib','.json']:
                manifest[str(path)]=sha(path)
    for name in ['sources.parquet','products.parquet','events.parquet','timestamps.npy','raw_values.npy']:
        manifest[str(V1/name)]=sha(V1/name)
    dump(manifest,ROOT/'previous_artifacts_sha256.json')
    protocol=load(ROOT/'protocol.json');protocol['training_outcome_availability_purge_seconds']=3900
    dump(protocol,ROOT/'protocol.json')
    out={'passed':True,'checks':checks,'folds':folds,'features':len(names),'sha256':sha(SHARED/'features.npy'),'checked_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    for task in TYPES+['life']:
        (ROOT/'evaluation'/task).mkdir(parents=True,exist_ok=True);dump(out,ROOT/'evaluation'/task/'preflight.json')
    dump(out,ROOT/'evaluation/preflight.json');print(out,flush=True)

if __name__=='__main__':main()
