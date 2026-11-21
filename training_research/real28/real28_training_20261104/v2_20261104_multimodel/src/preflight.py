"""Meaningful pre-training causality, target and architecture checks."""
import sys
sys.dont_write_bytecode=True
import torch
import numpy as np
import pandas as pd
from common_v2 import DATA,V1,EVAL,ROOT,MODELS,FIELDS,FOLDS,load,dump,sha,time_seconds
from prepare_v2 import event_targets,masked_window_statistics
from risk_models import RiskNet,distribution,Transform
from train_risk import fold_ids

def main():
    checks=[];torch.set_num_threads(2)
    y,m,event,delta,observed=event_targets(np.zeros(5,dtype=int),np.array([30,31,300,301,299]),np.array([300,300,300,300,298]))
    assert y[0,0]==1 and y[1,1]==1 and y[2,9]==1 and not event[3] and not event[4]
    assert m[4].sum()==9;checks.append('Exact horizon and interval censor masks')
    values=np.r_[np.zeros(150),np.full(150,10.),np.full(100,999.)];valid=np.r_[np.zeros(150,dtype=bool),np.ones(250,dtype=bool)]
    result=masked_window_statistics(values,valid,np.array([299]),np.array([0]),np.arange(400)/3600)
    assert abs(result[0][0]-10)<1e-7 and result[3][0]==150
    values[300:]=-999;again=masked_window_statistics(values,valid,np.array([299]),np.array([0]),np.arange(400)/3600)
    np.testing.assert_allclose(result,again);checks.append('Operating masks exclude unpowered zeros; future values do not affect features')
    for name in ['tcn','gru','patch_transformer']:
        model=RiskNet(name,266).cuda();x=torch.randn(4,58,120,device='cuda');s=torch.randn(4,266,device='cuda')
        out=model(x,s);assert out.shape==(4,10);p,b=distribution(out)
        assert torch.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        torch.testing.assert_close(b.sum(1),torch.ones(4,device='cuda'))
        out.square().mean().backward();assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
        del model
    checks.append('All three neural architectures have finite CUDA forward/backward and coherent event distributions')
    s=pd.read_parquet(DATA/'samples.parquet');t=np.load(DATA/'grid_t.npy',mmap_mode='r');block=np.load(DATA/'grid_block.npy',mmap_mode='r');part=np.load(DATA/'grid_part.npy',mmap_mode='r')
    end=s.end_index.to_numpy();assert np.all(end-119>=block[end]);assert np.all(t[end]-t[end-119]==119)
    assert np.all(part[end]==s.partition);assert len(load(DATA/'features.json')['raw_fields'])==28
    checks.append('All sequence windows have 120 real seconds and correct chronological partition')
    target=np.load(DATA/'targets.npz');np.testing.assert_array_equal(target['hazard_y'].sum(1)>0,s.event_in_300s)
    assert np.all(target['hazard_y']<=target['hazard_mask']);checks.append('Saved labels agree with event masks')
    p=pd.read_parquet(V1/'products.parquet');assert (p.end-p.start).dt.total_seconds().max()<3600
    u=p[p.label_valid.ne(1)].sort_values('start');us=time_seconds(u.start);ue=time_seconds(u.end);now=t[end];ui=np.searchsorted(us,now,'right');prev=np.maximum(ui-1,0)
    inside=(ui>0)&(ue[prev]>=now);assert (s.observed_seconds.to_numpy()[inside]==0).all()
    next_u=np.r_[us,np.iinfo('int64').max//2][ui];assert np.all(s.observed_seconds<=np.maximum(0,next_u-now))
    checks.append('All seven unknown products, including unavailable results, interrupt negative follow-up')
    assert s.loc[s.rul_valid].groupby('possible_cycle').partition.nunique().max()==1
    assert s.loc[s.rul_valid&s.rul_event].groupby('rul_endpoint_id').partition.nunique().max()==1
    checks.append('Lifetime reuse retains possible-cycle and weak-endpoint isolation')
    folds=[]
    for f in FOLDS:
        tr,val=fold_ids(s,f);cut=time_seconds(f['train_end']);assert (now[tr]+300<cut).all() and (now[val]>=cut+7200).all()
        a=set(s.iloc[tr].loc[lambda z:z.event_in_300s,'event_id']);b=set(s.iloc[val].loc[lambda z:z.event_in_300s,'event_id']);assert not a&b
        folds.append({'fold':f['name'],'train_windows':len(tr),'validation_windows':len(val),'train_events':len(a),'validation_events':len(b)})
    checks.append('Forward folds purge 300 s targets and a conservative 7200 s source-history bound')
    features=np.load(DATA/'features.npy',mmap_mode='r');tr=np.flatnonzero(s.partition.eq(0))
    transform=Transform().fit(features[tr]);assert np.isfinite(transform.transform(features[tr[::100]])).all()
    checks.append('Fold-fitted imputation and scaling produce finite features')
    old=ROOT.parent/'models/20261104_v1';frozen=load(old/'pretest_freeze.json')
    assert all(sha(old/n)==h for n,h in frozen['artifact_sha256'].items());checks.append('All V1 frozen model files unchanged')
    # Finalize the conservative bound before any fitting begins.
    spec=load(ROOT/'protocol.json')
    if list(MODELS.rglob('*.pt')) or list(MODELS.rglob('*.joblib')):raise RuntimeError('Preflight protocol finalization must precede model fitting')
    spec['max_feature_history_seconds']=7200;spec['risk_split']='chronological/event isolated; 7200 s history purge includes completed-product duration; NOT independent electrode identity'
    dump(spec,ROOT/'protocol.json')
    out={'passed':True,'checks':checks,'folds':folds,'protocol_sha256':sha(ROOT/'protocol.json'),'samples_sha256':sha(DATA/'samples.parquet'),'features_sha256':sha(DATA/'features.npy')}
    dump(out,EVAL/'preflight.json');print(out,flush=True)

if __name__=='__main__':main()
