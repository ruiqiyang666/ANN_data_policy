"""Train distinct CPU/GPU methods using fixed forward folds, without history access."""
import sys
sys.dont_write_bytecode=True
import torch
import argparse,time,json,random
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from common_v2 import DATA,MODELS,EVAL,LOGS,FOLDS,RISK_NAMES,protocol,load,dump,status,sha,time_seconds
from metrics_v2 import eligible,ranking,threshold_from_oof
from risk_models import Cache,Transform,ExtraTransform,cpu_model,row_weights,RiskNet,distribution,fit_auxiliary,auxiliary_predict

NEURAL={'tcn','gru','patch_transformer'}

def fold_ids(samples,fold):
    t=time_seconds(samples.timestamp);cut=time_seconds(fold['train_end']);end=time_seconds(fold['validation_end'])
    tr=np.flatnonzero((t<cut-300)&samples.partition.eq(0).to_numpy())
    val=np.flatnonzero((t>=cut+7200)&(t<end-300)&samples.partition.eq(0).to_numpy())
    return tr,val

def set_seed():
    torch.manual_seed(42);torch.cuda.manual_seed_all(42);np.random.seed(42);random.seed(42)

@torch.no_grad()
def neural_predict(model,cache,ids,transform,norm,batch=256):
    model.eval();risks=[];times=[]
    for start in range(0,len(ids),batch):
        ii=ids[start:start+batch];stats=transform.transform(cache.features[ii]);x,z=cache.batch(ii,stats,norm)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=model.name!='gru'):logits=model(x,z)
        p,t=distribution(logits);risks.append(p.cpu().numpy());times.append(t.cpu().numpy())
    return {'risk':np.concatenate(risks),'time':np.concatenate(times)}

def train_neural(name,cache,ids,transform,path,epochs=6):
    set_seed();rng=np.random.default_rng(42);model=RiskNet(name,len(transform.mean)).cuda();norm=cache.sequence_norm(ids)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
    y=cache.samples.iloc[ids].event_in_300s.to_numpy();pos=ids[y];neg=ids[~y];p=len(pos)/len(ids)
    assert len(pos)>0 and len(neg)>0
    stats=transform.transform(cache.features[ids]);positions=np.full(len(cache.samples),-1,dtype=int);positions[ids]=np.arange(len(ids))
    weights=row_weights(cache.samples.iloc[ids]);global_weights=np.ones(len(cache.samples),dtype='float32');global_weights[ids]=weights
    with torch.no_grad():
        prior=np.clip(cache.targets['hazard_y'][ids].sum(0)/np.maximum(cache.targets['hazard_mask'][ids].sum(0),1)*8,1e-4,.1)
        model.head[-1].bias.copy_(torch.from_numpy(np.log(prior/(1-prior))).cuda())
    history=[];batch=256;steps=int(np.ceil(len(ids)/batch))
    for epoch in range(epochs):
        started=time.perf_counter();model.train();losses=[]
        for step in range(steps):
            npos=64;chosen=np.r_[rng.choice(pos,npos),rng.choice(neg,batch-npos)];rng.shuffle(chosen)
            event=cache.samples.event_in_300s.to_numpy()[chosen]
            importance=np.where(event,p/.25,(1-p)/.75)*global_weights[chosen]
            x,z=cache.batch(chosen,stats[positions[chosen]],norm)
            target=torch.from_numpy(cache.targets['hazard_y'][chosen]).cuda();mask=torch.from_numpy(cache.targets['hazard_mask'][chosen]).cuda()
            weight=torch.from_numpy(importance.astype('float32')).cuda();opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=name!='gru'):out=model(x,z)
            per=(torch.nn.functional.binary_cross_entropy_with_logits(out.float(),target,reduction='none')*mask).sum(1)
            loss=(per*weight).sum()/weight.sum().clamp_min(1)
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss for {name}')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
        entry={'epoch':epoch+1,'loss':float(np.mean(losses)),'seconds':time.perf_counter()-started,'gpu_tensor_peak_mib':torch.cuda.max_memory_allocated()/1024**2}
        history.append(entry);dump(history,path/'epochs.json');status('risk_neural_epoch',model=name,fit=path.name,**entry)
    torch.save({'name':name,'nstats':len(transform.mean),'state_dict':model.state_dict(),'epochs':epochs,'seed':42},path/'model.pt')
    joblib.dump(norm,path/'sequence_normalization.joblib');return model,norm,history

def fit_one(name,cache,tr,val,label):
    path=MODELS/'risk'/name/label;path.mkdir(parents=True,exist_ok=True)
    saved=EVAL/f'{name}_{label}.npz'
    if (path/'fit.json').exists() and saved.exists():return dict(np.load(saved))
    full=eligible(cache.samples.iloc[tr]);tr=tr[full]
    started=time.perf_counter();transform=Transform().fit(cache.features[tr]);x=transform.transform(cache.features[tr]);z=transform.transform(cache.features[val])
    joblib.dump(transform,path/'preprocess.joblib');aux=fit_auxiliary(cache,tr,x);joblib.dump(aux,path/'auxiliary.joblib')
    auxpred=auxiliary_predict(aux,z)
    if name in NEURAL:
        if (path/'model.pt').exists() and (path/'sequence_normalization.joblib').exists():
            state=torch.load(path/'model.pt',map_location='cuda',weights_only=True)
            assert state['name']==name and state['epochs']==6
            model=RiskNet(name,state['nstats']).cuda();model.load_state_dict(state['state_dict'])
            norm=joblib.load(path/'sequence_normalization.joblib');history=load(path/'epochs.json')
            status('risk_checkpoint_reused_for_evaluation',model=name,fit=label,epochs=len(history))
        else:model,norm,history=train_neural(name,cache,tr,transform,path)
        pred=neural_predict(model,cache,val,transform,norm);del model;torch.cuda.empty_cache()
    else:
        extra=ExtraTransform(name).fit(x);model=cpu_model(name)
        with threadpool_limits(limits=4):model.fit(extra.transform(x),cache.samples.iloc[tr].event_in_300s,row_weights(cache.samples.iloc[tr]))
        pred={'risk':model.predict_proba(extra.transform(z))[:,1],'time':auxpred['time']}
        joblib.dump({'model':model,'extra':extra},path/'model.joblib');history=[]
    pred['type']=auxpred['type'];pred['eta']=(pred['time']*np.arange(15,300,30)).sum(1);pred['ids']=val
    np.savez(saved,**pred)
    metrics=ranking(cache.samples.iloc[val],pred['risk'])
    dump({'name':name,'fit':label,'train_windows':len(tr),'evaluation_windows':len(val),'seconds':time.perf_counter()-started,
          'metrics':metrics,'epochs':len(history),'positive_cost':8,'normalization_fit':'this fold training only',
          'model_sha256':sha(path/('model.pt' if name in NEURAL else 'model.joblib'))},path/'fit.json')
    status('risk_fit_complete',model=name,fit=label,metrics=metrics,seconds=time.perf_counter()-started)
    return pred

def calibrate(raw,y):
    logit=np.log(np.clip(raw,1e-6,1-1e-6)/np.clip(1-raw,1e-6,1))[:,None]
    model=LogisticRegression(C=.1,max_iter=300).fit(logit,y)
    if model.coef_[0,0]<=0:
        # Reject an inversion of a failed ranking model; keep its original order.
        return {'kind':'identity','reason':'fitted calibration slope nonpositive'}
    return {'kind':'sigmoid','model':model}

def apply_calibration(cal,p):
    if cal['kind']=='identity':return p
    z=np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1))[:,None]
    return cal['model'].predict_proba(z)[:,1]

def run_method(name,cache):
    destination=MODELS/'risk'/name;destination.mkdir(parents=True,exist_ok=True)
    if (destination/'complete.json').exists():return
    preds=[];fold_scores=[]
    for fold in FOLDS:
        tr,val=fold_ids(cache.samples,fold);pred=fit_one(name,cache,tr,val,fold['name']);preds.append(pred)
        fold_scores.append(ranking(cache.samples.iloc[val],pred['risk']))
    ids=np.concatenate([p['ids'] for p in preds]);risk=np.concatenate([p['risk'] for p in preds]);rows=cache.samples.iloc[ids]
    ok=eligible(rows);cal=calibrate(risk[ok],rows.event_in_300s.to_numpy()[ok]);joblib.dump(cal,destination/'calibration.joblib')
    calibrated=apply_calibration(cal,risk);threshold,curve=threshold_from_oof(rows,calibrated)
    np.savez(EVAL/f'{name}_oof.npz',ids=ids,risk=risk,calibrated_risk=calibrated,time=np.concatenate([p['time'] for p in preds]),type=np.concatenate([p['type'] for p in preds]))
    dump(curve,destination/'oof_threshold_search.json')
    # Fit all pre-October training data; evaluate only development selection now.
    tr=np.flatnonzero(cache.samples.partition.eq(0));val=np.flatnonzero(cache.samples.partition.eq(1))
    final_pred=fit_one(name,cache,tr,val,'final')
    summary={'name':name,'algorithm_family':name,'fold_scores':fold_scores,
             'mean_fold_event_macro_ap':float(np.mean([m['event_macro_ap'] for m in fold_scores if m['event_macro_ap'] is not None])),
             'pooled_oof':ranking(rows,risk),'oof_calibrated':ranking(rows,calibrated),
             'development_alarm_policy':threshold,'calibration_status':'fit on OOF development labels; Oct8-9 untouched for diagnostics',
             'threshold_selection_is_not_unbiased_validation':True,'selection':ranking(cache.samples.iloc[val],final_pred['risk']),
             'history_period_accessed':False,'actual_distinct_model':True,'status':'trained'}
    dump(summary,destination/'complete.json');status('risk_method_complete',**summary)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='+',default=RISK_NAMES);args=parser.parse_args()
    protocol()
    if not (DATA/'targets_finalized.json').exists():raise RuntimeError('Data target masks must be finalized before training')
    if not (EVAL/'preflight.json').exists():raise RuntimeError('Run preflight tests before training')
    torch.set_num_threads(4);torch.set_num_interop_threads(2);cache=Cache()
    for name in args.models:run_method(name,cache)
    status('risk_batch_complete',models=args.models)

if __name__=='__main__':main()
