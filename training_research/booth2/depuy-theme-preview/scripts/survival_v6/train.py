"""v6: direct censored quantiles, mixtures, survival forest and separate wall-clock warnings."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import numpy as np
import pandas as pd
from scipy.special import ndtr, logsumexp
from joblib import Parallel, delayed, dump, load
import v5 as V

H=V.H
SMALL=V.SMALL
PHYS=V.END+["age_h","since_restart_h","cum_low53_h","cum_vsd_hi_h","tte51_h","v_first1h","vsd_slope_3h","v_slope_10h","v60"]
NAMES=["km_landmark","km_age","ipcw_hgb6","ipcw_hgb32","cqrnn6","cqrnn32","mix6","mix32","rsf32"]
TAU=np.arange(1,20)/20
Q=np.array([.1,.5,.9])


def cqr_weights(taus,qhat):
    return (taus<qhat)*1.0+(taus>=qhat)*(taus-qhat)/(1-qhat)


def prep(frame):
    frame=frame.copy()
    frame["log_vsd_30m"]=np.log(frame.vsd_mean_30m.clip(lower=.02))
    return frame


def qoutputs(values,taus):
    values=np.maximum.accumulate(np.maximum(values,1e-5),axis=1)
    tail=values[:,-1]+np.maximum(2*(values[:,-1]-values[:,0]),24)
    curves=np.empty((len(values),len(H)));qs=np.empty((len(values),3))
    for i,row in enumerate(values):
        curves[i]=np.interp(H,np.r_[0,row,tail[i]],np.r_[0,taus,1])
        qs[i]=np.interp(Q,taus,row)
    return curves,qs


def surv_outputs(times,surv,tail):
    curve=np.empty((len(surv),len(H)));quant=np.empty((len(surv),3));unresolved=np.zeros_like(quant,dtype=bool)
    ix=np.searchsorted(times,H,side="right")-1
    curve[:]=np.where(ix[None,:]>=0,1-surv[:,np.maximum(ix,0)],0)
    for j,p in enumerate(Q):
        crossed=surv<=1-p
        where=crossed.argmax(1)
        unresolved[:,j]=~crossed.any(1)
        quant[:,j]=np.where(unresolved[:,j],tail,np.asarray(times)[where])
    return curve,quant,unresolved


def forward_net(state,z):
    s={k:np.array(v) for k,v in state.items()}
    hidden=np.tanh(z@s["0.weight"].T+s["0.bias"])
    return hidden@s["2.weight"].T+s["2.bias"]


def mixture_outputs(parameters):
    def cdf(t):
        values=[]
        for mu,sigma,prob in parameters:
            t=np.asarray(t)
            if t.ndim==1:
                norm=(np.log(np.maximum(t,1e-12))[None,:,None]-mu[:,None,:])/sigma[None,None,:]
            else:
                norm=(np.log(np.maximum(t,1e-12))[:,:,None]-mu[:,None,:])/sigma[None,None,:]
            values.append(np.sum(ndtr(norm)*prob[:,None,:],axis=2))
        return np.mean(values,axis=0)
    curves=cdf(H)
    lo=np.full((len(curves),3),1e-8);hi=np.full_like(lo,1e5)
    for _ in range(60):
        mid=np.sqrt(lo*hi);below=cdf(mid)<Q
        lo=np.where(below,mid,lo);hi=np.where(below,hi,mid)
    return curves,(lo+hi)/2


def predict_model(model,frame):
    name=model["family"]
    if name=="km_landmark":
        times,surv=np.array(model["times"]),np.array(model["survival"])
        c,q,u=surv_outputs(times,np.repeat(surv[None,:],len(frame),0),model["tail"])
        return c,q,dict(unresolved=u.mean(0).tolist())
    if name=="km_age":
        times,surv=np.array(model["times"]),np.array(model["survival"])
        age=frame.age_h.to_numpy()
        km=(times,surv)
        raw_denom=V.km_value(km,age)
        unsupported=raw_denom<=0
        denom=np.maximum(raw_denom,1e-8)
        c=1-V.km_value(km,age[:,None]+H)/denom[:,None]
        q=np.empty((len(age),3));u=np.zeros_like(q,dtype=bool)
        for i,a in enumerate(age):
            for j,p in enumerate(Q):
                available=np.flatnonzero((times>a)&(surv<=denom[i]*(1-p)))
                q[i,j]=times[available[0]]-a if len(available) else max(model["tail"]-a,.001)
                u[i,j]=not len(available)
        # Beyond empirical support, probabilities are unidentified. Keep the declared
        # sentinel quantiles and use the corresponding point-mass display convention.
        # This is flagged, not represented as a calibrated extrapolation.
        c[unsupported]=(H[None,:]>=q[unsupported,1,None]).astype(float)
        return np.clip(c,0,1),q,dict(unresolved=u.mean(0).tolist(),unsupported_age_fraction=float(unsupported.mean()))
    x=frame[model["features"]].to_numpy(float)
    z=V.scaled(x,np.array(model["median"]),np.array(model["spread"]))
    if name.startswith("ipcw_hgb"):
        values=np.column_stack([m.predict(z) for m in model["estimators"]])
        values=np.sort(np.maximum(values,.001),axis=1)
        c,q=qoutputs(values,model["taus"])
    elif name.startswith("cqrnn"):
        values=np.mean([np.cumsum(np.logaddexp(0,forward_net(s,z))+.0001,axis=1)*10 for s in model["states"]],axis=0)
        c,q=qoutputs(values,TAU)
    elif name.startswith("mix"):
        params=[]
        for member in model["states"]:
            raw=forward_net(member["network"],z)
            mu=np.clip(raw[:,:2],-7,5)
            sigma=np.exp(np.clip(member["log_sigma"],-1.5,1.2))
            prob=np.exp(raw[:,2:]-logsumexp(raw[:,2:],axis=1,keepdims=True))
            params.append((mu,sigma,prob))
        c,q=mixture_outputs(params)
    else:
        forest=model["estimator"]
        curves=[];quants=[];unresolved=[]
        for start in range(0,len(z),512):
            survival=forest.predict_survival_function(z[start:start+512],return_array=True)
            a,b,u=surv_outputs(forest.unique_times_,survival,model["tail"])
            curves.append(a);quants.append(b);unresolved.append(u)
        return np.concatenate(curves),np.concatenate(quants),dict(unresolved=np.concatenate(unresolved).mean(0).tolist())
    return c,q,{}


def fit_cqr(z,t,e,w):
    import torch
    from torch import nn
    torch.set_num_threads(1)
    xt=torch.tensor(z,dtype=torch.float32);yt=torch.tensor(t[:,None]/10,dtype=torch.float32)
    event=torch.tensor(e[:,None],dtype=torch.bool);wt=torch.tensor(w[:,None]/w.sum(),dtype=torch.float32)
    taus=torch.tensor(TAU[None,:],dtype=torch.float32)
    ystar=max(80,2*float(t.max()))/10
    models=[];history=[]
    for seed in (17,43):
        torch.manual_seed(seed)
        net=nn.Sequential(nn.Linear(z.shape[1],12),nn.Tanh(),nn.Linear(12,len(TAU)))
        # Training-only starting grid; the architecture enforces increasing positive quantiles.
        initial=np.maximum(np.quantile(t[e],TAU)/10,np.arange(1,20)*.002)
        delta=np.diff(np.r_[0,initial]);bias=np.log(np.expm1(np.maximum(delta,.002)))
        with torch.no_grad():
            net[2].weight.mul_(.05);net[2].bias.copy_(torch.tensor(bias,dtype=torch.float32))
        opt=torch.optim.AdamW(net.parameters(),lr=.015,weight_decay=.03)
        logs=[]
        for epoch in range(350):
            pred=torch.cumsum(nn.functional.softplus(net(xt))+.0001,dim=1)
            with torch.no_grad():
                nearest=(pred-yt).abs().argmin(1)
                qhat=taus[0,nearest,None]
                weights=cqr_weights(taus,qhat)
            error=yt-pred;check=torch.maximum(taus*error,(taus-1)*error)
            far=ystar-pred;tail=torch.maximum(taus*far,(taus-1)*far)
            loss=(wt*torch.where(event,check,weights*check+(1-weights)*tail)).sum()/len(TAU)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
            if epoch%50==0 or epoch==349: logs.append([epoch,float(loss.detach())])
        models.append({k:v.detach().numpy() for k,v in net.state_dict().items()});history.append(logs)
    return models,history


def fit_mix(z,t,e,w):
    import torch
    from torch import nn
    torch.set_num_threads(1)
    xt=torch.tensor(z,dtype=torch.float32);yt=torch.tensor(np.log(np.maximum(t,1e-6))[:,None],dtype=torch.float32)
    et=torch.tensor(e,dtype=torch.bool);wt=torch.tensor(w/w.sum(),dtype=torch.float32)
    models=[];history=[]
    for seed in (17,43):
        torch.manual_seed(seed)
        net=nn.Sequential(nn.Linear(z.shape[1],8),nn.Tanh(),nn.Linear(8,4))
        ls=nn.Parameter(torch.tensor([-.1,.1],dtype=torch.float32))
        with torch.no_grad():
            net[2].weight.mul_(.1)
            net[2].bias[:2].copy_(torch.tensor(np.log(np.quantile(t[e],[.25,.75])),dtype=torch.float32))
            net[2].bias[2:].zero_()
        opt=torch.optim.AdamW(list(net.parameters())+[ls],lr=.012,weight_decay=.04);logs=[]
        for epoch in range(300):
            raw=net(xt);mu=raw[:,:2].clamp(-7,5);sigma=ls.clamp(-1.5,1.2).exp()
            logp=nn.functional.log_softmax(raw[:,2:],dim=1);zscore=(yt-mu)/sigma
            logdensity=-.5*zscore.square()-sigma.log()-yt-.5*np.log(2*np.pi)
            logsurvival=torch.special.log_ndtr(-zscore)
            loglike=torch.where(et,torch.logsumexp(logp+logdensity,dim=1),torch.logsumexp(logp+logsurvival,dim=1))
            loss=-(wt*loglike).sum()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(net.parameters())+[ls],5);opt.step()
            if epoch%50==0 or epoch==299:logs.append([epoch,float(loss.detach())])
        models.append(dict(network={k:v.detach().numpy() for k,v in net.state_dict().items()},log_sigma=ls.detach().numpy()));history.append(logs)
    return models,history


def fit_predict(frame,cycles,test_indices,name,path=None):
    tr,w=V.train_rows(frame,cycles);te=frame.loc[test_indices]
    model=dict(family=name,training_cycles=cycles)
    if name.startswith("km"):
        if name=="km_age":
            q=tr[tr.left_censored.eq(0)].groupby("cycle_id").first()
            times,surv=V.km_fit(q.life_h.to_numpy(),q.event.to_numpy(),np.ones(len(q)))
        else:
            times,surv=V.km_fit(tr.rul_h.to_numpy(),tr.event.to_numpy(),w)
        model.update(times=times,survival=surv,tail=float(times.max()+24))
    else:
        cols=SMALL if name.endswith("6") else PHYS
        med,spread=V.scaler(tr[cols].to_numpy(float));z=V.scaled(tr[cols].to_numpy(float),med,spread)
        t=tr.rul_h.to_numpy();e=tr.event.to_numpy(bool)
        model.update(features=cols,median=med,spread=spread)
        if name.startswith("ipcw"):
            from sklearn.ensemble import HistGradientBoostingRegressor
            km=V.km_fit(t,~e,w);iw=w/np.maximum(V.km_value(km,t,left=True),.1)
            taus=np.array([.1,.25,.5,.75,.9]);estimators=[]
            for tau in taus:
                m=HistGradientBoostingRegressor(loss="quantile",quantile=tau,max_iter=120,max_leaf_nodes=7,min_samples_leaf=100,
                     l2_regularization=10,learning_rate=.05,early_stopping=False,random_state=20261120)
                m.fit(z[e],t[e],sample_weight=iw[e]);estimators.append(m)
            model.update(estimators=estimators,taus=taus)
        elif name.startswith("cqr"):
            states,history=fit_cqr(z,t,e,w);model.update(states=states,history=history)
        elif name.startswith("mix"):
            states,history=fit_mix(z,t,e,w);model.update(states=states,history=history)
        else:
            from sksurv.ensemble import RandomSurvivalForest
            from sksurv.util import Surv
            pick=np.concatenate([np.flatnonzero(tr.cycle_id.to_numpy()==c)[::5] for c in cycles])
            target=Surv.from_arrays(e[pick],np.ceil(t[pick]*10)/10)
            m=RandomSurvivalForest(n_estimators=80,max_depth=5,min_samples_leaf=25,max_features=.7,n_jobs=1,random_state=20261120)
            m.fit(z[pick],target,sample_weight=w[pick]);model.update(estimator=m,tail=float(np.max(target['time'])+24))
    c,q,details=predict_model(model,te)
    assert np.isfinite(c).all() and np.isfinite(q).all()
    assert np.all(np.diff(c,axis=1)>=-1e-6) and np.all(np.diff(q,axis=1)>=-1e-6)
    if path:
        dump(model,path.with_suffix('.joblib'),compress=3)
        V.save_json(path,dict(family=name,features=model.get('features',[]),training_cycles=cycles,details=details,history=model.get('history'),
                              artifact=path.with_suffix('.joblib').name,sha256=V.digest(path.with_suffix('.joblib'))))
    return c.astype('float32'),q.astype('float32'),details


def warning_fit(frame,cycles,ix,name,path=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    q,w=V.train_rows(frame,cycles)
    cols=SMALL if name=='warn_logit6' else PHYS
    med,spread=V.scaler(q[cols].to_numpy(float));z=V.scaled(q[cols].to_numpy(float),med,spread)
    y=q.y_eol_60m.to_numpy(float);known=np.isfinite(y)&(y>=0)
    y=y[known].astype(int)
    if len(np.unique(y))<2:
        model=dict(constant=float(y.mean()) if len(y) else 0.)
        score=np.full(len(ix),model['constant'])
    else:
        m=LogisticRegression(C=.05,max_iter=1000) if name=='warn_logit6' else HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,
             max_depth=3,min_samples_leaf=80,l2_regularization=10,learning_rate=.05,early_stopping=False,random_state=20261120)
        m.fit(z[known],y,sample_weight=w[known])
        score=m.predict_proba(V.scaled(frame.loc[ix,cols].to_numpy(float),med,spread))[:,1]
        model=dict(estimator=m,features=cols,median=med,spread=spread)
    if path:dump(model,path,compress=3)
    return score.astype('float32')


def fold(data,out,tag,tr,te):
    started=time.time();frame=prep(pd.read_parquet(Path(data)/'frame.parquet'))
    directory=Path(out)/'folds'/tag;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'result.json').exists():return json.loads((directory/'result.json').read_text(encoding='utf-8'))
    ti=frame.index[frame.cycle_id.isin(tr)].to_numpy();hi=frame.index[frame.cycle_id.isin(te)].to_numpy()
    arrays=dict(train_index=ti,test_index=hi);result=dict(tag=tag,training=tr,held_out=te,models={})
    for name in NAMES:
        rows=[]
        for val in V.inner_groups(tr):
            fit=[c for c in tr if c not in val];vi=frame.index[frame.cycle_id.isin(val)].to_numpy()
            c,q,_=fit_predict(frame,fit,vi,name);rows+=V.metrics(frame,vi,c,q,fit)
        inner=V.summarize(rows)
        c,q,details=fit_predict(frame,tr,hi,name,directory/f'{name}.json')
        result['models'][name]=dict(inner=inner,selection_loss=inner['mae']+.5*inner['last2h'],test=V.metrics(frame,hi,c,q,tr),details=details)
        arrays[name+'_curve']=c;arrays[name+'_quant']=q
        print(tag,name,'complete',flush=True)
    result['selected']=min(NAMES,key=lambda n:result['models'][n]['selection_loss'])
    for name in ('warn_logit6','warn_hgb32'):
        inner=np.zeros(len(frame),dtype='float32')
        for val in V.inner_groups(tr):
            fit=[c for c in tr if c not in val];vi=frame.index[frame.cycle_id.isin(val)].to_numpy()
            inner[vi]=warning_fit(frame,fit,vi,name)
        arrays[name+'_inner']=inner[ti];arrays[name+'_score']=warning_fit(frame,tr,hi,name,directory/f'{name}.joblib')
    result['seconds']=time.time()-started
    np.savez_compressed(directory/'predictions.npz',**arrays);V.save_json(directory/'result.json',result)
    return result


def alarm_fold(data,out,r,protocol):
    ev=V.load_ev(Path(data)/'evaluation.py');frame=pd.read_parquet(Path(data)/'frame.parquet')
    events=pd.read_csv(Path(data)/'events.csv',parse_dates=['time']);products=pd.read_csv(Path(data)/'products.csv',low_memory=False)
    def hours(cycles):return float(products[products.cycle_id.isin(cycles)&products.supervised_product_eligible.eq(1)].spray_samples.sum()/3600)
    with np.load(Path(out)/'folds'/r['tag']/'predictions.npz',allow_pickle=False) as a:
        ti,hi=a['train_index'],a['test_index'];ans={}
        for name in ('warn_logit6','warn_hgb32'):
            inner=ev.select_threshold(frame.loc[ti].assign(split='validation'),a[name+'_inner'],events[events.cycle_id.isin(r['training'])].assign(split='validation'),
                                      hours(r['training']),grid=np.array(protocol['alarm_threshold_grid']),false_budget=.05)
            m,rows,alarms=ev.evaluate(frame.loc[hi],a[name+'_score'],inner['threshold'],events[events.cycle_id.isin(r['held_out'])],hours(r['held_out']))
            ans[name]=dict(inner=inner['metrics'],threshold=inner['threshold'],test=m,events=rows)
        selected=max(ans,key=lambda n:ev.selection_key(ans[n]['inner'],false_budget=.05))
    return dict(tag=r['tag'],models=ans,selected=selected)


def collect(data,out,results,protocol,jobs):
    alarms=Parallel(n_jobs=min(jobs,4))(delayed(alarm_fold)(data,out,r,protocol) for r in results)
    report=dict(protocol=protocol,runs=results,comparison={},warning={})
    frame=pd.read_parquet(Path(data)/'frame.parquet');ledger={}
    for mode in ('loco','forward'):
        folds=[r for r in results if r['tag'].startswith(mode)];table={}
        for name in NAMES+['nested_selected']:
            rows=[]
            for r in folds:
                n=r['selected'] if name=='nested_selected' else name;rows+=r['models'][n]['test']
                if mode=='loco':
                    with np.load(Path(out)/'folds'/r['tag']/'predictions.npz',allow_pickle=False) as a:
                        for suffix,width in (('curve',len(H)),('quant',3)):
                            ledger.setdefault(name+'_'+suffix,np.zeros((len(frame),width),dtype='float32'))[a['test_index']]=a[n+'_'+suffix]
            table[name]=dict(life=V.summarize(rows),cycles=rows)
        report['comparison'][mode]=table
        subset=[r for r in alarms if r['tag'].startswith(mode)];warn={}
        for name in ('warn_logit6','warn_hgb32','nested_selected'):
            entries=[r['models'][r['selected'] if name=='nested_selected' else name] for r in subset]
            totals={k:sum(float(r['test'][k]) for r in entries) for k in ('total_events','hit20','late','missed','false_alarms','unknown_alarms','operating_hours')}
            totals['false_per_100h']=100*totals['false_alarms']/totals['operating_hours']
            warn[name]=dict(totals=totals,folds=entries)
        report['warning'][mode]=warn
    np.savez_compressed(Path(out)/'predictions.npz',**ledger);V.save_json(Path(out)/'report.json',report)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--jobs',type=int,default=6);p.add_argument('--smoke',action='store_true');p.add_argument('--collect',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads(Path(__file__).with_name('protocol.json').read_text(encoding='utf-8'))
    import scipy,sklearn,torch,sksurv
    V.save_json(a.out/'protocol.json',protocol)
    provenance=dict(sources={p.name:V.digest(p) for p in Path(__file__).parent.glob('*.py')},protocol_sha256=V.digest(Path(__file__).with_name('protocol.json')),
      inputs={p.name:V.digest(p) for p in a.data.iterdir() if p.is_file()},versions=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,torch=torch.__version__,sksurv=sksurv.__version__))
    V.save_json(a.out/'provenance.json',provenance)
    frame=prep(pd.read_parquet(a.data/'frame.parquet'));cycles=sorted(frame.cycle_id.unique())
    if a.smoke:
        ix=frame.index[frame.cycle_id.eq(cycles[-1])][::97]
        for name in NAMES:
            c,q,_=fit_predict(frame,cycles[:4],ix,name,a.out/f'{name}.json')
            reloaded=load(a.out/f'{name}.joblib');rc,rq,_=predict_model(reloaded,frame.loc[ix])
            assert np.allclose(c,rc,atol=2e-6) and np.allclose(q,rq,rtol=2e-5,atol=1e-4)
            print('smoke',name,float(np.median(q[:,1])),flush=True)
        for name in ('warn_logit6','warn_hgb32'):
            s=warning_fit(frame,cycles[:4],ix,name)
            assert np.isfinite(s).all() and np.all((s>=0)&(s<=1))
            print('smoke',name,len(s),flush=True)
        return
    tasks=[(f'loco-{c}',[v for v in cycles if v!=c],[c]) for c in cycles]
    tasks += [(f'forward-{i}',[c for c in cycles if c<held[0]],held) for i,held in enumerate(protocol['forward_blocks'])]
    if a.collect:
        results=[json.loads((a.out/'folds'/tag/'result.json').read_text(encoding='utf-8')) for tag,_,_ in tasks]
    else:
        results=Parallel(n_jobs=a.jobs)(delayed(fold)(str(a.data),str(a.out),tag,tr,te) for tag,tr,te in tasks)
    report=collect(a.data,a.out,results,protocol,a.jobs)
    for mode,table in report['comparison'].items():
        for name,r in table.items():print(mode,name,{k:round(v,4) for k,v in r['life'].items()},flush=True)
    print('COMPLETED',flush=True)


if __name__=='__main__':main()
