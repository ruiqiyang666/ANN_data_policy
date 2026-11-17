"""Bounded, validation-only selection. Never predicts the test split."""
import argparse, hashlib, json, platform, time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from modeling import ROOT,TARGETS,fit_preprocessor,encode,summaries,margin_score,RiskNet,predict_net,load_frames
from evaluation import select_threshold, POLICY
from dataclasses import asdict

def dump(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def operating_hours(p,split):
    q=p[p.split.eq(split)&p.supervised_product_eligible.eq(1)]
    return float(q.spray_samples.sum()/3600)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--overwrite-before-test',action='store_true');args=ap.parse_args()
    art=ROOT/'artifacts'; art.mkdir(exist_ok=True)
    if (art/'evaluation.json').exists():raise RuntimeError('Test has been opened: this experiment is frozen. Do not tune using test results.')
    if (art/'frozen_config.json').exists() and not args.overwrite_before_test:raise RuntimeError('Already frozen. Reproduction requires a separate named run, not overwriting results.')
    started=time.perf_counter(); torch.set_num_threads(4)
    seed=20261117; np.random.seed(seed);torch.manual_seed(seed)
    specs=[dict(name='margin',kind='margin')]
    specs += [dict(name='logistic_c'+str(c).replace('.','p'),kind='logistic',C=c) for c in [.03,.3,3.0]]
    specs += [dict(name=f'{kind}_{h}',kind=kind,hidden=h) for kind in ['tcn','gru'] for h in [8,16]]
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),seed=seed,model_specs=specs,
      targets=TARGETS,selectable_heads=['y_eol_30m','y_eol_45m','y_pre20_40m'],
      max_epochs=50,patience=9,batch_size=256,learning_rate=.001,weight_decay=.003,
      training_stride='one earliest available origin per calendar minute; no random row split',
      history_minutes=60,alarm_policy=asdict(POLICY),threshold_grid=[round(v,2) for v in np.arange(.05,1,.05)]+[1.000001],
      false_budget_per_production_hour=.1,selection_order=['false budget feasible','hit20','fewer false alarms','smaller successful excess','fewer late alarms','higher threshold'],
      stop_rule='8 fixed candidates; validation only. Freeze all models and policies before any new test prediction.',
      test_status='historical expanded holdout previously partly used by V1; no new V2 score opened during selection')
    dump(art/'experiment_plan.json',protocol)
    f,y,e,p=load_frames()
    # Keep test out of all preprocessing, representation fitting and scoring here.
    tv=f.split.isin(['train','validation']); f=f[tv].reset_index(drop=True); y=y[tv].reset_index(drop=True)
    pre=fit_preprocessor(f[f.split.eq('train')]);dump(art/'preprocessor.json',pre)
    x=encode(f,pre); z=summaries(x); yy=y[TARGETS].to_numpy(np.float32)
    train=f.split.eq('train').to_numpy(); val=f.split.eq('validation').to_numpy()
    minute=f.timestamp.dt.floor('min'); sample=~pd.DataFrame({'cycle':f.cycle_id,'minute':minute}).duplicated().to_numpy()
    ti=np.flatnonzero(train&sample&np.isfinite(yy).any(1)); vi=np.flatnonzero(val)
    ev=e[e.split.eq('validation')&e.is_eol.eq(1)].copy()
    vf=f.iloc[vi].reset_index(drop=True); hours=operating_hours(p,'validation')
    model_results={}; selection_rows=[]; histories={}
    for spec in specs:
        name,kind=spec['name'],spec['kind']; out=art/'models'/name;out.mkdir(parents=True,exist_ok=True)
        model_start=time.perf_counter();history=[]
        if kind=='margin':
            pv=margin_score(vf)[:,None];heads=['engineering_margin']
        elif kind=='logistic':
            models=[]
            for j in range(len(TARGETS)):
                valid=ti[np.isfinite(yy[ti,j])]
                m=make_pipeline(StandardScaler(),LogisticRegression(C=spec['C'],class_weight='balanced',solver='lbfgs',max_iter=1500,random_state=seed))
                m.fit(z[valid],yy[valid,j]);models.append(m)
            joblib.dump(models,out/'model.joblib')
            pv=np.column_stack([m.predict_proba(z[vi])[:,1] for m in models]);heads=TARGETS
        else:
            torch.manual_seed(seed);rng=np.random.default_rng(seed)
            net=RiskNet(kind,spec['hidden']);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.003)
            valid=np.isfinite(yy[ti]);pos=np.nansum(yy[ti],axis=0);neg=valid.sum(0)-pos
            pos_weight=torch.tensor(np.minimum(neg/np.maximum(pos,1),60),dtype=torch.float32)
            xt=torch.from_numpy(x[ti]); yt=torch.from_numpy(np.nan_to_num(yy[ti]));mt=torch.from_numpy(valid.astype(np.float32))
            vvalid=np.isfinite(yy[vi]); xv=torch.from_numpy(x[vi]);yv=torch.from_numpy(np.nan_to_num(yy[vi]));mv=torch.from_numpy(vvalid.astype(np.float32))
            def loss_fn(logit,target,mask):
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logit,target,pos_weight=pos_weight,reduction='none')
                return (loss*mask).sum()/mask.sum().clamp_min(1)
            best=float('inf');bad=0
            for epoch in range(1,51):
                net.train();total=0.;den=0
                order=rng.permutation(len(ti))
                for at in range(0,len(ti),256):
                    b=order[at:at+256];opt.zero_grad();loss=loss_fn(net(xt[b]),yt[b],mt[b]);loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
                    total+=float(loss.detach())*float(mt[b].sum());den+=float(mt[b].sum())
                net.eval();vl=0.;vd=0
                with torch.inference_mode():
                    for at in range(0,len(vi),1024):
                        mm=mv[at:at+1024]; ll=loss_fn(net(xv[at:at+1024]),yv[at:at+1024],mm)
                        vl+=float(ll)*float(mm.sum());vd+=float(mm.sum())
                row=dict(epoch=epoch,train_loss=total/max(den,1),validation_loss=vl/max(vd,1));history.append(row)
                if row['validation_loss']<best-1e-5:
                    best=row['validation_loss'];bad=0
                    torch.save(dict(state_dict=net.state_dict(),spec=spec,epoch=epoch),out/'best.pt')
                else:bad+=1
                if epoch%10==0:print(name,'epoch',epoch,'val_loss',round(row['validation_loss'],5),flush=True)
                if bad>=9:break
            torch.save(dict(state_dict=net.state_dict(),spec=spec,epoch=epoch),out/'last.pt')
            ck=torch.load(out/'best.pt',weights_only=False,map_location='cpu');net.load_state_dict(ck['state_dict'])
            pv=predict_net(net,x[vi]);heads=TARGETS
            dump(out/'history.json',history)
            spec.update(best_epoch=int(ck['epoch']),epochs_run=epoch,parameters=sum(q.numel() for q in net.parameters()))
        best_policy=None
        for j,head in enumerate(heads):
            if kind!='margin' and head not in protocol['selectable_heads']:continue
            chosen=select_threshold(vf,pv[:,j],ev,hours)
            row=dict(model=name,head=head,head_index=j,threshold=float(chosen['threshold']),metrics=chosen['metrics'],selection_key=list(chosen['selection_key']))
            selection_rows.append(row)
            if best_policy is None or tuple(row['selection_key'])>tuple(best_policy['selection_key']):best_policy=row
        spec['fit_seconds']=time.perf_counter()-model_start
        model_results[name]=best_policy
        dump(out/'validation_policy.json',best_policy)
        print(name,json.dumps(best_policy,ensure_ascii=False),flush=True)
    selected=max(specs,key=lambda s:(tuple(model_results[s['name']]['selection_key']),-specs.index(s)))['name']
    frozen=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),selected_model=selected,
      model_version='booth2-v2-'+datetime.now().strftime('%Y%m%dT%H%M%S'),
      model_specs=specs,thresholds={k:v['threshold'] for k,v in model_results.items()},
      policies={k:{j:v[j] for j in ['head','head_index','threshold']} for k,v in model_results.items()},
      validation_results=model_results,history_minutes=60,seed=seed,protocol_sha256=digest(art/'experiment_plan.json'),
      data_sha256={p.name:digest(p) for p in (ROOT/'data').iterdir() if p.is_file()},
      source_sha256={p.name:digest(p) for p in (ROOT/'src').glob('*.py') if p.name!='report.py'},alarm_policy=asdict(POLICY),
      fitted_file_sha256={str(p.relative_to(art)):digest(p) for p in (art/'models').rglob('*') if p.suffix in ['.pt','.joblib']},
      preprocessor_sha256=digest(art/'preprocessor.json'),test_used_for_selection=False)
    dump(art/'validation_search.json',selection_rows)
    dump(art/'frozen_config.json',frozen)
    dump(art/'run_log.json',dict(started_utc=protocol['created_utc'],frozen_utc=frozen['frozen_at_utc'],
      elapsed_seconds=time.perf_counter()-started,host=platform.node(),platform=platform.platform(),python=platform.python_version(),
      torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,device='cpu',threads=4,
      sampled_training_origins=len(ti),validation_origins=len(vi),server_execution=False,site_connected=False,selected_model=selected))
    print('FROZEN',selected,'seconds',round(time.perf_counter()-started,2),flush=True)

if __name__=='__main__':main()
