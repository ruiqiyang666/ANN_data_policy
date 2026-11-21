"""Run 20261112_v4: frozen v3 algorithms, expanding forward folds, blind week, final refit.

Nothing here selects an algorithm or a hyperparameter. Every fold fits its own
imputation/scaling/auxiliary heads on training rows only; calibration and alarm
thresholds for a fold come from EARLIER out-of-fold predictions only. The frozen
v3 models are scored on exactly the same blind rows as the v4 fold-f6 fit.
"""
import sys
sys.dont_write_bytecode=True
import torch
import os,time
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from common_v2 import ROOT,SHARED,DATA,MODELS,EVAL,TASK,TYPES,FOLDS,BLIND_FOLD,NEW_PARTITION,OLD_V3,load,dump,status,sha,time_seconds as sec
from metrics_v2 import eligible,ranking,alarms,threshold_from_oof
from risk_models import Cache,Transform,ExtraTransform,cpu_model,row_weights,fit_auxiliary,auxiliary_predict
from probability_guard import normalize_time
from life_models import landmark_weights,fit_model,predict_model
from train_stage_life import RAW,VARIANTS,stage_features,fit_stage
from train_life_v2 import evaluate as life_eval

THREADS=int(os.environ.get('V4_THREADS','8'))
ALGOS=['logistic','histgb','extra_trees']
RULE='last_product_rule'
RULE_COLUMN={'voltage':'product_voltage_mean','feedrate':'product_feedrate_mean','hopper_pressure':'product_pressure_mean'}
RULE_SCALE={'voltage':1.0,'feedrate':0.5,'hopper_pressure':20.0}

def fold_ids(samples,fold):
    t=sec(samples.timestamp);cut=sec(fold['train_end']);end=sec(fold['validation_end'])
    # 3900 s: a product result is known up to an hour after the 300 s target.
    # 10800 s: the longest causal feature history, so validation features never
    # summarise an interval whose labels were used for fitting.
    return np.flatnonzero(t<cut-3900),np.flatnonzero((t>=cut+10800)&(t<end-3900))

def calibrate(raw,y):
    if y.sum()==0 or (~y).sum()==0:return {'kind':'identity','reason':'single class in earlier out-of-fold labels'}
    logit=np.log(np.clip(raw,1e-6,1-1e-6)/np.clip(1-raw,1e-6,1))[:,None]
    model=LogisticRegression(C=.1,max_iter=300).fit(logit,y)
    if model.coef_[0,0]<=0:return {'kind':'identity','reason':'fitted calibration slope nonpositive'}
    return {'kind':'sigmoid','model':model}

def apply_calibration(cal,p):
    if cal['kind']=='identity':return np.asarray(p,dtype='float64')
    z=np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1))[:,None]
    return cal['model'].predict_proba(z)[:,1]

def fit_bundle(name,cache,tr):
    tr=tr[eligible(cache.samples.iloc[tr])];rows=cache.samples.iloc[tr];y=rows.event_in_300s.to_numpy()
    info={'train_windows':len(tr),'train_positive_windows':int(y.sum()),'train_events':int(rows.loc[y,'event_id'].nunique()),
          'first_timestamp':rows.timestamp.min() if len(rows) else None,'last_timestamp':rows.timestamp.max() if len(rows) else None}
    if y.sum()==0 or (~y).sum()==0:return None,{**info,'status':'not_fit_single_class_training_labels'}
    started=time.perf_counter();raw=np.asarray(cache.features[tr]);trans=Transform().fit(raw);x=trans.transform(raw);del raw
    aux=fit_auxiliary(cache,tr,x);extra=ExtraTransform(name).fit(x);model=cpu_model(name)
    if 'n_jobs' in model.get_params():model.set_params(n_jobs=THREADS)
    with threadpool_limits(limits=THREADS):model.fit(extra.transform(x),y,row_weights(rows))
    return {'transform':trans,'aux':aux,'model':model,'extra':extra},{**info,'status':'trained','seconds':time.perf_counter()-started}

def predict(bundle,cache,ids):
    z=bundle['transform'].transform(np.asarray(cache.features[ids]));aux=auxiliary_predict(bundle['aux'],z)
    with threadpool_limits(limits=THREADS):risk=bundle['model'].predict_proba(bundle['extra'].transform(z))[:,1]
    t=normalize_time(aux['time'])
    return {'ids':ids,'risk':risk.astype('float64'),'time':t,'type':aux['type'],'eta':(t*np.arange(15,300,30)).sum(1)}

def rule_probability(cache,names,ids):
    v=np.asarray(cache.features[ids][:,names.index(RULE_COLUMN[TASK])],dtype='float64')
    score={'voltage':51-v,'feedrate':np.maximum(9.1-v,v-11.1),'hopper_pressure':np.maximum(100-v,v-250)}[TASK]
    # Monotone squash only so the shared metric code accepts it; no fitting. No product yet -> lowest score.
    return np.where(np.isfinite(score),1/(1+np.exp(-np.clip(score/RULE_SCALE[TASK],-30,30))),0.0)

def score_block(rows,probability,events,threshold=None,pred=None):
    out={'ranking':ranking(rows,probability)}
    if threshold is not None:
        alarm,records,chosen=alarms(rows,probability,threshold,events)
        for record in records:
            e=events[events.event_id.eq(record['event_id'])].iloc[0];alert=record['alert_at']
            record['first_result_at']=e.first_result_at;record['failed_products']=int(e.failed_products)
            record['lead_to_product_start_s']=(e.first_product_start-alert).total_seconds() if alert is not None else None
            record['lead_to_result_s']=(e.first_result_at-alert).total_seconds() if alert is not None else None
        out.update(alarms=alarm,event_records=records,alert_timestamps=[str(v) for v in rows.timestamp.to_numpy()[chosen]])
    if pred is not None and 'eta' in pred:
        y=rows.event_in_300s.to_numpy();err=pred['eta'][y]-rows.eta_seconds.to_numpy()[y]
        out['eta_mae_s_all_positive_windows']=float(np.abs(err).mean()) if len(err) else None
        out['eta_is_conditional_on_future_event']=True
    return out

def pooled(cache,preds,upto):
    """Chronological out-of-fold predictions of folds strictly before `upto`."""
    parts=[preds[f['name']] for f in FOLDS[:upto] if f['name'] in preds]
    if not parts:return None,None
    ids=np.concatenate([p['ids'] for p in parts]);risk=np.concatenate([p['risk'] for p in parts])
    return cache.samples.iloc[ids],risk

def policy_from(cache,preds,upto):
    rows,risk=pooled(cache,preds,upto)
    if rows is None or not rows.event_in_300s.any():return None
    ok=eligible(rows);cal=calibrate(risk[ok],rows.event_in_300s.to_numpy()[ok]);best,_=threshold_from_oof(rows,apply_calibration(cal,risk))
    return {'calibration':cal,'threshold':best['threshold'],'oof_policy':best,'oof_folds':[f['name'] for f in FOLDS[:upto] if f['name'] in preds],
            'oof_events':int(rows.loc[rows.event_in_300s,'event_id'].nunique())}

def old_v3(cache,names,ids,events):
    """Frozen earlier-round models, scored on the identical blind rows."""
    old_names=load(OLD_V3/'data'/TASK/'features.json')['names']
    if old_names!=names:raise AssertionError('v4 feature order differs from the frozen v3 models')
    frozen=load(OLD_V3/'development_freeze.json');selected=frozen['risk_choices'][TASK];rows=cache.samples.iloc[ids];out={'selected_algorithm':selected}
    def run(path,name,label):
        bundle=joblib.load(path/'model.joblib');bundle.update(transform=joblib.load(path/'preprocess.joblib'),aux=joblib.load(path/'auxiliary.joblib'))
        pred=predict(bundle,cache,ids);root=OLD_V3/'models'/TASK/'risk'/name
        cal=joblib.load(root/'calibration.joblib');threshold=load(root/'complete.json')['development_alarm_policy']['threshold']
        prob=apply_calibration(cal,pred['risk']);np.savez(EVAL/f'old_v3_{label}_{name}_blind.npz',ids=ids,calibrated_risk=prob,**{k:v for k,v in pred.items() if k!='ids'})
        result=score_block(rows,prob,events,threshold,pred);result.update(algorithm=name,frozen_v3_threshold=threshold,model_sha256=sha(path/'model.joblib'))
        return result
    for name in ALGOS:out[f'isolated_{name}']={**run(OLD_V3/'models'/TASK/'risk'/name/'final',name,'isolated'),'trained_through':'2026-09-29'}
    out['full_refit']={**run(OLD_V3/'models'/'full_data'/TASK,selected,'full_refit'),'trained_through':'2026-10-23',
                       'note':'v3 attached no validation score to this refit; frozen calibration/threshold of the same algorithm applied as declared'}
    return out

def run_risk():
    torch.set_num_threads(4);cache=Cache();names=load(DATA/'features.json')['names'];events=pd.read_parquet(DATA/'events.parquet')
    protocol=load(ROOT/'protocol.json');selected=protocol['risk_selection'][TASK];preds={a:{} for a in ALGOS+[RULE]};report={'task':TASK,'selected_algorithm':selected,'folds':{}}
    for fold in FOLDS:
        tr,val=fold_ids(cache.samples,fold);vrows=cache.samples.iloc[val];entry={'train_end':fold['train_end'],'validation_end':fold['validation_end'],'validation_windows':len(val),
            'validation_events':int(vrows.loc[vrows.event_in_300s,'event_id'].nunique()),'validation_first':vrows.timestamp.min() if len(val) else None,'validation_last':vrows.timestamp.max() if len(val) else None,'models':{}}
        if len(val):preds[RULE][fold['name']]={'ids':val,'risk':rule_probability(cache,names,val)}
        for name in ALGOS:
            path=MODELS/'risk'/name/fold['name'];path.mkdir(parents=True,exist_ok=True);bundle,info=fit_bundle(name,cache,tr)
            if bundle is not None and len(val):
                joblib.dump(bundle,path/'bundle.joblib');pred=predict(bundle,cache,val);preds[name][fold['name']]=pred
                np.savez(EVAL/f'{name}_{fold["name"]}.npz',**pred);info['model_sha256']=sha(path/'bundle.joblib')
            entry['models'][name]=info;dump(info,path/'fit.json');status('v4_risk_fit',model=name,fit=fold['name'],**{k:v for k,v in info.items() if k in ['status','train_windows','train_events','seconds']})
        report['folds'][fold['name']]=entry
    # Prequential scoring: a fold's calibration and threshold use earlier folds only.
    for k,fold in enumerate(FOLDS):
        entry=report['folds'][fold['name']]
        for name in ALGOS+[RULE]:
            if fold['name'] not in preds[name]:continue
            pred=preds[name][fold['name']];rows=cache.samples.iloc[pred['ids']];policy=policy_from(cache,preds[name],k)
            if policy is None:result=score_block(rows,pred['risk'],events,None,pred);result['alarm_policy']='not available: no earlier out-of-fold events'
            else:
                prob=apply_calibration(policy['calibration'],pred['risk']) if name!=RULE else pred['risk']
                if name==RULE:
                    prows,prisk=pooled(cache,preds[name],k);best,_=threshold_from_oof(prows,prisk);policy={**policy,'threshold':best['threshold'],'oof_policy':best,'calibration':{'kind':'identity','reason':'rule score, no calibration'}}
                result=score_block(rows,prob,events,policy['threshold'],pred)
                result.update(threshold_from_folds=policy['oof_folds'],calibration_kind=policy['calibration']['kind'],oof_policy=policy['oof_policy'])
            entry.setdefault('scores',{})[name]=result
        # A constant score ranks nothing: its average precision equals the positive-window rate.
        if RULE in entry.get('scores',{}):entry['prevalence_baseline_ap']=entry['scores'][RULE]['ranking']['prevalence']
    blind_ids=preds[RULE][BLIND_FOLD]['ids'] if BLIND_FOLD in preds[RULE] else np.array([],int)
    part4=np.flatnonzero(cache.samples.partition.eq(NEW_PARTITION).to_numpy())
    report['blind']={'fold':BLIND_FOLD,'rows':len(blind_ids),'partition_rows':len(part4),'rows_equal_new_partition':bool(np.array_equal(blind_ids,part4)),
                     'old_v3':old_v3(cache,names,blind_ids,events) if len(blind_ids) else None}
    # Final refit on every labelled date; policy from all out-of-fold folds.
    all_ids=np.arange(len(cache.samples));report['final']={}
    for name in ALGOS:
        path=MODELS/'final'/name;path.mkdir(parents=True,exist_ok=True);bundle,info=fit_bundle(name,cache,all_ids);policy=policy_from(cache,preds[name],len(FOLDS))
        if bundle is None or policy is None:report['final'][name]={**info,'status':'not_available'};continue
        rows,risk=pooled(cache,preds[name],len(FOLDS));oof=apply_calibration(policy['calibration'],risk)
        bundle.update(calibration=policy['calibration'],threshold=policy['threshold'],feature_names=names,task=TASK,algorithm=name);joblib.dump(bundle,path/'bundle.joblib')
        info.update(model_sha256=sha(path/'bundle.joblib'),threshold=policy['threshold'],calibration_kind=policy['calibration']['kind'],oof_policy=policy['oof_policy'],
                    pooled_oof=ranking(rows,oof),oof_folds=policy['oof_folds'],oof_events=policy['oof_events'],validation_metrics=None,
                    note='refit on all dates through 2026-11-01; out-of-fold figures describe the earlier fold models, not this refit',production_ready=False)
        report['final'][name]=info;dump(info,path/'fit.json');status('v4_final_refit',model=name,train_windows=info['train_windows'],train_events=info['train_events'])
    counts=cache.samples.groupby('partition').agg(windows=('event_in_300s','size'),positive_windows=('event_in_300s','sum')).to_dict('index')
    for k in counts:counts[k]['events']=int(cache.samples.loc[cache.samples.partition.eq(k)&cache.samples.event_in_300s,'event_id'].nunique())
    report.update(partition_counts=counts,episodes_total=len(events),finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),production_ready=False)
    dump(report,EVAL/'v4_report.json');status('v4_risk_complete',task=TASK)

def life_fit(name,z,rows,task):
    lo=rows[f'rul_lower_{task}'].to_numpy();hi=rows[f'rul_upper_{task}'].to_numpy();ev=rows.rul_event.to_numpy();w=landmark_weights(rows)
    if task=='pieces':hi=np.where(ev,hi+1,hi)
    with threadpool_limits(limits=2):
        if name=='constant':return fit_model('constant',z,lo,hi,ev,w,task)
        dist=name.split('_')[1];reg={'001':.01,'01':.1,'1':1.}[name.split('_r')[1]]
        return {'name':'weibull_aft' if dist=='weibull' else 'lognormal_aft','model':fit_stage(z,lo,hi,ev,w,dist,reg),'stage_transform':True}

def run_life():
    torch.set_num_threads(2);s=pd.read_parquet(DATA/'samples.parquet');allx=np.load(SHARED/'features.npy',mmap_mode='r');names=load(DATA/'features.json')['names']
    cols=[names.index(n) for n in RAW];x=stage_features(np.asarray(allx[:,cols]));valid=s.rul_valid.to_numpy();group=s.possible_cycle.to_numpy();part=s.partition.to_numpy()
    sampled=(sec(s.timestamp)%60==0)|(s.rul_event.to_numpy()&(s.rul_lower_hours.to_numpy()<=5/60));methods=['constant']+VARIANTS
    folds=[]
    for g in sorted(np.unique(group[valid]))[1:]:folds.append((f'cycle{int(g)}',valid&(group<g)&sampled,valid&(group==g),'cycle-forward'))
    folds.append(('blind_new_week',valid&(part<NEW_PARTITION)&sampled,valid&(part==NEW_PARTITION),'fit on partitions 0-3, scored on the unseen week'))
    folds.append(('final',valid&sampled,np.zeros(len(s),bool),'all dates; no validation'))
    report={'task':'life','features':RAW,'selected':load(ROOT/'protocol.json')['life']['selected'],'folds':{},'methods':methods}
    support=[]
    for g in sorted(np.unique(group[valid])):
        m=valid&(group==g);rows=s[m];support.append({'possible_cycle':int(g),'partitions':sorted(int(v) for v in rows.partition.unique()),'valid_windows':int(m.sum()),
            'weak_endpoints':sorted(rows.loc[rows.rul_event,'rul_endpoint_id'].unique().tolist()),'event_windows':int(rows.rul_event.sum()),'first':rows.timestamp.min(),'last':rows.timestamp.max()})
    report['cycle_support']=support
    for label,trm,vam,kind in folds:
        tr=np.flatnonzero(trm);va=np.flatnonzero(vam);rows=s.iloc[tr];entry={'kind':kind,'train_windows':len(tr),'train_weak_endpoints':int(rows.loc[rows.rul_event,'rul_endpoint_id'].nunique()),
            'evaluation_windows':len(va),'evaluation_weak_endpoints':int(s.iloc[va].loc[s.iloc[va].rul_event,'rul_endpoint_id'].nunique()) if len(va) else 0,'methods':{}}
        report['folds'][label]=entry
        if entry['train_weak_endpoints']==0:entry['status']='not_fit_no_training_weak_endpoint';continue
        trans=Transform().fit(x[tr]);z=trans.transform(x[tr]);vz=trans.transform(x[va]) if len(va) else None
        for name in methods:
            dest=MODELS/'life'/name;dest.mkdir(parents=True,exist_ok=True);result={}
            for task in ['hours','pieces']:
                bundle=life_fit(name,z,rows,task);bundle.update(transform=trans,feature_indices=cols,task=task,variant=name);joblib.dump(bundle,dest/f'{label}_{task}.joblib')
                result[task]={'optimizer_success':bool(bundle['model']['optimizer_success']),'model_sha256':sha(dest/f'{label}_{task}.joblib'),
                              'beta':bundle['model']['beta'],'parameter':bundle['model']['parameter']}
                if len(va):
                    pred,tail=predict_model(bundle,vz,task);assert np.isfinite(pred).all()
                    result[task]['metrics']=life_eval(pred,s.iloc[va],task,tail);np.savez(EVAL/f'life_{name}_{label}_{task}.npz',ids=va,predictions=pred)
            entry['methods'][name]=result
        status('v4_life_fold',fold=label,train_windows=len(tr),evaluation_windows=len(va))
    # Frozen v3 life models on the unseen week.
    va=np.flatnonzero(valid&(part==NEW_PARTITION));old={}
    sources={'full_refit_stage_lognormal_r1':OLD_V3/'models/full_data/life/{task}.joblib','isolated_stage_lognormal_r1':OLD_V3/'models/life/life/stage_lognormal_r1/final_{task}.joblib',
             'isolated_constant':OLD_V3/'models/life/life/constant/final_{task}.joblib'}
    if len(va):
        for label,template in sources.items():
            old[label]={}
            for task in ['hours','pieces']:
                path=template.parent/template.name.format(task=task);bundle=joblib.load(path);raw=np.asarray(allx[va][:,bundle['feature_indices']])
                if bundle.get('stage_transform'):raw=stage_features(raw)
                pred,tail=predict_model(bundle,bundle['transform'].transform(raw),task)
                old[label][task]={'metrics':life_eval(pred,s.iloc[va],task,tail),'model_sha256':sha(path)};np.savez(EVAL/f'old_v3_{label}_blind_{task}.npz',ids=va,predictions=pred)
    report['blind_old_v3']=old
    counts={int(k):{'valid_windows':int((valid&(part==k)).sum()),'weak_endpoints':sorted(s.loc[valid&(part==k)&s.rul_event,'rul_endpoint_id'].unique().tolist())} for k in sorted(np.unique(part))}
    report.update(partition_counts=counts,confirmed_natural_failures=0,finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),production_ready=False)
    dump(report,EVAL/'v4_report.json');status('v4_life_complete')

if __name__=='__main__':
    if (EVAL/'v4_report.json').exists():print('Preserving completed v4 report for',TASK)
    elif TASK=='life':run_life()
    else:run_risk()
