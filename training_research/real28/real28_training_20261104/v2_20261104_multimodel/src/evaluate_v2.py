"""Freeze development choices, then perform an explicitly already-seen history comparison."""
import sys
sys.dont_write_bytecode=True
import torch
import numpy as np
import pandas as pd
import joblib,time
from common_v2 import ROOT,DATA,V1,MODELS,EVAL,RISK_NAMES,TYPES,FIELDS,load,dump,status,sha
from risk_models import Cache,RiskNet,auxiliary_predict
from train_risk import neural_predict,apply_calibration,NEURAL
from metrics_v2 import ranking,alarms,threshold_from_oof
from train_life_v2 import LIFE_NAMES,evaluate as life_evaluate
from life_models import predict_model

ALL_RISK=RISK_NAMES+['spline_logistic','rbf_logistic']

def freeze(cache):
    path=MODELS/'development_freeze.json'
    if path.exists():
        frozen=load(path)
        for name,digest in frozen['model_sha256'].items():assert sha(MODELS/name)==digest,name
        return frozen
    numerical_repairs=[]
    for name in NEURAL:
        for archive in EVAL.glob(f'{name}_*.npz'):
            arrays=dict(np.load(archive))
            if 'time' not in arrays:continue
            total=arrays['time'].sum(1,keepdims=True)
            if np.max(np.abs(total-1))>2e-6:
                arrays['time']=np.divide(arrays['time'],total,out=np.full_like(arrays['time'],.1),where=total>0)
                if 'eta' in arrays:arrays['eta']=(arrays['time']*np.arange(15,300,30)).sum(1)
                np.savez(archive,**arrays);numerical_repairs.append(archive.name)
    dump({'conditional_distribution_archives_normalized':numerical_repairs,'weights_changed':False,'reason':'same float32 boundary fix verified by probability_regression.json'},EVAL/'numerical_repair_audit.json')
    risk=[]
    for name in ALL_RISK:
        d=MODELS/'risk'/name;summary=load(d/'complete.json');p=np.load(EVAL/f'{name}_oof.npz');rows=cache.samples.iloc[p['ids']]
        # Refine numerical resolution uniformly for every algorithm. Budgets,
        # labels and policy are unchanged; no historical predictions are read.
        policy,curve=threshold_from_oof(rows,p['calibrated_risk'])
        dump({'initial_policy':summary['development_alarm_policy'],'refined_policy':policy,
              'reason':'The original quantile grid jumped from .9917 to1 and skipped feasible high-score operating points. Dense upper quantiles verified using OOF only.'},d/'alarm_resolution_audit.json')
        dump(curve,d/'oof_threshold_search_final.json')
        summary['development_alarm_policy']=policy;dump(summary,d/'complete.json');risk.append(summary)
    chosen=max(risk,key=lambda r:(r['mean_fold_event_macro_ap'],r['pooled_oof']['average_precision']))['name']
    lives=[load(MODELS/'life'/n/'complete.json') for n in LIFE_NAMES]
    constant=next(r for r in lives if r['name']=='constant');eligible=[];reviews=[]
    for r in lives:
        if r['name']=='constant':continue
        cv=[f for f in r['fits'] if f['fit']!='final' and f['task']=='hours']
        # Development-only extrapolation review, recorded before history access.
        unstable=any(f['metrics']['median_prediction_quantiles'][3]>400 for f in cv)
        near=[e['near_15min_mean_interval_distance'] for f in cv for e in f['metrics']['per_endpoint'] if e['near_15min_mean_interval_distance'] is not None]
        control_near=[e['near_15min_mean_interval_distance'] for f in constant['fits'] if f['fit']!='final' and f['task']=='hours' for e in f['metrics']['per_endpoint'] if e['near_15min_mean_interval_distance'] is not None]
        passes=(r['all_optimizers_converged'] and not unstable and r['forward_cycle_macro_log_distance']<constant['forward_cycle_macro_log_distance'] and np.mean(near)<=np.mean(control_near))
        reviews.append({'name':r['name'],'passes_constant_and_extrapolation_review':bool(passes),'p90_median_exceeds_400h':unstable,
                        'mean_near_15min_distance':float(np.mean(near)),'constant_mean_near_15min_distance':float(np.mean(control_near)),
                        'reason':'development extrapolation review; 400 h is a support-warning threshold, not a proven physical maximum'})
        if passes:eligible.append(r)
    life_choice=min(eligible,key=lambda r:r['forward_cycle_macro_log_distance'])['name'] if eligible else None
    model_files=[p for p in MODELS.rglob('*') if p.is_file() and p.suffix in ['.pt','.joblib']]
    frozen={'frozen_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'risk_choice':chosen,'life_choice':life_choice,
            'risk_methods':ALL_RISK,'life_methods':LIFE_NAMES,'risk_summaries':risk,'life_summaries':lives,'life_review':reviews,
            'history_already_seen_in_v1':True,'history_used_to_select_v2':False,
            'selection_rule':'mean forward-fold event-macro AP; lifetime must beat same-weight constant in log-distance and near-failure distance',
            'model_sha256':{str(p.relative_to(MODELS)):sha(p) for p in model_files},
            'data_sha256':{n:sha(DATA/n) for n in ['samples.parquet','features.npy','targets.npz']},
            'production_ready':False,'automatic_replacement':False}
    dump(frozen,path);return frozen

def predict_risk(name,cache,ids):
    path=MODELS/'risk'/name/'final';transform=joblib.load(path/'preprocess.joblib');z=transform.transform(cache.features[ids])
    aux=auxiliary_predict(joblib.load(path/'auxiliary.joblib'),z)
    if name in NEURAL:
        state=torch.load(path/'model.pt',map_location='cuda',weights_only=True);model=RiskNet(name,state['nstats']).cuda();model.load_state_dict(state['state_dict'])
        pred=neural_predict(model,cache,ids,transform,joblib.load(path/'sequence_normalization.joblib'));del model;torch.cuda.empty_cache()
    else:
        bundle=joblib.load(path/'model.joblib');pred={'risk':bundle['model'].predict_proba(bundle['extra'].transform(z))[:,1],'time':aux['time']}
    pred['type']=aux['type'];pred['eta']=(pred['time']*np.arange(15,300,30)).sum(1)
    pred['calibrated_risk']=apply_calibration(joblib.load(MODELS/'risk'/name/'calibration.joblib'),pred['risk'])
    return pred

def extra_metrics(cache,ids,pred,events):
    s=cache.samples.iloc[ids];positive=s.event_in_300s.to_numpy();per_type={}
    for j,key in enumerate(TYPES):
        masks=positive&cache.targets['type_y'][ids,j].astype(bool)
        per_type[key]={'positive_windows':int(masks.sum()),'events':int(s.loc[masks,'event_id'].nunique()),
                       'conditional_type_accuracy':float((pred['type'][masks].argmax(1)==j).mean()) if masks.any() else None}
    eta=pred['eta'][positive]-s.eta_seconds.to_numpy()[positive]
    return {'by_type':per_type,'conditional_eta_mae_all_event_windows_s':float(np.abs(eta).mean()) if len(eta) else None,
            'eta_includes_missed_events':True,'current_product_state_is_observation_not_trained_instantaneous_detector':True}

def main():
    if (EVAL/'comparison.json').exists():print('Comparison already saved; preserving original result.');return
    torch.set_num_threads(4);cache=Cache();frozen=freeze(cache)
    events=pd.read_parquet(V1/'events.parquet');report={'risk_choice':frozen['risk_choice'],'life_choice':frozen['life_choice'],
        'risk':{},'life':{},'history_already_seen':True,'history_used_for_selection':False,'freeze_sha256':sha(MODELS/'development_freeze.json'),
        'trained_risk_algorithms':len(ALL_RISK),'trained_lifetime_algorithms_excluding_constant':len(LIFE_NAMES)-1,'production_ready':False}
    old=pd.read_parquet(V1/'samples.parquet');old_times=set(old.loc[old.partition.eq(3),'timestamp'])
    allpred={}
    for name in ALL_RISK:
        report['risk'][name]={}
        threshold=next(r for r in frozen['risk_summaries'] if r['name']==name)['development_alarm_policy']['threshold']
        for part,label in [(1,'selection'),(2,'calibration_diagnostic'),(3,'historical_comparison')]:
            ids=np.flatnonzero(cache.samples.partition.eq(part));pred=predict_risk(name,cache,ids);s=cache.samples.iloc[ids]
            metrics=ranking(s,pred['calibrated_risk']);alarm,records,a=alarms(s,pred['calibrated_risk'],threshold,events)
            result={'ranking':metrics,'alarms':alarm,'auxiliary':extra_metrics(cache,ids,pred,events)}
            if part==3:
                common=s.timestamp.isin(old_times).to_numpy();ca,cr,_=alarms(s.loc[common],pred['calibrated_risk'][common],threshold,events)
                result['same_v1_landmarks']={'ranking':ranking(s.loc[common],pred['calibrated_risk'][common]),'alarms':ca}
                pd.DataFrame(cr).to_csv(EVAL/f'{name}_historical_common_events.csv',index=False,encoding='utf-8-sig')
                allpred[name]=(ids,pred)
            report['risk'][name][label]=result
            np.savez(EVAL/f'{name}_{label}.npz',ids=ids,**pred)
            pd.DataFrame(records).to_csv(EVAL/f'{name}_{label}_events.csv',index=False,encoding='utf-8-sig')
            s.iloc[a].assign(predicted_risk=pred['calibrated_risk'][a]).to_csv(EVAL/f'{name}_{label}_alerts.csv',index=False,encoding='utf-8-sig')
        status('risk_history_compared',name=name,metrics=report['risk'][name]['historical_comparison']['same_v1_landmarks'])
    ids=np.flatnonzero(cache.samples.partition.eq(3)&cache.samples.rul_valid)
    for name in LIFE_NAMES:
        report['life'][name]={}
        for task in ['hours','pieces']:
            b=joblib.load(MODELS/'life'/name/f'final_{task}.joblib');x=b['transform'].transform(cache.features[ids][:,b['feature_indices']]);pred,tail=predict_model(b,x,task)
            report['life'][name][task]=life_evaluate(pred,cache.samples.iloc[ids],task,tail)
            np.savez(EVAL/f'life_{name}_historical_{task}.npz',ids=ids,predictions=pred)
        status('life_history_compared',name=name,hour_distance=report['life'][name]['hours']['macro_interval_distance'])
    dump(report,EVAL/'comparison.json')
    contract={'raw_fields':FIELDS,'raw_field_count':28,'sequence_channels':58,'feature_count':cache.features.shape[1],
              'risk_method':frozen['risk_choice'],'lifetime_method':frozen['life_choice'],
              'future_seconds':300,'history_seconds':120,'pressure_forecast':False,'gun_health':None,'tank_health':None,
              'risk_probability_status':'development-calibrated; sparse-event historical comparison only',
              'life_status':'weak-proxy experiment; not validated natural electrode life',
              'lifetime_selection_status':'passed development constant baseline review' if frozen['life_choice'] else 'no learned model passed development constant baseline review',
              'model_intervals':'distribution quantiles, not calibrated coverage','automatic_replacement':False,
              'current_anomaly':'latest known completed product result with timestamp; not a new independent instantaneous detector',
              'artifact_status':'offline_experimental','original_dashboard_modified':False}
    dump(contract,MODELS/'output_contract.json')
    status('all_models_compared',risk_methods=len(ALL_RISK),life_methods=len(LIFE_NAMES)-1,production_ready=False)

if __name__=='__main__':main()
