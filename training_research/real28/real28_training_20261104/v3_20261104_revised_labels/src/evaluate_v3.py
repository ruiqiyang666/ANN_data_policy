"""Freeze training-only choices before any revised historical predictions."""
import sys
sys.dont_write_bytecode=True
import torch
import os,time
import numpy as np
import pandas as pd
import joblib
from common_v2 import ROOT,SHARED,DATA,MODELS,EVAL,TASK,TYPES,FIELDS,load,dump,sha
from risk_models import Cache,RiskNet,auxiliary_predict
from train_risk import neural_predict,apply_calibration,NEURAL
from metrics_v2 import ranking,alarms
from train_life_v2 import LIFE_NAMES,evaluate as life_eval
from life_models import predict_model
from train_stage_life import VARIANTS,stage_features
from probability_guard import normalize_time
LIFE_NAMES=LIFE_NAMES+VARIANTS

RISK={'voltage':['logistic','histgb','extra_trees','random_forest','tcn','gru','patch_transformer','tcn_ssl'],
      'feedrate':['logistic','histgb','extra_trees'],'hopper_pressure':['logistic','histgb','extra_trees']}

def freeze():
    file=ROOT/'development_freeze.json'
    if file.exists():
        f=load(file)
        for path,digest in f['artifact_sha256'].items():assert sha(ROOT/path)==digest,path
        return f
    choices={};risk={}
    for task,names in RISK.items():
        risk[task]=[load(ROOT/'models'/task/'risk'/n/'complete.json') for n in names]
        choices[task]=max(risk[task],key=lambda x:(x['mean_fold_event_macro_ap'],x['pooled_oof']['average_precision']))['name']
    lives=[load(ROOT/'models/life/life'/n/'complete.json') for n in LIFE_NAMES]
    constant=next(v for v in lives if v['name']=='constant')
    def near(v):
        a=[e['near_15min_mean_interval_distance'] for fit in v['fits'] if fit['fit']!='final' and fit['task']=='hours' for e in fit['metrics']['per_endpoint'] if e['near_15min_mean_interval_distance'] is not None]
        return float(np.mean(a))
    reviews=[]
    for v in lives:
        if v['name']=='constant':continue
        cv=[x for x in v['fits'] if x['fit']!='final' and x['task']=='hours']
        extrapolation=any(x['metrics']['median_prediction_quantiles'][3]>400 for x in cv)
        passes=v['all_optimizers_converged'] and not extrapolation and v['forward_cycle_macro_log_distance']<constant['forward_cycle_macro_log_distance'] and near(v)<=near(constant)
        reviews.append({'name':v['name'],'passes':bool(passes),'near_15min_distance_hours':near(v),'constant_near_distance_hours':near(constant),
                        'p90_median_above_400h':extrapolation,'support_warning_is_not_physical_maximum':True})
    passed=[v for v in lives if any(r['name']==v['name'] and r['passes'] for r in reviews)]
    life_choice=min(passed,key=lambda x:x['forward_cycle_macro_log_distance'])['name'] if passed else None
    artifacts={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'models').rglob('*') if p.is_file() and p.suffix in ['.joblib','.pt']}
    frozen={'frozen_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'risk_choices':choices,'life_choice':life_choice,
            'risk_summaries':risk,'life_summaries':lives,'life_gate':reviews,'artifact_sha256':artifacts,
            'protocol_sha256':sha(ROOT/'protocol.json'),'features_sha256':sha(SHARED/'features.npy'),
            'data_sha256':{str(p.relative_to(ROOT)):sha(p) for p in SHARED.rglob('*') if p.is_file() and p.suffix in ['.parquet','.npz','.csv','.json']},
            'training_source_sha256':{name:sha(ROOT/'src'/name) for name in ['common_v2.py','prepare_v3.py','risk_models.py','train_risk.py','metrics_v2.py','train_life_v2.py','life_models.py','train_ssl.py','train_stage_life.py','refit_latest.py']},
            'history_already_seen':True,'history_used_for_method_selection':False,'production_ready':False}
    dump(frozen,file);return frozen

def risk_prediction(name,cache,ids):
    path=MODELS/'risk'/name/'final';trans=joblib.load(path/'preprocess.joblib');z=trans.transform(cache.features[ids]);aux=auxiliary_predict(joblib.load(path/'auxiliary.joblib'),z)
    if name in NEURAL|{'tcn_ssl'}:
        state=torch.load(path/'model.pt',map_location='cuda',weights_only=True);model=RiskNet('tcn' if name=='tcn_ssl' else name,state['nstats']).cuda();model.load_state_dict(state['state_dict'])
        pred=neural_predict(model,cache,ids,trans,joblib.load(path/'sequence_normalization.joblib'));del model;torch.cuda.empty_cache()
    else:
        bundle=joblib.load(path/'model.joblib');pred={'risk':bundle['model'].predict_proba(bundle['extra'].transform(z))[:,1],'time':aux['time']}
    pred['time']=normalize_time(pred['time'])
    pred['eta']=(pred['time']*np.arange(15,300,30)).sum(1);pred['calibrated_risk']=apply_calibration(joblib.load(MODELS/'risk'/name/'calibration.joblib'),pred['risk'])
    return pred

def main():
    torch.set_num_threads(4);frozen=freeze()
    if (EVAL/'comparison.json').exists():print('Preserving completed comparison',TASK);return
    out={'task':TASK,'history_used_for_selection':False,'historical_already_seen':True,'freeze_sha256':sha(ROOT/'development_freeze.json')}
    if TASK=='life':
        samples=pd.read_parquet(DATA/'samples.parquet');features=np.load(SHARED/'features.npy',mmap_mode='r');out['algorithms']={}
        for name in LIFE_NAMES:
            out['algorithms'][name]={}
            for task in ['hours','pieces']:
                bundle=joblib.load(MODELS/'life'/name/f'final_{task}.joblib')
                for part,label in [(1,'development'),(3,'historical')]:
                    ids=np.flatnonzero(samples.partition.eq(part)&samples.rul_valid);raw=features[ids][:,bundle['feature_indices']]
                    if bundle.get('stage_transform'):raw=stage_features(raw)
                    z=bundle['transform'].transform(raw);pred,tail=predict_model(bundle,z,task)
                    metric=life_eval(pred,samples.iloc[ids],task,tail);out['algorithms'][name][f'{label}_{task}']=metric
                    np.savez(EVAL/f'{name}_{label}_{task}.npz',ids=ids,predictions=pred)
            print('Compared life',name,flush=True)
    else:
        cache=Cache();events=pd.read_parquet(DATA/'events.parquet');out['algorithms']={};out['selected_algorithm']=frozen['risk_choices'][TASK]
        for name in RISK[TASK]:
            summary=load(MODELS/'risk'/name/'complete.json');th=summary['development_alarm_policy']['threshold'];out['algorithms'][name]={}
            for part,label in [(1,'development'),(2,'calibration_diagnostic'),(3,'historical')]:
                ids=np.flatnonzero(cache.samples.partition.eq(part));s=cache.samples.iloc[ids];pred=risk_prediction(name,cache,ids)
                alarm,records,alert_ids=alarms(s,pred['calibrated_risk'],th,events)
                for record in records:
                    e=events[events.event_id.eq(record['event_id'])].iloc[0];record['result_available_at']=e.first_result_at
                    record['lead_to_product_start_s']=record['lead_to_result_s']
                    record['lead_to_result_s']=(e.first_result_at-record['alert_at']).total_seconds() if record['alert_at'] is not None else None
                y=s.event_in_300s.to_numpy();eta=pred['eta'][y]-s.eta_seconds.to_numpy()[y]
                out['algorithms'][name][label]={'ranking':ranking(s,pred['calibrated_risk']),'alarms':alarm,
                     'eta_mae_s_all_positive_windows':float(np.abs(eta).mean()) if len(eta) else None,
                     'eta_is_conditional_on_future_event':True,'event_records':records}
                np.savez(EVAL/f'{name}_{label}.npz',ids=ids,**pred)
                pd.DataFrame(records).to_csv(EVAL/f'{name}_{label}_events.csv',index=False,encoding='utf-8-sig')
                s.iloc[alert_ids].assign(probability=pred['calibrated_risk'][alert_ids]).to_csv(EVAL/f'{name}_{label}_alerts.csv',index=False,encoding='utf-8-sig')
            print('Compared',TASK,name,flush=True)
    dump(out,EVAL/'comparison.json')
    dump({'task':TASK,'status':'evaluated','production_ready':False,'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')},ROOT/f'execution_status_{TASK}.json')

if __name__=='__main__':main()
