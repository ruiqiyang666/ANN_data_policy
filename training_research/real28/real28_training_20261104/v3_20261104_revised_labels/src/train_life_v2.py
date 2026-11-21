"""Train six lifetime families on cycle-isolated weak/censored data."""
import sys
sys.dont_write_bytecode=True
import torch
import argparse,time
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from common_v2 import DATA,MODELS,EVAL,SHARED,load,dump,status,sha,time_seconds
from risk_models import Transform
from life_models import landmark_weights,fit_model,predict_model

LIFE_NAMES=['constant','lognormal_aft','weibull_aft','spline_aft','mlp_aft','logistic_hazard','boosted_hazard','forest_hazard']
LIFE_FEATURES=['powered_Voltage_mean_300s','powered_Voltage_std_300s','powered_Voltage_slope_per_hour_900s','powered_Voltage_slope_per_hour_3600s',
 'powered_Current_mean_300s','powered_Argon_mean_300s','powered_Nitrogen_mean_300s','powered_Cooling_water_inlet_temp._mean_900s',
 'powered_Cooling_water_outlet_temp._mean_900s','product_voltage_mean','product_voltage_std','matched_product_voltage_delta',
 'matched_product_voltage_slope_per_hour','matched_product_count','observed_operation_hours','observed_operation_amp_hours','history_Voltage_mean_7200s','history_Voltage_slope_per_hour_7200s','history_observed_powered_hours','matched_long_peak_drop','matched_long_voltage_slope','matched_long_voltage_std','matched_long_count','matched_long_margin_51','phase_rising','phase_falling','phase_near_peak']

def evaluate(pred,s,task,tail=None):
    lo=s[f'rul_lower_{task}'].to_numpy();hi=s[f'rul_upper_{task}'].to_numpy();ev=s.rul_event.to_numpy();median=pred[:,1]
    records=[];offset=1/60 if task=='hours' else 1
    for endpoint in s.loc[ev,'rul_endpoint_id'].unique():
        mask=ev&s.rul_endpoint_id.eq(endpoint).to_numpy();near=mask&(s.rul_lower_hours.to_numpy()<=.25)
        distance=np.maximum(np.maximum(lo[mask]-median[mask],median[mask]-hi[mask]),0)
        log_distance=np.maximum(np.maximum(np.log(lo[mask]+offset)-np.log(median[mask]+offset),np.log(median[mask]+offset)-np.log(hi[mask]+offset)),0)
        nd=np.maximum(np.maximum(lo[near]-median[near],median[near]-hi[near]),0)
        records.append({'endpoint':endpoint,'windows':int(mask.sum()),'mean_interval_distance':float(distance.mean()),
                        'mean_log_interval_distance':float(log_distance.mean()),'near_15min_windows':int(near.sum()),
                        'near_15min_mean_interval_distance':float(nd.mean()) if near.any() else None,
                        'overestimate_fraction':float((median[mask]>hi[mask]).mean()),
                        'model_interval_overlap_fraction':float(((pred[mask,0]<=hi[mask])&(pred[mask,2]>=lo[mask])).mean())})
    return {'windows':len(s),'weak_endpoints':len(records),'per_endpoint':records,
            'macro_interval_distance':float(np.mean([r['mean_interval_distance'] for r in records])) if records else None,
            'macro_log_interval_distance':float(np.mean([r['mean_log_interval_distance'] for r in records])) if records else None,
            'median_prediction_quantiles':np.quantile(median,[0,.1,.5,.9,1]),
            'censor_median_below_observed_lower_fraction':float((median[~ev]<lo[~ev]).mean()) if (~ev).any() else None,
            'median_tail_extrapolation_fraction':float(tail[:,1].mean()) if tail is not None else 0,
            'confirmed_natural_failure_error':None,'calibrated_interval_coverage':None}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='+',default=LIFE_NAMES);args=parser.parse_args()
    if not (EVAL/'preflight.json').exists():raise RuntimeError('Preflight required')
    torch.set_num_threads(2)
    s=pd.read_parquet(DATA/'samples.parquet');allx=np.load(SHARED/'features.npy',mmap_mode='r');names=load(DATA/'features.json')['names'];cols=[names.index(n) for n in LIFE_FEATURES]
    x=np.asarray(allx[:,cols]);valid=s.rul_valid.to_numpy();group=s.possible_cycle.to_numpy();part=s.partition.to_numpy()
    # Thin ordinary landmarks to 60s while retaining every near-terminal landmark.
    sampled=(time_seconds(s.timestamp)%60==0)|(s.rul_event.to_numpy()&(s.rul_lower_hours.to_numpy()<=5/60))
    folds=[('cycle1',np.flatnonzero(valid&(part==0)&(group<1)&sampled),np.flatnonzero(valid&(part==0)&(group==1))),
           ('cycle2',np.flatnonzero(valid&(part==0)&(group<2)&sampled),np.flatnonzero(valid&(part==0)&(group==2))),
           ('final',np.flatnonzero(valid&(part==0)&sampled),np.flatnonzero(valid&(part==1)))]
    dump({'features':LIFE_FEATURES,'folds':[{'name':n,'train_windows':len(tr),'evaluation_windows':len(va)} for n,tr,va in folds],
          'weighting':'equal possible-cycle and event/censor x remaining-observation stratum; near terminal retained at10s, ordinary sampled60s',
          'selection':'forward-cycle mean log distance to weak interval, requiring superiority to same-weight constant control; not history scores',
          'physical_constraints':'positive survival and censor likelihood; no forced monotonic raw voltage','history_accessed':False},MODELS/'life_protocol.json')
    for name in args.models:
        destination=MODELS/'life'/name;destination.mkdir(parents=True,exist_ok=True)
        if (destination/'complete.json').exists():continue
        summaries=[]
        for label,tr,val in folds:
            trans=Transform().fit(x[tr]);z=trans.transform(x[tr]);vz=trans.transform(x[val]);rows=s.iloc[tr];weight=landmark_weights(rows)
            for task in ['hours','pieces']:
                path=destination/f'{label}_{task}.joblib';metricpath=destination/f'{label}_{task}.json'
                if path.exists() and metricpath.exists():summaries.append(load(metricpath));continue
                started=time.perf_counter();lo=rows[f'rul_lower_{task}'].to_numpy();hi=rows[f'rul_upper_{task}'].to_numpy();ev=rows.rul_event.to_numpy()
                if task=='pieces':hi=np.where(ev,hi+1,hi)
                with threadpool_limits(limits=2):bundle=fit_model(name,z,lo,hi,ev,weight,task)
                bundle.update(transform=trans,feature_indices=cols,task=task)
                pred,tail=predict_model(bundle,vz,task)
                if not np.isfinite(pred).all():raise RuntimeError(f'Invalid life predictions: {name}')
                metrics=evaluate(pred,s.iloc[val],task,tail)
                summary={'name':name,'fit':label,'task':task,'train_windows':len(tr),'seconds':time.perf_counter()-started,'metrics':metrics,
                         'optimizer_success':bundle['model']['optimizer_success'],'history_period_accessed':False}
                joblib.dump(bundle,path);np.savez(EVAL/f'life_{name}_{label}_{task}.npz',ids=val,predictions=pred)
                dump(summary,metricpath);summaries.append(summary);status('life_fit_complete',**summary)
        oof=[v for v in summaries if v['fit']!='final' and v['task']=='hours']
        out={'name':name,'status':'trained','real_algorithm_family':name,'constant_is_control_not_counted':name=='constant',
             'forward_cycle_macro_log_distance':float(np.mean([v['metrics']['macro_log_interval_distance'] for v in oof])),
             'forward_cycle_macro_hour_distance':float(np.mean([v['metrics']['macro_interval_distance'] for v in oof])),
             'fits':summaries,'all_optimizers_converged':all(v['optimizer_success'] for v in summaries),'history_period_accessed':False}
        dump(out,destination/'complete.json');status('life_method_complete',name=name,score=out['forward_cycle_macro_log_distance'])
    status('life_batch_complete',models=args.models)

if __name__=='__main__':main()
