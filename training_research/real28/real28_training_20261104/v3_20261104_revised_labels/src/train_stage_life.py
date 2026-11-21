"""Four-feature sign-constrained AFT sensitivity study; two existing families.

Declared before historical predictions. This adds six implementations, not six
independent algorithm families or six new independent validation endpoints.
"""
import sys
sys.dont_write_bytecode=True
import torch
import time
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from common_v2 import ROOT,SHARED,DATA,MODELS,EVAL,TASK,load,dump,time_seconds,sha,status
from risk_models import Transform
from life_models import nll,landmark_weights,predict_model
from train_life_v2 import evaluate

RAW=['product_voltage_mean','matched_long_peak_drop','powered_Voltage_slope_per_hour_300s','powered_Voltage_std_300s']
VARIANTS=[f'stage_{dist}_r{reg}' for dist in ['lognormal','weibull'] for reg in ['001','01','1']]
def stage_features(x):
    return np.column_stack([np.log1p(np.maximum(x[:,0]-51,0)),np.log1p(np.maximum(x[:,1],0)),np.log1p(np.maximum(-x[:,2],0)),np.log1p(np.maximum(x[:,3],0))]).astype('float32')

def fit_stage(x,lo,hi,event,w,dist,reg):
    design=np.column_stack([np.ones(len(x)),x]);initial=np.zeros(6);initial[0]=np.log(max(np.median((lo[event]+hi[event])/2),.001))
    def objective(theta):return float(w@nll(design@theta[:-1],np.exp(theta[-1]),lo,hi,event,dist)+reg*np.sum(theta[1:-1]**2))
    result=minimize(objective,initial,method='L-BFGS-B',bounds=[(-12,12),(0,5),(-5,0),(-5,0),(-5,0),(-2.5,2)],options={'maxiter':600,'ftol':1e-9})
    return {'beta':result.x[:-1],'parameter':float(np.exp(result.x[-1])),'distribution':dist,'optimizer_success':bool(result.success),
            'message':str(result.message),'objective':float(result.fun),'iterations':int(result.nit),'penalty':reg,'sign_constraints':['margin+','drop-','decline_speed-','volatility-']}

def main():
    if TASK!='life':raise RuntimeError('Requires TRAIN_TASK=life')
    torch.set_num_threads(2);s=pd.read_parquet(DATA/'samples.parquet');allx=np.load(SHARED/'features.npy',mmap_mode='r');names=load(DATA/'features.json')['names'];cols=[names.index(n) for n in RAW]
    x=stage_features(allx[:,cols]);valid=s.rul_valid.to_numpy();group=s.possible_cycle.to_numpy();part=s.partition.to_numpy()
    sampled=(time_seconds(s.timestamp)%60==0)|(s.rul_event.to_numpy()&(s.rul_lower_hours.to_numpy()<=5/60))
    folds=[('cycle1',np.flatnonzero(valid&(part==0)&(group<1)&sampled),np.flatnonzero(valid&(part==0)&(group==1))),
           ('cycle2',np.flatnonzero(valid&(part==0)&(group<2)&sampled),np.flatnonzero(valid&(part==0)&(group==2))),
           ('final',np.flatnonzero(valid&(part==0)&sampled),np.flatnonzero(valid&(part==1)))]
    for variant in VARIANTS:
        dest=MODELS/'life'/variant;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'complete.json').exists():continue
        dist=variant.split('_')[1];reg={'001':.01,'01':.1,'1':1.}[variant.split('_r')[1]];summaries=[]
        for label,tr,va in folds:
            trans=Transform().fit(x[tr]);z=trans.transform(x[tr]);vz=trans.transform(x[va]);rows=s.iloc[tr];w=landmark_weights(rows)
            for task in ['hours','pieces']:
                started=time.perf_counter();lo=rows[f'rul_lower_{task}'].to_numpy();hi=rows[f'rul_upper_{task}'].to_numpy();ev=rows.rul_event.to_numpy()
                if task=='pieces':hi=np.where(ev,hi+1,hi)
                with threadpool_limits(limits=2):model=fit_stage(z,lo,hi,ev,w,dist,reg)
                bundle={'name':'weibull_aft' if dist=='weibull' else 'lognormal_aft','model':model,'transform':trans,'feature_indices':cols,'stage_transform':True,'task':task}
                pred,tail=predict_model(bundle,vz,task);assert np.isfinite(pred).all()
                metrics=evaluate(pred,s.iloc[va],task,tail)
                summary={'name':variant,'fit':label,'task':task,'train_windows':len(tr),'seconds':time.perf_counter()-started,'metrics':metrics,'optimizer_success':model['optimizer_success'],'history_period_accessed':False}
                joblib.dump(bundle,dest/f'{label}_{task}.joblib');dump(summary,dest/f'{label}_{task}.json');np.savez(EVAL/f'life_{variant}_{label}_{task}.npz',ids=va,predictions=pred)
                summaries.append(summary);status('stage_life_fit_complete',name=variant,fit=label,task=task,seconds=summary['seconds'])
        cv=[v for v in summaries if v['fit']!='final' and v['task']=='hours']
        dump({'name':variant,'status':'trained','real_algorithm_family':dist+'_aft','initialization_variant_not_extra_family':True,'fits':summaries,
              'forward_cycle_macro_log_distance':float(np.mean([v['metrics']['macro_log_interval_distance'] for v in cv])),
              'forward_cycle_macro_hour_distance':float(np.mean([v['metrics']['macro_interval_distance'] for v in cv])),
              'all_optimizers_converged':all(v['optimizer_success'] for v in summaries),'history_period_accessed':False,'features':RAW,'penalty':reg},dest/'complete.json')
    status('stage_life_batch_complete',variants=VARIANTS)
if __name__=='__main__':main()
