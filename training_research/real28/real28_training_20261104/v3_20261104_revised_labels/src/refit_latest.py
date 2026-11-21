"""Refit frozen choices on all available dates; no historical scoring of these fits.

These are separate research candidates. Evaluation numbers belong to the earlier
date-isolated models and must never be attributed to these all-data refits.
"""
import sys
sys.dont_write_bytecode=True
import torch
import os,time,subprocess
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from common_v2 import ROOT,SHARED,TASK,DATA,load,dump,sha,time_seconds
from risk_models import Cache,Transform,ExtraTransform,cpu_model,row_weights,fit_auxiliary
from train_risk import train_neural,NEURAL
from metrics_v2 import eligible
from life_models import landmark_weights,fit_model
from train_stage_life import stage_features,fit_stage

def worker():
    torch.set_num_threads(4);frozen=load(ROOT/'development_freeze.json');dest=ROOT/'models/full_data'/TASK;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'fit.json').exists():return
    started=time.perf_counter()
    if TASK=='life':
        name=frozen['life_choice']
        if name is None:
            dump({'task':TASK,'status':'skipped','main_fits':0,'reason':'no method passed development gate'},dest/'fit.json');return
        template=joblib.load(ROOT/'models/life/life'/name/'final_hours.joblib');s=pd.read_parquet(DATA/'samples.parquet')
        mask=s.rul_valid&((time_seconds(s.timestamp)%60==0)|(s.rul_event&(s.rul_lower_hours<=5/60)))
        rows=s.loc[mask];features=np.load(SHARED/'features.npy',mmap_mode='r');x=np.asarray(features[np.flatnonzero(mask)][:,template['feature_indices']])
        if template.get('stage_transform'):x=stage_features(x)
        trans=Transform().fit(x);z=trans.transform(x);w=landmark_weights(rows)
        for task in ['hours','pieces']:
            lo=rows[f'rul_lower_{task}'].to_numpy();hi=rows[f'rul_upper_{task}'].to_numpy();ev=rows.rul_event.to_numpy()
            if task=='pieces':hi=np.where(ev,hi+1,hi)
            with threadpool_limits(limits=2):
                if template.get('stage_transform'):
                    model=fit_stage(z,lo,hi,ev,w,template['model']['distribution'],template['model']['penalty'])
                    bundle={'name':template['name'],'model':model,'stage_transform':True}
                else:bundle=fit_model(name,z,lo,hi,ev,w,task)
            bundle.update(transform=trans,feature_indices=template['feature_indices'],task=task)
            joblib.dump(bundle,dest/f'{task}.joblib')
        count=2;events=int(rows.loc[rows.rul_event,'rul_endpoint_id'].nunique());normstatus='weak interval/censor likelihood; zero individually confirmed natural failures'
    else:
        name=frozen['risk_choices'][TASK];cache=Cache();ids=np.flatnonzero(eligible(cache.samples));rows=cache.samples.iloc[ids]
        trans=Transform().fit(cache.features[ids]);x=trans.transform(cache.features[ids]);joblib.dump(trans,dest/'preprocess.joblib')
        aux=fit_auxiliary(cache,ids,x);joblib.dump(aux,dest/'auxiliary.joblib')
        if name=='tcn_ssl':
            import train_ssl as ssl
            ssl.train.RiskNet=ssl.factory
            model,norm,h=ssl.pretrain_then_fit(name,cache,ids,trans,dest)
            del model;torch.cuda.empty_cache()
        elif name in NEURAL:
            model,norm,h=train_neural(name,cache,ids,trans,dest);del model;torch.cuda.empty_cache()
        else:
            extra=ExtraTransform(name).fit(x);model=cpu_model(name)
            with threadpool_limits(limits=4):model.fit(extra.transform(x),rows.event_in_300s,row_weights(rows))
            joblib.dump({'model':model,'extra':extra},dest/'model.joblib')
        count=1;events=int(rows.loc[rows.event_in_300s,'event_id'].nunique());normstatus='raw weighted score; calibration on future data not yet verified for this refit'
    files={p.name:sha(p) for p in dest.iterdir() if p.suffix in ['.pt','.joblib']}
    dump({'task':TASK,'status':'trained_research_candidate','algorithm':name,'main_fits':count,'train_windows':len(rows),'event_or_weak_endpoint_count':events,
          'first_timestamp':rows.timestamp.min(),'last_timestamp':rows.timestamp.max(),'seconds':time.perf_counter()-started,
          'normalization_fit':'all included dates, AFTER method freeze and isolated evaluation','validation_metrics':None,
          'previous_validation_scores_do_not_apply_to_this_refit':True,'output_status':normstatus,'production_ready':False,'artifact_sha256':files},dest/'fit.json')
    print('All-data refit completed',TASK,name,'events',events,flush=True)

def main():
    if '--worker' in sys.argv:worker();return
    if not (ROOT/'evaluation/verification_final.json').exists():raise RuntimeError('Finish isolated comparisons and verification first')
    for task in ['voltage','feedrate','hopper_pressure','life']:
        env=os.environ.copy();env.update(TRAIN_TASK=task,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        with (ROOT/'logs'/f'all_data_{task}.log').open('a',encoding='utf-8') as stream:
            p=subprocess.run([sys.executable,'-B',__file__,'--worker'],env=env,stdout=stream,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError('All-data refit failed: '+task)
    fits={task:load(ROOT/'models/full_data'/task/'fit.json') for task in ['voltage','feedrate','hopper_pressure','life']}
    dump({'fits':fits,'main_fits':sum(v['main_fits'] for v in fits.values()),'refit_after_frozen_evaluation':True,'production_ready':False,'finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')},ROOT/'all_data_refits.json')
    print('All all-data research candidates saved',flush=True)
if __name__=='__main__':main()
