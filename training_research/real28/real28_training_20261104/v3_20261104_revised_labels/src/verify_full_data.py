"""Reload latest candidates and exercise real predictions; no accuracy claims."""
import sys
sys.dont_write_bytecode=True
import os
os.environ.setdefault('LOKY_MAX_CPU_COUNT','4')
import torch
import numpy as np
import pandas as pd
import joblib
from common_v2 import ROOT,SHARED,load,dump,sha
from risk_models import RiskNet,Cache,auxiliary_predict
from train_risk import neural_predict
from life_models import predict_model
from train_stage_life import stage_features

def main():
    torch.set_num_threads(4);latest=load(ROOT/'all_data_refits.json');features=np.load(SHARED/'features.npy',mmap_mode='r');result={}
    for task,meta in latest['fits'].items():
        if meta['status']=='skipped':continue
        dest=ROOT/'models/full_data'/task;s=pd.read_parquet(SHARED/task/'samples.parquet');ids=np.linspace(0,len(s)-1,19,dtype=int)
        if task=='life':
            for target in ['hours','pieces']:
                b=joblib.load(dest/f'{target}.joblib');raw=features[ids][:,b['feature_indices']]
                if b.get('stage_transform'):raw=stage_features(raw)
                z=b['transform'].transform(raw);p,_=predict_model(b,z,target)
                assert np.isfinite(p).all() and (p>=0).all() and (np.diff(p,axis=1)>=0).all()
            result[task]={'reload_and_predict_passed':True,'rows':len(ids),'targets':['hours','pieces']}
        else:
            trans=joblib.load(dest/'preprocess.joblib');z=trans.transform(features[ids]);aux=auxiliary_predict(joblib.load(dest/'auxiliary.joblib'),z)
            if (dest/'model.joblib').exists():
                b=joblib.load(dest/'model.joblib');p=b['model'].predict_proba(b['extra'].transform(z))[:,1]
            else:
                state=torch.load(dest/'model.pt',map_location='cuda',weights_only=True);model=RiskNet('tcn' if meta['algorithm']=='tcn_ssl' else meta['algorithm'],state['nstats']).cuda();model.load_state_dict(state['state_dict'])
                p=neural_predict(model,Cache(),ids,trans,joblib.load(dest/'sequence_normalization.joblib'))['risk'];del model;torch.cuda.empty_cache()
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            np.testing.assert_allclose(aux['time'].sum(1),1,atol=2e-6)
            result[task]={'reload_and_predict_passed':True,'rows':len(ids),'raw_scores_not_validated_calibrated_probabilities':True}
    dump({'passed':True,'checks':result,'not_validation_metrics':True,'refit_manifest_sha256':sha(ROOT/'all_data_refits.json')},ROOT/'evaluation/full_data_prediction_smoke.json')
    print(result,flush=True)
if __name__=='__main__':main()
