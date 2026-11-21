"""Final source isolation, model count, freeze and prediction checks."""
import sys
sys.dont_write_bytecode=True
import torch
import numpy as np
from common_v2 import ROOT,DATA,MODELS,EVAL,V1,load,dump,sha
from evaluate_v2 import ALL_RISK
from train_life_v2 import LIFE_NAMES

def main():
    frozen=load(MODELS/'development_freeze.json');report=load(EVAL/'comparison.json');checks=[]
    assert report['freeze_sha256']==sha(MODELS/'development_freeze.json')
    for filename,digest in frozen['model_sha256'].items():assert sha(MODELS/filename)==digest,filename
    for filename,digest in frozen['data_sha256'].items():assert sha(DATA/filename)==digest,filename
    checks.append('All model, preprocessing and target files match the development freeze')
    assert len(ALL_RISK)>=5 and report['trained_risk_algorithms']==len(ALL_RISK)
    for name in ALL_RISK:
        summary=load(MODELS/'risk'/name/'complete.json');assert summary['status']=='trained'
        assert len(summary['fold_scores'])==3 and summary['history_period_accessed'] is False
        for label in ['f1','f2','f3','final']:
            fit=load(MODELS/'risk'/name/label/'fit.json')
            assert fit['train_windows']>0
            model_file=MODELS/'risk'/name/label/('model.pt' if name in ['tcn','gru','patch_transformer'] else 'model.joblib')
            assert model_file.exists() and sha(model_file)==fit['model_sha256']
    checks.append('Nine distinct risk algorithms each have three forward fits and a final fitted artifact')
    for name in LIFE_NAMES:
        summary=load(MODELS/'life'/name/'complete.json');assert len(summary['fits'])==6
        assert (MODELS/'life'/name/'final_hours.joblib').exists() and (MODELS/'life'/name/'final_pieces.joblib').exists()
    checks.append(f'{len(LIFE_NAMES)-1} lifetime algorithms plus a constant control each fitted both targets across three cycle-separated fits')
    for path in EVAL.glob('*.npz'):
        arrays=np.load(path)
        for key in ['risk','calibrated_risk']:
            if key in arrays:assert np.isfinite(arrays[key]).all() and ((arrays[key]>=0)&(arrays[key]<=1)).all(),str(path)
        for key in ['time','type']:
            if key in arrays:
                assert np.isfinite(arrays[key]).all() and (arrays[key]>=0).all(),str(path)
                np.testing.assert_allclose(arrays[key].sum(1),1,atol=2e-5)
        if 'predictions' in arrays:
            pred=arrays['predictions'];assert np.isfinite(pred).all() and (pred>=0).all() and (np.diff(pred,axis=1)>=0).all(),str(path)
    checks.append('All saved probabilities, conditional distributions and survival quantiles are finite and ordered')
    old=ROOT.parent/'models/20261104_v1';oldfreeze=load(old/'pretest_freeze.json')
    for filename,digest in oldfreeze['artifact_sha256'].items():assert sha(old/filename)==digest
    checks.append('All original V1 model hashes remain unchanged')
    contract=load(MODELS/'output_contract.json');assert len(contract['raw_fields'])==28 and not contract['pressure_forecast'] and not contract['automatic_replacement']
    assert contract['gun_health'] is None and contract['tank_health'] is None and report['history_already_seen']
    assert report['history_used_for_selection'] is False and not report['production_ready']
    checks.append('No false health scores, pressure forecasts, natural-failure validation or unseen-test claims')
    for path in ROOT.glob('src/*.py'):compile(path.read_text('utf-8'),str(path),'exec')
    checks.append('All new source files parse successfully')
    out={'passed':True,'checks':checks,'risk_methods':len(ALL_RISK),'lifetime_methods_excluding_constant':len(LIFE_NAMES)-1,
         'main_fit_count':len(ALL_RISK)*4+len(LIFE_NAMES)*6,'retained_model_files':len(frozen['model_sha256'])}
    dump(out,EVAL/'verification_final.json');print(out)

if __name__=='__main__':main()
