"""Export the time-isolated neural models, keeping v4 AFT and recorded rows."""
import sys
sys.dont_write_bytecode=True
import os,json,hashlib,shutil
from pathlib import Path
import joblib,numpy as np
BASE=Path('/old/v4');ROOT=Path('/new');OUT=ROOT/'bundle'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(d,p):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
assert json.loads((ROOT/'status.json').read_text())['stage']=='complete'
assert not OUT.exists(),'Never overwrite an export'
shutil.copytree(BASE/'bundle',OUT)
comparison=json.loads((ROOT/'comparison.json').read_text())
checks={}
for task in ['voltage','feedrate','hopper_pressure']:
    b=joblib.load(ROOT/'models'/task/'f6/export.joblib')
    b['trained_through']='2026-10-23'
    joblib.dump(b,OUT/'models'/f'{task}.joblib',compress=3)
    z=np.load(OUT/'replay/features.npy')
    z=np.clip((np.where(np.isfinite(z),z,b['center'])-b['mean'])/b['scale'],-12,12).astype('float32')
    pred=b['model'].predict_proba(z)[:,1]
    saved=np.load(ROOT/f'{task}_f6.npz')['risk']
    assert np.max(np.abs(pred-saved))<1e-6
    checks[task]={'risk_max_abs_error':float(np.max(np.abs(pred-saved))),'epochs':b['epochs'],
                  'network':[292,64,32,1],'train_windows':b['train_windows'],'historical_comparison':comparison[task]['neural_holdout']}
contract=json.loads((OUT/'contract.json').read_text())
contract.update(model_name='28 参数神经网络（实验）',run='20261112_neural',
    neural={'architecture':[292,64,32,1],'activation':'relu','input_channels':28,'display_task':'voltage'},
    risk_calibration='Earlier-fold calibration and thresholds. Historical holdout already inspected in v4; no new blind evaluation.',
    why_this_fit='Requested neural alternative; trained through 2026-10-23, evaluated on the already-inspected later week. Not selected for superior test performance.')
contract['unavailable_outputs']=[x for x in contract['unavailable_outputs'] if x['field']!='neural']
contract['limits']=['实验模型，不能作为自动换枪依据。','历史周已在此前分析中查看，不是新增盲测。','寿命仍由弱标签 AFT 估计；风险网络不能改善寿命标签。']
contract['neural_comparison']=comparison
contract['parity_with_training']=checks
write(contract,OUT/'contract.json')
m=json.loads((OUT/'manifest.json').read_text())
m['files']={p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='manifest.json'}
m['version']='real28-neural-'+hashlib.sha256(json.dumps(m['files'],sort_keys=True).encode()).hexdigest()[:12]
m['panel_adapter_sha256']=sha(ROOT/'src/real28.py')
write(m,OUT/'manifest.json')
OUT.rename(ROOT/m['version'])
write(dict(version=m['version'],parity=checks),ROOT/'export.json')
print(json.dumps(dict(version=m['version'],checks=checks)))
