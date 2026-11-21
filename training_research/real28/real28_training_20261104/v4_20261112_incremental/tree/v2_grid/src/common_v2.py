"""Isolated paths, fixed protocol and serialization for second-round experiments."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json, hashlib, time, os
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
V1 = PARENT / 'data' / '20261112_v4'
DATA = ROOT / 'data'
MODELS = ROOT / 'models'
EVAL = ROOT / 'evaluation'
LOGS = ROOT / 'logs'
for p in (DATA, MODELS, EVAL, LOGS): p.mkdir(parents=True, exist_ok=True)
FIELDS = ['Argon','Nitrogen','AirJet','Silvent','Current','Power','Voltage',
 'Cooling_water_flow','Conductivity','Cooling_water_inlet_temp.','Cooling_water_outlet_temp.',
 'Feedrate_1','Disk_1','Carrier_gas_1','Feed_Factor_1','Press._Hopper_1','Stirrer_1','Weight_Hopper_1',
 'Feedrate_2','Disk_2','Carrier_gas_2','Feed_Factor_2','Press._Hopper_2','Stirrer_2','Weight_Hopper_2',
 'Station_1','Station_2','O2_Sensor']
TYPES = ['voltage','feedrate','hopper_pressure']
RISK_NAMES = ['logistic','histgb','extra_trees','random_forest','tcn','gru','patch_transformer']
FOLDS = [
 {'name':'f1','train_end':'2026-09-01','validation_end':'2026-09-08'},
 {'name':'f2','train_end':'2026-09-08','validation_end':'2026-09-15'},
 {'name':'f3','train_end':'2026-09-15','validation_end':'2026-09-30'},
]
def plain(x):
    if isinstance(x, dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)): return [plain(v) for v in x]
    if isinstance(x, np.ndarray): return plain(x.tolist())
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (bool,np.bool_)): return bool(x)
    if isinstance(x, (float,np.floating)): return float(x) if np.isfinite(x) else None
    if isinstance(x, Path): return str(x)
    if hasattr(x, 'isoformat'): return x.isoformat()
    return x
def dump(obj, path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(plain(obj),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)
def load(path): return json.loads(Path(path).read_text('utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def status(stage,**kwargs):
    entry={'stage':stage,'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kwargs}
    dump(entry,ROOT/'execution_status.json'); print(json.dumps(plain(entry),ensure_ascii=False),flush=True)
def time_seconds(x):
    return np.asarray(x,dtype='datetime64[s]').astype('int64')

def protocol():
    path=ROOT/'protocol.json'
    if path.exists(): return load(path)
    obj={'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'raw_parameters':28,
         'history_seconds':120,'max_feature_history_seconds':7200,'forecast_seconds':300,'stride_seconds':10,
         'risk_models':RISK_NAMES,'rolling_folds':FOLDS,
         'final_fit_end':'2026-09-30','development_selection':['2026-10-04','2026-10-08'],
         'calibration':['2026-10-08','2026-10-10'],'historical_comparison':['2026-10-10','2026-10-24'],
         'already_seen_historical_period':True,'risk_split':'chronological with full event isolation and conservative 7200 s validation history purge (including completed-product duration); NOT independent electrode identity',
         'lifetime_split':'reuse V1 weak/censored lifetime labels with possible cycles wholly disjoint',
         'neural_epochs_per_fit':6,'neural_batch_size':256,'seed':42,
         'selection_rule':'mean rolling-fold event-macro AP, then pooled window AP; history comparison never selects a method',
         'alarm_threshold_policy':'fit on pooled training-period OOF predictions; maximize event recall under 20 false episodes/100 observable powered hours, with rising edge and 300 s cooldown',
         'alarm_budget_is_experimental':True,'probability_calibration':'regularized sigmoid on pooled OOF; October8-9 reserved for separate calibration diagnostics',
         'lifetime_models':['lognormal_aft','weibull_aft','spline_aft','mlp_aft','logistic_hazard','boosted_hazard'],
         'pressure_forecast':False,'health_scores_trained':False,'automatic_replacement':False}
    dump(obj,path); return obj
