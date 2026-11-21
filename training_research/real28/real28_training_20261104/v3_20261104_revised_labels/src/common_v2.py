"""Isolated paths, fixed protocol and serialization for second-round experiments."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json, hashlib, time, os
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
V1 = PARENT / 'data' / '20261104_v1'
SHARED = ROOT / 'data'
GRID = PARENT / 'v2_20261104_multimodel' / 'data'
TASK = os.environ.get('TRAIN_TASK','voltage')
DATA = SHARED / TASK
MODELS = ROOT / 'models' / TASK
EVAL = ROOT / 'evaluation' / TASK
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
    dump(entry,ROOT/f'execution_status_{TASK}.json'); print(json.dumps(plain(entry),ensure_ascii=False),flush=True)
def time_seconds(x):
    return np.asarray(x,dtype='datetime64[s]').astype('int64')

def protocol(): return load(ROOT/'protocol.json')
