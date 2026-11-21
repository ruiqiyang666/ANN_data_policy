"""Paths and serialization for the isolated real-28 experiment."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
SOURCE = PROJECT / "Real-time Data"
RUN = "20261112_v4"
DATA = ROOT / "data" / RUN
MODEL = ROOT / "models" / RUN
EVAL = ROOT / "evaluation" / RUN
LOG = ROOT / "logs" / RUN
FIELDS = ['Argon','Nitrogen','AirJet','Silvent','Current','Power','Voltage',
 'Cooling_water_flow','Conductivity','Cooling_water_inlet_temp.','Cooling_water_outlet_temp.',
 'Feedrate_1','Disk_1','Carrier_gas_1','Feed_Factor_1','Press._Hopper_1','Stirrer_1','Weight_Hopper_1',
 'Feedrate_2','Disk_2','Carrier_gas_2','Feed_Factor_2','Press._Hopper_2','Stirrer_2','Weight_Hopper_2',
 'Station_1','Station_2','O2_Sensor']
TYPES = ['voltage','feedrate','hopper_pressure']
PARTITIONS = ['train','selection','calibration','test','new_blind']

def plain(x):
    if isinstance(x, dict): return {str(k): plain(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)): return [plain(v) for v in x]
    if isinstance(x,np.ndarray): return plain(x.tolist())
    if isinstance(x,(np.bool_,bool)): return bool(x)
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(np.floating,float)): return float(x) if np.isfinite(x) else None
    if isinstance(x,Path): return str(x)
    if hasattr(x,'isoformat'): return x.isoformat()
    return x

def dump(obj, path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(plain(obj),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temporary,path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def status(stage, **kwargs):
    rec={'stage':stage,'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kwargs}
    dump(rec,ROOT/'execution_status.json')
    print(json.dumps(plain(rec),ensure_ascii=False),flush=True)

def ensure_dirs():
    for p in [DATA,MODEL,EVAL,LOG]: p.mkdir(parents=True,exist_ok=True)
