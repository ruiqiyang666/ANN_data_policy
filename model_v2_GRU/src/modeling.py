"""Causal sensor-only representations and small risk models for Booth2 V2."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['Voltage', 'Current', 'Power', 'Station_1', 'Station_2',
          'Feedrate_1', 'Feedrate_2', 'Press._Hopper_1', 'Press._Hopper_2']
VALUE_FIELDS = FIELDS + ['active_feedrate', 'active_pressure', 'Voltage_mean60s', 'Voltage_std60s']
TARGETS = ['y_eol_30m', 'y_eol_45m', 'y_eol_60m', 'y_pre20_40m']

def seconds(t):
    return np.asarray(t, dtype='datetime64[s]').astype(np.int64)

def raw_values(frame):
    a = frame[FIELDS].to_numpy(np.float32)
    on1, on2 = a[:,5] > 0, a[:,6] > 0
    single = on1 ^ on2
    feed = np.where(single, np.where(on1, a[:,5], a[:,6]), np.nan)
    pressure = np.where(single, np.where(on1, a[:,7], a[:,8]), np.nan)
    return np.column_stack([a, feed, pressure, frame[['Voltage_mean60s','Voltage_std60s']]]).astype(np.float32)

def fit_preprocessor(frame):
    a = raw_values(frame)
    med = np.nanmedian(a, axis=0)
    q = np.nanpercentile(a, [25,75], axis=0)
    floor = np.array([.2,1,.1,50,50,.1,.1,1,1,.1,1,.2,.1])
    return dict(fields=VALUE_FIELDS, median=med.tolist(), scale=np.maximum(q[1]-q[0],floor).tolist(),
                history_minutes=60, grid_seconds=60, max_age_seconds=60, fit_split='train')

def sequence_indices(frame, history_minutes=60):
    t = seconds(frame.timestamp)
    indices = np.full((len(frame),history_minutes+1), -1, np.int32)
    age = np.ones(indices.shape, np.float32)
    for _,ix in frame.groupby('cycle_id',sort=False).indices.items():
        local = t[ix]
        grid = local[:,None]-np.arange(history_minutes,-1,-1)[None,:]*60
        j = np.searchsorted(local,grid,side='right')-1
        delta = grid-local[np.maximum(j,0)]
        good = (j>=0)&(delta>=0)&(delta<=60)
        indices[ix] = np.where(good,ix[np.maximum(j,0)],-1)
        age[ix] = np.where(good,delta/60,1)
    return indices,age

def encode(frame, pre):
    ix, age = sequence_indices(frame,pre['history_minutes'])
    a = raw_values(frame)
    med,scale = np.array(pre['median']),np.array(pre['scale'])
    obs = np.isfinite(a)
    z = np.clip((np.where(obs,a,med)-med)/scale,-12,12).astype(np.float32)
    mask = obs[np.maximum(ix,0)] & (ix[:,:,None]>=0)
    values = np.where(mask,z[np.maximum(ix,0)],0)
    return np.concatenate([values,mask.astype(np.float32),age[:,:,None]],axis=2).astype(np.float32)

def summaries(x):
    n = len(VALUE_FIELDS)
    v,m = x[:,:,:n],x[:,:,n:2*n]
    cols = [v[:,-1],m[:,-1]]
    for width in [6,16,31,61]:
        a,b = v[:,-width:],m[:,-width:]
        count = np.maximum(b.sum(1),1)
        mean = (a*b).sum(1)/count
        sd = np.sqrt((((a-mean[:,None])**2)*b).sum(1)/count)
        cols.extend([mean,sd,v[:,-1]-mean,b.mean(1)])
    # Temporal slopes for voltage and its causal 60s mean; no calendar/date feature.
    for channel in [0,11]:
        for width in [16,61]:
            a,b = v[:,-width:,channel],m[:,-width:,channel]
            t = np.arange(-width+1,1,dtype=np.float32)
            count=np.maximum(b.sum(1),1)
            tm=(b*t).sum(1)/count; am=(a*b).sum(1)/count
            slope=((a-am[:,None])*(t-tm[:,None])*b).sum(1)/np.maximum(((t-tm[:,None])**2*b).sum(1),1)
            cols.append(slope[:,None])
    return np.concatenate(cols,axis=1).astype(np.float32)

def margin_score(frame):
    # Fixed engineering score; only the threshold is fitted on validation.
    v = frame.Voltage_mean60s.fillna(frame.Voltage).to_numpy(float)
    spread = frame.Voltage_std60s.fillna(0).to_numpy(float)
    return (1/(1+np.exp(-np.clip((53.0-v)/.8+.2*spread,-30,30)))).astype(np.float32)

class CausalBlock(nn.Module):
    def __init__(self,h,d):
        super().__init__(); self.pad=2*d
        self.conv=nn.Conv1d(h,h,3,dilation=d,padding=self.pad)
        self.drop=nn.Dropout(.15); self.act=nn.GELU()
    def forward(self,x):
        return x+self.drop(self.act(self.conv(x)[:,:,:-self.pad]))

class RiskNet(nn.Module):
    def __init__(self,kind,hidden):
        super().__init__();self.kind=kind
        channels=len(VALUE_FIELDS)*2+1
        if kind=='tcn':
            self.input=nn.Conv1d(channels,hidden,1)
            self.body=nn.Sequential(*[CausalBlock(hidden,d) for d in [1,2,4,8,16]])
        else:
            self.body=nn.GRU(channels,hidden,batch_first=True)
        self.head=nn.Linear(hidden,len(TARGETS))
    def forward(self,x):
        h=self.body(self.input(x.transpose(1,2)))[:,:,-1] if self.kind=='tcn' else self.body(x)[0][:,-1]
        return self.head(h)

def predict_net(net,x,batch=1024):
    net.eval(); out=[]
    with torch.inference_mode():
        for begin in range(0,len(x),batch):
            out.append(torch.sigmoid(net(torch.from_numpy(x[begin:begin+batch]))).numpy())
    return np.concatenate(out) if out else np.empty((0,len(TARGETS)))

def load_frames():
    f=pd.read_csv(ROOT/'data/anchors.csv.gz',parse_dates=['timestamp'])
    y=pd.read_csv(ROOT/'data/targets.csv.gz')
    assert f.anchor_id.equals(y.anchor_id)
    e=pd.read_csv(ROOT/'data/events.csv',parse_dates=['time'])
    p=pd.read_csv(ROOT/'data/products.csv.gz',low_memory=False,parse_dates=['start','end'])
    return f,y,e,p
