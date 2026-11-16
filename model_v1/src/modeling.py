"""Shared causal representation and frozen predictors; no labels enter this module."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import torch
from torch import nn

FIELDS = ['Voltage','Current','Power','Station_1','Station_2','Feedrate_1','Feedrate_2','Press._Hopper_1','Press._Hopper_2']
VALUE_FIELDS = FIELDS + ['active_feedrate','active_pressure']
TARGETS = ['eol_15m','eol_30m','eol_60m',
           'adjacent_voltage_15m','adjacent_voltage_30m','adjacent_voltage_60m',
           'adjacent_any_15m','adjacent_any_30m','adjacent_any_60m',
           'next_any','next_voltage','next_feedrate','next_pressure']
GROUPS = [slice(0,3),slice(3,6),slice(6,9)]

def seconds(values):
    return np.asarray(values,dtype='datetime64[s]').astype('int64')

def raw_values(frame):
    a=frame.reindex(columns=FIELDS).to_numpy(dtype=np.float32)
    # A hopper contributes only at positive feed; neither/dual activity stays missing.
    on1=np.isfinite(a[:,5]) & (a[:,5]>0)
    on2=np.isfinite(a[:,6]) & (a[:,6]>0)
    single=on1 ^ on2
    feed=np.where(single,np.where(on1,a[:,5],a[:,6]),np.nan)
    pressure=np.where(single,np.where(on1,a[:,7],a[:,8]),np.nan)
    return np.column_stack([a,feed,pressure]).astype(np.float32)

def fit_preprocessor(frame):
    a=raw_values(frame)
    median=np.nanmedian(a,axis=0)
    q=np.nanpercentile(a,[25,75],axis=0)
    scale=np.maximum(q[1]-q[0],np.array([.2,1,.1,50,50,.1,.1,1,1,.1,1]))
    return dict(fields=VALUE_FIELDS,median=median.tolist(),scale=scale.tolist(),
                history_minutes=30,grid_seconds=60,max_age_seconds=60,fit_split='train')

def sequence_indices(frame,history_minutes=30,max_age_seconds=60):
    """Per-origin one-minute calendar grid. No cross-cycle or future lookup."""
    t=seconds(frame.timestamp)
    indices=np.full((len(frame),history_minutes+1),-1,dtype=np.int32)
    age=np.ones(indices.shape,dtype=np.float32)
    offsets=np.arange(history_minutes,-1,-1)*60
    for _,positions in frame.groupby('cycle_id',sort=False).indices.items():
        positions=np.sort(positions)
        local=t[positions]
        grid=local[:,None]-offsets[None,:]
        found=np.searchsorted(local,grid,side='right')-1
        safe=np.maximum(found,0)
        delta=grid-local[safe]
        good=(found>=0)&(delta>=0)&(delta<=max_age_seconds)
        indices[positions]=np.where(good,positions[safe],-1)
        age[positions]=np.where(good,delta/max_age_seconds,1)
    return indices,age

def encode_sequences(frame,preprocessor,indices=None,age=None):
    if indices is None: indices,age=sequence_indices(frame)
    a=raw_values(frame)
    med=np.asarray(preprocessor['median'],dtype=np.float32)
    scale=np.asarray(preprocessor['scale'],dtype=np.float32)
    observed=np.isfinite(a)
    z=np.clip((np.where(observed,a,med)-med)/scale,-10,10)
    values=z[np.maximum(indices,0)]
    mask=observed[np.maximum(indices,0)] & (indices[:,:,None]>=0)
    values=np.where(mask,values,0)
    return np.concatenate([values,mask.astype(np.float32),age[:,:,None]],axis=2).astype(np.float32)

def summaries(x):
    n=len(VALUE_FIELDS)
    values=x[:,:,:n]; mask=x[:,:,n:2*n]
    count=np.maximum(mask.sum(axis=1),1)
    mean=(values*mask).sum(axis=1)/count
    var=(((values-mean[:,None,:])**2)*mask).sum(axis=1)/count
    # Current, trailing mean, dispersion and departure from trailing mean.
    return np.concatenate([values[:,-1],mean,np.sqrt(var),values[:,-1]-mean,
                           mask.mean(axis=1),mask[:,-1]],axis=1).astype(np.float32)

def margin_probabilities(frame,x):
    a=raw_values(frame)
    v=a[:,0]; f=a[:,-2]; p=a[:,-1]
    # Predetermined engineering scores, subsequently calibrated on validation.
    mean_z=summaries(x)[:,11]
    # Recent voltage decline relative to causal trailing mean in standardized units.
    decline=np.maximum(mean_z-x[:,-1,0],0)
    voltage=(53-np.nan_to_num(v,nan=55))/.8 + .35*decline
    feed=(np.abs(np.nan_to_num(f,nan=10.1)-10.1)-.85)/.2
    pressure=np.maximum((105-np.nan_to_num(p,nan=180))/12,(np.nan_to_num(p,nan=180)-242)/12)
    sigmoid=lambda a: 1/(1+np.exp(-np.clip(a,-30,30)))
    out=np.zeros((len(frame),len(TARGETS)),dtype=np.float32)
    for sl in GROUPS:
        base=voltage if sl.start<6 else np.maximum.reduce([voltage,feed,pressure])
        out[:,sl]=sigmoid(base[:,None]+np.array([-2.2,-1.6,-.9]))
    out[:,9]=sigmoid(np.maximum.reduce([voltage-2,feed,pressure]))
    out[:,10]=sigmoid(voltage-2)
    out[:,11]=sigmoid(feed);out[:,12]=sigmoid(pressure)
    return out

class CausalBlock(nn.Module):
    def __init__(self,channels,dilation):
        super().__init__()
        self.pad=2*dilation
        self.conv=nn.Conv1d(channels,channels,3,dilation=dilation,padding=self.pad)
        self.act=nn.GELU();self.drop=nn.Dropout(.1)
    def forward(self,x):
        h=self.conv(x)[:,:,:-self.pad]
        return x+self.drop(self.act(h))

class RiskNet(nn.Module):
    def __init__(self,kind='tcn',channels=23,hidden=16):
        super().__init__();self.kind=kind
        if kind=='tcn':
            self.input=nn.Conv1d(channels,hidden,1)
            self.body=nn.Sequential(*[CausalBlock(hidden,d) for d in [1,2,4,8]])
        elif kind=='gru': self.body=nn.GRU(channels,hidden,batch_first=True)
        else: raise ValueError(kind)
        self.head=nn.Linear(hidden,len(TARGETS))
    def forward(self,x):
        if self.kind=='tcn': h=self.body(self.input(x.transpose(1,2)))[:,:,-1]
        else: h=self.body(x)[0][:,-1]
        return self.head(h)

def monotone(p):
    p=np.clip(p,1e-6,1-1e-6).copy()
    for sl in GROUPS: p[:,sl]=np.maximum.accumulate(p[:,sl],axis=1)
    return p

def apply_calibration(p,calibrators):
    out=p.copy()
    for j,c in enumerate(calibrators):
        if c is not None:
            logits=np.log(np.clip(p[:,j],1e-6,1-1e-6)/np.clip(1-p[:,j],1e-6,1))[:,None]
            out[:,j]=c.predict_proba(logits)[:,1]
    return monotone(out)

def remaining_quantiles(probabilities):
    # Restricted (0,60min] cumulative risk, linear interpolation within bins.
    p=np.maximum.accumulate(np.asarray(probabilities,dtype=float))
    times=np.array([0,15,30,60],dtype=float)
    cdf=np.r_[0,p]
    return {f'remaining_minutes_q{int(q*100)}':
            float(np.interp(q,cdf,times)) if p[-1]>=q else None for q in [.1,.5,.9]}

class Predictor:
    def __init__(self,project_dir):
        self.root=Path(project_dir)
        self.config=json.loads((self.root/'artifacts/frozen_config.json').read_text(encoding='utf-8'))
        self.pre=json.loads((self.root/'artifacts/preprocessor.json').read_text(encoding='utf-8'))
        self.name=self.config['selected_model'];self.model_version=self.config['model_version']
        self.version=self.model_version
        self.thresholds=self.config['thresholds'][self.name]
        self.calibrators=joblib.load(self.root/'artifacts/calibrators.joblib')[self.name]
        self.baselines=joblib.load(self.root/'artifacts/baselines.joblib')
        self.net=None
        if self.name in ['tcn','gru']:
            self.net=RiskNet(self.name,hidden=self.config['hidden'])
            ck=torch.load(self.root/f'artifacts/{self.name}/best.pt',map_location='cpu',weights_only=False)
            self.net.load_state_dict(ck['model']);self.net.eval()
    def predict(self,history_dataframe):
        frame=history_dataframe.copy().sort_values('timestamp').reset_index(drop=True)
        if frame.empty: raise ValueError('No causal history')
        frame['timestamp']=pd.to_datetime(frame.timestamp)
        x=encode_sequences(frame,self.pre)[-1:]
        if self.net is not None:
            with torch.inference_mode(): p=torch.sigmoid(self.net(torch.from_numpy(x))).numpy()
        elif self.name=='margin': p=margin_probabilities(frame.tail(1),x)
        elif self.name=='event_rate': p=self.baselines['event_rate'][None,:]
        else:
            z=summaries(x)
            p=np.array([[m.predict_proba(z)[0,1] if hasattr(m,'predict_proba') else float(m)
                         for m in self.baselines['logistic']]],dtype=np.float32)
        p=apply_calibration(p,self.calibrators)[0]
        result=dict(zip(TARGETS,map(float,p)))
        result.update(remaining_quantiles(p[:3]))
        result['model_version']=self.model_version
        return result
