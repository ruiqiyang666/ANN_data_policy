"""Independent risk algorithms sharing causal features and declared auxiliary targets."""
import sys
sys.dont_write_bytecode=True
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier,ExtraTreesClassifier,RandomForestClassifier
from sklearn.preprocessing import SplineTransformer
from sklearn.kernel_approximation import RBFSampler
from common_v2 import DATA,TYPES,load

class Transform:
    def fit(self,x):
        with np.errstate(all='ignore'):
            self.center=np.nanmedian(x,axis=0);self.center=np.nan_to_num(self.center)
        a=np.where(np.isfinite(x),x,self.center)
        self.mean=a.mean(0);self.scale=a.std(0);self.scale=np.where(self.scale>1e-4,self.scale,1)
        return self
    def transform(self,x):return np.clip((np.where(np.isfinite(x),x,self.center)-self.mean)/self.scale,-12,12).astype('float32')

def row_weights(samples):
    y=samples.event_in_300s.to_numpy();w=np.ones(len(samples),dtype='float64')
    if y.any():
        counts=samples.loc[y,'event_id'].value_counts();median=float(counts.median())
        w[y]=8*median/samples.loc[y,'event_id'].map(counts).to_numpy()
    return w

def cpu_model(name):
    if name in ['logistic','spline_logistic','rbf_logistic']:return LogisticRegression(C=.1,max_iter=400,solver='lbfgs')
    if name=='histgb':return HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,max_depth=4,min_samples_leaf=30,l2_regularization=5,learning_rate=.05,early_stopping=False,random_state=42)
    if name=='extra_trees':return ExtraTreesClassifier(n_estimators=160,max_depth=16,min_samples_leaf=20,max_features=.7,n_jobs=4,random_state=42)
    if name=='random_forest':return RandomForestClassifier(n_estimators=160,max_depth=12,min_samples_leaf=20,max_features=.7,max_samples=.7,n_jobs=4,random_state=42)
    raise ValueError(name)

class ExtraTransform:
    def __init__(self,name):self.name=name;self.method=None
    def fit(self,x):
        if self.name=='rbf_logistic':self.method=RBFSampler(gamma=1/x.shape[1],n_components=384,random_state=42).fit(x)
        if self.name=='spline_logistic':
            # Curated physical and available-product features; no labels select columns.
            names=load(DATA/'features.json')['names']
            self.indices=[i for i,n in enumerate(names) if n.startswith(('now_Voltage','now_Current','now_Argon','now_Nitrogen','now_Feedrate','now_Press.','matched_product_','product_voltage','product_feedrate','product_pressure'))]
            self.method=SplineTransformer(n_knots=4,degree=2,include_bias=False,extrapolation='linear').fit(x[:,self.indices])
        return self
    def transform(self,x):
        if self.method is None:return x
        if self.name=='spline_logistic':return self.method.transform(x[:,self.indices]).astype('float32')
        return self.method.transform(x).astype('float32')

class CausalBlock(nn.Module):
    def __init__(self,c,d):
        super().__init__();self.d=d;self.a=nn.Conv1d(c,c,3,dilation=d);self.b=nn.Conv1d(c,c,3,dilation=d);self.drop=nn.Dropout(.15)
    def forward(self,x):return x+self.drop(F.gelu(self.b(F.pad(self.drop(F.gelu(self.a(F.pad(x,(2*self.d,0))))),(2*self.d,0)))))

class RiskNet(nn.Module):
    def __init__(self,name,nstats):
        super().__init__();self.name=name;self.nstats=nstats
        if name=='tcn':
            self.encoder=nn.Sequential(nn.Conv1d(58,32,1),*[CausalBlock(32,2**i) for i in range(5)]);width=64
        elif name=='gru':
            self.encoder=nn.GRU(58,48,num_layers=1,batch_first=True);width=96
        elif name=='patch_transformer':
            self.patch=nn.Linear(58*10,48);self.position=nn.Parameter(torch.zeros(1,12,48))
            layer=nn.TransformerEncoderLayer(48,4,128,dropout=.15,batch_first=True,norm_first=True)
            self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False);width=96
        else:raise ValueError(name)
        self.statistics=nn.Sequential(nn.Linear(nstats,48),nn.LayerNorm(48),nn.GELU())
        self.head=nn.Sequential(nn.Linear(width+48,64),nn.GELU(),nn.Dropout(.15),nn.Linear(64,10))
    def forward(self,x,stats):
        if self.name=='tcn':
            seq=self.encoder(x);emb=torch.cat([seq[:,:,-1],seq.mean(-1)],1)
        elif self.name=='gru':
            seq,_=self.encoder(x.transpose(1,2));emb=torch.cat([seq[:,-1],seq.mean(1)],1)
        else:
            patches=x.transpose(1,2).reshape(len(x),12,580);seq=self.encoder(self.patch(patches)+self.position);emb=torch.cat([seq[:,-1],seq.mean(1)],1)
        return self.head(torch.cat([emb,self.statistics(stats)],1))

def distribution(logits):
    h=torch.sigmoid(logits.float());s=torch.cumprod(1-h,dim=1);m=h*torch.cat([torch.ones_like(s[:,:1]),s[:,:-1]],1)
    total=m.sum(1);risk=total.clamp(0,1)
    conditional=torch.where(total[:,None]>0,m/total[:,None].clamp_min(torch.finfo(m.dtype).tiny),torch.full_like(m,.1))
    return risk,conditional

class Cache:
    def __init__(self):
        self.samples=pd.read_parquet(DATA/'samples.parquet');self.features=np.load(DATA/'features.npy',mmap_mode='r')
        self.x=np.load(DATA/'grid_x.npy',mmap_mode='r');self.seen=np.load(DATA/'grid_seen.npy',mmap_mode='r')
        self.targets=dict(np.load(DATA/'targets.npz'));self.ends=self.samples.end_index.to_numpy();self.offset=np.arange(-119,1)
    def sequence_norm(self,ids):
        raw=np.array(self.x[self.ends[ids]])
        center=np.nanmedian(raw,axis=0);scale=np.nanstd(raw,axis=0)
        return {'center':np.nan_to_num(center).astype('float32'),'scale':np.where(scale>.01,scale,1).astype('float32')}
    def batch(self,ids,stats,norm,device='cuda'):
        loc=self.ends[ids,None]+self.offset
        raw=self.x[loc];seen=self.seen[loc]
        vals=np.nan_to_num((raw-norm['center'])/norm['scale'],nan=0,posinf=0,neginf=0).clip(-12,12)
        powered=(raw[:,:,4]>100)&(raw[:,:,5]>5);spray=powered&((raw[:,:,25]>50)|(raw[:,:,26]>50))
        inp=np.concatenate([vals,seen,powered[:,:,None],spray[:,:,None]],2).transpose(0,2,1).astype('float32',copy=True)
        return torch.from_numpy(inp).to(device),torch.from_numpy(stats).to(device)

def fit_auxiliary(cache,ids,stats):
    event=cache.samples.iloc[ids].event_in_300s.to_numpy();eids=ids[event];x=stats[event]
    times=np.argmax(cache.targets['hazard_y'][eids],axis=1);types=np.argmax(cache.targets['type_y'][eids],axis=1)
    result={}
    for name,y in [('time',times),('type',types)]:
        unique=np.unique(y)
        if len(unique)<2:result[name]={'constant':int(unique[0])}
        else:result[name]=LogisticRegression(C=.05,max_iter=350).fit(x,y)
    return result

def auxiliary_predict(aux,stats):
    result={}
    for name,size in [('time',10),('type',3)]:
        model=aux[name];prob=np.zeros((len(stats),size),dtype='float32')
        if isinstance(model,dict):prob[:,model['constant']]=1
        else:prob[:,model.classes_.astype(int)]=model.predict_proba(stats)
        result[name]=prob
    return result
