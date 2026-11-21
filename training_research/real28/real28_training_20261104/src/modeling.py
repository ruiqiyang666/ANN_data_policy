"""Small causal TCN and likelihoods used by the real-device experiment."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class CausalBlock(nn.Module):
    def __init__(self,channels,dilation,dropout=.1):
        super().__init__();self.dilation=dilation
        self.conv1=nn.Conv1d(channels,channels,3,dilation=dilation)
        self.conv2=nn.Conv1d(channels,channels,3,dilation=dilation)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        z=self.dropout(F.gelu(self.conv1(F.pad(x,(2*self.dilation,0)))))
        z=self.dropout(F.gelu(self.conv2(F.pad(z,(2*self.dilation,0)))))
        return x+z

class RealTCN(nn.Module):
    def __init__(self,stats_dim,channels=32):
        super().__init__();self.stats_dim=stats_dim
        self.input_projection=nn.Conv1d(56,channels,1)
        self.blocks=nn.Sequential(*[CausalBlock(channels,2**i) for i in range(5)])
        self.reconstruction=nn.Conv1d(channels,28,1)
        self.fusion=nn.Sequential(nn.Linear(channels*2+stats_dim,64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(.1))
        self.hazard=nn.Linear(64,10);self.type_head=nn.Linear(64,3);self.current=nn.Linear(64,3)
    def encode(self,x):
        sequence=self.blocks(self.input_projection(x))
        representation=torch.cat([sequence[:,:,-1],sequence.mean(dim=-1)],dim=1)
        return sequence,representation
    def forward(self,x,stats):
        sequence,embedding=self.encode(x)
        z=self.fusion(torch.cat([embedding,stats],dim=1))
        return {'hazard':self.hazard(z),'type':self.type_head(z),'current':self.current(z),'embedding':embedding}

def masked_mean(loss,mask,weight=None):
    if weight is not None:mask=mask*weight[:,None]
    return (loss*mask).sum()/mask.sum().clamp_min(1)

def multitask_loss(out,target,importance=None):
    loss=masked_mean(F.binary_cross_entropy_with_logits(out['hazard'].float(),target['hazard_y'],reduction='none'),target['hazard_mask'],importance)
    event=target['hazard_y'].sum(dim=1)>0
    if event.any():
        loss=loss+.25*F.binary_cross_entropy_with_logits(out['type'][event].float(),target['type_y'][event])
    loss=loss+.25*masked_mean(F.binary_cross_entropy_with_logits(out['current'].float(),target['current_y'],reduction='none'),target['current_mask'],importance)
    return loss

def hazard_distribution(logits):
    h=torch.sigmoid(logits.float())
    survival=torch.cumprod(1-h,dim=1)
    mass=h*torch.cat([torch.ones_like(h[:,:1]),survival[:,:-1]],dim=1)
    probability=mass.sum(dim=1)
    midpoints=torch.arange(15,300,30,device=logits.device,dtype=torch.float32)
    eta=(mass*midpoints).sum(dim=1)/probability.clamp_min(1e-12)
    return probability,eta,mass

class CachedData:
    """Batched memory-mapped slicing; windows are not materialized on disk."""
    def __init__(self,data_dir,device='cuda'):
        import json,pandas as pd
        self.path=data_dir;self.device=device
        self.samples=pd.read_parquet(data_dir/'samples.parquet')
        self.x=np.load(data_dir/'grid_values.npy',mmap_mode='r')
        self.observed=np.load(data_dir/'grid_observed.npy',mmap_mode='r')
        self.stats=np.load(data_dir/'sample_stats.npy',mmap_mode='r')
        self.ends=self.samples.end_index.to_numpy()
        self.targets=dict(np.load(data_dir/'targets.npz'))
        norm=json.loads((data_dir/'normalization.json').read_text('utf-8'))
        self.center=np.array(norm['center'],dtype='float32');self.scale=np.array(norm['scale'],dtype='float32')
        self.offsets=np.arange(-119,1)
    def batch(self,ids):
        locations=self.ends[ids,None]+self.offsets
        v=np.nan_to_num((self.x[locations]-self.center)/self.scale,nan=0,posinf=0,neginf=0).clip(-20,20)
        mask=self.observed[locations].astype('float32')
        seq=np.concatenate([v,mask],axis=-1).transpose(0,2,1).copy()
        tensor=torch.from_numpy(seq).to(self.device)
        stats=torch.from_numpy(np.array(self.stats[ids])).to(self.device)
        targets={k:torch.from_numpy(v[ids]).to(self.device) for k,v in self.targets.items()}
        return tensor,stats,targets

def weighted_stratified_ids(ids,event,batch_size,rng):
    """10% positive draw with inverse sampling weights for the original prevalence."""
    positive=ids[event[ids]];negative=ids[~event[ids]]
    if not len(positive) or not len(negative):
        return rng.choice(ids,batch_size,replace=True),np.ones(batch_size,dtype='float32')
    q=max(.10,len(positive)/len(ids));npos=max(1,min(batch_size-1,round(batch_size*q)));q=npos/batch_size
    chosen=np.r_[rng.choice(positive,npos,replace=True),rng.choice(negative,batch_size-npos,replace=True)]
    p=len(positive)/len(ids)
    weight=np.r_[np.full(npos,p/q),np.full(batch_size-npos,(1-p)/(1-q))].astype('float32')
    order=rng.permutation(batch_size)
    return chosen[order],weight[order]
