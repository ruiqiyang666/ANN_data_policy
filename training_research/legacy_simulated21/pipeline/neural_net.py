"""Small local multitask networks. No downloaded weights or executable model code."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

ARCHS={'mlp':'MTL-MLP','tcn':'Residual-TCN','patch':'Patch-Attention','multiscale':'Gated-Multiscale'}
SCALE=np.array([35.,2500.,100.,100.,.2],dtype=np.float32)

class CausalBlock(nn.Module):
    def __init__(self,dilation):
        super().__init__();self.pad=2*dilation
        self.conv=nn.Conv1d(32,32,3,dilation=dilation)
        self.norm=nn.GroupNorm(4,32)
    def forward(self,x):return x+F.gelu(self.norm(self.conv(F.pad(x,(self.pad,0)))))

class MultiTaskNet(nn.Module):
    def __init__(self,arch):
        super().__init__();assert arch in ARCHS;self.arch=arch
        if arch=='tcn':
            self.encoder=nn.Sequential(nn.Conv1d(30,32,1),CausalBlock(1),CausalBlock(2),CausalBlock(4),CausalBlock(8))
        elif arch=='patch':
            self.patch=nn.Linear(30*10,64);self.position=nn.Parameter(torch.zeros(1,6,64))
            layer=nn.TransformerEncoderLayer(64,4,128,.1,batch_first=True,norm_first=True)
            self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        elif arch=='multiscale':
            self.branches=nn.ModuleList([nn.Sequential(nn.Conv1d(30,32,3,padding=1),nn.GELU(),nn.Conv1d(32,32,3,padding=1),nn.GELU()) for _ in range(3)])
            self.gate=nn.Linear(88,3)
        width=88 if arch=='mlp' else 152
        self.hidden1=nn.Linear(width,64);self.hidden2=nn.Linear(64,32)
        self.dropout=nn.Dropout(.1)
        # One dense output layer contains 5 regressions, 6 current logits,
        # 6 future logits and an ETA logit. Losses remain task-specific.
        self.output=nn.Linear(32,18)
    def representation(self,seq,flat):
        if self.arch=='mlp':return flat
        if self.arch=='tcn':
            z=self.encoder(seq.transpose(1,2));z=torch.cat([z[:,:,-1],z.mean(-1)],-1)
        elif self.arch=='patch':
            z=self.encoder(self.patch(seq.reshape(-1,6,300))+self.position)
            z=z.mean(1)
        else:
            x=seq.transpose(1,2);parts=[]
            for size,branch in zip([1,5,15],self.branches):
                h=branch(x if size==1 else F.avg_pool1d(x,size,size))
                parts.append(torch.cat([h[:,:,-1],h.mean(-1)],-1))
            z=(torch.stack(parts,1)*self.gate(flat).softmax(-1)[:,:,None]).sum(1)
        return torch.cat([flat,z],-1)
    def forward(self,seq,flat,activations=False):
        z=self.representation(seq,flat);a=F.relu(self.hidden1(z));b=F.relu(self.hidden2(self.dropout(a)));o=self.output(self.dropout(b))
        return (o,[z,a,b,o]) if activations else o

def decode(raw,pressure):
    reg=raw[:,:5]*SCALE
    reg[:,4]=pressure+reg[:,4]
    reg[:,:2]=np.maximum(reg[:,:2],0);reg[:,2:4]=np.clip(reg[:,2:4],0,100);reg[:,4]=np.maximum(reg[:,4],0)
    def softmax(x):
        a=np.exp(x-x.max(1,keepdims=True));return a/a.sum(1,keepdims=True)
    return reg,softmax(raw[:,5:11]),softmax(raw[:,11:17]),300/(1+np.exp(-np.clip(raw[:,17],-30,30)))

def normalize_inputs(seq,flat,stats):
    seq=np.array(seq,dtype=np.float32,copy=True);flat=np.array(flat,dtype=np.float32,copy=True)
    seq[...,:19]=(seq[...,:19]-np.asarray(stats['mean'][:19]))/np.asarray(stats['std'][:19])
    flat=(flat-np.asarray(stats['mean']))/np.asarray(stats['std'])
    return np.clip(seq,-12,12).astype(np.float32),np.clip(flat,-12,12).astype(np.float32)

def encode_frame(frame):
    from core import NUMERIC,STATES,feature_rows
    f=frame.iloc[-60:]
    a=f[NUMERIC].to_numpy(np.float32)
    state=np.stack([(f.operation_state==s).to_numpy(np.float32) for s in STATES],1)
    msg=f.machine_message.fillna('').astype(str).str.startswith('SIM_STATE_').to_numpy(np.float32)[:,None]
    return np.concatenate([a,state,msg],1)[None],feature_rows(frame,[len(frame)-1])
