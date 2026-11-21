"""Causal lifetime features and bounded neural lifespan experiments. No future inputs."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import torch
import numpy as np
import pandas as pd
import json,time,gc,argparse
from pathlib import Path
from torch import nn
from torch.nn import functional as F
from core import ROOT,NUMERIC,ENERGIZED

HISTORY_NAMES=['observed_hours','energized_hours','completed_parts','baseline_samples']+[f'ema300_{k}' for k in NUMERIC]+[f'ema1800_{k}' for k in NUMERIC]+[f'drift_{k}' for k in NUMERIC]

def history_features(frame,indices=None):
    """Every row depends only on its prefix; initial baseline freezes after 240 arc samples."""
    active=frame.operation_state.isin(ENERGIZED).to_numpy();n=len(frame)
    count=np.cumsum(active);x=frame[NUMERIC].astype(float)
    masked=x.where(np.broadcast_to(active[:,None],x.shape));short=masked.ewm(alpha=2/301,adjust=False,ignore_na=True).mean().fillna(0).to_numpy()
    long=masked.ewm(alpha=2/1801,adjust=False,ignore_na=True).mean().fillna(0).to_numpy()
    initial=active&(count<=240);baseline=np.cumsum(x.to_numpy()*initial[:,None],axis=0)/np.maximum(np.cumsum(initial),1)[:,None]
    st=frame.operation_state.to_numpy();parts=np.cumsum((st=='powder_clear')&np.r_[False,st[:-1]=='spraying'])
    z=np.column_stack([np.arange(1,n+1)/3600,count/3600,parts,np.minimum(count,240)/240,short,long,short-baseline]).astype(np.float32)
    return z if indices is None else z[indices]

class LifetimePredictor:
    def __init__(self,folder):self.path=Path(folder)/'lifetime.pt';self.stamp=None
    def predict(self,flat,history):
        stamp=self.path.stat().st_mtime_ns
        if stamp!=self.stamp:
            self.c=torch.load(self.path,map_location='cpu',weights_only=True);c=self.c
            self.net=LifetimeNet(c['width'],c['kind'],c['bins']);self.net.load_state_dict(c['state_dict']);self.net.eval();self.stamp=stamp
        c=self.c;h=np.clip((np.asarray(history)-c['history_mean'])/c['history_std'],-12,12).astype(np.float32)
        x=np.c_[flat,h[None]] if c['history'] else flat
        with torch.inference_mode():raw,acts=self.net(torch.tensor(x),trace=True)
        values=np.maximum(0,raw.numpy()[0]*c['scales']);values[2]=min(100,values[2])
        acts=acts[1:];names=['历史表征','隐藏层','寿命输出']
        graph={'layers':[{'name':name,'units':a.shape[1]} for name,a in zip(names,acts)],'connections':[{'from':i,'to':i+1,'weights':layer.weight.detach().numpy().tolist()} for i,layer in enumerate([self.net.hidden2,self.net.output])]}
        return values,c,graph,[{'name':name,'values':a[0].numpy().tolist()} for name,a in zip(names,acts)]

class HistoryState:
    """Constant-memory online equivalent; reset on stream gaps/replacement."""
    def __init__(self):self.n=0;self.arc=0;self.parts=0;self.previous=None;self.sum=np.zeros(19);self.short=np.zeros(19);self.long=np.zeros(19)
    def step(self,row):
        self.n+=1;state=row['operation_state'];x=np.array([row[k] for k in NUMERIC],float)
        if state in ENERGIZED:
            self.arc+=1
            if self.arc<=240:self.sum+=x
            if self.arc==1:self.short=x.copy();self.long=x.copy()
            else:self.short+=(x-self.short)*2/301;self.long+=(x-self.long)*2/1801
        self.parts+=int(state=='powder_clear' and self.previous=='spraying');self.previous=state
        return np.r_[self.n/3600,self.arc/3600,self.parts,min(self.arc,240)/240,self.short,self.long,self.short-self.sum/max(min(self.arc,240),1)].astype(np.float32)

class LifetimeNet(nn.Module):
    def __init__(self,width,kind='residual',bins=None,outputs=3):
        super().__init__();self.kind=kind
        self.register_buffer('bins',torch.tensor(bins,dtype=torch.float32) if bins is not None else torch.empty(0))
        dim=width if kind=='residual' else width*8
        self.hidden1=nn.Linear(dim,192);self.hidden2=nn.Linear(192,96);self.output=nn.Linear(96,outputs)
    def forward(self,x,trace=False):
        z=x if self.kind=='residual' else ((x[:,:,None]-self.bins[:,:-1])/(self.bins[:,1:]-self.bins[:,:-1]).clamp_min(.02)).clamp(0,1).flatten(1)
        a=F.silu(self.hidden1(z));b=F.silu(self.hidden2(a));o=self.output(b)
        return (o,[z,a,b,o]) if trace else o

def prepare_history(folder,targets):
    from neural_train import resources
    path=folder/'history.npy'
    if path.exists():return np.load(path,mmap_mode='r')
    import pyarrow.parquet as pq
    manifest=json.loads((ROOT/'results/数据清单.json').read_text(encoding='utf8'));pf=pq.ParquetFile(ROOT/'results/模拟数据.parquet')
    z=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(targets),len(HISTORY_NAMES)))
    for g in manifest['guns']:
        ids=np.flatnonzero(targets[:,11]==g['gun_id'])
        if not len(ids):continue
        frame=pf.read_row_group(g['row_group'],columns=NUMERIC+['operation_state']).to_pandas()
        z[ids]=history_features(frame,targets[ids,12].astype(int));del frame
        if g['gun_id']%10==0:print('History features gun',g['gun_id'],flush=True);resources('history_features')
    z.flush();return z

def run():
    from neural_train import resources,PEAK
    torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.cuda.set_per_process_memory_fraction(.5)
    folder=ROOT/'results/neural';lock=folder/'training.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        t=np.load(folder/'targets.npy',mmap_mode='r');split=np.load(folder/'split.npy',mmap_mode='r');flat=np.load(folder/'flat.npy',mmap_mode='r')
        h=prepare_history(folder,t);train=np.flatnonzero((split==0)&t[:,8].astype(bool));val=np.flatnonzero((split==1)&t[:,8].astype(bool))
        hm=np.mean(h[train],0);hs=np.maximum(np.std(h[train],0),.01)
        x=np.c_[flat,np.clip((h-hm)/hs,-12,12)].astype(np.float32);y=np.array(t[:,[0,1,2]],np.float32);scales=np.array([35,2500,100],np.float32)
        # Both architecture and history ablation fixed before any new test evaluation.
        experiments=[('recent_only','residual',False,314),('history_mlp','residual',True,314),('history_ple','ple',True,314),('history_ple_seed2','ple',True,2718)]
        summaries=[];best=float('inf');chosen=None;started=time.time()
        for name,kind,history,seed in experiments:
            torch.manual_seed(seed);np.random.seed(seed);xx=x if history else x[:,:88];bins=np.quantile(xx[train],np.linspace(0,1,9),axis=0).T.astype(np.float32) if kind=='ple' else None
            net=LifetimeNet(xx.shape[1],kind,bins).cuda();opt=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,40,eta_min=.00003)
            bestval=float('inf');bad=0;log=[];start=time.time()
            def predict(ids):
                net.eval();out=[]
                with torch.inference_mode():
                    for ix in np.array_split(ids,max(1,int(np.ceil(len(ids)/1024)))):out.append(net(torch.from_numpy(xx[ix]).cuda()).cpu().numpy()*scales)
                return np.maximum(0,np.concatenate(out))
            for epoch in range(1,41):
                net.train();order=np.random.default_rng(seed+epoch).permutation(train);total=0
                for begin in range(0,len(order),512):
                    ix=order[begin:begin+512];xb=torch.from_numpy(xx[ix]).cuda();yb=torch.from_numpy(y[ix]/scales).cuda();opt.zero_grad(set_to_none=True)
                    pred=net(xb);err=F.smooth_l1_loss(pred,yb,reduction='none',beta=.05);loss=err[:,0].mean()+.35*err[:,1].mean()+.15*err[:,2].mean()
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                    loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1);opt.step();total+=float(loss.detach())*len(ix)
                    if begin%(512*160)==0:resources('lifespan_training',name,epoch)
                pv=predict(val);mae=float(np.mean(abs(pv[:,0]-y[val,0])));sched.step();log.append({'epoch':epoch,'loss':total/len(train),'validation_rul_mae':mae})
                print(name,epoch,'validation RUL',round(mae,4),flush=True)
                if mae<bestval-.001:
                    bestval=mae;bad=0;checkpoint={'state_dict':{k:v.detach().cpu() for k,v in net.state_dict().items()},'kind':kind,'width':xx.shape[1],'bins':bins.tolist() if bins is not None else None,'history':history,'history_mean':hm.tolist(),'history_std':hs.tolist(),'scales':scales.tolist(),'name':name,'epoch':epoch,'validation_mae':mae}
                    torch.save(checkpoint,folder/(name+'.pt'))
                else:bad+=1
                if bad>=7:break
            record={'id':name,'kind':kind,'history':history,'validation_mae':bestval,'seconds':time.time()-start,'epochs':log};summaries.append(record)
            if bestval<best:best=bestval;chosen=name
            (folder/'lifetime_progress.json').write_text(json.dumps({'candidates':summaries,'selected':chosen}),encoding='utf8')
            del net,opt;torch.cuda.empty_cache();gc.collect()
        # Freeze validation choice BEFORE reading test targets for scoring.
        selection={'selected':chosen,'criterion':'lowest validation RUL MAE','validation_mae':best,'test_used_for_selection':False,'candidates':summaries}
        (folder/'lifetime_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf8')
        c=torch.load(folder/(chosen+'.pt'),map_location='cpu',weights_only=True);net=LifetimeNet(c['width'],c['kind'],c['bins']).cuda();net.load_state_dict(c['state_dict']);xx=x if c['history'] else x[:,:88]
        cal=np.flatnonzero((split==2)&t[:,8].astype(bool));test=np.flatnonzero((split==3)&t[:,8].astype(bool));pc=predict(cal);pt=predict(test)
        quantile=np.quantile(abs(pc[:,:2]-y[cal,:2]),.9,axis=0);c['conformal']=quantile.tolist();c['version']='lifetime-20261007';torch.save(c,folder/'lifetime.pt')
        from sklearn.metrics import r2_score
        metrics={}
        for j,name in enumerate(['rul_hours','rul_pieces','gun_health']):metrics[name]={'mae':float(np.mean(abs(pt[:,j]-y[test,j]))),'rmse':float(np.sqrt(np.mean((pt[:,j]-y[test,j])**2))),'r2':float(r2_score(y[test,j],pt[:,j])),'rows':len(test)}
        ev=pd.DataFrame({'gun_id':t[test,11].astype(int),'seq':t[test,13].astype(int)})
        for j,key in enumerate(['rul_hours','rul_pieces','gun_health']):ev['y_'+key]=y[test,j];ev['pred_'+key]=pt[:,j]
        ev.to_parquet(folder/'lifetime_evaluation.parquet',index=False)
        report={**selection,'test':metrics,'coverage':np.mean(abs(pt[:,:2]-y[test,:2])<=quantile,axis=0).tolist(),'resources':PEAK,'seconds':time.time()-started,'features':88+len(HISTORY_NAMES),'history_features':HISTORY_NAMES,'limitations':['Previously inspected test set reused for comparability; not a fresh confirmatory holdout.','Cumulative context starts at observation start, not inferred physical age; cold-start uncertainty remains.']}
        (folder/'lifetime_summary.json').write_text(json.dumps(report,indent=2),encoding='utf8');print('FINAL',json.dumps(metrics),flush=True);resources('ready',chosen)
    finally:lock.unlink(missing_ok=True)

def train_auxiliary():
    from neural_train import resources,PEAK
    torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.cuda.set_per_process_memory_fraction(.5)
    folder=ROOT/'results/neural';lock=folder/'training.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        x=np.load(folder/'flat.npy',mmap_mode='r');t=np.load(folder/'targets.npy',mmap_mode='r');split=np.load(folder/'split.npy',mmap_mode='r')
        tr=np.flatnonzero(split==0);val=np.flatnonzero(split==1);bins=np.quantile(x[tr],np.linspace(0,1,9),axis=0).T.astype(np.float32);y=np.array(t[:,3],np.float32)/100
        candidates=[];winner=None;best=1e9
        def predict(ids):
            net.eval();p=[]
            with torch.inference_mode():
                for i in range(0,len(ids),1024):p.append(net(torch.tensor(np.array(x[ids[i:i+1024]])).cuda()).cpu().numpy()[:,0]*100)
            return np.clip(np.concatenate(p),0,100)
        for seed in [314,2718]:
            torch.manual_seed(seed);net=LifetimeNet(88,'ple',bins,1).cuda();opt=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.001);local=1e9;bad=0;history=[]
            for epoch in range(1,31):
                net.train();order=np.random.default_rng(seed+epoch).permutation(tr)
                for start in range(0,len(order),512):
                    ix=order[start:start+512];opt.zero_grad(set_to_none=True);pred=net(torch.tensor(np.array(x[ix])).cuda())[:,0];loss=F.smooth_l1_loss(pred,torch.tensor(y[ix]).cuda(),beta=.02)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite auxiliary loss')
                    loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
                    if start%(512*160)==0:resources('tank_health_training',str(seed),epoch)
                mae=float(np.mean(abs(predict(val)-y[val]*100)));history.append({'epoch':epoch,'validation_mae':mae});print('Tank',seed,epoch,mae,flush=True)
                if mae<local-.005:
                    local=mae;bad=0;c={'state_dict':{k:v.detach().cpu() for k,v in net.state_dict().items()},'bins':bins.tolist(),'seed':seed,'validation_mae':mae,'epoch':epoch};torch.save(c,folder/f'tank_{seed}.pt')
                else:bad+=1
                if bad>=6:break
            candidates.append({'seed':seed,'validation_mae':local,'epochs':history})
            if local<best:best=local;winner=seed
            del opt,net;torch.cuda.empty_cache()
        (folder/'tank_selection.json').write_text(json.dumps({'seed':winner,'validation_mae':best,'candidates':candidates,'criterion':'validation tank MAE only'}),encoding='utf8')
        c=torch.load(folder/f'tank_{winner}.pt',map_location='cpu',weights_only=True);net=LifetimeNet(88,'ple',c['bins'],1).cuda();net.load_state_dict(c['state_dict']);ids=np.flatnonzero(split==3);pred=predict(ids);truth=y[ids]*100
        from sklearn.metrics import r2_score
        metrics={'mae':float(np.mean(abs(pred-truth))),'rmse':float(np.sqrt(np.mean((pred-truth)**2))),'r2':float(r2_score(truth,pred)),'rows':len(ids)}
        c['metrics']=metrics;torch.save(c,folder/'tank_health.pt');pd.DataFrame({'gun_id':t[ids,11].astype(int),'seq':t[ids,13].astype(int),'pred_tank_health':pred}).to_parquet(folder/'tank_evaluation.parquet',index=False)
        (folder/'tank_summary.json').write_text(json.dumps({'test':metrics,'validation_mae':best,'seed':winner,'resources':PEAK,'candidates':candidates},indent=2),encoding='utf8');print('FINAL TANK',metrics,flush=True);resources('ready','tank_health')
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':
    import sys
    train_auxiliary() if '--tank' in sys.argv else run()
