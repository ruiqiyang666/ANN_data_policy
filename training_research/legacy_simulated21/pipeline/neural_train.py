"""Resource bounded, whole-gun holdout neural experiments, using existing 1 Hz data."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import argparse,gc,json,math,time,sys,subprocess,hashlib
from pathlib import Path
# Windows CUDA DLLs must initialize before PyArrow's native dependencies.
import torch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import psutil
from torch.nn import functional as F
from sklearn.metrics import mean_absolute_error,r2_score,f1_score,confusion_matrix,roc_auc_score
from neural_net import MultiTaskNet,ARCHS,SCALE,decode,normalize_inputs
from core import ROOT,NUMERIC,STATES,REG_TARGETS,feature_rows,dump

OUT=ROOT/'results'/'neural'
COLS=REG_TARGETS+['y_anomaly','y_next_anomaly_300s','y_anomaly_eta_s','y_rul_valid','y_pressure_valid','y_anomaly_valid','gun_id','local_index','seq','state_index','y_root_parameter']
PEAK={'rss_gb':0.,'gpu_allocated_gb':0.,'gpu_reserved_gb':0.,'gpu_temp_c':0,'minimum_available_ram_gb':100.}
def resources(stage='',arch='',epoch=0):
    ram=psutil.virtual_memory();rss=psutil.Process().memory_info().rss/2**30
    PEAK['rss_gb']=max(PEAK['rss_gb'],rss);PEAK['minimum_available_ram_gb']=min(PEAK['minimum_available_ram_gb'],ram.available/2**30)
    allocated=torch.cuda.memory_allocated()/2**30;reserved=torch.cuda.memory_reserved()/2**30
    PEAK['gpu_allocated_gb']=max(PEAK['gpu_allocated_gb'],allocated);PEAK['gpu_reserved_gb']=max(PEAK['gpu_reserved_gb'],reserved)
    try:
        v=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,memory.free','--format=csv,noheader,nounits'],text=True,timeout=5,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).strip().split(',')
        temp=int(v[0]);free=int(v[1])/1024;PEAK['gpu_temp_c']=max(PEAK['gpu_temp_c'],temp)
    except (OSError,ValueError,subprocess.SubprocessError):temp=None;free=None
    info={'stage':stage,'architecture':arch,'epoch':epoch,'rss_gb':round(rss,2),'available_ram_gb':round(ram.available/2**30,2),'gpu_allocated_gb':round(allocated,2),'gpu_reserved_gb':round(reserved,2),'gpu_temp_c':temp,'updated':time.time()}
    dump(OUT/'progress.json',info)
    if ram.available<5*2**30 or rss>10 or (free is not None and free<2):raise RuntimeError('Resource guard stopped safely before exhaustion: '+json.dumps(info))
    if temp is not None and temp>=83:
        print('Thermal guard: cooling pause',flush=True);time.sleep(15)
        if temp>=87:raise RuntimeError('GPU temperature guard stopped training safely.')
    return info

def prepare():
    manifest=json.loads((ROOT/'results'/'数据清单.json').read_text(encoding='utf8'))
    groups=[g for g in manifest['guns'] if g['split']!='replay']
    cal=[g['gun_id'] for g in groups if g['split']=='calibration']
    validation=set(cal[::2]);calibration=set(cal[1::2])
    splits={s:[] for s in ['train','validation','calibration','test']}
    for g in groups:splits['validation' if g['gun_id'] in validation else 'calibration' if g['gun_id'] in calibration else g['split']].append(g['gun_id'])
    source=ROOT/'results'/'模拟数据.parquet'
    identity={'size':source.stat().st_size,'mtime_ns':source.stat().st_mtime_ns,'sampling':1,'window':60,'stride':60,'splits':splits,'cache_version':1}
    if (OUT/'cache.json').exists():
        c=json.loads((OUT/'cache.json').read_text(encoding='utf8'))
        if c['identity']==identity and all((OUT/(s+'.npy')).exists() for s in ['sequence','flat','targets','split']):return c
        raise RuntimeError('Existing cache differs; explicit cache review required.')
    pf=pq.ParquetFile(source);sums=np.zeros(88);squares=np.zeros(88);count=0;total=0
    for g in groups:
        idx=np.arange(59,g['rows']-60,60);total+=len(idx)
        if g['gun_id'] not in splits['train']:continue
        f=pf.read_row_group(g['row_group']).to_pandas();x=feature_rows(f,idx).astype(np.float64)
        sums+=x.sum(0);squares+=(x*x).sum(0);count+=len(x)
        resources('scaling');del f,x
    mean=sums/count;std=np.sqrt(np.maximum(squares/count-mean**2,0));std=np.maximum(std,.01)
    stats={'mean':mean.tolist(),'std':std.tolist()}
    seq=np.lib.format.open_memmap(OUT/'sequence.npy',mode='w+',dtype='float16',shape=(total,60,30))
    flat=np.lib.format.open_memmap(OUT/'flat.npy',mode='w+',dtype='float32',shape=(total,88))
    target=np.lib.format.open_memmap(OUT/'targets.npy',mode='w+',dtype='float64',shape=(total,len(COLS)))
    split=np.lib.format.open_memmap(OUT/'split.npy',mode='w+',dtype='uint8',shape=(total,))
    offset=0
    for g in groups:
        f=pf.read_row_group(g['row_group']).to_pandas();idx=np.arange(59,len(f)-60,60)
        x=feature_rows(f,idx);a=f[NUMERIC].to_numpy(np.float32)
        st=np.stack([(f.operation_state==s).to_numpy(np.float32) for s in STATES],1)
        msg=f.machine_message.fillna('').astype(str).str.startswith('SIM_STATE_').to_numpy(np.float32)[:,None]
        channels=np.concatenate([a,st,msg],1);raw=channels[idx[:,None]-np.arange(59,-1,-1)]
        q,z=normalize_inputs(raw,x,stats);n=len(idx);end=offset+n
        seq[offset:end]=q;flat[offset:end]=z
        f['local_index']=np.arange(len(f));f['state_index']=f.operation_state.map({s:i for i,s in enumerate(STATES)})
        target[offset:end]=f.iloc[idx][COLS].to_numpy(float)
        split[offset:end]=next(i for i,(name,guns) in enumerate(splits.items()) if g['gun_id'] in guns)
        offset=end
        if g['gun_id']%10==0:print('Cached gun',g['gun_id'],'windows',offset,flush=True);resources('caching')
        del f,x,a,st,msg,channels,raw,q,z;gc.collect()
    for a in [seq,flat,target,split]:a.flush()
    c={'identity':identity,'stats':stats,'columns':COLS,'rows':total,'train_windows':count,'splits':splits}
    dump(OUT/'cache.json',c);return c

def open_data():return [np.load(OUT/(n+'.npy'),mmap_mode='r') for n in ['sequence','flat','targets','split']]
def raw_predict(net,arrays,ids,batch=256):
    seq,flat,_,_=arrays;pred=[];net.eval()
    with torch.inference_mode():
        for begin in range(0,len(ids),batch):
            ix=ids[begin:begin+batch]
            q=torch.from_numpy(np.array(seq[ix],dtype=np.float32)).cuda();x=torch.from_numpy(np.array(flat[ix])).cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):o=net(q,x)
            pred.append(o.float().cpu().numpy())
    return np.concatenate(pred)

def metrics(raw,t,pressure):
    pred,prob,future,eta=decode(raw,pressure);result={'regression':{}}
    for j,key in enumerate(REG_TARGETS):
        valid=t[:,8].astype(bool) if j<2 else t[:,9].astype(bool) if j==4 else np.ones(len(t),bool)
        valid &= np.isfinite(t[:,j])
        y=t[valid,j];p=pred[valid,j]
        result['regression'][key]={'mae':float(np.mean(abs(y-p))),'rmse':float(np.sqrt(np.mean((y-p)**2))),'r2':float(r2_score(y,p)),'rows':int(valid.sum())}
    for name,truth,p,valid in [('current',t[:,5].astype(int),prob,np.ones(len(t),bool)),('future',t[:,6].astype(int),future,t[:,10].astype(bool))]:
        y=truth[valid];p=p[valid];labels=p.argmax(1);normal=y==0;binary=(y!=0).astype(int);risk=1-p[:,0]
        result[name]={'accuracy':float(np.mean(labels==y)),'macro_f1':float(f1_score(y,labels,labels=np.arange(6),average='macro',zero_division=0)),'false_positive_rate':float(np.mean(labels[normal]!=0)),'brier':float(np.mean((risk-binary)**2)),'auc':float(roc_auc_score(binary,risk)) if len(set(binary))>1 else None,'confusion':confusion_matrix(y,labels,labels=np.arange(6)).tolist(),'rows':len(y)}
    ev=np.isfinite(t[:,7])&(t[:,5]==0)&t[:,10].astype(bool)
    result['eta']={'mae_seconds':float(np.mean(abs(eta[ev]-t[ev,7]))) if ev.any() else None,'rows':int(ev.sum()),'scope':'Only current-normal windows with a future event; not an event detection score.'}
    result['pressure_persistence_mae']=float(np.mean(abs(t[t[:,9].astype(bool),4]-pressure[t[:,9].astype(bool)])))
    r=result['regression'];result['selection_score']=float((r['y_rul_hours']['mae']/5+r['y_rul_pieces']['mae']/400+r['y_gun_health']['mae']/12+r['y_tank_health']['mae']/6+r['y_pressure_60s']['mae']/.06)/5+(1-result['current']['macro_f1'])+result['future']['brier']+((result['eta']['mae_seconds'] or 0)/300)*.2)
    return result

def train_one(arch,arrays,cache,epochs,batch):
    torch.manual_seed(20261007);np.random.seed(20261007)
    seq,flat,targets,split=arrays;ids=np.flatnonzero(split==0);validation=np.flatnonzero(split==1)
    t=np.array(targets);pressure=flat[:,14]*cache['stats']['std'][14]+cache['stats']['mean'][14]
    net=MultiTaskNet(arch).cuda();optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0005)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=.00005)
    counts=np.bincount(t[ids,5].astype(int),minlength=6);weights=np.sqrt(counts.sum()/np.maximum(counts,1));weights/=weights.mean()
    classweights=torch.tensor(weights,dtype=torch.float32,device='cuda')
    best=math.inf;bad=0;history=[];start=time.time();checkpoint=OUT/(arch+'.pt')
    for epoch in range(1,epochs+1):
        order=np.random.default_rng(20261007+epoch).permutation(ids);net.train();loss_sum=0.;seen=0
        for begin in range(0,len(order),batch):
            ix=order[begin:begin+batch];y=np.array(t[ix],dtype=np.float32);reg=y[:,:5].copy();reg[:,4]-=pressure[ix];reg/=SCALE
            valid=np.isfinite(reg);valid[:,:2]&=y[:,8:9].astype(bool);valid[:,4]&=y[:,9].astype(bool)
            yr=torch.from_numpy(np.nan_to_num(reg)).cuda();mask=torch.from_numpy(valid).cuda()
            q=torch.from_numpy(np.array(seq[ix],dtype=np.float32)).cuda();x=torch.from_numpy(np.array(flat[ix])).cuda()
            current=torch.from_numpy(y[:,5].astype(np.int64)).cuda();future=torch.from_numpy(y[:,6].astype(np.int64)).cuda();fv=torch.from_numpy(y[:,10].astype(bool)).cuda()
            eta_mask=torch.from_numpy(np.isfinite(y[:,7])&(y[:,5]==0)&y[:,10].astype(bool)).cuda();eta=torch.from_numpy(np.nan_to_num(y[:,7])/300).cuda()
            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    o=net(q,x);lr=(F.smooth_l1_loss(o[:,:5],yr,reduction='none')*mask).sum()/mask.sum().clamp_min(1)
                    lc=F.cross_entropy(o[:,5:11],current,weight=classweights)
                    lf=F.cross_entropy(o[:,11:17][fv],future[fv]) if fv.any() else o.sum()*0
                    le=F.smooth_l1_loss(o[:,17].sigmoid()[eta_mask],eta[eta_mask]) if eta_mask.any() else o.sum()*0
                    loss=lr*4+lc+lf*.7+le*.5
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
            except torch.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache();raise RuntimeError('CUDA allocation guard: stopped safely; retry with smaller batch after reviewing resource log.')
            loss_sum+=float(loss.detach())*len(ix);seen+=len(ix)
            if begin%(batch*150)==0:resources('training',arch,epoch)
        raw=raw_predict(net,arrays,validation,batch);score=metrics(raw,t[validation],pressure[validation]);schedule.step()
        history.append({'epoch':epoch,'loss':loss_sum/seen,'validation':score,'seconds':time.time()-start})
        print(arch,'epoch',epoch,'loss',round(loss_sum/seen,4),'val score',round(score['selection_score'],4),'RUL MAE',round(score['regression']['y_rul_hours']['mae'],3),'F1',round(score['current']['macro_f1'],3),flush=True)
        if score['selection_score']<best-.0001:
            best=score['selection_score'];bad=0
            torch.save({'state_dict':{k:v.detach().cpu() for k,v in net.state_dict().items()},'arch':arch,'epoch':epoch,'stats':cache['stats'],'validation':score},checkpoint)
        else:bad+=1
        dump(OUT/(arch+'.json'),{'architecture':ARCHS[arch],'parameters':sum(p.numel() for p in net.parameters()),'history':history,'seconds':time.time()-start,'best_score':best})
        resources('validation',arch,epoch)
        if bad>=4:break
    del net,optimizer;gc.collect();torch.cuda.empty_cache()
    return {'id':arch,'name':ARCHS[arch],'validation_score':best,'seconds':time.time()-start,'epochs':len(history)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=16);p.add_argument('--batch',type=int,default=256);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    OUT.mkdir(exist_ok=True);lock=OUT/'training.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        torch.set_num_threads(4);torch.set_num_interop_threads(2)
        assert torch.cuda.is_available(),'CUDA required for this bounded experiment'
        torch.cuda.set_per_process_memory_fraction(.5);torch.backends.cuda.matmul.allow_tf32=True
        resources('preparing');cache=prepare();arrays=open_data()
        if a.smoke:
            ix=np.flatnonzero(arrays[3]==0)[:32]
            for arch in ARCHS:
                net=MultiTaskNet(arch).cuda();raw=raw_predict(net,arrays,ix,32)
                assert raw.shape==(len(ix),18) and np.isfinite(raw).all()
                q=torch.tensor(np.array(arrays[0][ix],dtype=np.float32),device='cuda');x=torch.tensor(np.array(arrays[1][ix]),device='cuda')
                net.train();loss=net(q,x).square().mean();loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
                print('Smoke passed',arch,resources('smoke',arch),flush=True);del net,q,x,loss;gc.collect();torch.cuda.empty_cache()
            dump(OUT/'smoke.json',{'passed':True,'resources':PEAK});return
        assert (OUT/'smoke.json').exists(),'Run smoke first'
        results=[]
        for arch in ARCHS:results.append(train_one(arch,arrays,cache,a.epochs,a.batch))
        # The winner is frozen using validation before touching test targets.
        selected=min(results,key=lambda r:r['validation_score'])['id'];dump(OUT/'selection.json',{'selected':selected,'models':results,'criterion':'Fixed multitask validation score; test not used for selection.'})
        t=np.array(arrays[2]);pressure=arrays[1][:,14]*cache['stats']['std'][14]+cache['stats']['mean'][14]
        calibration=np.flatnonzero(arrays[3]==2);test=np.flatnonzero(arrays[3]==3)
        for r in results:
            c=torch.load(OUT/(r['id']+'.pt'),map_location='cpu',weights_only=True);net=MultiTaskNet(r['id']).cuda();net.load_state_dict(c['state_dict'])
            raw=raw_predict(net,arrays,test);r['test']=metrics(raw,t[test],pressure[test]);r['best_epoch']=c['epoch']
            if r['id']==selected:
                cr=raw_predict(net,arrays,calibration);cp=decode(cr,pressure[calibration])[0];cv=t[calibration,8].astype(bool)
                residual=abs(cp[cv,:2]-t[calibration[cv],:2]);level=min(1,math.ceil((len(residual)+1)*.9)/len(residual));half=np.quantile(residual,level,axis=0,method='higher')
                pred,prob,future,eta=decode(raw,pressure[test]);tv=t[test,8].astype(bool)
                c.update({'conformal':half.tolist(),'model_name':ARCHS[r['id']],'version':'p2-neural-20261007-'+r['id']})
                torch.save(c,OUT/'active.pt')
                r['range_coverage']=np.mean(abs(pred[tv,:2]-t[test[tv],:2])<=half,axis=0).tolist()
                ev=pd.DataFrame(t[test],columns=COLS)
                for j,k in enumerate(REG_TARGETS):ev['pred_'+k[2:]]=pred[:,j]
                ev['pred_anomaly']=prob.argmax(1);ev['anomaly_probability']=1-prob[:,0];ev['pred_next_anomaly']=future.argmax(1);ev['forecast_probability']=1-future[:,0];ev['pred_eta_seconds']=eta
                ev.to_parquet(OUT/'evaluation.parquet',index=False)
            del net;gc.collect();torch.cuda.empty_cache();resources('testing',r['id'])
        summary={'selected':selected,'name':ARCHS[selected],'version':'p2-neural-20261007-'+selected,'models':results,'splits':cache['splits'],'windows':{name:int(np.sum(arrays[3]==i)) for i,name in enumerate(cache['splits'])},'input_parameters':21,'sampling_seconds':1,'window_seconds':60,'training_stride_seconds':60,'resources':PEAK,'selection':'Validation only; disjoint whole-gun calibration and test sets.','hardware':{'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cpu_threads':4,'batch':a.batch},'limitations':['Synthetic-only evaluation; no production accuracy claim.','Architectures are task adaptations, not exact reproductions of published benchmarks.','Future events partly stochastic in the simulator; uncertainty is irreducible.']}
        dump(OUT/'summary.json',summary);resources('ready',selected);print(json.dumps(summary,ensure_ascii=False),flush=True)
    finally:
        lock.unlink(missing_ok=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():dump(OUT/'progress.json',{'stage':'stopped','error':str(e),'resources':PEAK,'updated':time.time()})
        raise
