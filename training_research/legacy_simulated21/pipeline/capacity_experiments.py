"""Bounded causal capacity-head experiments. Never mutates serving checkpoints."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import torch
import numpy as np
import json,time,gc,hashlib
from lifecycle_neural import LifetimeNet
from neural_train import resources,PEAK
from core import ROOT

def run(regularized=False):
    torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.cuda.set_per_process_memory_fraction(.45)
    folder=ROOT/'results/neural';lock=folder/'training.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    started=time.time();tag='regularized' if regularized else 'capacity'
    try:
        t=np.load(folder/'targets.npy',mmap_mode='r');sp=np.load(folder/'split.npy',mmap_mode='r');flat=np.load(folder/'flat.npy',mmap_mode='r');h=np.load(folder/'history.npy',mmap_mode='r')
        ids={k:np.flatnonzero((sp==n)&t[:,8].astype(bool)) for k,n in [('train',0),('validation',1),('calibration',2),('test',3)]}
        c=torch.load(folder/'lifetime.pt',map_location='cpu',weights_only=True);hm=np.array(c['history_mean']);hs=np.array(c['history_std']);scales=np.array(c['scales'],np.float32)
        x=np.c_[flat,np.clip((h-hm)/hs,-12,12)].astype(np.float32)
        y=np.array(t[:,:3],np.float32);age=np.c_[h[:,1],h[:,2],np.zeros(len(h))].astype(np.float32)
        bins=np.array(c['bins'],np.float32)
        baseline=LifetimeNet(c['width'],c['kind'],c['bins']).cuda();baseline.load_state_dict(c['state_dict']);baseline.eval()
        def predict(net,ix,capacity=False):
            net.eval();out=[]
            with torch.inference_mode():
                for start in range(0,len(ix),1024):
                    batch=ix[start:start+1024];p=net(torch.from_numpy(x[batch]).cuda()).cpu().numpy()*scales
                    if capacity:p-=age[batch]
                    p=np.maximum(0,p);p[:,2]=np.minimum(p[:,2],100);out.append(p)
            return np.concatenate(out)
        def metrics(pred,ix):
            truth=y[ix];error=pred-truth
            return {'mae':np.mean(abs(error),axis=0).tolist(),'rmse':np.sqrt(np.mean(error**2,axis=0)).tolist(),'r2':(1-np.sum(error**2,axis=0)/np.sum((truth-truth.mean(0))**2,axis=0)).tolist(),'near_end_mae':float(abs(error[truth[:,0]<=4,0]).mean()),'rows':len(ix)}
        bv=metrics(predict(baseline,ids['validation']),ids['validation']);del baseline;torch.cuda.empty_cache()
        plan={'candidates':[{'name':'capacity_ple','capacity':True,'mse_weight':0.0},{'name':'capacity_balanced','capacity':True,'mse_weight':.4},{'name':'rul_balanced','capacity':False,'mse_weight':.4}], 'seed':20261008,'epochs':40,'patience':8,'validation_baseline':bv,'selection':'validation RUL MAE; validation RUL RMSE cannot exceed baseline; health MAE <= baseline * 1.03','release_gate':'selected candidate test MAE and RMSE must improve, health MAE <= baseline * 1.03, near-end MAE <= baseline * 1.10','test_previously_seen':True}
        if regularized:
            plan['candidates']=[{'name':'ple_noise','capacity':False,'mse_weight':0.,'noise':.04},{'name':'ple_consistent','capacity':False,'mse_weight':0.,'temporal':.5},{'name':'capacity_noise','capacity':True,'mse_weight':0.,'noise':.04},{'name':'ple_finetune','capacity':False,'mse_weight':0.,'noise':.02,'temporal':.5,'finetune':True}]
            plan['temporal_scope']='Previous training window from the same gun only; no validation/test windows used for training.'
        (folder/(tag+'_plan.json')).write_text(json.dumps(plan,indent=2),encoding='utf8');records=[];chosen=None;best=bv['mae'][0]
        for spec in plan['candidates']:
            name=spec['name'];torch.manual_seed(plan['seed']);net=LifetimeNet(x.shape[1],'ple',bins).cuda()
            if spec.get('finetune'):net.load_state_dict(c['state_dict'])
            opt=torch.optim.AdamW(net.parameters(),lr=.0001 if spec.get('finetune') else .0006,weight_decay=.01 if regularized else .002);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,40,eta_min=.00002)
            yy=(y+age if spec['capacity'] else y)/scales;local=1e9;bad=0;history=[];bestmetrics=None
            for epoch in range(1,41):
                net.train();order=np.random.default_rng(plan['seed']+epoch).permutation(ids['train']);loss_sum=0
                for start in range(0,len(order),512):
                    ix=order[start:start+512];xb=torch.from_numpy(x[ix]).cuda();yb=torch.from_numpy(yy[ix]).cuda();opt.zero_grad(set_to_none=True)
                    if spec.get('noise'):xb=xb+torch.randn_like(xb)*spec['noise']
                    out=net(xb);losses=torch.nn.functional.smooth_l1_loss(out,yb,reduction='none',beta=.04)+spec['mse_weight']*(out-yb).square()
                    loss=(losses*torch.tensor([1,.35,.15],device='cuda')).sum(1).mean()
                    if spec.get('temporal'):
                        previous=np.maximum(ix-1,0);valid=(sp[previous]==0)&(t[previous,11]==t[ix,11])&t[previous,8].astype(bool)
                        if valid.any():
                            prev=net(torch.from_numpy(x[previous[valid]]).cuda());delta=torch.from_numpy((yy[ix[valid]]-yy[previous[valid]])).cuda()
                            loss+=spec['temporal']*torch.nn.functional.smooth_l1_loss(out[torch.tensor(valid,device='cuda')]-prev,delta,beta=.02)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step();loss_sum+=float(loss.detach())*len(ix)
                    if start%(512*160)==0:resources('capacity_training',name,epoch)
                v=metrics(predict(net,ids['validation'],spec['capacity']),ids['validation']);sched.step();history.append({'epoch':epoch,'loss':loss_sum/len(order),'validation':v});print(name,epoch,'MAE',round(v['mae'][0],4),'RMSE',round(v['rmse'][0],4),flush=True)
                if v['mae'][0]<local-.001:
                    local=v['mae'][0];bad=0;bestmetrics=v
                    ck={**c,'state_dict':{k:v.detach().cpu() for k,v in net.state_dict().items()},'target_mode':'capacity' if spec['capacity'] else 'rul','name':name,'epoch':epoch,'validation_mae':local,'version':'capacity-20261007'}
                    torch.save(ck,folder/(name+'.pt'))
                else:bad+=1
                (folder/(tag+'_progress.json')).write_text(json.dumps({'model':name,'epoch':epoch,'completed':records,'history':history}),encoding='utf8')
                if bad>=8:break
            record={'name':name,'capacity':spec['capacity'],'validation':bestmetrics,'epochs':history};records.append(record)
            if local<best and bestmetrics['rmse'][0]<=bv['rmse'][0] and bestmetrics['mae'][2]<=bv['mae'][2]*1.03:best=local;chosen=record
            del net,opt;torch.cuda.empty_cache();gc.collect()
        selection={'selected':chosen['name'] if chosen else 'existing_lifetime','candidates':records,'criterion':plan['selection'],'frozen_before_test':True}
        (folder/(tag+'_selection.json')).write_text(json.dumps(selection,indent=2),encoding='utf8')
        result={**selection,'baseline_validation':bv,'active_weights_changed':False,'seconds':time.time()-started,'resources':PEAK}
        if chosen:
            old=LifetimeNet(c['width'],c['kind'],c['bins']).cuda();old.load_state_dict(c['state_dict']);bt=metrics(predict(old,ids['test']),ids['test']);del old
            ck=torch.load(folder/(chosen['name']+'.pt'),map_location='cpu',weights_only=True);net=LifetimeNet(ck['width'],ck['kind'],ck['bins']).cuda();net.load_state_dict(ck['state_dict'])
            pt=predict(net,ids['test'],chosen['capacity']);test=metrics(pt,ids['test']);pc=predict(net,ids['calibration'],chosen['capacity'])
            q=np.quantile(abs(pc[:,:2]-y[ids['calibration'],:2]),.9,axis=0);ck['conformal']=q.tolist();torch.save(ck,folder/(tag+'_candidate.pt'))
            result.update(test=test,baseline_test=bt,coverage=np.mean(abs(pt[:,:2]-y[ids['test'],:2])<=q,0).tolist(),release_gate_passed=test['mae'][0]<bt['mae'][0] and test['rmse'][0]<bt['rmse'][0] and test['mae'][2]<=bt['mae'][2]*1.03 and test['near_end_mae']<=bt['near_end_mae']*1.10)
            import pandas as pd
            ev=pd.DataFrame({'gun_id':t[ids['test'],11].astype(int),'seq':t[ids['test'],13].astype(int)})
            for j,key in enumerate(['rul_hours','rul_pieces','gun_health']):ev['y_'+key]=y[ids['test'],j];ev['pred_'+key]=pt[:,j]
            ev.to_parquet(folder/(tag+'_evaluation.parquet'),index=False)
        else:result['release_gate_passed']=False
        (folder/(tag+'_study.json')).write_text(json.dumps(result,indent=2),encoding='utf8');print('FINAL',json.dumps({k:v for k,v in result.items() if k!='candidates'}),flush=True)
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':
    import sys
    run('--regularized' in sys.argv)
