"""Validation-first blend of direct RUL and cumulative-capacity neural heads."""
import torch
import numpy as np,json,time
from lifecycle_neural import LifetimeNet
from neural_train import resources,PEAK
from core import ROOT

def run():
    torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.cuda.set_per_process_memory_fraction(.45)
    folder=ROOT/'results/neural';lock=folder/'training.lock'
    with lock.open('x') as f:f.write('capacity_blend')
    try:
        flat=np.load(folder/'flat.npy',mmap_mode='r');h=np.load(folder/'history.npy',mmap_mode='r');t=np.load(folder/'targets.npy',mmap_mode='r');sp=np.load(folder/'split.npy',mmap_mode='r')
        ids={k:np.flatnonzero((sp==v)&t[:,8].astype(bool)) for k,v in [('validation',1),('calibration',2),('test',3)]}
        def predict(name,split):
            c=torch.load(folder/(name+'.pt'),map_location='cpu',weights_only=True);net=LifetimeNet(c['width'],c['kind'],c['bins']).cuda();net.load_state_dict(c['state_dict']);net.eval();p=[]
            with torch.inference_mode():
                for start in range(0,len(ids[split]),1024):
                    ix=ids[split][start:start+1024];z=np.clip((h[ix]-c['history_mean'])/c['history_std'],-12,12).astype(np.float32);out=net(torch.tensor(np.c_[flat[ix],z]).cuda()).cpu().numpy()*c['scales']
                    if c.get('target_mode')=='capacity':out[:,:2]-=h[ix][:,[1,2]]
                    out=np.maximum(0,out);out[:,2]=np.minimum(100,out[:,2]);p.append(out)
            del net;torch.cuda.empty_cache();resources('capacity_blend',name);return np.concatenate(p)
        def metrics(p,split):
            y=t[ids[split],:3];err=p-y
            return {'mae':np.mean(abs(err),0).tolist(),'rmse':np.sqrt(np.mean(err**2,0)).tolist(),'r2':(1-np.sum(err**2,0)/np.sum((y-y.mean(0))**2,0)).tolist(),'near_end_mae':float(abs(err[y[:,0]<=4,0]).mean()),'rows':len(y)}
        names=['capacity_ple','capacity_noise'];alphas=[.25,.5,.75];plan={'experts':names,'weights':alphas,'criterion':'Validation RUL MAE, with RMSE and health nonregression; candidate set fixed before test scoring','test_previously_seen':True}
        (folder/'capacity_blend_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf8');base=predict('lifetime','validation');bm=metrics(base,'validation');best=bm['mae'][0];chosen=None;grid=[]
        for name in names:
            expert=predict(name,'validation')
            for alpha in alphas:
                m=metrics((1-alpha)*base+alpha*expert,'validation');record={'expert':name,'alpha':alpha,'metrics':m};grid.append(record)
                if m['mae'][0]<best and m['rmse'][0]<=bm['rmse'][0] and m['mae'][2]<=bm['mae'][2]*1.03:best=m['mae'][0];chosen=record
        selection={**plan,'selected':chosen,'grid':grid,'baseline_validation':bm,'frozen_before_test':True}
        (folder/'capacity_blend_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf8');result={**selection,'active_weights_changed':False,'release_gate_passed':False}
        if chosen:
            alpha=chosen['alpha'];test_base=predict('lifetime','test');p=(1-alpha)*test_base+alpha*predict(chosen['expert'],'test');bt=metrics(test_base,'test');mt=metrics(p,'test');cal=(1-alpha)*predict('lifetime','calibration')+alpha*predict(chosen['expert'],'calibration');q=np.quantile(abs(cal[:,:2]-t[ids['calibration'],:2]),.9,0)
            result.update(test=mt,baseline_test=bt,coverage=np.mean(abs(p[:,:2]-t[ids['test'],:2])<=q,0).tolist(),release_gate_passed=mt['mae'][0]<bt['mae'][0] and mt['rmse'][0]<bt['rmse'][0] and mt['mae'][2]<=bt['mae'][2]*1.03 and mt['near_end_mae']<=bt['near_end_mae']*1.10)
            c=torch.load(folder/'lifetime.pt',map_location='cpu',weights_only=True);c['ensemble_member']=torch.load(folder/(chosen['expert']+'.pt'),map_location='cpu',weights_only=True);c['ensemble_alpha']=alpha;c['conformal']=q.tolist();c['version']='capacity-blend-20261007';torch.save(c,folder/'capacity_blend_candidate.pt')
            import pandas as pd
            ev=pd.DataFrame({'gun_id':t[ids['test'],11].astype(int),'seq':t[ids['test'],13].astype(int)})
            for j,key in enumerate(['rul_hours','rul_pieces','gun_health']):ev['y_'+key]=t[ids['test'],j];ev['pred_'+key]=p[:,j]
            ev.to_parquet(folder/'capacity_blend_evaluation.parquet',index=False)
        result['resources']=PEAK;(folder/'capacity_blend_study.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result),flush=True)
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':run()
