"""Fixed-weight lifetime ensemble study; never replaces the active checkpoint."""
import torch
import numpy as np,json,time
from lifecycle_neural import LifetimeNet
from core import ROOT
from neural_train import resources,PEAK

def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.cuda.set_per_process_memory_fraction(.5)
    folder=ROOT/'results/neural';lock=folder/'training.lock'
    with lock.open('x') as f:f.write('ensemble study')
    try:
        flat=np.load(folder/'flat.npy',mmap_mode='r');h=np.load(folder/'history.npy',mmap_mode='r');t=np.load(folder/'targets.npy',mmap_mode='r');sp=np.load(folder/'split.npy',mmap_mode='r')
        groups={name:np.flatnonzero((sp==code)&t[:,8].astype(bool)) for name,code in [('validation',1),('calibration',2),('test',3)]}
        names=['history_mlp','history_ple','history_ple_seed2'];preds={}
        for name in names:
            c=torch.load(folder/(name+'.pt'),map_location='cpu',weights_only=True);net=LifetimeNet(c['width'],c['kind'],c['bins']).cuda();net.load_state_dict(c['state_dict']);net.eval();preds[name]={}
            for group,ids in groups.items():
                values=[]
                with torch.inference_mode():
                    for start in range(0,len(ids),1024):
                        ix=ids[start:start+1024];z=np.clip((h[ix]-c['history_mean'])/c['history_std'],-12,12);x=np.c_[flat[ix],z].astype(np.float32)
                        values.append(np.maximum(0,net(torch.tensor(x).cuda()).cpu().numpy()[:,0]*35))
                preds[name][group]=np.concatenate(values)
            resources('ensemble_evaluation',name);del net;torch.cuda.empty_cache()
        y=np.asarray(t[groups['validation'],0]);grid=[]
        for i in range(5):
            for j in range(5-i):
                w=np.array([i,j,4-i-j])/4;p=sum(w[k]*preds[n]['validation'] for k,n in enumerate(names));grid.append({'weights':w.tolist(),'validation_mae':float(abs(p-y).mean())})
        best=min(grid,key=lambda r:r['validation_mae']);selection={'models':names,'selected':best,'grid':grid,'criterion':'validation RUL MAE; weights grid fixed at quarters','test_seen_during_selection':False}
        (folder/'ensemble_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf8')
        w=best['weights'];blend={g:sum(w[k]*preds[n][g] for k,n in enumerate(names)) for g in groups};test=groups['test'];truth=np.asarray(t[test,0]);base=preds['history_ple']['test'];err=abs(blend['test']-truth);baseerr=abs(base-truth)
        calerr=abs(blend['calibration']-t[groups['calibration'],0]);half=float(np.quantile(calerr,.9));guns=t[test,11].astype(int);unique=np.unique(guns)
        paired=np.array([np.mean(err[guns==g]-baseerr[guns==g]) for g in unique]);boot=np.random.default_rng(711).choice(paired,(5000,len(paired)),replace=True).mean(1)
        result={**selection,'test_mae':float(err.mean()),'current_model_test_mae':float(baseerr.mean()),'gun_mean_mae_change':float(paired.mean()),'paired_gun_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist(),'range_half_width_h':half,'range_test_coverage':float(np.mean(err<=half)),'near_end_mae':float(err[truth<=4].mean()),'resources':PEAK,'active_weights_changed':False,'limitations':['Previously inspected synthetic test set reused.','Bootstrap is descriptive, not an independent external validation.']}
        (folder/'ensemble_study.json').write_text(json.dumps(result,indent=2),encoding='utf8');resources('ready','ensemble_study');print(json.dumps(result),flush=True)
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
