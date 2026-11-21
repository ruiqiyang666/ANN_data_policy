"""Run baselines, causal self-supervision and TCN adaptation on local CUDA.

Windows: import torch before pandas to avoid the reproduced c10.dll import-order
failure in the existing environment. All fitted artifacts stay in the run dir.
"""
import sys
sys.dont_write_bytecode=True
import torch
import argparse,json,math,time,random
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,log_loss
import joblib
from common import ROOT,DATA,MODEL,EVAL,LOG,dump,sha,status,ensure_dirs
from modeling import RealTCN,CachedData,multitask_loss,hazard_distribution,weighted_stratified_ids

def seed_all(seed):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def score(samples,prob):
    eligible=samples.event_in_300s.to_numpy()|(samples.observed_seconds.to_numpy()>=300)
    y=samples.event_in_300s.to_numpy()[eligible];p=np.clip(prob[eligible],1e-8,1-1e-8)
    return dict(n=int(len(y)),positives=int(y.sum()),prevalence=float(y.mean()),
        average_precision=float(average_precision_score(y,p)) if y.any() else None,
        roc_auc=float(roc_auc_score(y,p)) if y.any() and (~y).any() else None,
        brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p,labels=[False,True])))

@torch.no_grad()
def predict(model,cache,ids,batch_size=256):
    model.eval();arrays={k:[] for k in ['risk','eta','type_probability','current_probability','embedding']}
    for start in range(0,len(ids),batch_size):
        x,stats,_=cache.batch(ids[start:start+batch_size])
        with torch.autocast('cuda',dtype=torch.bfloat16):out=model(x,stats)
        p,eta,_=hazard_distribution(out['hazard'])
        values={'risk':p,'eta':eta,'type_probability':torch.sigmoid(out['type'].float()),
                'current_probability':torch.sigmoid(out['current'].float()),'embedding':out['embedding'].float()}
        for k,v in values.items():arrays[k].append(v.cpu().numpy())
    return {k:np.concatenate(v) for k,v in arrays.items()}

def save_checkpoint(model,path,**metadata):
    torch.save({'state_dict':model.state_dict(),'stats_dim':model.stats_dim,**metadata},path)

def load_model(path,device='cuda'):
    state=torch.load(path,map_location=device,weights_only=True)
    model=RealTCN(state['stats_dim']).to(device);model.load_state_dict(state['state_dict'])
    return model

def baselines(cache,train,selection):
    if (MODEL/'baseline_metrics.json').exists():return
    x=np.asarray(cache.stats);y=cache.samples.event_in_300s.to_numpy()
    full=(cache.samples.observed_seconds.to_numpy()>=300)|y
    tr=train[full[train]];sel=selection[full[selection]]
    results={}
    for name,model in [('logistic',LogisticRegression(C=.1,max_iter=500)),
                       ('hist_tree',HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,max_depth=3,l2_regularization=10,learning_rate=.05,early_stopping=False,random_state=42))]:
        started=time.perf_counter();model.fit(x[tr],y[tr])
        prob=model.predict_proba(x[selection])[:,1]
        joblib.dump(model,MODEL/f'{name}.joblib')
        results[name]={**score(cache.samples.iloc[selection],prob),'seconds':time.perf_counter()-started}
        np.savez(EVAL/f'{name}_selection.npz',ids=selection,risk=prob)
        status('baseline_fitted',model=name,selection=results[name])
    dump(results,MODEL/'baseline_metrics.json')

def reconstruction_batch(model,cache,ids,rng):
    x,_,_=cache.batch(ids)
    # Short blocks within historical data only; invalid source cells never supervise.
    blocks=rng.random((len(ids),1,15))<.15
    if not blocks.any():blocks[0,0,7]=True
    m=torch.from_numpy(np.repeat(blocks,8,axis=2)).to('cuda')
    supervise=m&x[:,28:].bool()
    target=x[:,:28].clone()
    masked=x.clone();masked[:,:28]=masked[:,:28].masked_fill(m,0)
    masked[:,28:]=masked[:,28:].masked_fill(m,0)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        sequence,_=model.encode(masked);recon=model.reconstruction(sequence)
    loss=torch.nn.functional.smooth_l1_loss(recon.float(),target,reduction='none')
    return (loss*supervise).sum()/supervise.sum().clamp_min(1)

def pretrain(cache,train,max_epochs=20,batch_size=128):
    path=MODEL/'self_supervised_encoder.pt'
    if path.exists() and (MODEL/'self_supervised_metrics.json').exists():return
    seed_all(42);rng=np.random.default_rng(42)
    times=cache.samples.timestamp.astype('int64').to_numpy()//10**9
    cut=int(np.quantile(times[train],.9))
    fit=train[times[train]<cut];valid=train[times[train]>cut+119]
    valid=rng.choice(valid,min(2048,len(valid)),replace=False)
    model=RealTCN(cache.stats.shape[1]).cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    best=float('inf');stale=0;history=[]
    for epoch in range(max_epochs):
        started=time.perf_counter();model.train();order=rng.permutation(fit);losses=[]
        for a in range(0,len(order),batch_size):
            opt.zero_grad(set_to_none=True);loss=reconstruction_batch(model,cache,order[a:a+batch_size],rng)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite pretraining loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
        model.eval();vrng=np.random.default_rng(17);val=[]
        with torch.no_grad():
            for a in range(0,len(valid),batch_size):val.append(float(reconstruction_batch(model,cache,valid[a:a+batch_size],vrng)))
        metric=float(np.mean(val));entry=dict(epoch=epoch+1,train_loss=float(np.mean(losses)),validation_reconstruction_loss=metric,
                seconds=time.perf_counter()-started,peak_gpu_MiB=torch.cuda.max_memory_allocated()/1024**2)
        history.append(entry);status('self_supervised_pretraining',**entry)
        dump(history,LOG/'self_supervised_epochs.json')
        if metric<best-1e-5:
            best=metric;stale=0;save_checkpoint(model,path,stage='self_supervised',epoch=epoch+1,seed=42)
        else:stale+=1
        if stale>=5:break
    dump({'epochs':history,'train_windows':len(fit),'monitor_windows':len(valid),'monitor_partition':'internal training time block',
          'best_loss':best,'parameters':sum(p.numel() for p in model.parameters())},MODEL/'self_supervised_metrics.json')
    del model,opt;torch.cuda.empty_cache()

def train_model(name,cache,train,selection,pretrained,seed=42,max_epochs=30,batch_size=128):
    path=MODEL/f'{name}.pt';metrics_path=MODEL/f'{name}_metrics.json'
    if path.exists() and metrics_path.exists():return json.loads(metrics_path.read_text('utf-8'))
    seed_all(seed);rng=np.random.default_rng(seed)
    model=RealTCN(cache.stats.shape[1]).cuda()
    if pretrained:
        state=torch.load(MODEL/'self_supervised_encoder.pt',map_location='cuda',weights_only=True)['state_dict']
        model.load_state_dict({k:v for k,v in state.items() if k.startswith(('input_projection.','blocks.'))},strict=False)
    pos=cache.targets['hazard_y'][train].sum(axis=0)
    tot=cache.targets['hazard_mask'][train].sum(axis=0)
    prior=np.clip(pos/np.maximum(tot,1),1e-5,.1)
    with torch.no_grad():model.hazard.bias.copy_(torch.from_numpy(np.log(prior/(1-prior))).cuda())
    encoder_params=list(model.input_projection.parameters())+list(model.blocks.parameters())
    enc_ids={id(p) for p in encoder_params}
    head_params=[p for p in model.parameters() if id(p) not in enc_ids]
    opt=torch.optim.AdamW([{'params':encoder_params,'lr':1e-4 if pretrained else 1e-3},
                          {'params':head_params,'lr':1e-3}],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=3)
    event=cache.samples.event_in_300s.to_numpy()
    best=-1;stale=0;history=[];steps=math.ceil(len(train)/batch_size)
    for epoch in range(max_epochs):
        start=time.perf_counter();model.train();losses=[]
        for step in range(steps):
            ids,importance=weighted_stratified_ids(train,event,batch_size,rng)
            x,stats,target=cache.batch(ids);opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(x,stats)
            loss=multitask_loss(out,target,torch.from_numpy(importance).cuda())
            if not torch.isfinite(loss):raise RuntimeError(f'Non-finite loss: {name}')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            if not torch.isfinite(norm):raise RuntimeError('Non-finite gradient')
            opt.step();losses.append(float(loss.detach()))
            if epoch==0 and step==119:
                torch.cuda.synchronize();status('training_throughput',model=name,steps=120,seconds=time.perf_counter()-start,
                    peak_gpu_MiB=torch.cuda.max_memory_allocated()/1024**2)
        pred=predict(model,cache,selection);metrics=score(cache.samples.iloc[selection],pred['risk'])
        ap=metrics['average_precision'] or 0.;scheduler.step(ap)
        entry=dict(epoch=epoch+1,loss=float(np.mean(losses)),selection=metrics,seconds=time.perf_counter()-start,
                   peak_gpu_MiB=torch.cuda.max_memory_allocated()/1024**2)
        history.append(entry);status('supervised_training',model=name,**entry);dump(history,LOG/f'{name}_epochs.json')
        if ap>best+1e-5:
            best=ap;stale=0;save_checkpoint(model,path,model_name=name,epoch=epoch+1,seed=seed,pretrained=pretrained)
            np.savez(EVAL/f'{name}_selection.npz',ids=selection,**pred)
        else:stale+=1
        if stale>=5:break
    result={'name':name,'seed':seed,'pretrained':pretrained,'epochs':history,'best_selection_AP':best,
            'parameter_count':sum(p.numel() for p in model.parameters()),'checkpoint_sha256':sha(path),
            'probability_calibration':'not independently calibrated; no independent calibration group remained',
            'initial_pretraining_seed':42 if pretrained else None}
    dump(result,metrics_path);del model,opt;torch.cuda.empty_cache();return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--extra-seeds',action='store_true');args=parser.parse_args()
    ensure_dirs();torch.set_num_threads(8);torch.set_num_interop_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('Expected local CUDA GPU not available')
    cache=CachedData(DATA);s=cache.samples
    train=np.flatnonzero(s.partition.to_numpy()==0);selection=np.flatnonzero(s.partition.to_numpy()==1)
    readiness=json.loads((DATA/'windows_ready.json').read_text('utf-8'))
    config={'run':'20261104_v1','models':['b0_tcn','b1_ssl_tcn'],'batch_size':128,'max_epochs':30,'pretraining_max_epochs':20,
        'seed':42,'extra_adaptation_seeds':[43,44] if args.extra_seeds else [],'pretraining_seed':42,
        'input_fields':28,'history_seconds':120,'future_seconds':300,'bin_seconds':30,
        'loss':'masked survival likelihood with inverse stratified-sampling weights; conditional type BCE; current BCE',
        'split':readiness,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'environment':'isolated venv entry; existing system site packages read-only',
        'test_accessed_for_selection':False,'pressure_forecast':False}
    dump(config,MODEL/'training_config.json');status('training_started',gpu=config['gpu'],train_windows=len(train),selection_windows=len(selection))
    baselines(cache,train,selection)
    b0=train_model('b0_tcn',cache,train,selection,False)
    pretrain(cache,train)
    b1=train_model('b1_ssl_tcn',cache,train,selection,True)
    selected='b1_ssl_tcn' if b1['best_selection_AP']>b0['best_selection_AP'] else 'b0_tcn'
    dump({'selected':selected,'basis':'seed42 selection AP; no test access','b0_AP':b0['best_selection_AP'],'b1_AP':b1['best_selection_AP']},MODEL/'selected_model.json')
    if args.extra_seeds:
        for seed in [43,44]:train_model(f'b1_ssl_tcn_seed{seed}',cache,train,selection,True,seed=seed)
    status('neural_training_complete',selected=selected,final_test_pending=True)

if __name__=='__main__':main()
