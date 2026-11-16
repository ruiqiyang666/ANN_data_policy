"""Train all small candidates, calibrate/select on validation, freeze before testing."""
from pathlib import Path
import argparse,json,time,hashlib,random,platform
import numpy as np
import pandas as pd
import joblib,torch
from torch.utils.data import TensorDataset,DataLoader
from sklearn.linear_model import LogisticRegression
from modeling import TARGETS,RiskNet,summaries,margin_probabilities,apply_calibration
from evaluation import probability_metrics,endpoint_events,choose_threshold

ROOT=Path(__file__).resolve().parents[1]

def infer(net,x,batch=1024):
    net.eval();out=[]
    with torch.inference_mode():
        for start in range(0,len(x),batch):out.append(torch.sigmoid(net(torch.from_numpy(x[start:start+batch]))).numpy())
    return np.concatenate(out)

def neural(kind,x,y,train_idx,val_idx,cfg,directory,epochs,resume=False):
    directory.mkdir(parents=True,exist_ok=True)
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    net=RiskNet(kind,hidden=cfg['hidden']);optimizer=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    mask=np.isfinite(y[train_idx]);clean=np.nan_to_num(y[train_idx],nan=0)
    pos=(clean*mask).sum(axis=0);neg=((1-clean)*mask).sum(axis=0)
    weight=torch.tensor(np.clip(neg/np.maximum(pos,1),1,15),dtype=torch.float32)
    dataset=TensorDataset(torch.from_numpy(x[train_idx]),torch.from_numpy(clean),torch.from_numpy(mask.astype(np.float32)))
    generator=torch.Generator().manual_seed(cfg['seed'])
    loader=DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=True,generator=generator,num_workers=0)
    history=[];best=float('inf');bad=0;start_epoch=1
    if resume and (directory/'last.pt').exists():
        ck=torch.load(directory/'last.pt',weights_only=False,map_location='cpu')
        net.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);history=ck['history']
        best=ck['best'];bad=ck['bad'];start_epoch=ck['epoch']+1
        torch.set_rng_state(ck['rng']);generator.set_state(ck['loader_rng'])
    valmask=torch.from_numpy(np.isfinite(y[val_idx]).astype(np.float32))
    valy=torch.from_numpy(np.nan_to_num(y[val_idx],nan=0))
    valx=torch.from_numpy(x[val_idx])
    for epoch in range(start_epoch,epochs+1):
        start=time.perf_counter();net.train();loss_sum=0.;samples=0
        for bx,by,bm in loader:
            optimizer.zero_grad();logits=net(bx)
            losses=torch.nn.functional.binary_cross_entropy_with_logits(logits,by,pos_weight=weight,reduction='none')
            loss=(losses*bm).sum()/bm.sum().clamp_min(1)
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);optimizer.step()
            loss_sum+=loss.item()*len(bx);samples+=len(bx)
        net.eval()
        with torch.inference_mode():
            logits=net(valx)
            losses=torch.nn.functional.binary_cross_entropy_with_logits(logits,valy,pos_weight=weight,reduction='none')
            val_loss=float((losses*valmask).sum()/valmask.sum().clamp_min(1))
        record=dict(epoch=epoch,train_loss=loss_sum/samples,val_loss=val_loss,seconds=time.perf_counter()-start)
        history.append(record);print(kind,json.dumps(record),flush=True)
        if val_loss<best-1e-5:best=val_loss;bad=0;improved=True
        else:bad+=1;improved=False
        ck=dict(model=net.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,history=history,best=best,bad=bad,
                rng=torch.get_rng_state(),loader_rng=generator.get_state(),config=cfg,kind=kind)
        torch.save(ck,directory/'last.pt')
        if improved:torch.save(ck,directory/'best.pt')
        (directory/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        if bad>=cfg['patience']:break
    net.load_state_dict(torch.load(directory/'best.pt',weights_only=False,map_location='cpu')['model'])
    return net,history

def train_baselines(frame,x,y,train_idx):
    z=summaries(x);models=[]
    for j in range(len(TARGETS)):
        ix=train_idx[np.isfinite(y[train_idx,j])]
        if len(np.unique(y[ix,j]))<2:models.append(float(np.mean(y[ix,j])) if len(ix) else 0.)
        else:
            # Unweighted regularized logistic, no fit on validation or test.
            m=LogisticRegression(C=.15,max_iter=500,random_state=20261116)
            m.fit(z[ix],y[ix,j]);models.append(m)
    rates=np.nanmean(y[train_idx],axis=0)
    baselines=dict(logistic=models,event_rate=rates)
    predictions=dict(event_rate=np.tile(rates,(len(frame),1)),margin=margin_probabilities(frame,x),
        logistic=np.column_stack([m.predict_proba(z)[:,1] if hasattr(m,'predict_proba') else np.repeat(m,len(frame)) for m in models]))
    return baselines,predictions

def calibrate(p,y,val_idx):
    calibrators=[]
    for j in range(len(TARGETS)):
        ix=val_idx[np.isfinite(y[val_idx,j])]
        if len(np.unique(y[ix,j]))<2 or np.std(p[ix,j])<1e-7:
            calibrators.append(None);continue
        logits=np.log(np.clip(p[ix,j],1e-6,1-1e-6)/np.clip(1-p[ix,j],1e-6,1))[:,None]
        m=LogisticRegression(C=.1,max_iter=200)
        m.fit(logits,y[ix,j]);calibrators.append(m)
    return calibrators

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,default=ROOT)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    root=args.project;art=root/'artifacts';cfg=json.loads((root/'configs/train.json').read_text(encoding='utf-8'))
    torch.set_num_threads(cfg['threads'])
    data=np.load(art/'dataset.npz');x=data['x'];y=data['y']
    frame=pd.read_csv(art/'anchors.csv.gz',parse_dates=['timestamp'])
    cycles=pd.read_csv(art/'cycles.csv',parse_dates=['eol_endpoint'])
    products=pd.read_csv(art/'products.csv.gz',low_memory=False)
    train_idx=np.flatnonzero(frame.split.eq('train')&frame.timestamp.dt.second.eq(0))
    val_idx=np.flatnonzero(frame.split.eq('validation'))
    # Training sample order can shuffle within train; the split itself never shuffles.
    if args.smoke:
        train_idx=train_idx[np.linspace(0,len(train_idx)-1,min(len(train_idx),1800),dtype=int)]
        val_idx=val_idx[::10]
        net,h=neural('tcn',x,y,train_idx,val_idx,cfg,art/'smoke',2)
        estimate=sum(r['seconds'] for r in h)/len(h)*len(np.flatnonzero(frame.split.eq('train')&frame.timestamp.dt.second.eq(0)))/len(train_idx)*cfg['max_epochs']*2
        result=dict(status='只完成试运行，未冻结正式模型',history=h,estimated_full_two_models_seconds=estimate,
                    cpu=platform.processor(),torch=torch.__version__,cuda_available=torch.cuda.is_available(),train_origins=len(train_idx))
        (art/'smoke_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(result,ensure_ascii=False),flush=True);return
    start=time.perf_counter()
    baselines,predictions=train_baselines(frame,x,y,train_idx)
    histories={};parameters={}
    for kind in ['tcn','gru']:
        net,h=neural(kind,x,y,train_idx,val_idx,cfg,art/kind,cfg['max_epochs'],args.resume)
        predictions[kind]=infer(net,x);histories[kind]=h
        parameters[kind]=sum(p.numel() for p in net.parameters())
    joblib.dump(baselines,art/'baselines.joblib')
    calibrators={name:calibrate(p,y,val_idx) for name,p in predictions.items()}
    joblib.dump(calibrators,art/'calibrators.joblib')
    calibrated={name:apply_calibration(p,calibrators[name]) for name,p in predictions.items()}
    np.savez_compressed(art/'predictions_frozen.npz',**calibrated)
    vf=frame.iloc[val_idx].reset_index(drop=True);events=endpoint_events(cycles,'validation')
    hours=products.loc[products.split.eq('validation')&products.supervised_product_eligible.eq(1),'spray_samples'].sum()/3600
    comparison=[];thresholds={}
    for name,p in calibrated.items():
        summary=choose_threshold(vf,p[val_idx,2],events,y[val_idx,2],hours)
        metrics=probability_metrics(y[val_idx,2],p[val_idx,2])
        summary.update(name=name,pr_auc=metrics['pr_auc'],brier=metrics['brier'])
        comparison.append(summary);thresholds[name]={'eol_60m':summary['threshold']}
        # Separate future adjacent-voltage policy, selected only on validation events.
        adjacent=pd.read_csv(art/'target_events.csv')
        adjacent=adjacent[adjacent.split.eq('validation')&adjacent.kind.eq('adjacent_two_voltage')].copy()
        adjacent['time']=pd.to_datetime(adjacent.time,format='%d %m %Y %H:%M:%S');adjacent['evidence']='diagnostic'
        av=choose_threshold(vf,p[val_idx,5],adjacent,y[val_idx,5],hours)
        thresholds[name]['adjacent_voltage_60m']=av['threshold']
    def key(r):
        return (r['false_alarms_per_hour']<=.5,r['hit30'] or 0,r['hit15'] or 0,
                -(r['missed'] or 0),-r['false_alarms_per_hour'],r['pr_auc'] or 0,-r['brier'])
    winner=max(comparison,key=key)['name']
    stamp=pd.Timestamp.now().isoformat()
    model_hash=hashlib.sha256((json.dumps(cfg,sort_keys=True)+winner+stamp).encode()).hexdigest()[:12]
    frozen=dict(cfg,selected_model=winner,model_version='booth2-1116-'+model_hash,trained_at=stamp,
        status='完成正式训练',thresholds=thresholds,validation_comparison=comparison,
        elapsed_seconds=time.perf_counter()-start,parameters=parameters,
        tcn_receptive_field_grid_steps=31,tcn_receptive_field_calendar_minutes=30,
        training_origins=int(len(train_idx)),validation_origins=int(len(val_idx)),
        calibration='Platt on validation only; same validation reused for calibration and model/threshold selection',
        test_frozen_at=stamp,compute='CPU torch, 4 threads; GPU not needed for small models')
    (art/'frozen_config.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FROZEN',json.dumps(frozen,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
