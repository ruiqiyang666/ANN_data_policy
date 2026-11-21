"""Training-only masked-channel pretraining, followed by the same TCN risk fit.

This is an extra initialization comparison, NOT an eighth algorithm family.
"""
import sys
sys.dont_write_bytecode=True
import torch
import time,subprocess
import numpy as np
import train_risk as train
from risk_models import RiskNet as BaseNet,Cache
from common_v2 import ROOT,TASK,dump,status,load

INITIAL=None
original_train=train.train_neural
def factory(name,nstats):
    m=BaseNet('tcn' if name=='tcn_ssl' else name,nstats)
    if name=='tcn_ssl' and INITIAL is not None:m.encoder.load_state_dict(INITIAL)
    return m

def pretrain_then_fit(name,cache,ids,transform,path,epochs=6):
    global INITIAL
    train.set_seed();torch.cuda.set_per_process_memory_fraction(.5)
    rng=np.random.default_rng(42);norm=cache.sequence_norm(ids);model=BaseNet('tcn',len(transform.mean)).cuda()
    head=torch.nn.Conv1d(32,28,1).cuda();opt=torch.optim.AdamW([*model.encoder.parameters(),*head.parameters()],lr=5e-4,weight_decay=1e-3)
    history=[]
    for epoch in range(3):
        started=time.perf_counter();losses=[]
        for step in range(64):
            temp=int(subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0]) if step%32==0 else 0
            if temp>=87:raise RuntimeError('SSL thermal abort')
            deadline=time.monotonic()+600
            while temp>=83:
                if temp>=87 or time.monotonic()>deadline:raise RuntimeError('SSL thermal abort or cooling timeout')
                status('ssl_cooling_pause',temperature_c=temp);time.sleep(10)
                temp=int(subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0])
            chosen=rng.choice(ids,128);stats=transform.transform(cache.features[chosen]);x,_=cache.batch(chosen,stats,norm)
            target=x[:,:28].detach().clone();known=x[:,28:56]>0;hidden=(torch.rand_like(target)<.15)&known
            corrupt=x.clone();corrupt[:,:28].masked_fill_(hidden,0);corrupt[:,28:56].masked_fill_(hidden,0)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=head(model.encoder(corrupt))
            loss=torch.nn.functional.smooth_l1_loss(pred.float()[hidden],target[hidden])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite SSL loss')
            loss.backward();torch.nn.utils.clip_grad_norm_([*model.encoder.parameters(),*head.parameters()],1);opt.step();losses.append(float(loss.detach()))
        entry={'epoch':epoch+1,'masked_huber_loss':float(np.mean(losses)),'seconds':time.perf_counter()-started,'training_windows_only':True}
        history.append(entry);dump(history,path/'self_supervised_epochs.json');status('tcn_ssl_pretrain_epoch',fit=path.name,**entry)
    INITIAL={k:v.detach().cpu().clone() for k,v in model.encoder.state_dict().items()}
    torch.save({'encoder':INITIAL,'reconstruction_head':{k:v.detach().cpu() for k,v in head.state_dict().items()},'train_windows':len(ids),'epochs':3},path/'self_supervised_encoder.pt')
    del model,head,opt;torch.cuda.empty_cache()
    result=original_train(name,cache,ids,transform,path,epochs);INITIAL=None;return result

def main():
    if TASK!='voltage':raise RuntimeError('SSL comparison is voltage-only')
    torch.set_num_threads(4);torch.set_num_interop_threads(2)
    train.RiskNet=factory;train.NEURAL.add('tcn_ssl');train.train_neural=pretrain_then_fit
    train.run_method('tcn_ssl',Cache())
    status('ssl_comparison_complete')
if __name__=='__main__':main()
