import sys
sys.dont_write_bytecode=True
import numpy as np
import torch
from life_models import fit_linear,nll,build_hazard_rows,piecewise_quantiles,torch_lognormal_nll,HOUR_EDGES,PIECE_EDGES
from common_v2 import EVAL,dump

def main():
    rng=np.random.default_rng(42);x=rng.normal(size=(400,1));true=np.exp(1+.8*x[:,0]+rng.normal(0,.25,len(x)));censor=4.
    event=true<=censor;lo=np.where(event,true*.999,censor);hi=np.where(event,true*1.001,np.inf)
    model=fit_linear(x,lo,hi,event,np.ones(len(x))/len(x),penalty=0)
    assert model['optimizer_success'] and abs(model['beta'][1]-.8)<.15,model
    checks=['Interval and right-censored AFT recovers a known synthetic covariate effect']
    loss=nll(np.log(np.array([1.,4.])),.5,np.array([2.,2.]),np.array([np.inf,np.inf]),np.array([False,False]))
    assert loss[0]>loss[1];checks.append('Censor likelihood rewards survival past observation without fake event labels')
    mu=torch.zeros(4,requires_grad=True);a=torch.tensor([0.,.49,1.,50.]);b=torch.tensor([1.,.51,float('inf'),51.]);e=torch.tensor([True,True,False,True])
    l=torch_lognormal_nll(mu,torch.tensor(1.),a,b,e).sum();l.backward();assert torch.isfinite(l) and torch.isfinite(mu.grad).all()
    checks.append('Neural survival likelihood remains differentiable for zero counts, narrow intervals and censoring')
    rows,bins,y,w,amb=build_hazard_rows(np.ones((2,1)),np.array([0.,.49]),np.array([1.,.51]),np.ones(2,dtype=bool),np.ones(2),PIECE_EDGES)
    assert y[rows==0].any()
    rows,bins,y,w,amb=build_hazard_rows(np.ones((1,1)),np.array([.49]),np.array([.51]),np.ones(1,dtype=bool),np.ones(1),HOUR_EDGES)
    assert not y.any() and amb==1;checks.append('Hazard expansion retains zero pieces and treats boundary-straddling intervals as partial observation')
    h=1-np.exp(-2*np.diff(HOUR_EDGES));p,tail=piecewise_quantiles(h[None,:],HOUR_EDGES)
    np.testing.assert_allclose(p[0],-np.log1p(-np.array([.1,.5,.9]))/2,rtol=1e-7)
    checks.append('Piecewise hazard quantiles agree with a known exponential survival law')
    dump({'passed':True,'checks':checks,'synthetic_beta':model['beta']},EVAL/'life_preflight.json');print(checks)

if __name__=='__main__':main()
