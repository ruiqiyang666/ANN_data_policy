"""Learn a known censored lognormal relation; separate from the equipment dataset."""
import numpy as np
from scipy.special import ndtri
import train as T
import v5 as V
from pathlib import Path

rng=np.random.default_rng(9471)
x=rng.uniform(-1.5,1.5,(1600,1));true=np.exp(1+.6*x[:,0]+.3*rng.normal(size=len(x)))
censor=rng.exponential(7,size=len(x))+.5
t=np.minimum(true,censor);e=true<=censor;w=np.ones(len(x))
grid=np.linspace(-1.2,1.2,51)[:,None]
expected=np.exp(1+.6*grid[:,0]);expected_q=np.exp(1+.6*grid+.3*ndtri(T.Q))
results={}
for family,fit in [('cqrnn',T.fit_cqr),('mix',T.fit_mix)]:
    states,history=fit(x,t,e,w)
    if family=='cqrnn':
        values=np.mean([np.cumsum(np.logaddexp(0,T.forward_net(s,grid))+.0001,axis=1)*10 for s in states],axis=0)
        c,q=T.qoutputs(values,T.TAU)
    else:
        from scipy.special import logsumexp
        parameters=[]
        for state in states:
            raw=T.forward_net(state['network'],grid)
            parameters.append((np.clip(raw[:,:2],-7,5),np.exp(np.clip(state['log_sigma'],-1.5,1.2)),
                               np.exp(raw[:,2:]-logsumexp(raw[:,2:],axis=1,keepdims=True))))
        c,q=T.mixture_outputs(parameters)
    relative=float(np.mean(np.abs(q[:,1]-expected)/expected))
    assert relative<.25,(family,relative)
    assert np.corrcoef(q[:,1],expected)[0,1]>.95
    assert np.all(np.diff(q,axis=1)>0) and np.all(np.diff(c,axis=1)>=0)
    results[family]=dict(median_relative_error=relative,grid_quantile_mae=float(np.abs(q-expected_q).mean()),
                        censored_fraction=float((~e).mean()))
    print(family,results[family],flush=True)
V.save_json(Path('synthetic-checks.json'),results)
