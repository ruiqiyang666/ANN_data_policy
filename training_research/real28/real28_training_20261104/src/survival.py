"""Regularized lognormal AFT with interval and right-censored likelihood."""
import numpy as np
from scipy.special import log_ndtr
from scipy.optimize import minimize

def logdiff(a,b):
    return a+np.log(-np.expm1(np.minimum(b-a,-1e-14)))

def lognormal_nll(mu,sigma,lower,upper,event):
    lo=np.asarray(lower,dtype=float);hi=np.asarray(upper,dtype=float)
    zlo=(np.log(np.maximum(lo,1e-300))-mu)/sigma
    zhi=(np.log(np.maximum(np.where(event,hi,lo),1e-300))-mu)/sigma
    cdf=logdiff(log_ndtr(zhi),log_ndtr(zlo))
    sf=logdiff(log_ndtr(-zlo),log_ndtr(-zhi))
    interval=np.where(zlo>0,sf,cdf)
    return np.where(event,-interval,-log_ndtr(-zlo))

def fit_aft(features,lower,upper,event,groups,penalty=1.0):
    x=np.asarray(features,dtype='float64');design=np.column_stack([np.ones(len(x)),x])
    _,inverse,counts=np.unique(groups,return_inverse=True,return_counts=True)
    weight=1/counts[inverse];weight/=weight.sum()
    informative=np.asarray(event,dtype=bool)
    if not informative.any():raise ValueError('AFT requires at least one observed endpoint proxy')
    init=np.zeros(design.shape[1]+1)
    init[0]=np.median(np.log(np.maximum((lower[informative]+upper[informative])/2,1e-3)))
    init[-1]=0
    def objective(theta):
        mu=design@theta[:-1];sigma=np.exp(theta[-1])
        nll=lognormal_nll(mu,sigma,lower,upper,informative)
        return float(np.sum(weight*nll)+penalty*np.mean(theta[1:-1]**2))
    result=minimize(objective,init,method='L-BFGS-B',bounds=[(-12,12)]+[(-5,5)]*x.shape[1]+[(-2.5,1.5)],
                    options={'maxiter':300,'ftol':1e-10})
    if not np.isfinite(result.fun):raise RuntimeError('AFT fit produced invalid objective')
    return {'coefficients':result.x[:-1].tolist(),'sigma':float(np.exp(result.x[-1])),
            'optimizer_success':bool(result.success),'optimizer_message':str(result.message),'objective':float(result.fun),
            'iterations':int(result.nit),'penalty':penalty,'groups':len(counts),'event_windows':int(informative.sum()),
            'interval_quantiles':[.1,.9],'independent_interval_calibration':False}

def predict_aft(model,features,pieces=False):
    beta=np.array(model['coefficients']);mu=beta[0]+np.asarray(features)@beta[1:];sigma=model['sigma']
    result=np.exp(np.clip(mu[:,None]+sigma*np.array([-1.2815515655,0,1.2815515655]),-20,20))
    if pieces:result=np.floor(result)
    return result

def interval_distance(pred,lo,hi):return np.maximum(np.maximum(lo-pred,pred-hi),0)
