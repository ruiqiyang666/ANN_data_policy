"""Six censor-aware lifetime model families and a same-weight constant control."""
import sys
sys.dont_write_bytecode=True
import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr
from sklearn.preprocessing import SplineTransformer,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier,ExtraTreesClassifier
import torch
from torch import nn

HOUR_EDGES=np.array([0,1/60,5/60,.25,.5,1,2,4,8,16,32,64.])
PIECE_EDGES=np.array([0,1,2,5,10,20,50,100,200,500,1000,2000,4000.])

def logdiff(a,b):return a+np.log(-np.expm1(np.minimum(b-a,-1e-12)))

def nll(mu,param,lo,hi,event,distribution='lognormal'):
    high=np.where(event,hi,lo+1)
    if distribution=='weibull':
        slo=-np.exp(np.clip(param*(np.log(np.maximum(lo,1e-100))-mu),-100,25))
        shi=-np.exp(np.clip(param*(np.log(np.maximum(high,1e-100))-mu),-100,25))
        return np.where(event,-logdiff(slo,shi),-slo)
    a=(np.log(np.maximum(lo,1e-100))-mu)/param;b=(np.log(np.maximum(high,1e-100))-mu)/param
    interval=np.where(a>0,logdiff(log_ndtr(-a),log_ndtr(-b)),logdiff(log_ndtr(b),log_ndtr(a)))
    return np.where(event,-interval,-log_ndtr(-a))

def landmark_weights(samples):
    # Equal possible-cycle weight, then equal censor/event × horizon-stratum weight.
    event=samples.rul_event.to_numpy();h=samples.rul_lower_hours.to_numpy()
    bucket=np.digitize(h,[5/60,.25,1,4,16]);stratum=bucket+event.astype(int)*6
    group=samples.possible_cycle.to_numpy();weight=np.zeros(len(samples))
    for g in np.unique(group):
        gm=group==g;strata=np.unique(stratum[gm])
        for k in strata:
            hit=gm&(stratum==k);weight[hit]=1/(len(np.unique(group))*len(strata)*hit.sum())
    return weight

def fit_linear(x,lo,hi,event,weight,distribution='lognormal',penalty=.25):
    design=np.column_stack([np.ones(len(x)),x]);initial=np.zeros(design.shape[1]+1)
    initial[0]=np.log(np.maximum(np.median((lo[event]+hi[event])/2),.001));initial[-1]=0
    def objective(theta):
        loss=nll(design@theta[:-1],np.exp(theta[-1]),lo,hi,event,distribution)
        regularizer=penalty*np.sum(theta[1:-1]**2) if x.shape[1] else 0
        return float(weight@loss+regularizer)
    fit=minimize(objective,initial,method='L-BFGS-B',bounds=[(-12,12)]+[(-5,5)]*x.shape[1]+[(-2.5,2)],options={'maxiter':350,'ftol':1e-8})
    if not np.isfinite(fit.fun):raise RuntimeError('Nonfinite AFT optimization')
    return {'beta':fit.x[:-1],'parameter':float(np.exp(fit.x[-1])),'distribution':distribution,
            'optimizer_success':bool(fit.success),'message':str(fit.message),'objective':float(fit.fun),'iterations':int(fit.nit),'penalty':penalty}

class LifeMLP(nn.Module):
    def __init__(self,d):
        super().__init__();self.net=nn.Sequential(nn.Linear(d,16),nn.Tanh(),nn.Linear(16,8),nn.Tanh(),nn.Linear(8,1));self.log_sigma=nn.Parameter(torch.tensor(.5))
    def forward(self,x):return self.net(x).squeeze(1)

def torch_lognormal_nll(mu,sigma,lo,hi,event):
    a=(lo.clamp_min(1e-30).log()-mu)/sigma;b=(torch.where(event,hi,lo+1).clamp_min(1e-30).log()-mu)/sigma
    def difference(u,v):return u+torch.log(-torch.expm1((v-u).clamp_max(-1e-7)))
    interval=torch.where(a>0,difference(torch.special.log_ndtr(-a),torch.special.log_ndtr(-b)),difference(torch.special.log_ndtr(b),torch.special.log_ndtr(a)))
    return torch.where(event,-interval,-torch.special.log_ndtr(-a))

def fit_mlp(x,lo,hi,event,weight):
    torch.manual_seed(42);torch.set_num_threads(2);model=LifeMLP(x.shape[1])
    with torch.no_grad():model.net[-1].bias.fill_(np.log(max(np.median((lo[event]+hi[event])/2),.001)))
    opt=torch.optim.AdamW(model.parameters(),lr=.006,weight_decay=.003)
    tx=torch.tensor(x,dtype=torch.float32);tl=torch.tensor(lo,dtype=torch.float32);th=torch.tensor(hi,dtype=torch.float32);te=torch.tensor(event);tw=torch.tensor(weight,dtype=torch.float32)
    best=float('inf');state=None;stale=0;history=[]
    for epoch in range(250):
        opt.zero_grad(set_to_none=True);mu=model(tx);sigma=model.log_sigma.clamp(-2,2).exp()
        loss=(torch_lognormal_nll(mu,sigma,tl,th,te)*tw).sum()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite neural survival loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();value=float(loss.detach());history.append(value)
        if value<best-1e-5:best=value;state={k:v.detach().clone() for k,v in model.state_dict().items()};stale=0
        else:stale+=1
        if stale>=30:break
    model.load_state_dict(state)
    return {'state_dict':state,'features':x.shape[1],'history':history,'objective':best,'distribution':'lognormal','optimizer_success':True}

def build_hazard_rows(x,lo,hi,event,weight,edges):
    n=len(edges)-1;event_bin=np.searchsorted(edges,lo,'right')-1;upper_bin=np.searchsorted(edges,hi,'left')-1
    known=event&(event_bin==upper_bin)&(event_bin>=0)&(event_bin<n)
    survived=edges[1:][None,:]<=lo[:,None]
    y=np.zeros((len(x),n),dtype=bool)
    y[np.flatnonzero(known),event_bin[known]]=True;mask=survived|y
    rows,bins=np.nonzero(mask)
    return rows,bins,y[rows,bins],weight[rows],int((event&~known).sum())

def hazard_design(x,bins,edges,linear):
    if linear:return np.column_stack([x,np.eye(len(edges)-1,dtype='float32')[bins]])
    return np.column_stack([x,np.log1p(edges[1:])[bins]])

def fit_hazard(name,x,lo,hi,event,weight,task):
    edges=HOUR_EDGES if task=='hours' else PIECE_EDGES
    rows,bins,y,w,ambiguous=build_hazard_rows(x,lo,hi,event,weight,edges)
    linear=name=='logistic_hazard';design=hazard_design(x[rows],bins,edges,linear)
    # sklearn regularization is calibrated to ordinary sample-weight scale.
    w=w/w.mean()
    if linear:model=LogisticRegression(C=.1,max_iter=350)
    elif name=='forest_hazard':model=ExtraTreesClassifier(n_estimators=100,max_depth=10,min_samples_leaf=40,max_features=.8,n_jobs=2,random_state=42)
    else:model=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=7,max_depth=3,min_samples_leaf=30,l2_regularization=5,learning_rate=.05,early_stopping=False,random_state=42)
    model.fit(design,y,sample_weight=w)
    return {'model':model,'edges':edges,'linear':linear,'ambiguous_interval_landmarks_as_censor':ambiguous,
            'expanded_rows':len(rows),'distribution':'piecewise_exponential','optimizer_success':True}

def hazard_probabilities(model,x):
    nb=len(model['edges'])-1;pieces=[]
    for j in range(nb):
        z=hazard_design(x,np.full(len(x),j),model['edges'],model['linear']);pieces.append(model['model'].predict_proba(z)[:,1])
    return np.clip(np.column_stack(pieces),1e-6,1-1e-6)

def piecewise_quantiles(h,edges):
    h=np.clip(h,1e-9,1-1e-9)
    cumulative=np.cumsum(-np.log1p(-h),axis=1);previous=np.column_stack([np.zeros(len(h)),cumulative[:,:-1]])
    rates=-np.log1p(-h)/np.diff(edges)
    out=[];tail=[]
    for q in [.1,.5,.9]:
        required=-np.log1p(-q);idx=(cumulative<required).sum(1);b=np.minimum(idx,len(edges)-2);row=np.arange(len(h))
        answer=edges[b]+(required-previous[row,b])/rates[row,b]
        out.append(answer);tail.append(idx>=len(edges)-1)
    return np.column_stack(out),np.column_stack(tail)

def predict_model(bundle,x,task):
    name=bundle['name'];model=bundle['model'];tail=None
    if name in ['logistic_hazard','boosted_hazard','forest_hazard']:
        h=hazard_probabilities(model,x);prediction,tail=piecewise_quantiles(h,model['edges'])
    else:
        z=x
        if name=='constant':z=x[:,:0]
        if name=='spline_aft':z=bundle['basis_scaler'].transform(bundle['basis'].transform(x))
        if name=='mlp_aft':
            net=LifeMLP(model['features']);net.load_state_dict(model['state_dict']);net.eval()
            with torch.no_grad():mu=net(torch.tensor(z,dtype=torch.float32)).numpy();param=float(net.log_sigma.clamp(-2,2).exp())
        else:mu=model['beta'][0]+z@model['beta'][1:];param=model['parameter']
        if model['distribution']=='weibull':log_q=mu[:,None]+np.log(-np.log1p(-np.array([.1,.5,.9])))/param
        else:log_q=mu[:,None]+param*np.array([-1.2815515655,0,1.2815515655])
        prediction=np.exp(np.clip(log_q,-25,25))
    if task=='pieces':prediction=np.floor(prediction)
    return prediction,tail

def fit_model(name,x,lo,hi,event,weight,task):
    bundle={'name':name}
    if name in ['logistic_hazard','boosted_hazard','forest_hazard']:bundle['model']=fit_hazard(name,x,lo,hi,event,weight,task)
    elif name=='mlp_aft':bundle['model']=fit_mlp(x,lo,hi,event,weight)
    else:
        z=x
        if name=='constant':z=x[:,:0]
        if name=='spline_aft':
            basis=SplineTransformer(n_knots=3,degree=2,include_bias=False,extrapolation='linear').fit(x)
            scaler=StandardScaler().fit(basis.transform(x));z=scaler.transform(basis.transform(x));bundle.update(basis=basis,basis_scaler=scaler)
        bundle['model']=fit_linear(z,lo,hi,event,weight,'weibull' if name=='weibull_aft' else 'lognormal',penalty=.05)
    return bundle
