"""Round 4: is the sensor life model better than trivial baselines? Proper quantile scoring."""
import sys; sys.path.insert(0,'research')
import numpy as np, pandas as pd, data as D, rul_bench as R, rul_bench2 as R2
pd.set_option('display.width',250)
f=D.build().reset_index(drop=True)
z=np.load('research/out/round3_v3.npz')
Q=np.array([.1,.5,.9])
GRID=np.r_[np.arange(0,2,1/60),np.arange(2,60,.05)]
def pinball(truth,pred):
    d=truth[:,None]-pred
    return np.maximum(Q*d,(Q-1)*d).mean(0)
def mixture(age,p,k,lam,end_h=1.0):
    """F(t)=p*U(0,end_h)+(1-p)*Weibull residual given age; quantiles on a grid."""
    out=np.zeros((len(age),3))
    S_w=np.exp(-(((age[:,None]+GRID[None,:])/lam)**k)+((age[:,None]/lam)**k))
    F=p[:,None]*np.minimum(GRID[None,:]/end_h,1)+(1-p[:,None])*(1-S_w)
    for j,q in enumerate(Q):
        out[:,j]=GRID[np.minimum((F<q).sum(1),len(GRID)-1)]
    return out
def ema(x,a):
    y=np.empty_like(x); s=x[0]
    for i,v in enumerate(x):
        s=s+a*(v-s); y[i]=s
    return y
rows=[]
cyc=f.groupby('cycle_id').agg(life=('life_h','first'),ev=('event','first'))
for held in cyc.index:
    te=f[(f.cycle_id==held)&f.rul_h.notna()]; truth=te.rul_h.to_numpy(); age=te.age_h.to_numpy()
    tr=f[(f.cycle_id!=held)&f.rul_h.notna()]; trc=cyc.drop(held)
    k,lam=R.weibull_fit(trc.life,trc.ev)
    p=z['S:warn_cal'][te.index]
    cand={}
    const=np.quantile(tr[tr.event==1].rul_h,Q); cand['constant']=np.tile(const,(len(te),1))
    # empirical residual life of observed cycles that outlived this age
    obs=trc[trc.ev==1].life.to_numpy()
    emp=np.array([np.quantile(obs[obs>a]-a,Q) if (obs>a).sum()>=2 else [np.nan]*3 for a in age]); 
    emp=pd.DataFrame(emp).ffill().to_numpy(); cand['empirical_age']=emp
    cand['weibull_age']=np.column_stack([R.weibull_residual(age,k,lam,q) for q in Q])
    cand['weibull+warn']=mixture(age,p,k,lam)
    cand['weibull+warn_ema.3']=mixture(age,ema(p,.3),k,lam)
    cand['weibull+warn_max3']=mixture(age,pd.Series(p).rolling(18,min_periods=1).max().to_numpy(),k,lam)
    kc,lc=1.0,const[1]/np.log(2)
    cand['expo+warn']=mixture(age,p,1.0,np.mean(tr[tr.event==1].rul_h)*1.0)
    for n in ['hazard_lin','hazard_lin~s0.01','fused','smooth_then_cap~s0.01']:
        cand[n]=z[f'{n}@{held}'][:,2:5].astype(float)
    for n,pr in cand.items():
        row=dict(model=n,cycle=held,event=int(cyc.ev[held]),k=k,lam=lam)
        if cyc.ev[held]:
            err=np.abs(pr[:,1]-truth); pb=pinball(truth,pr)
            row.update(mae=err.mean(),mae5=err[truth<=5].mean(),mae2=err[truth<=2].mean(),mae1=err[truth<=1].mean(),
                       pin10=pb[0],pin50=pb[1],pin90=pb[2],cover=((pr[:,0]<=truth)&(truth<=pr[:,2])).mean(),width=(pr[:,2]-pr[:,0]).mean(),
                       jump=np.abs(np.diff(pr[:,1])).mean()*360)
        else: row.update(under=(pr[:,1]<truth).mean())
        rows.append(row)
r=pd.DataFrame(rows); ev=r[r.event==1]
s=ev.groupby('model')[['mae','mae5','mae2','mae1','pin10','pin50','pin90','cover','width','jump']].mean()
s['pin_sum']=s.pin10+s.pin50+s.pin90; s['cens_under']=r[r.event==0].groupby('model').under.mean()
print(s.sort_values('pin_sum').round(3).to_string())
print(r.groupby('cycle')[['k','lam']].first().round(2).T.to_string())
print(ev.pivot(index='cycle',columns='model',values='mae')[['constant','weibull_age','weibull+warn','hazard_lin~s0.01','smooth_then_cap~s0.01']].round(2).to_string())
