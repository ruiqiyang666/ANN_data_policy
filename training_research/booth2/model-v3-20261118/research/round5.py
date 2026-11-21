"""Round 5: causal smoothing of the life quantiles without the floor trap."""
import sys; sys.path.insert(0,'research')
import numpy as np, pandas as pd, data as D
pd.set_option('display.width',250)
f=D.build().reset_index(drop=True); z=np.load('research/out/round3_v3.npz'); Q=np.array([.1,.5,.9])
def log_ema(x,a):
    y=np.empty_like(x); s=np.log(np.maximum(x[0],.02))
    for i,v in enumerate(x):
        s+=a*(np.log(max(v,.02))-s); y[i]=np.exp(s)
    return y
def lin_count(x,a):
    y=np.empty_like(x); s=x[0]
    for i,v in enumerate(x):
        if i: s=max(s-D.STEP_H,0.0)
        s+=a*(v-s); y[i]=s
    return y
def med_roll(x,n): return pd.Series(x).rolling(n,min_periods=1).median().to_numpy()
def pinball(t,p):
    d=t[:,None]-p; return np.maximum(Q*d,(Q-1)*d).mean(0).sum()
rows=[]
cyc=f.groupby('cycle_id').event.first()
for c in cyc.index:
    base=z[f'hazard_lin@{c}'].astype(float); fused=z[f'fused@{c}'][:,2:5].astype(float); truth=base[:,1]; raw=base[:,2:5]
    cand={'raw':raw,'fused':fused}
    for a in (.02,.01,.005,.003,.0015):
        s=np.column_stack([log_ema(raw[:,j],a) for j in range(3)]); cand[f'logema{a}']=s; cand[f'logema{a}+cap']=np.minimum(s,fused)
        s=np.column_stack([lin_count(raw[:,j],a) for j in range(3)]); cand[f'lincount{a}+cap']=np.minimum(s,fused)
    for n in (180,360,1080):
        s=np.column_stack([med_roll(raw[:,j],n) for j in range(3)]); cand[f'median{n}+cap']=np.minimum(s,fused)
    for n,p in cand.items():
        if cyc[c]:
            e=np.abs(p[:,1]-truth)
            rows.append(dict(model=n,mae=e.mean(),mae5=e[truth<=5].mean(),mae2=e[truth<=2].mean(),mae1=e[truth<=1].mean(),pin=pinball(truth,p),
                             cover=((p[:,0]<=truth)&(truth<=p[:,2])).mean(),p10=np.median(p[:,0]),p50=np.median(p[:,1]),p90=np.median(p[:,2]),jump=np.abs(np.diff(p[:,1])).mean()*360))
print(pd.DataFrame(rows).groupby('model').mean().sort_values('pin').round(3).to_string())
