import pandas as pd, numpy as np, sys
pd.set_option('display.width',250); pd.set_option('display.max_columns',80)
R='repo/'
a=pd.read_csv(R+'model_v2_GRU/data/anchors.csv.gz',parse_dates=['timestamp'])
c=pd.read_csv(R+'model_v2_GRU/data/cycles.csv',parse_dates=['usable_start','usable_end','eol_endpoint'])
p=pd.read_csv(R+'model_v2_GRU/data/products.csv.gz',parse_dates=['start','end'],low_memory=False)
f=pd.read_csv(R+'training_data/features_10s.csv.gz')
print('anchors in features:',a.anchor_id.isin(f.anchor_id).mean(), len(f))
rows=[]
for r in c.itertuples():
    q=a[a.cycle_id==r.cycle_id]; pp=p[(p.cycle_id==r.cycle_id)]
    end=r.eol_endpoint if pd.notna(r.eol_endpoint) else r.usable_end
    qq=q[q.timestamp<end]; pe=pp[pp.start<end]
    v=qq.Voltage_mean60s
    n=len(qq)
    rows.append(dict(cycle=r.cycle_id,split=r.split,eol=int(r.eol_strict_observed),end_type=r.end_type[:14],left_c=r.left_censored,
        cal_h=round((end-r.usable_start).total_seconds()/3600,1),proc_h=round(n*10/3600,2),products=len(pe),
        spray_h=round(pe.spray_samples.sum()/3600,2),rec_h=r.recorded_old_electrode_hours,obs_on_h=round(r.observed_on_hours,1),
        v_first10=round(v.iloc[:max(n//10,1)].median(),2),v_mid=round(v.iloc[n//2-n//20:n//2+n//20].median(),2),v_last10=round(v.iloc[-max(n//10,1):].median(),2),
        v_last1h=round(v.iloc[-360:].median(),2),vstd_first=round(qq.Voltage_std60s.iloc[:n//10].median(),3),vstd_last=round(qq.Voltage_std60s.iloc[-n//10:].median(),3),
        cur=qq.Current.median(), vfail=int(pe.voltage_fail.eq(1).sum())))
print(pd.DataFrame(rows).to_string())
# voltage decile trajectory per cycle
print('\nVoltage_mean60s median by life decile (processing-time fraction)')
for r in c.itertuples():
    end=r.eol_endpoint if pd.notna(r.eol_endpoint) else r.usable_end
    q=a[(a.cycle_id==r.cycle_id)&(a.timestamp<end)]
    d=np.array_split(q.Voltage_mean60s.to_numpy(),10)
    print(r.cycle_id, int(r.eol_strict_observed), ' '.join(f'{np.median(x):5.1f}' for x in d), ' | cur', ' '.join(f'{int(np.median(x))}' for x in np.array_split(q.Current.to_numpy(),5)))
