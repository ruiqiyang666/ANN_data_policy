"""Causal operating-state features, complete chronological risk data and V1 life join."""
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from common_v2 import DATA,V1,MODELS,FIELDS,TYPES,dump,status,protocol,sha,time_seconds

def assign_part(t):
    out=np.full(len(t),-1,dtype=np.int8)
    for k,(a,b) in enumerate([('2026-08-15','2026-09-30'),('2026-10-04','2026-10-08'),('2026-10-08','2026-10-10'),('2026-10-10','2026-10-24'),('2026-10-27','2026-11-02')]):
        out[(t>=time_seconds(a))&(t<time_seconds(b))]=k
    return out

def event_targets(now, next_at, observable):
    delta=next_at-now;observed=np.maximum(0,np.minimum(300,observable-now))
    event=(delta>0)&(delta<=300)&(delta<=observed)
    b=np.clip((np.maximum(delta,1)-1)//30,0,9).astype(int)
    bins=np.arange(10)[None,:]
    mask=np.where(event[:,None],bins<=b[:,None],bins<(observed//30)[:,None])
    y=event[:,None]&(bins==b[:,None])
    return y.astype('float32'),mask.astype('float32'),event,delta,observed

def masked_window_statistics(values, valid, ends, left, clock):
    """Statistics fit only observed operating rows, with slopes per clock hour."""
    a=np.where(valid,values,0).astype('float64')
    def csum(x): return np.r_[0.,np.cumsum(x,dtype='float64')]
    c=csum(valid);sx=csum(a);sx2=csum(a*a);st=csum(clock*valid);st2=csum(clock*clock*valid);stx=csum(clock*a)
    n=c[ends+1]-c[left];total=sx[ends+1]-sx[left]
    mean=total/np.maximum(n,1)
    std=np.sqrt(np.maximum(0,(sx2[ends+1]-sx2[left])/np.maximum(n,1)-mean*mean))
    tsum=st[ends+1]-st[left];txsum=stx[ends+1]-stx[left]
    den=st2[ends+1]-st2[left]-tsum*tsum/np.maximum(n,1)
    slope=(txsum-tsum*mean)/np.maximum(den,1e-8)
    mean[n<1]=np.nan;std[n<2]=np.nan;slope[(n<10)|(den<1e-6)]=np.nan
    return mean,std,slope,n

def main():
    protocol()
    rebuilding='--rebuild-before-training' in sys.argv
    if rebuilding and (list(MODELS.rglob('*.pt')) or list(MODELS.rglob('*.joblib'))):raise RuntimeError('Cannot rebuild after training artifacts exist')
    if (DATA/'ready.json').exists() and not rebuilding: print('Immutable V2 feature data already present.');return
    if rebuilding:
        for name in ['ready.json','targets_finalized.json']:
            path=DATA/name
            if path.exists():path.unlink()
    raw_t=np.load(V1/'timestamps.npy',mmap_mode='r');raw_x=np.load(V1/'raw_values.npy',mmap_mode='r')
    raw_part=assign_part(raw_t)
    cuts=np.r_[0,np.flatnonzero((np.diff(raw_t)>2)|(np.diff(raw_part)!=0))+1,len(raw_t)]
    ts=[];xs=[];ms=[];bs=[];be=[];parts=[];cursor=0
    for a,b in zip(cuts[:-1],cuts[1:]):
        if raw_part[a]<0: continue
        grid=np.arange(raw_t[a],raw_t[b-1]+1,dtype='int64');ix=np.searchsorted(raw_t[a:b],grid,'right')-1
        v=np.asarray(raw_x[a:b][ix]).copy();seen=(raw_t[a:b][ix]==grid)[:,None]&np.isfinite(v)
        for c in range(28):
            prev=np.maximum.accumulate(np.where(seen[:,c],np.arange(len(grid)),-1))
            v[:,c]=np.where(prev>=0,v[np.maximum(prev,0),c],np.nan)
        ts.append(grid);xs.append(v);ms.append(seen);bs.append(np.full(len(grid),cursor,dtype='int64'))
        be.append(np.full(len(grid),cursor+len(grid)-1,dtype='int64'));parts.append(np.full(len(grid),raw_part[a],dtype='int8'))
        cursor+=len(grid)
    t=np.concatenate(ts);x=np.concatenate(xs);seen=np.concatenate(ms);block=np.concatenate(bs);endblock=np.concatenate(be);part=np.concatenate(parts)
    del ts,xs,ms,bs,be,parts
    index=np.arange(len(t));newblock=index==block
    powered=(x[:,4]>100)&(x[:,5]>5)&seen[:,4]&seen[:,5]
    prior_power=np.r_[False,powered[:-1]]&~newblock
    off_start=np.maximum.accumulate(np.where((~powered&prior_power)|newblock,index,0))
    restart=powered&~prior_power&((index-off_start)>=60)
    operation_start=np.maximum.accumulate(np.where(newblock|restart,index,0))
    spray=powered&(((x[:,25]>50)&seen[:,25])|((x[:,26]>50)&seen[:,26]))
    active1=spray&(x[:,11]>0)&seen[:,11];active2=spray&(x[:,18]>0)&seen[:,18]
    ends=np.flatnonzero((t%10==0)&powered&(index-block>=119))
    now=t[ends]
    feature_arrays=[];names=[]
    def add(name,value): names.append(name);feature_arrays.append(np.asarray(value,dtype='float32'))
    for c,name in enumerate(FIELDS):
        add('now_'+name,np.where(seen[ends,c],x[ends,c],np.nan));add('observed_'+name,seen[ends,c])
    for name,mask in [('powered',powered),('spray',spray),('hopper1_active',active1),('hopper2_active',active2)]: add('now_'+name,mask[ends])
    add('observed_operation_hours',(now-t[operation_start[ends]])/3600)
    # This is observed operating-segment history, never asserted true gun age.
    energy=np.r_[0.,np.cumsum(np.where(powered,x[:,5],0),dtype=float)]
    charge=np.r_[0.,np.cumsum(np.where(powered,x[:,4],0),dtype=float)]
    add('observed_operation_kwh',(energy[ends+1]-energy[operation_start[ends]])/3600)
    add('observed_operation_amp_hours',(charge[ends+1]-charge[operation_start[ends]])/3600)
    clock=index.astype('float64')/3600
    scopes=[('powered',powered,[0,1,4,5,6,7,9,10]),('spray',spray,[6]),
            ('active1',active1,[11,13,15]),('active2',active2,[18,20,22])]
    for scope,mask,cols in scopes:
        for c in cols:
            for window in [300,900,3600]:
                left=np.maximum(operation_start[ends],np.searchsorted(t,now-window+1))
                mean,std,slope,n=masked_window_statistics(x[:,c],mask&seen[:,c],ends,left,clock)
                for label,values in [('mean',mean),('std',std),('slope_per_hour',slope),('coverage',n/window)]: add(f'{scope}_{FIELDS[c]}_{label}_{window}s',values)
        status('feature_scope_complete',scope=scope,features=len(names),windows=len(ends))
    # Available completed products only; history is capped at 1800 s and stale
    # product summaries expire after 1800 s, keeping total context <=3600 s.
    all_products=pd.read_parquet(V1/'products.parquet');p=all_products[all_products.label_available_at.notna()].sort_values('label_available_at').reset_index(drop=True)
    pt=time_seconds(p.label_available_at);ps=time_seconds(p.start)
    pi=np.searchsorted(pt,now,'right')-1;safe=np.maximum(pi,0);prev=p.iloc[safe]
    usable=(pi>=0)&prev.label_valid.eq(1).to_numpy()&(pt[safe]>=t[operation_start[ends]])&(now-pt[safe]<=1800)&(ps[safe]>=now-3600)
    fields=['voltage_mean','voltage_std','voltage_min','voltage_max','feedrate_mean','pressure_mean','current_median','argon_median','nitrogen_median','hopper','station','coverage','voltage_fail','feedrate_fail','hopper_pressure_fail']
    for f in fields: add('product_'+f,np.where(usable,prev[f].to_numpy(dtype=float),np.nan))
    add('product_available',usable);add('product_age_seconds',np.where(usable,now-pt[safe],np.nan))
    trends=np.full((len(p),6),np.nan)
    pv=p.voltage_mean.to_numpy();pc=p.current_median.to_numpy();pa=p.argon_median.to_numpy();pn=p.nitrogen_median.to_numpy();pok=p.label_valid.eq(1).to_numpy()
    for j in range(len(p)):
        a=np.searchsorted(pt,pt[j]-1800);q=np.arange(a,j+1)
        at_index=max(0,np.searchsorted(t,pt[j],'right')-1)
        q=q[ps[q]>=t[operation_start[at_index]]]
        q=q[pok[q]&(np.abs(pc[q]-pc[j])<=10)&(np.abs(pa[q]-pa[j])<=1)&(np.abs(pn[q]-pn[j])<=.5)]
        if len(q)<3:continue
        u=(pt[q]-pt[q[-1]])/3600;v=pv[q];center=u-u.mean();den=np.sum(center**2)
        if den<=1e-8: continue
        trends[j]=[len(q),np.sum(center*(v-v.mean()))/den,v[-1]-v[0],np.std(v),v[-1]-51,(pt[q[-1]]-pt[q[0]])/3600]
    for j,n in enumerate(['count','voltage_slope_per_hour','voltage_delta','voltage_std','voltage_margin_51','observed_hours']): add('matched_product_'+n,np.where(usable,trends[safe,j],np.nan))
    stats=np.column_stack(feature_arrays).astype('float32');del feature_arrays,clock
    events=pd.read_parquet(V1/'events.parquet').sort_values('event_at')
    event_at=time_seconds(events.event_at);ei=np.searchsorted(event_at,now,'right')
    next_at=np.r_[event_at,np.iinfo('int64').max//2][ei]
    unknown=all_products[all_products.label_valid.ne(1)].sort_values('start');us=time_seconds(unknown.start);ue=time_seconds(unknown.end)
    ui=np.searchsorted(us,now,'right');obs_until=t[endblock[ends]].copy()
    if len(us):
        obs_until=np.minimum(obs_until,np.r_[us,np.iinfo('int64').max//2][ui]);up=np.maximum(ui-1,0)
        obs_until=np.where((ui>0)&(ue[up]>=now),now,obs_until)
    hy,hm,event,delta,observed=event_targets(now,next_at,obs_until)
    ids=np.r_[events.event_id.to_numpy(),np.array([''],dtype=object)][ei]
    type_y=np.column_stack([np.r_[events.first_types.str.split('|').map(lambda v:key in v),False][ei]&event for key in TYPES]).astype('float32')
    s=pd.DataFrame({'timestamp':pd.to_datetime(now,unit='s'),'end_index':ends,'partition':part[ends],
                    'event_in_300s':event,'observed_seconds':observed,'event_id':np.where(event,ids,''),
                    'eta_seconds':np.where(event,delta,np.nan),'context_start':pd.to_datetime(t[block[ends]],unit='s')})
    oldlife=pd.read_parquet(V1/'rul_samples.parquet');keep=['timestamp','possible_cycle']+[c for c in oldlife if c.startswith('rul_')]
    s=s.merge(oldlife[keep],on='timestamp',how='left',validate='one_to_one')
    # New chronological anomaly rows outside the old cycle-safe life split do not
    # acquire synthetic life labels.
    for flag in ['rul_valid','rul_event']:s[flag]=s[flag].eq(True)
    for name,array in [('grid_t',t),('grid_x',x),('grid_seen',seen),('grid_part',part),('grid_block',block),('grid_operation_start',operation_start),('features',stats)]: np.save(DATA/f'{name}.npy',array)
    s.to_parquet(DATA/'samples.parquet',index=False);np.savez(DATA/'targets.npz',hazard_y=hy,hazard_mask=hm,type_y=type_y)
    dump({'names':names,'raw_fields':FIELDS,'missing':'NaN; imputation and scaling fit separately on each training fold','operation_history':'resets after observed unpowered interval >=60s or data gap; not true electrode age'},DATA/'features.json')
    counts={str(k):{'windows':int(s.partition.eq(k).sum()),'positive_windows':int((s.partition.eq(k)&s.event_in_300s).sum()),
                   'events':s.loc[s.partition.eq(k)&s.event_in_300s,'event_id'].nunique(),
                   'life_windows':int((s.partition.eq(k)&s.rul_valid).sum())} for k in range(5)}
    ready={'grid_rows':len(t),'features':len(names),'counts':counts,'raw_parameters':28,'data_source':str(V1),
           'source_manifest_sha256':sha(V1/'sources.parquet'),'life_labels_sha256':sha(V1/'rul_samples.parquet'),
           'features_sha256':sha(DATA/'features.npy'),'samples_sha256':sha(DATA/'samples.parquet')}
    dump(ready,DATA/'ready.json');status('v2_data_ready',**ready)

if __name__=='__main__':main()
