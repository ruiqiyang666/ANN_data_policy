"""Revised, evidence-traceable weak labels and causal multiscale features."""
import sys
sys.dont_write_bytecode=True
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from common_v2 import ROOT,V1,GRID,SHARED,FIELDS,TYPES,dump,sha,time_seconds as sec

def targets(now,next_at,observable):
    delta=next_at-now;observed=np.maximum(0,np.minimum(300,observable-now))
    event=(delta>0)&(delta<=300)&(delta<=observed)
    b=np.clip((np.maximum(delta,1)-1)//30,0,9).astype(int);bins=np.arange(10)[None,:]
    mask=np.where(event[:,None],bins<=b[:,None],bins<(observed//30)[:,None])
    return (event[:,None]&(bins==b[:,None])).astype('float32'),mask.astype('float32'),event,delta,observed

def window_stats(v,valid,ends,left,clock):
    def cs(a):return np.r_[0.,np.cumsum(a,dtype=float)]
    def diff(a):return a[ends+1]-a[left]
    a=np.where(valid,v,0);n=diff(cs(valid));mean=diff(cs(a))/np.maximum(n,1)
    std=np.sqrt(np.maximum(0,diff(cs(a*a))/np.maximum(n,1)-mean*mean))
    st=diff(cs(clock*valid));den=diff(cs(clock*clock*valid))-st*st/np.maximum(n,1)
    slope=(diff(cs(clock*a))-st*mean)/np.maximum(den,1e-8)
    mean[n<1]=np.nan;std[n<2]=np.nan;slope[(n<10)|(den<1e-6)]=np.nan
    return mean,std,slope,n

def build_episodes(p,key):
    """Two observed normal products resolve an episode; overnight alone does not.

    This is a retrospective target definition, never a live gun-change claim.
    Known ongoing results suppress new-event followup until observed recovery.
    """
    flag={'voltage':'voltage_fail','feedrate':'feedrate_fail','hopper_pressure':'hopper_pressure_fail'}[key]
    episodes=[];active=None;normals=0
    for r in p.sort_values('start').itertuples():
        if r.label_valid!=1 or pd.isna(r.label_available_at):
            # Unknown does not prove recovery.
            continue
        if getattr(r,flag)==1:
            if active is None:
                active={'event_id':f'{key}_{len(episodes)+1:03d}','first_product_start':r.start,
                        'event_at':r.start,'first_result_at':r.label_available_at,'first_types':key,
                        'product_id':r.product_id,'recovery_at':pd.NaT,'failed_products':0}
                episodes.append(active)
            active['failed_products']+=1;normals=0
        elif active is not None:
            normals+=1
            if normals>=2:
                active['recovery_at']=r.label_available_at;active=None;normals=0
    return pd.DataFrame(episodes)

def endpoint_review(p,c):
    terminal=pd.read_parquet(V1/'natural_endpoint_evidence.parquet').copy()
    for i,r in terminal.iterrows():
        # Expand onset uncertainty to a preceding low product separated by at
        # most one normal within 15 minutes. Do not include distant early dips.
        a=p[(p.start<r.failure_lower)&(p.start>=r.failure_lower-pd.Timedelta(minutes=15))].sort_values('start')
        preceding=a.tail(2)
        prior=preceding[preceding.label_valid.eq(1)&preceding.voltage_fail.eq(1)]
        first=prior.iloc[0] if len(prior) else p[p.product_id.eq(r.product_id)].iloc[0]
        terminal.loc[i,'terminal_product_id']=r.product_id
        terminal.loc[i,'product_id']=first.product_id
        terminal.loc[i,'failure_lower']=first.start
        terminal.loc[i,'eligible']=True
        terminal.loc[i,'early_instability_status']='sustained terminal low-voltage sequence plus stop/recovery; no blanket 10h exclusion'
        terminal.loc[i,'evidence']='weak interval from associated first instability to sustained low-product result; NOT confirmed natural failure'
    terminal.to_parquet(SHARED/'natural_endpoint_evidence.parquet',index=False)
    terminal.to_csv(SHARED/'natural_endpoint_evidence.csv',index=False,encoding='utf-8-sig')
    c=c.copy();c['review_decision']='uncertain intervention: right censor, not failure'
    c.loc[c.candidate_kind.eq('after_voltage_failure'),'review_decision']='inferred low-voltage stop/recovery; possible cycle boundary'
    c.loc[c.candidate_id.eq('R009'),'review_decision']='company-confirmed ordinary wait; no replacement/censor boundary'
    c.to_csv(SHARED/'candidate_review.csv',index=False,encoding='utf-8-sig')
    return terminal,c

def lifetime(s,p,c,terminal,t,x,block,block_end):
    now=sec(s.timestamp);ends=s.end_index.to_numpy();observable=t[block_end[ends]].copy()
    inf=np.iinfo('int64').max//2
    unknown=p[p.label_valid.ne(1)].sort_values('start');us=sec(unknown.start);ue=sec(unknown.end)
    ui=np.searchsorted(us,now,'right');observable=np.minimum(observable,np.r_[us,inf][ui])
    if len(us):observable=np.where((ui>0)&(ue[np.maximum(ui-1,0)]>=now),now,observable)
    amb=c[c.candidate_kind.ne('after_voltage_failure')&c.candidate_id.ne('R009')].sort_values('start')
    ai=sec(amb.start);observable=np.minimum(observable,np.r_[ai,inf][np.searchsorted(ai,now,'right')])
    low=p[p.label_valid.eq(1)&p.voltage_fail.eq(1)].sort_values('start');lt=sec(low.start);li=np.searchsorted(lt,now,'right')
    lowids=np.r_[low.product_id.to_numpy(),['']][li]
    lower=np.minimum(observable,np.r_[lt,inf][li]);upper=lower.copy();event=np.zeros(len(s),bool);endpoint=np.full(len(s),'',object)
    for r in terminal.itertuples():
        lo=int(sec(r.failure_lower));hi=int(sec(r.failure_upper))
        hit=(lowids==r.product_id)&(lo>now)&(hi<=observable)
        event[hit]=True;lower[hit]=lo;upper[hit]=hi;endpoint[hit]=r.candidate_id
    available=p[p.label_available_at.notna()].sort_values('label_available_at');at=sec(available.label_available_at)
    j=np.searchsorted(at,now,'right')-1;prev=available.iloc[np.maximum(j,0)]
    rc=c[c.candidate_kind.eq('after_voltage_failure')].sort_values('resume_at');resumes=sec(rc.resume_at)
    group=np.searchsorted(resumes,now,'right');last=np.r_[np.iinfo('int64').min//2,resumes][group]
    previous=(j>=0)&prev.label_valid.eq(1).to_numpy()&prev.voltage_fail.eq(0).to_numpy()&(at[np.maximum(j,0)]>=np.maximum(t[block[ends]],last))
    le=sec(low.end);previous&=~((li>0)&(le[np.maximum(li-1,0)]>=now))
    powered=(x[:,4]>100)&(x[:,5]>5);energy=np.r_[0,np.cumsum(powered,dtype=float)]
    loix=np.minimum(np.searchsorted(t,lower,'left'),len(t));hiix=np.minimum(np.searchsorted(t,upper,'right'),len(t))
    loh=np.maximum(0,energy[loix]-energy[ends+1])/3600;hih=np.maximum(loh,energy[hiix]-energy[ends+1])/3600
    pe=np.sort(sec(p[p.left_boundary_observed&p.right_boundary_observed].end))
    lop=np.maximum(0,np.searchsorted(pe,lower,'left')-np.searchsorted(pe,now,'right'))
    hip=np.maximum(lop,np.searchsorted(pe,upper,'right')-np.searchsorted(pe,now,'right'))
    valid=previous&np.where(event,loh>0,loh>=1/60)
    # A possible cycle belongs entirely to its last chronological partition.
    for g in np.unique(group):valid[(group==g)&s.partition.ne(s.loc[group==g,'partition'].max()).to_numpy()]=False
    s['possible_cycle']=group;s['rul_valid']=valid;s['rul_event']=event;s['rul_lower_hours']=loh
    s['rul_upper_hours']=np.where(event,hih,np.inf);s['rul_lower_pieces']=lop;s['rul_upper_pieces']=np.where(event,hip,np.inf)
    s['rul_endpoint_id']=endpoint;s['rul_observation_end']=pd.to_datetime(lower,unit='s')
    return s

def main():
    if (SHARED/'ready.json').exists():print('Prepared data already immutable');return
    if list((ROOT/'models').rglob('*.joblib')):raise RuntimeError('Cannot rebuild after fitting')
    SHARED.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT/'protocol.json').read_text('utf-8'))  # declared before preparation
    prepared=json.loads((V1/'prepared.json').read_text('utf-8'))
    p=pd.read_parquet(V1/'products.parquet').sort_values('start');c=pd.read_parquet(V1/'replacement_candidates.parquet')
    terminal,c=endpoint_review(p,c)
    t=np.load(GRID/'grid_t.npy',mmap_mode='r');x=np.load(GRID/'grid_x.npy',mmap_mode='r');seen=np.load(GRID/'grid_seen.npy',mmap_mode='r')
    block=np.load(GRID/'grid_block.npy',mmap_mode='r');s=pd.read_parquet(GRID/'samples.parquet');ends=s.end_index.to_numpy();now=sec(s.timestamp)
    cuts=np.r_[np.flatnonzero(np.diff(block)!=0)+1,len(t)];block_end=cuts[np.searchsorted(cuts,np.arange(len(t)),'right')]-1
    features=np.load(GRID/'features.npy',mmap_mode='r');names=json.loads((GRID/'features.json').read_text('utf-8'))['names'];extra=[]
    def add(n,a):names.append(n);extra.append(np.asarray(a,dtype='float32'))
    powered=(x[:,4]>100)&(x[:,5]>5)&seen[:,4]&seen[:,5];clock=np.arange(len(t),dtype=float)/3600
    left=np.maximum(block[ends],np.searchsorted(t,now-7199))
    for col in [6,4,0,1]:
        mean,std,slope,n=window_stats(x[:,col],powered&seen[:,col],ends,left,clock)
        for label,a in [('mean',mean),('std',std),('slope_per_hour',slope),('coverage',n/7200)]:add(f'history_{FIELDS[col]}_{label}_7200s',a)
        if col==6:add('history_observed_powered_hours',n/3600)
    available=p[p.label_available_at.notna()].sort_values('label_available_at').reset_index(drop=True);pt=sec(available.label_available_at)
    ps=sec(available.start);pv=available.voltage_mean.to_numpy();pc=available.current_median.to_numpy();pa=available.argon_median.to_numpy();pn=available.nitrogen_median.to_numpy();pok=available.label_valid.eq(1).to_numpy()
    # Compute as of each landmark, not as of the previous product; stale history expires causally.
    long=np.full((len(s),9),np.nan,dtype='float32')
    for j,n in enumerate(now):
        a=np.searchsorted(pt,max(n-7200,int(t[block[ends[j]]])));b=np.searchsorted(pt,n,'right')
        q=np.arange(a,b)
        if len(q)<3:continue
        last=b-1;q=q[pok[q]&(ps[q]>=n-10800)&(np.abs(pc[q]-pc[last])<=10)&(np.abs(pa[q]-pa[last])<=1)&(np.abs(pn[q]-pn[last])<=.5)]
        if len(q)<3:continue
        u=(pt[q]-pt[q[-1]])/3600;v=pv[q];center=u-u.mean();den=center@center
        if den<=1e-8:continue
        slope=float(center@(v-v.mean())/den);drop=float(v.max()-v[-1]);margin=float(v[-1]-51)
        long[j]=[drop,slope,float(v.std()),len(q),margin,float(slope>.1),float(slope<-.1),float(drop<.25),float((n-pt[q[-1]])/3600)]
    for j,n in enumerate(['peak_drop','voltage_slope','voltage_std','count','margin_51','rising','falling','near_peak','age_hours']):add(('phase_'+n if n in ['rising','falling','near_peak'] else 'matched_long_'+n),long[:,j])
    np.save(SHARED/'features.npy',np.column_stack([features,*extra]).astype('float32'));del features,extra,long
    spec={'names':names,'raw_fields':FIELDS,'history_bound_seconds':10800,'pause_handling':'2h history crosses ordinary unpowered pauses, resets at observed data gaps; no true gun-age assertion'}
    s=lifetime(s.copy(),p,c,terminal,t,x,block,block_end)
    for task in TYPES+['life']:
        dest=SHARED/task;dest.mkdir(exist_ok=True);dump(spec,dest/'features.json')
        rows=s.copy()
        if task!='life':
            events=build_episodes(p,task);events.to_parquet(dest/'events.parquet',index=False)
            events.to_csv(dest/'events.csv',index=False,encoding='utf-8-sig');et=sec(events.event_at);ix=np.searchsorted(et,now,'right');inf=np.iinfo('int64').max//2
            nxt=np.r_[et,inf][ix];obs=t[block_end[ends]].copy()
            u=p[p.label_valid.ne(1)].sort_values('start');us=sec(u.start);ue=sec(u.end);ui=np.searchsorted(us,now,'right')
            obs=np.minimum(obs,np.r_[us,inf][ui]);obs=np.where((ui>0)&(ue[np.maximum(ui-1,0)]>=now),now,obs)
            for r in events.itertuples():
                start=int(sec(r.first_result_at));stop=int(sec(r.recovery_at)) if pd.notna(r.recovery_at) else inf
                obs=np.where((now>=start)&(now<stop),now,obs)
            hy,hm,event,delta,observed=targets(now,nxt,obs)
            rows['event_in_300s']=event;rows['observed_seconds']=observed;rows['eta_seconds']=np.where(event,delta,np.nan)
            rows['event_id']=np.where(event,np.r_[events.event_id.to_numpy(),['']][ix],'')
            ty=np.zeros((len(s),3),dtype='float32');ty[:,TYPES.index(task)]=event
            np.savez(dest/'targets.npz',hazard_y=hy,hazard_mask=hm,type_y=ty)
        rows.to_parquet(dest/'samples.parquet',index=False)
        dump({'finalized':True,'task':task},dest/'targets_finalized.json')
        counts={str(k):{'windows':int(rows.partition.eq(k).sum()),'events':int(rows.loc[rows.partition.eq(k)&rows.event_in_300s,'event_id'].nunique()) if task!='life' else None,
                           'life_valid':int((rows.partition.eq(k)&rows.rul_valid).sum()),'weak_endpoints':int(rows.loc[rows.partition.eq(k)&rows.rul_valid&rows.rul_event,'rul_endpoint_id'].nunique())} for k in range(5)}
        dump({'task':task,'counts':counts},dest/'ready.json');print(task,counts,flush=True)
    dump(spec,SHARED/'features.json')
    dump({'source_manifest_sha256':sha(V1/'sources.parquet'),'feature_sha256':sha(SHARED/'features.npy'),'features':len(names),'raw_rows':prepared['rows'],'main_csv':prepared['source_files'],'rows_through_20261023':prepared['rows_through_20261023'],'confirmed_natural_failures':0},SHARED/'ready.json')

if __name__=='__main__':main()
