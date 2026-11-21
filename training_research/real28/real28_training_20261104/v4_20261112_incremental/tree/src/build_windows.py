"""Causal feature cache, event/censor targets and conservative group splits."""
import sys
sys.dont_write_bytecode=True
import json
import numpy as np
import pandas as pd
from common import ROOT,DATA,FIELDS,TYPES,PARTITIONS,dump,status,sha

def seconds(x):
    return pd.to_datetime(x).to_numpy(dtype='datetime64[s]').astype('int64')

def scalar_time(x): return pd.Timestamp(x).value//10**9

def lookup_ids(values,indices):
    # np.r_[array, ''] treats the string as an indexing directive in NumPy.
    return np.concatenate([np.asarray(values,dtype=object),np.array([''],dtype=object)])[indices]

def terminal_chain(before):
    """Trace consecutive low products back to the last known normal/unknown.

    The one-hour comparison limit screens a replacement, not failure onset.
    A low-voltage evening followed by another low start the next morning must
    not be assigned a new failure onset on that following morning.
    """
    tail=[]
    for item in before.iloc[::-1].to_dict('records'):
        if item['label_valid']==1 and item['voltage_fail']==1:tail.append(item)
        else:break
    return tail

SPLIT_START=np.array([scalar_time(x) for x in ['2026-08-15','2026-10-04','2026-10-08','2026-10-10','2026-10-27']])
SPLIT_END=np.array([scalar_time(x) for x in ['2026-09-30','2026-10-08','2026-10-10','2026-10-24','2026-11-02']])-1

def assign_parts(t):
    part=np.full(len(t),-1,dtype=np.int8)
    for k,(a,b) in enumerate(zip(SPLIT_START,SPLIT_END)):part[(t>=a)&(t<=b)]=k
    return part

def hazard_targets(t, event_at, observable_until, horizon=300, bin_seconds=30):
    """Bin masks include the event bin; a censor inside a bin contributes none."""
    delta=event_at-t
    observed=np.maximum(0,np.minimum(observable_until-t,horizon))
    event=(delta>0)&(delta<=horizon)&(delta<=observed)
    idx=np.clip((np.maximum(delta,1)-1)//bin_seconds,0,horizon//bin_seconds-1).astype('int64')
    bins=np.arange(horizon//bin_seconds)[None,:]
    mask=bins<(observed//bin_seconds)[:,None]
    mask=np.where(event[:,None],bins<=idx[:,None],mask)
    target=(event[:,None]&(bins==idx[:,None])).astype('float32')
    return target,mask.astype('float32'),event,delta,observed

def rolling_stats(t,x,observed,ends,block_start):
    """Right aligned, past-only summaries, bounded by continuity AND partition."""
    cols=[6,4,0,1,7,9,10,11,18,15,22]
    out=[];names=[]
    a=x[:,cols].astype('float64');valid=np.isfinite(a)&observed[:,cols]
    finite=np.where(valid,a,0.0)
    sums=np.vstack([np.zeros((1,len(cols))),np.cumsum(finite,axis=0)])
    squares=np.vstack([np.zeros((1,len(cols))),np.cumsum(finite**2,axis=0)])
    counts=np.vstack([np.zeros((1,len(cols))),np.cumsum(valid,axis=0)])
    for window in [300,900,3600]:
        left=np.maximum(block_start[ends],np.searchsorted(t,t[ends]-window+1))
        n=counts[ends+1]-counts[left]
        mean=(sums[ends+1]-sums[left])/np.maximum(n,1)
        var=(squares[ends+1]-squares[left])/np.maximum(n,1)-mean**2
        out.extend([mean,np.sqrt(np.maximum(var,0)),np.minimum(n/window,1)])
        names.extend([f'{FIELDS[c]}_{name}_{window}s' for name in ['mean','std','coverage'] for c in cols])
    return np.concatenate(out,axis=1).astype('float32'),names

def main():
    if (DATA/'windows_ready.json').exists():print('Immutable window cache already exists.');return
    meta=json.loads((DATA/'prepared.json').read_text('utf-8'))
    raw_t=np.load(DATA/'timestamps.npy',mmap_mode='r');raw_x=np.load(DATA/'raw_values.npy',mmap_mode='r')
    p=pd.read_parquet(DATA/'products.parquet');events=pd.read_parquet(DATA/'events.parquet')
    candidates=pd.read_parquet(DATA/'replacement_candidates.parquet')
    inferred=candidates[candidates.candidate_kind.eq('after_voltage_failure')].sort_values('start').copy()
    resume=seconds(inferred.resume_at)
    # These groups are POSSIBLE electrode cycles, never asserted true gun IDs.
    raw_group=np.searchsorted(resume,raw_t,side='right')
    raw_part=assign_parts(raw_t)
    group_owner=np.array([raw_part[raw_group==g].max() for g in range(len(resume)+1)],dtype=np.int8)
    retained=(raw_part>=0)&(raw_part==group_owner[raw_group])
    # All gaps >2 s interrupt observable history; small gaps are represented with masks.
    splits=np.r_[0,np.flatnonzero((np.diff(raw_t)>2)|(np.diff(raw_part)!=0)|(np.diff(raw_group)!=0))+1,len(raw_t)]
    ts=[];xs=[];masks=[];starts=[];stops=[];parts=[];groups=[];cursor=0
    for a,b in zip(splits[:-1],splits[1:]):
        if not retained[a]:continue
        grid=np.arange(raw_t[a],raw_t[b-1]+1,dtype='int64')
        idx=np.searchsorted(raw_t[a:b],grid,side='right')-1
        vals=np.asarray(raw_x[a:b][idx]).copy()
        seen=(raw_t[a:b][idx]==grid)[:,None]&np.isfinite(vals)
        # Forward fill invalid cells within this block only; original validity retained.
        for c in range(28):
            good=seen[:,c]
            last=np.maximum.accumulate(np.where(good,np.arange(len(grid)),-1))
            vals[:,c]=np.where(last>=0,vals[np.maximum(last,0),c],np.nan)
        ts.append(grid);xs.append(vals);masks.append(seen)
        starts.append(np.full(len(grid),cursor,dtype='int64'));stops.append(np.full(len(grid),cursor+len(grid)-1,dtype='int64'))
        parts.append(np.full(len(grid),raw_part[a],dtype='int8'));groups.append(np.full(len(grid),raw_group[a],dtype='int16'))
        cursor+=len(grid)
    t=np.concatenate(ts);x=np.concatenate(xs);observed=np.concatenate(masks)
    block_start=np.concatenate(starts);block_end=np.concatenate(stops)
    part=np.concatenate(parts);group=np.concatenate(groups)
    powered=(x[:,4]>100)&(x[:,5]>5)
    # Uniform 10-second landmarks, full 120-second causal context, powered target scope.
    ends=np.flatnonzero((t%10==0)&powered&(np.arange(len(t))-block_start>=119)&observed[:,4]&observed[:,5])
    sample_t=t[ends];sample_part=part[ends];sample_group=group[ends]
    history_fraction=np.mean(observed[ends],axis=1)
    stats,names=rolling_stats(t,x,observed,ends,block_start)
    p=p.sort_values('label_available_at',na_position='last').copy()
    usable=p.label_available_at.notna()
    available=p[usable].copy();pt=seconds(available.label_available_at)
    last=np.searchsorted(pt,sample_t,side='right')-1
    previous=available.iloc[np.maximum(last,0)]
    previous_in_block=(last>=0)&(pt[np.maximum(last,0)]>=t[block_start[ends]])
    previous_valid=previous_in_block&previous.label_valid.eq(1).to_numpy()
    extras=previous[['voltage_mean','feedrate_mean','pressure_mean','current_median','argon_median','nitrogen_median']].to_numpy(dtype='float32')
    extras[~previous_valid]=0
    age=np.where(previous_in_block,np.minimum(sample_t-pt[np.maximum(last,0)],3600)/3600,1)
    stats=np.column_stack([stats,extras,previous_valid,age,history_fraction]).astype('float32')
    names+=['last_product_'+n for n in ['voltage','feedrate','pressure','current','argon','nitrogen']]+['last_product_valid','last_product_age_hours','current_sensor_fraction']
    current_y=previous[['voltage_fail','feedrate_fail','hopper_pressure_fail']].fillna(0).to_numpy(dtype='float32')
    current_y[~previous_valid]=0
    current_mask=np.repeat(previous_valid[:,None],3,axis=1).astype('float32')
    event_at=seconds(events.event_at);ev_index=np.searchsorted(event_at,sample_t,side='right')
    next_at=np.r_[event_at,np.iinfo('int64').max//2][ev_index]
    # Unknown product intervals stop negative follow-up; never turn missing labels into normal.
    unknown=p[p.label_valid.ne(1)].sort_values('start')
    unknown_start=seconds(unknown.start);unknown_end=seconds(unknown.end)+1
    next_u=np.searchsorted(unknown_start,sample_t,side='right')
    unknown_cut=np.r_[unknown_start,np.iinfo('int64').max//2][next_u]
    if len(unknown):
        prev_u=np.maximum(next_u-1,0)
        unknown_cut=np.where((next_u>0)&(unknown_end[prev_u]>=sample_t),sample_t,unknown_cut)
    obs_until=np.minimum(t[block_end[ends]],unknown_cut)
    y,mask,event,delta,obs=hazard_targets(sample_t,next_at,obs_until)
    type_y=np.zeros((len(ends),3),dtype='float32')
    for j,key in enumerate(TYPES):
        flags=events.first_types.str.split('|').map(lambda a:key in a).to_numpy()
        type_y[:,j]=np.r_[flags,False][ev_index]&event
    # Natural endpoint proxies retain candidate evidence and the early-instability guard.
    pp=p.sort_values('start').copy(); terminal=[]
    raw_power=(raw_x[:,4]>100)&(raw_x[:,5]>5)
    raw_energy=np.r_[0,np.cumsum(raw_power,dtype='float64')]
    previous_resume=int(raw_t[0])
    for _,r in inferred.iterrows():
        before=pp[pp.end.lt(r.start)&pp.start.ge(pd.Timestamp(previous_resume,unit='s'))]
        tail=terminal_chain(before)
        if len(tail)<2:continue
        first=tail[-1]
        observed_hours=(raw_energy[np.searchsorted(raw_t,scalar_time(first['start']))]-raw_energy[np.searchsorted(raw_t,previous_resume)])/3600
        eligible=observed_hours>=10
        terminal.append(dict(candidate_id=r.candidate_id,product_id=first['product_id'],
            failure_lower=first['start'],failure_upper=first['label_available_at'],
            replacement_start=r.start,replacement_end=r.end,
            observed_powered_hours_since_previous_candidate=observed_hours,
            eligible=eligible,evidence='weak low-voltage-stop/recovery proxy; no individual natural-failure confirmation',
            early_instability_status='not recently after previous inferred change' if eligible else 'excluded: fewer than 10 observed powered hours since preceding candidate',
            physical_cycle_identity_confirmed=False))
        previous_resume=scalar_time(r.resume_at)
    terminal=pd.DataFrame(terminal)
    terminal.to_csv(DATA/'natural_endpoint_evidence.csv',index=False,encoding='utf-8-sig')
    terminal.to_parquet(DATA/'natural_endpoint_evidence.parquet',index=False)
    # A known low product (including a non-terminal one) ends an at-risk prefix.
    low=pp[pp.label_valid.eq(1)&pp.voltage_fail.eq(1)].copy()
    low_at=seconds(low.start);next_low=np.searchsorted(low_at,sample_t,side='right')
    low_cut=np.r_[low_at,np.iinfo('int64').max//2][next_low]
    end_time=np.minimum(obs_until,low_cut)
    energy=np.r_[0,np.cumsum(powered,dtype='float64')]
    upper_time=end_time.copy();rul_event=np.zeros(len(ends),dtype=bool);endpoint_id=np.full(len(ends),'',dtype=object)
    term_map={r['product_id']:r for r in terminal[terminal.eligible].to_dict('records')}
    next_low_ids=lookup_ids(low.product_id.to_numpy(),next_low)
    for pid,r in term_map.items():
        lo=scalar_time(r['failure_lower']);hi=scalar_time(r['failure_upper'])
        hit=(next_low_ids==pid)&(lo>sample_t)&(hi<=obs_until)
        rul_event[hit]=True;end_time[hit]=lo;upper_time[hit]=hi;endpoint_id[hit]=r['candidate_id']
    lower_ix=np.searchsorted(t,end_time,side='left');upper_ix=np.searchsorted(t,upper_time,side='right')
    # Half-open integration uses observations strictly AFTER landmark t.
    lower_hours=np.maximum(0,energy[np.minimum(lower_ix,len(t))]-energy[ends+1])/3600
    upper_hours=np.maximum(lower_hours,energy[np.minimum(upper_ix,len(t))]-energy[ends+1])/3600
    product_end=np.sort(seconds(pp[pp.left_boundary_observed&pp.right_boundary_observed].end))
    lower_pieces=np.maximum(0,np.searchsorted(product_end,end_time,side='left')-np.searchsorted(product_end,sample_t,side='right'))
    upper_pieces=np.maximum(lower_pieces,np.searchsorted(product_end,upper_time,side='right')-np.searchsorted(product_end,sample_t,side='right'))
    rul_valid=previous_valid&previous.voltage_fail.eq(0).to_numpy()&np.where(rul_event,lower_hours>0,lower_hours>=1/60)
    # Reject clocks in an already-started low product; no false 'not yet failed' state.
    low_end=seconds(low.end)
    if len(low):
        li=np.maximum(next_low-1,0)
        rul_valid&=~((next_low>0)&(low_end[li]>=sample_t))
    # Whole possible-cycle groups are assigned to a single partition.
    for k in range(5):
        assert np.all((sample_t[sample_part==k]>=SPLIT_START[k]+119)&(sample_t[sample_part==k]<=SPLIT_END[k]))
    train=sample_part==0
    center=np.nanmedian(x[ends[train]],axis=0)
    scale=np.nanstd(x[ends[train]],axis=0);scale=np.where(scale>.01,scale,1)
    stat_center=np.nanmean(stats[train],axis=0);stat_scale=np.nanstd(stats[train],axis=0);stat_scale=np.where(stat_scale>.001,stat_scale,1)
    normalized=np.nan_to_num((stats-stat_center)/stat_scale,nan=0,posinf=0,neginf=0).clip(-20,20).astype('float32')
    dump({'fields':FIELDS,'center':center,'scale':scale,'stats_names':names,'stats_center':stat_center,'stats_scale':stat_scale,
          'fit_partition':'train','window_seconds':120,'step_seconds':10,'time_basis':'Asia/Shanghai instrument clock; future window uses natural seconds'},DATA/'normalization.json')
    for name,array in [('grid_t',t),('grid_values',x.astype('float32')),('grid_observed',observed),('grid_part',part),('grid_group',group),('grid_block_start',block_start),('sample_stats',normalized)]:np.save(DATA/f'{name}.npy',array)
    samples=pd.DataFrame(dict(end_index=ends,timestamp=pd.to_datetime(sample_t,unit='s'),partition=sample_part,possible_cycle=sample_group,
        event_in_300s=event,event_id=np.where(event,lookup_ids(events.event_id.to_numpy(),ev_index),''),
        next_event_at=np.where(event,next_at,0),eta_seconds=np.where(event,delta,np.nan),observed_seconds=obs,
        rul_valid=rul_valid,rul_event=rul_event,rul_lower_hours=lower_hours,rul_upper_hours=np.where(rul_event,upper_hours,np.inf),
        rul_lower_pieces=lower_pieces,rul_upper_pieces=np.where(rul_event,upper_pieces,np.inf),rul_endpoint_id=endpoint_id))
    samples.to_parquet(DATA/'samples.parquet',index=False)
    np.savez(DATA/'targets.npz',hazard_y=y,hazard_mask=mask,type_y=type_y,current_y=current_y,current_mask=current_mask)
    counts={PARTITIONS[k]:dict(windows=int((sample_part==k).sum()),positive_windows=int((event&(sample_part==k)).sum()),
        independent_events=int(samples.loc[(sample_part==k)&event,'event_id'].nunique()),rul_windows=int((rul_valid&(sample_part==k)).sum()),
        rul_event_windows=int((rul_valid&rul_event&(sample_part==k)).sum()),
        rul_endpoints=int(samples.loc[rul_valid&rul_event&(sample_part==k),'rul_endpoint_id'].nunique())) for k in range(5)}
    ready={'partition_counts':counts,'grid_rows':len(t),'windows':len(ends),'stats_features':stats.shape[1],
       'raw_rows_removed_by_group_isolation':int((~retained).sum()),'possible_cycle_owner':group_owner,
       'physical_cycle_identity_confirmed':False,'grouping_basis':'conservative inferred post-low-voltage change boundaries, not true gun IDs',
       'natural_endpoint_candidates':len(terminal),'natural_endpoint_eligible':int(terminal.eligible.sum()),
       'natural_endpoint_evidence':'all weak proxies; early-age observation guard does not prove true electrode age',
       'data_manifest_sha256':meta['sources_sha256'],'normalization_sha256':sha(DATA/'normalization.json')}
    dump(ready,DATA/'windows_ready.json');status('windows_prepared',**ready)

if __name__=='__main__':main()
