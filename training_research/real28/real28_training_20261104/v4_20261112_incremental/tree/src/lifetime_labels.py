"""Separate lifetime labels: uncertain interventions truncate follow-up.

This file does not alter the frozen anomaly/representation training inputs.
"""
import numpy as np
import pandas as pd
from common import DATA,dump,sha
from build_windows import seconds,scalar_time,lookup_ids

def make_lifetime_labels():
    out=DATA/'rul_samples.parquet'
    if out.exists():return pd.read_parquet(out)
    s=pd.read_parquet(DATA/'samples.parquet').copy();p=pd.read_parquet(DATA/'products.parquet').sort_values('start')
    candidates=pd.read_parquet(DATA/'replacement_candidates.parquet').sort_values('start')
    terminal=pd.read_parquet(DATA/'natural_endpoint_evidence.parquet')
    t=np.load(DATA/'grid_t.npy',mmap_mode='r');x=np.load(DATA/'grid_values.npy',mmap_mode='r');block=np.load(DATA/'grid_block_start.npy',mmap_mode='r')
    ends=s.end_index.to_numpy();now=t[ends]
    blockcuts=np.r_[np.flatnonzero(np.diff(block)!=0)+1,len(t)]
    end_idx=blockcuts[np.searchsorted(blockcuts,ends,side='right')]-1
    observable=t[end_idx].copy()
    # Unknown product evidence also truncates lifetime follow-up.
    unknown=p[p.label_valid.ne(1)]
    starts=seconds(unknown.start);ue=seconds(unknown.end)
    index=np.searchsorted(starts,now,side='right')
    observable=np.minimum(observable,np.r_[starts,np.iinfo('int64').max//2][index])
    if len(starts):
        prev=np.maximum(index-1,0)
        observable=np.where((index>0)&(ue[prev]>=now),now,observable)
    # Baseline shifts do NOT reset age or prove changes, but their possible
    # intervention makes linking a precise future lifetime across them unsafe.
    ambiguous=candidates[candidates.candidate_kind.ne('after_voltage_failure')]
    intervention_at=seconds(ambiguous.start)
    ai=np.searchsorted(intervention_at,now,side='right')
    observable=np.minimum(observable,np.r_[intervention_at,np.iinfo('int64').max//2][ai])
    low=p[p.label_valid.eq(1)&p.voltage_fail.eq(1)]
    low_at=seconds(low.start);li=np.searchsorted(low_at,now,side='right')
    low_end=seconds(low.end);low_ids=lookup_ids(low.product_id.to_numpy(),li)
    lower=np.minimum(observable,np.r_[low_at,np.iinfo('int64').max//2][li]);upper=lower.copy()
    event=np.zeros(len(s),bool);endpoint=np.full(len(s),'',object)
    for r in terminal[terminal.eligible].itertuples():
        lo=scalar_time(r.failure_lower);hi=scalar_time(r.failure_upper)
        hit=(low_ids==r.product_id)&(lo>now)&(hi<=observable)
        event[hit]=True;lower[hit]=lo;upper[hit]=hi;endpoint[hit]=r.candidate_id
    available=p[p.label_available_at.notna()].sort_values('label_available_at')
    at=seconds(available.label_available_at);previous=np.searchsorted(at,now,side='right')-1
    rows=available.iloc[np.maximum(previous,0)]
    resumes=seconds(candidates.resume_at);ri=np.searchsorted(resumes,now,side='right')
    last_resume=np.r_[np.iinfo('int64').min//2,resumes][ri]
    previous_ok=(previous>=0)&rows.label_valid.eq(1).to_numpy()&rows.voltage_fail.eq(0).to_numpy()
    previous_ok&=(at[np.maximum(previous,0)]>=np.maximum(t[block[ends]],last_resume))
    if len(low):previous_ok&=~((li>0)&(low_end[np.maximum(li-1,0)]>=now))
    powered=(x[:,4]>100)&(x[:,5]>5);energy=np.r_[0,np.cumsum(powered,dtype='float64')]
    lower_ix=np.minimum(np.searchsorted(t,lower,'left'),len(t));upper_ix=np.minimum(np.searchsorted(t,upper,'right'),len(t))
    loh=np.maximum(0,energy[lower_ix]-energy[ends+1])/3600
    hih=np.maximum(loh,energy[upper_ix]-energy[ends+1])/3600
    prod_end=np.sort(seconds(p[p.left_boundary_observed&p.right_boundary_observed].end))
    lop=np.maximum(0,np.searchsorted(prod_end,lower,'left')-np.searchsorted(prod_end,now,'right'))
    hip=np.maximum(lop,np.searchsorted(prod_end,upper,'right')-np.searchsorted(prod_end,now,'right'))
    valid=previous_ok&np.where(event,loh>0,loh>=1/60)
    s['rul_valid']=valid;s['rul_event']=event;s['rul_lower_hours']=loh;s['rul_upper_hours']=np.where(event,hih,np.inf)
    s['rul_lower_pieces']=lop;s['rul_upper_pieces']=np.where(event,hip,np.inf);s['rul_endpoint_id']=endpoint
    s['rul_observation_end']=pd.to_datetime(lower,unit='s')
    s.to_parquet(out,index=False)
    counts={str(k):{'valid':int((valid&s.partition.eq(k)).sum()),'event_windows':int((valid&event&s.partition.eq(k)).sum()),
        'endpoints':int(s.loc[valid&event&s.partition.eq(k),'rul_endpoint_id'].nunique())} for k in range(5)}
    dump({'counts':counts,'ambiguous_interventions_censor_followup':len(ambiguous),'normalization_changed':False,
          'anomaly_targets_changed':False,'true_natural_failures_confirmed':0,
          'assumption':'available product normality supports an operational, weak survival prefix; no proven electrode identity',
          'file_sha256':sha(out)},DATA/'rul_label_audit.json')
    return s

if __name__=='__main__':
    make_lifetime_labels();print((DATA/'rul_label_audit.json').read_text('utf-8'))
