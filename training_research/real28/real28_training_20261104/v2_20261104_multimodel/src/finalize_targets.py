"""Finalize masks from ALL unknown products, including unavailable result times."""
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from common_v2 import DATA,V1,TYPES,load,dump,sha,time_seconds,status
from prepare_v2 import event_targets

def main():
    if (DATA/'targets_finalized.json').exists():return
    s=pd.read_parquet(DATA/'samples.parquet');t=np.load(DATA/'grid_t.npy',mmap_mode='r');block=np.load(DATA/'grid_block.npy',mmap_mode='r')
    end=s.end_index.to_numpy();now=t[end];cuts=np.r_[np.flatnonzero(np.diff(block)!=0)+1,len(t)]
    until=t[cuts[np.searchsorted(cuts,end,'right')]-1].copy()
    all_products=pd.read_parquet(V1/'products.parquet');u=all_products[all_products.label_valid.ne(1)].sort_values('start')
    us=time_seconds(u.start);ue=time_seconds(u.end);j=np.searchsorted(us,now,'right')
    until=np.minimum(until,np.r_[us,np.iinfo('int64').max//2][j]);prev=np.maximum(j-1,0)
    until=np.where((j>0)&(ue[prev]>=now),now,until)
    e=pd.read_parquet(V1/'events.parquet').sort_values('event_at');at=time_seconds(e.event_at);i=np.searchsorted(at,now,'right')
    nxt=np.r_[at,np.iinfo('int64').max//2][i];hy,hm,event,delta,observed=event_targets(now,nxt,until)
    changed=int(np.sum(s.observed_seconds.to_numpy()!=observed))
    s['event_in_300s']=event;s['observed_seconds']=observed;s['eta_seconds']=np.where(event,delta,np.nan)
    s['event_id']=np.where(event,np.r_[e.event_id.to_numpy(),np.array([''],dtype=object)][i],'')
    ty=np.column_stack([np.r_[e.first_types.str.split('|').map(lambda a:k in a),False][i]&event for k in TYPES]).astype('float32')
    s.to_parquet(DATA/'samples.parquet',index=False);np.savez(DATA/'targets.npz',hazard_y=hy,hazard_mask=hm,type_y=ty)
    ready=load(DATA/'ready.json');ready['samples_sha256']=sha(DATA/'samples.parquet');ready['targets_sha256']=sha(DATA/'targets.npz');dump(ready,DATA/'ready.json')
    result={'all_unknown_products':len(u),'without_available_result_time':int(u.label_available_at.isna().sum()),'changed_observation_masks':changed,'before_any_training':True}
    dump(result,DATA/'targets_finalized.json');status('targets_finalized',**result)

if __name__=='__main__':main()
