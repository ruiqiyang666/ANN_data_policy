"""Build targets and causal fixed-calendar windows from the immutable 11_16 handoff."""
from pathlib import Path
import argparse,hashlib,json,shutil
import numpy as np
import pandas as pd
from modeling import FIELDS,TARGETS,fit_preprocessor,encode_sequences,seconds

FMT='%d %m %Y %H:%M:%S'
DEFAULT_DATA=Path(__file__).resolve().parents[2]/'training_data'
ROOT=Path(__file__).resolve().parents[1]

def window_target(delta,horizon,window):
    delta=np.asarray(delta,dtype=float);horizon=np.asarray(horizon,dtype=float)
    positive=(delta>0)&(delta<=window)&(delta<horizon)
    return np.where(positive,1,np.where(horizon>window,0,np.nan)).astype(np.float32)

def construct(data):
    f=pd.read_csv(data/'features_10s.csv.gz')
    t=pd.read_csv(data/'targets_10s.csv.gz')
    p=pd.read_csv(data/'products.csv',low_memory=False);c=pd.read_csv(data/'electrode_cycles.csv')
    assert f.anchor_id.equals(t.anchor_id) and f.timestamp.equals(t.timestamp)
    assert c.groupby('cycle_id').split.nunique().max()==1
    assert f.groupby('cycle_id').split.nunique().max()==1
    frame=f[['anchor_id','timestamp','cycle_id','split']].copy()
    frame['timestamp']=pd.to_datetime(frame.timestamp,format=FMT)
    for col in FIELDS: frame[col]=f[col+'_last']
    for col in ['start','end']: p[col]=pd.to_datetime(p[col],format=FMT)
    c['eol_endpoint']=pd.to_datetime(c.eol_endpoint,format=FMT)
    # Conservative start confirmation latency; current product final data is never used.
    now=seconds(frame.timestamp)
    keep=np.ones(len(frame),dtype=bool)
    for cycle,idx in frame.groupby('cycle_id',sort=False).indices.items():
        q=p[p.cycle_id.eq(cycle)].sort_values('start')
        starts=seconds(q.start);ends=seconds(q.end)
        j=np.searchsorted(starts,now[idx],side='right')-1
        safe=np.maximum(j,0)
        keep[idx]=(j>=0)&(now[idx]<=ends[safe])&(now[idx]-starts[safe]>=3)
    frame=frame.loc[keep].reset_index(drop=True);t=t.loc[keep].reset_index(drop=True)
    now=seconds(frame.timestamp);h=t.observation_horizon_seconds.to_numpy(float)
    y=np.full((len(frame),len(TARGETS)),np.nan,dtype=np.float32)
    delta=t.rul_strict_seconds.to_numpy(float)
    y[:,0:3]=np.stack([window_target(delta,h,m*60) for m in [15,30,60]],axis=1)
    # Weak endpoint/unknown-damage periods must not become strict negative evidence.
    unknown_cycles=c.loc[c.eol_sensor_observed.eq(1)|c.end_type.eq('replacement_after_reported_damage'),'cycle_id']
    y[frame.cycle_id.isin(unknown_cycles),:3]=np.nan
    for start,col in [(3,'adjacent_two_voltage_seconds'),(6,'adjacent_two_any_seconds')]:
        y[:,start:start+3]=np.stack([window_target(t[col],h,m*60) for m in [15,30,60]],axis=1)
    next_product=np.full(len(frame),'',dtype=object)
    # Predict the first future START, not the unfinished current product's final label.
    for cycle,idx in frame.groupby('cycle_id',sort=False).indices.items():
        q=p[p.cycle_id.eq(cycle)].sort_values('start').reset_index(drop=True)
        starts=seconds(q.start);ends=seconds(q.end)
        j=np.searchsorted(starts,now[idx],side='right');safe=np.minimum(j,len(q)-1)
        valid=(j<len(q)) & q.supervised_product_eligible.to_numpy()[safe].astype(bool)
        valid &= (ends[safe]-now[idx]>0)&(ends[safe]-now[idx]<h[idx])
        valid &= ends[safe]-now[idx]<=3600
        next_product[idx]=np.where(valid,q.product_id.to_numpy()[safe],'')
        for k,col in enumerate(['failure','voltage_fail','feedrate_fail','hopper_pressure_fail']):
            vals=q[col].to_numpy(float)[safe]
            y[idx,9+k]=np.where(valid,vals,np.nan)
    frame['next_product_id']=next_product
    frame['observation_horizon_seconds']=h
    frame['strict_remaining_seconds']=t.rul_strict_seconds
    frame['weak_remaining_seconds']=t.rul_sensor_seconds
    for name in ['adjacent_two_voltage_seconds','adjacent_two_any_seconds']: frame[name]=t[name]
    return frame,y,p,c

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=DEFAULT_DATA)
    ap.add_argument('--output',type=Path,default=ROOT/'artifacts');args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    frame,y,p,c=construct(args.data)
    pre=fit_preprocessor(frame.loc[frame.split.eq('train')])
    x=encode_sequences(frame,pre)
    # The masks and clock offset are explicitly represented. Indices never cross cycles.
    assert np.isfinite(x).all()
    for sl in [slice(0,3),slice(3,6),slice(6,9)]:
        q=y[:,sl];a=np.nan_to_num(q,nan=-1)
        assert not ((a[:,:-1]==1)&(a[:,1:]==0)).any()
    np.savez_compressed(args.output/'dataset.npz',x=x,y=y)
    frame.to_csv(args.output/'anchors.csv.gz',index=False,compression='gzip')
    p.to_csv(args.output/'products.csv.gz',index=False,compression='gzip')
    c.to_csv(args.output/'cycles.csv',index=False)
    shutil.copyfile(args.data/'target_events.csv',args.output/'target_events.csv')
    (args.output/'preprocessor.json').write_text(json.dumps(pre,ensure_ascii=False,indent=2),encoding='utf-8')
    counts={s:{'anchors':int(frame.split.eq(s).sum()),'targets':{
        name:{'eligible':int(np.isfinite(y[frame.split.eq(s),j]).sum()),
              'positive':int(np.nansum(y[frame.split.eq(s),j]))} for j,name in enumerate(TARGETS)}}
        for s in ['train','validation','test']}
    files=[]
    for name in ['data_manifest.json','split_summary.csv','electrode_cycles.csv','maintenance_events.csv',
                 'products.csv','target_events.csv','features_10s.csv.gz','targets_10s.csv.gz',
                 'live_input_schema.json','sensor_minutes.csv.gz','forecast_windows.csv']:
        path=args.data/name
        if not path.is_file():continue  # Optional minute/forecast metadata are not training inputs.
        files.append(dict(name=name,size_bytes=path.stat().st_size,sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()))
    manifest=dict(data_version='11_16数据处理',data_path=str(args.data),counts=counts,files=files,
        exclusions='N000499 and ineligible products inherited; first 3 seconds of product start removed for pending-start latency',
        inputs=FIELDS,targets=TARGETS,strict_unknown_cycles=unknown_cycle_names(c),
        time_clock='device calendar seconds, not spraying time or PLC Gun_Hours')
    (args.output/'data_audit.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(shape=x.shape,counts=counts),ensure_ascii=False),flush=True)

def unknown_cycle_names(c):
    return c.loc[c.eol_sensor_observed.eq(1)|c.end_type.eq('replacement_after_reported_damage'),'cycle_id'].tolist()

if __name__=='__main__':main()
