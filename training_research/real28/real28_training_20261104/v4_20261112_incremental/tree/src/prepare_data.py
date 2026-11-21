"""Recompute main-device labels with traceable revisions; no source writes.

Only parse_csv/extract_products and replacement_windows are reused from the
read-only ruiqi snapshot. Its obsolete natural-failure classifier is NOT used.
"""
import sys
sys.dont_write_bytecode=True
from collections import defaultdict
from datetime import datetime
import json
import numpy as np
import pandas as pd
from common import ROOT,SOURCE,DATA,FIELDS,TYPES,dump,sha,status,ensure_dirs
sys.path.insert(0,str(ROOT/'reference'))
import label_ann as core
import revise_labels as reference

FOLDERS=['20260815-20260911','20260912-20261009','20261010-20261016','20261019-20261023','20261027-20261101']

def state_runs(d):
    stamps=d.timestamp.to_numpy(dtype='datetime64[s]').astype('int64')
    on=((d.Current>100)&(d.Power>5)).to_numpy()
    cuts=np.r_[0,np.flatnonzero((on[1:]!=on[:-1])|(np.diff(stamps)>2))+1,len(d)]
    return [dict(start=pd.Timestamp(int(stamps[a]),unit='s'),end=pd.Timestamp(int(stamps[b-1]),unit='s'),
                 powered=bool(on[a]),samples=b-a) for a,b in zip(cuts[:-1],cuts[1:])]

def make_events(products):
    groups=[]
    for row in products[products.failure.eq(1)].to_dict('records'):
        if groups and (row['start']-groups[-1][-1]['end']).total_seconds()<=600 and row['power_episode_id']==groups[-1][-1]['power_episode_id']:
            groups[-1].append(row)
        else: groups.append([row])
    events=[]
    for i,members in enumerate(groups):
        first,last=members[0],members[-1]
        ev=dict(event_id=f'EV{i+1:04d}',start=first['start'],end=last['end'],
                event_at=first['label_available_at'],first_product_start=first['start'],
                last_label_at=last['label_available_at'],first_product_id=first['product_id'],
                product_ids=';'.join(r['product_id'] for r in members),count=len(members),
                first_types=first['failure_type'],types='|'.join(k for k in TYPES if any(k in r['failure_type'].split('|') for r in members)),
                evidence='HMI product rules plus documented company correction; not physical root-cause confirmation')
        events.append(ev)
        products.loc[products.product_id.isin([r['product_id'] for r in members]),'event_id']=ev['event_id']
    return pd.DataFrame(events)

def main():
    ensure_dirs()
    if (DATA/'prepared.json').exists():
        print('Prepared data already exists; keeping immutable run data.',flush=True); return
    cfg=json.loads((ROOT/'reference/config.json').read_text(encoding='utf-8-sig'))
    cfg.update(analysis_start='2026-08-15 00:00:00',analysis_end_exclusive='2026-11-02 00:00:00',minimum_sampling_coverage=0.0,
               version='real28-20261112-incremental-v4')
    cfg['revision_notes']={'missing':'actual observed means; at most 2 missing points and coverage >= .95, other validity checks retained',
       'natural_failure':'low voltage is not automatically electrode natural failure',
       'pressure_override':'P003507 confirmed not failed; raw HMI mean unchanged',
       'source_branch':'origin/ruiqi cf771037eee26d8912c512cddf87db58dd11cfb1'}
    dump(cfg,DATA/'label_rules.json')
    byday=defaultdict(list); manifest=[]
    files=sorted([p for folder in FOLDERS for p in (SOURCE/folder).rglob('*.csv')],
                 key=lambda p:(datetime.strptime(p.stem,'%d_%b_%YH%H'),str(p)))
    for i,p in enumerate(files):
        day=datetime.strptime(p.stem,'%d_%b_%YH%H').date()
        byday[day].append((i,p))
    days=sorted(byday)
    def load_day(day):
        frames=[]
        for i,path in byday[day]:
            data=path.read_bytes(); d=core.parse_csv(data)
            if d.timestamp.isna().any(): raise ValueError(f'Invalid timestamp: {path}')
            if list(d.columns[:30])!=['Date-Time','Time_s',*FIELDS]: raise ValueError('Unexpected column order')
            import hashlib
            manifest.append(dict(source_id=f'S{i+1:04d}',path=path.relative_to(SOURCE).as_posix(),bytes=len(data),
                                 sha256=hashlib.sha256(data).hexdigest(),rows=len(d),
                                 invalid_cells=int(d[FIELDS].isna().sum().sum())))
            if not len(d): continue
            d['source_id']=f'S{i+1:04d}';d['source_row']=np.arange(len(d))+2
            frames.append(d)
        if not frames:return pd.DataFrame()
        d=pd.concat(frames,ignore_index=True).sort_values(['timestamp','source_id','source_row'],kind='stable')
        exact=d.duplicated(['timestamp',*FIELDS]); d=d[~exact].copy()
        d['timestamp_conflict']=d.timestamp.duplicated(keep=False)
        if d.timestamp_conflict.any(): raise ValueError('Conflicting raw timestamps need explicit resolution')
        return d.reset_index(drop=True)

    current=load_day(days[0]);previous=None
    values=[];times=[];states=[];products=[];daily=[]
    for di,day in enumerate(days):
        nxt=load_day(days[di+1]) if di+1<len(days) else None
        if not len(current): current=nxt;continue
        context=[q for q in [previous,current,nxt.iloc[:3600] if nxt is not None else None] if q is not None and len(q)]
        context=pd.concat(context,ignore_index=True).sort_values('timestamp').reset_index(drop=True)
        rows=[r for r in core.extract_products(context,cfg) if r['start'].date()==day]
        for r in rows:
            r['expected_samples']=int((r['end']-r['start']).total_seconds())+1
            r['missing_samples']=r['expected_samples']-r['samples']
            r['raw_rule_failure']=r['failure'];r['raw_pressure_fail']=r['hopper_pressure_fail']
            r['company_override']=''
            if r['missing_samples']>2 or r['coverage']<.95:
                r['quality_flags']=';'.join(filter(None,[r['quality_flags'],'missing_beyond_company_examples']))
                r['label_valid']=0;r['failure']=None;r['failure_type']='unknown'
            r['product_id']=f"DEV1_{r['start']:%Y%m%dT%H%M%S}_S{r['station']}"
        products.extend(rows)
        values.append(current[FIELDS].to_numpy(dtype='float32'))
        times.append(current.timestamp.to_numpy(dtype='datetime64[s]').astype('int64'))
        states.extend(state_runs(current))
        daily.append(dict(date=str(day),rows=len(current),products=len(rows),invalid_cells=int(current[FIELDS].isna().sum().sum())))
        previous=current[current.timestamp>=current.timestamp.iloc[-1].floor('h')].copy()
        current=nxt
        status('preparing_data',day=str(day),days_done=di+1,days_total=len(days),products=len(products))
    p=pd.DataFrame(products).sort_values(['start','station']).reset_index(drop=True)
    p['legacy_id']=''
    late=p.index[p.start.ge('2026-09-12')]
    p.loc[late,'legacy_id']=[f'P{i+1:06d}' for i in range(len(late))]
    legacy=pd.read_csv(ROOT/'reference'/'label_samples.csv')
    legacy['start']=pd.to_datetime(legacy.start,format=core.TIME_FORMAT)
    legacy['end']=pd.to_datetime(legacy.end,format=core.TIME_FORMAT)
    comparison=legacy.merge(p,on=['start','end','station'],suffixes=('_old','_new'),validate='one_to_one')
    if len(comparison)!=len(legacy): raise AssertionError(f'Legacy mappings {len(comparison)} / {len(legacy)}')
    if not comparison.sample_id.eq(comparison.legacy_id).all(): raise AssertionError('Legacy IDs shifted unexpectedly')
    mean_errors={}
    for a,b in [('voltage_mean_v','voltage_mean'),('feedrate_mean_g_min','feedrate_mean'),('hopper_pressure_mean_mbar','pressure_mean')]:
        err=np.abs(comparison[a].to_numpy()-comparison[b].to_numpy())
        mean_errors[a]=float(np.nanmax(err))
        if not np.allclose(comparison[a],comparison[b],atol=1e-7,rtol=1e-9,equal_nan=True): raise AssertionError(f'Legacy mean mismatch: {a}')
    corrected=p.legacy_id.eq('P003507')
    if corrected.sum()!=1: raise AssertionError('Company correction must match exactly once')
    p.loc[corrected,['hopper_pressure_fail','failure']]=0
    p.loc[corrected,'failure_type']='none'
    p.loc[corrected,'company_override']='20261103 PLC=250mbar; explicitly not failed; HMI mean retained'
    merged=[]
    for r in sorted(states,key=lambda r:r['start']):
        if merged and r['powered']==merged[-1]['powered'] and (r['start']-merged[-1]['end']).total_seconds()<=2:
            merged[-1]['end']=r['end'];merged[-1]['samples']+=r['samples']
        else:merged.append(r.copy())
    state=pd.DataFrame(merged)
    state['duration_seconds']=(state.end-state.start).dt.total_seconds()+1
    powered=state[state.powered].reset_index(drop=True)
    ix=np.searchsorted(powered.start.to_numpy(),p.start.to_numpy(),side='right')-1
    p['power_episode_id']=np.maximum(ix,0)+1
    p['power_episode_start']=powered.start.iloc[np.maximum(ix,0)].to_numpy()
    p['event_id']=''
    events=make_events(p)
    audit,candidates=reference.replacement_windows(p,state,cfg)
    # Old references are evidence only; do not run their outdated cause classifier.
    candidates['natural_failure_confirmed']=False
    candidates['ignition_template_status']='not available; no further company labels expected'
    candidates['natural_evidence']='candidate only; early electrode instability must be considered separately'
    t=np.concatenate(times); v=np.concatenate(values)
    if np.any(np.diff(t)<=0): raise AssertionError('Raw timeline is not strictly increasing')
    old_rows=int((t<pd.Timestamp('2026-10-24').value//10**9).sum())
    # The audited 2026-10-23 delivery must be reproduced exactly inside the larger scope.
    if old_rows!=5_402_169 or len(files)!=1635: raise AssertionError(f'Main source scope changed: {old_rows} rows through 20261023, {len(files)} files')
    np.save(DATA/'timestamps.npy',t);np.save(DATA/'raw_values.npy',v)
    for name,frame in [('products',p),('events',events),('power_states',state),('stop_audit',audit),('replacement_candidates',candidates),('daily',pd.DataFrame(daily)),('sources',pd.DataFrame(manifest))]:
        frame.to_parquet(DATA/f'{name}.parquet',index=False)
        frame.to_csv(DATA/f'{name}.csv',index=False,encoding='utf-8-sig')
    comp=legacy.merge(p,on=['start','end','station'],suffixes=('_old','_new'),validate='one_to_one')
    comp[['sample_id','product_id','failure_type_old','failure_type_new','label_valid_new','company_override']].to_csv(DATA/'legacy_mapping.csv',index=False,encoding='utf-8-sig')
    meta={'rows':len(t),'source_files':len(files),'fields':FIELDS,'raw_invalid_cells':int(np.isnan(v).sum()),
        'products':len(p),'valid_products':int(p.label_valid.sum()),'failed_products':int(p.failure.eq(1).sum()),
        'product_counts_by_type':p.failure_type.value_counts().to_dict(),'events':len(events),
        'late_products':len(late),'early_failed_products':int((p.start.lt('2026-09-12')&p.failure.eq(1)).sum()),
        'late_failed_products':int((p.start.ge('2026-09-12')&p.failure.eq(1)).sum()),
        'candidate_counts':candidates.candidate_kind.value_counts().to_dict(),
        'legacy_rows_matched':len(comp),'legacy_mean_max_abs_errors':mean_errors,
        'legacy_unknown_now_valid':int((comp.failure_type_old.eq('unknown')&comp.label_valid_new.eq(1)).sum()),
        'company_corrections':1,'labels_are_weak_process_labels':True,
        'sources_sha256':sha(DATA/'sources.csv'),'rules_sha256':sha(DATA/'label_rules.json'),
        'rows_through_20261023':old_rows,'new_rows_20261027_20261101':int(len(t)-old_rows),
        'scope':'main device 2026-08-15 through 2026-11-01; Voltage Sample excluded'}
    dump(meta,DATA/'prepared.json');status('raw_and_product_preparation_complete',**meta)

if __name__=='__main__':main()
