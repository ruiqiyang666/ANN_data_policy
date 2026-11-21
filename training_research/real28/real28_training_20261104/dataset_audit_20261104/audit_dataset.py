"""Read-only audit of main-device training data and separate voltage reference."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
PROJECT = ROOT.parent
SOURCE = PROJECT/'Real-time Data'
V1 = ROOT/'data/20261104_v1'
V2 = ROOT/'v2_20261104_multimodel'
sys.path.insert(0,str(ROOT/'reference'))
import label_ann as core
cfg=json.loads((V1/'label_rules.json').read_text(encoding='utf-8'))
products=pd.read_parquet(V1/'products.parquet').sort_values('start').reset_index(drop=True)
events=pd.read_parquet(V1/'events.parquet')
endpoints=pd.read_csv(V1/'natural_endpoint_evidence.csv',parse_dates=['failure_lower','failure_upper','replacement_start','replacement_end'])
sources=pd.read_parquet(V1/'sources.parquet')
raw=np.load(V1/'raw_values.npy',mmap_mode='r')
times=np.load(V1/'timestamps.npy',mmap_mode='r')
fields=json.loads((V1/'prepared.json').read_text(encoding='utf-8'))['fields']
result={}

print('Verifying main source manifest...',flush=True)
changed=[]
for r in sources.itertuples():
    path=SOURCE/r.path
    if hashlib.sha256(path.read_bytes()).hexdigest()!=r.sha256:changed.append(r.path)
assert not changed,changed
assert not sources.path.str.contains('Voltage Sample',regex=False).any()
gaps=np.diff(times)-1
invalid={key:int((~np.isfinite(raw[:,i])).sum()) for i,key in enumerate(fields)}
result['source_scope']=dict(files=len(sources),rows=len(times),all_source_hashes_match=True,
    voltage_reference_excluded=True,invalid_cells=sum(invalid.values()),invalid_by_field=invalid,
    missing_seconds=int(gaps[gaps>0].sum()),gap_count=int((gaps>0).sum()),
    gaps_over_2_missing_seconds=int((gaps>2).sum()),duplicates=int((np.diff(times)==0).sum()))
critical=['Voltage','Current','Argon','Nitrogen','Feedrate_1','Feedrate_2','Press._Hopper_1','Press._Hopper_2']
checked=0
for item in sources.iloc[np.linspace(0,len(sources)-1,12).astype(int)].itertuples():
    d=pd.read_csv(SOURCE/item.path,sep=';',skipinitialspace=True)
    if d.empty:continue
    stamp=pd.to_datetime(d['Date-Time'],format='%d %m %Y %H:%M:%S').astype('int64').to_numpy()//10**9
    ids=np.array([0,len(d)//2,len(d)-1]);indices=np.searchsorted(times,stamp[ids])
    assert np.array_equal(times[indices],stamp[ids])
    vals=d[critical].iloc[ids].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    assert np.allclose(vals,raw[indices][:,[fields.index(c) for c in critical]],equal_nan=True)
    checked+=len(ids)
result['source_scope']['raw_cache_crosscheck']=dict(rows=checked,critical_channels=8)
result['products']=dict(total=len(products),valid=int(products.label_valid.sum()),
    failures=products.failure_type.value_counts().to_dict(),
    averaging_scope_disagreements=int(products.averaging_scope_disagreement.sum()),
    company_corrections=products.loc[products.company_override.ne(''),['legacy_id','product_id','pressure_mean','raw_pressure_fail','hopper_pressure_fail','company_override']].to_dict('records'),
    marginal_pressure_failures=int((products.hopper_pressure_fail.eq(1)&np.minimum(abs(products.pressure_mean-100),abs(products.pressure_mean-250)).le(.5)).sum()),
    marginal_feed_failures=int((products.feedrate_fail.eq(1)&np.minimum(abs(products.feedrate_mean-9.1),abs(products.feedrate_mean-11.1)).le(.1)).sum()),
    recovered_legacy_unknown_count=int(pd.read_csv(V1/'legacy_mapping.csv').query("failure_type_old == 'unknown' and label_valid_new == 1").shape[0]))
result['event_types']=events.first_types.value_counts().to_dict()

samples=pd.read_parquet(V2/'data/samples.parquet')
features=np.load(V2/'data/features.npy',mmap_mode='r')
names=json.loads((V2/'data/features.json').read_text(encoding='utf-8'))['names']
getfeature=lambda key:features[:,names.index(key)]
emap=events.set_index('event_id')
positive=samples.event_in_300s
start=samples.loc[positive,'event_id'].map(emap.first_product_start)
result['prediction_target']=dict(positive_windows=int(positive.sum()),
    positive_windows_after_product_started=int((samples.loc[positive,'timestamp']>=start).sum()),
    observable_event_types_by_partition={str(part):events.loc[events.event_id.isin(frame.loc[frame.event_in_300s,'event_id']),'first_types'].value_counts().to_dict() for part,frame in samples.groupby('partition')})
old_life=pd.read_parquet(V1/'rul_samples.parquet')
wait_at=pd.Timestamp('2026-10-14 13:40:45')
cut=old_life[old_life.rul_valid&old_life.rul_observation_end.eq(wait_at)]
result['company_reviewed_wait']=dict(start=wait_at,censored_windows=len(cut),
    affected_from=cut.timestamp.min(),affected_to=cut.timestamp.max(),
    company_evidence='Company reply paragraph23 and image8/image9: automatic standby after >15min waiting; rising normal voltage is not evidence of gun replacement',
    qualification='Review intervention evidence and censoring policy; this audit does not assert an unobserved physical action or change existing labels')
result['feature_availability']={}
for key in ['observed_operation_hours','matched_product_count','matched_product_voltage_slope_per_hour','powered_Voltage_mean_300s','powered_Voltage_coverage_3600s']:
    x=getfeature(key)
    result['feature_availability'][key]=dict(missing_fraction=float((~np.isfinite(x)).mean()),
        positive_missing_fraction=float((~np.isfinite(x[positive])).mean()),
        finite_median=float(np.nanmedian(x)),finite_p90=float(np.nanquantile(x,.9)))

print('Auditing failure endpoints and causal voltage trends...',flush=True)
profiles=[]
for ep in endpoints.itertuples():
    first=products.loc[products.product_id.eq(ep.product_id)].iloc[0]
    pre=products[(products.end<ep.failure_lower)&(products.end>=ep.failure_lower-pd.Timedelta(hours=4))&products.label_valid.eq(1)].copy()
    recipe=(abs(pre.current_median-first.current_median)<=10)&(abs(pre.argon_median-first.argon_median)<=1)&(abs(pre.nitrogen_median-first.nitrogen_median)<=.5)
    matched=pre[recipe]
    item=dict(candidate=ep.candidate_id,eligible=ep.eligible,first_low=ep.failure_lower,
        terminal_voltage=first.voltage_mean,current=first.current_median,argon=first.argon_median,nitrogen=first.nitrogen_median,
        observed_hours_since_previous=ep.observed_powered_hours_since_previous_candidate,
        previous_products=len(pre),recipe_matched=len(matched))
    for minutes in [15,30,60,180]:
        q=matched[matched.end>=ep.failure_lower-pd.Timedelta(minutes=minutes)]
        span=(q.end.max()-q.end.min()).total_seconds()/3600 if len(q)>1 else 0
        slope=float(np.polyfit((q.end-ep.failure_lower).dt.total_seconds()/3600,q.voltage_mean,1)[0]) if len(q)>=3 and span>0 else None
        item[f'pre_{minutes}m']=dict(products=len(q),voltage_median=float(q.voltage_mean.median()) if len(q) else None,slope_v_per_hour=slope)
    used=samples.rul_valid&samples.rul_event&samples.rul_endpoint_id.eq(ep.candidate_id)
    near=used&samples.rul_lower_hours.le(.25)
    item['life_event_windows']=int(used.sum());item['near_15min_windows']=int(near.sum())
    item['near_matched_trend_missing_fraction']=float((~np.isfinite(getfeature('matched_product_voltage_slope_per_hour')[near])).mean()) if near.any() else None
    item['near_operation_history_max_hours']=float(np.nanmax(getfeature('observed_operation_hours')[near])) if near.any() else None
    profiles.append(item)
result['endpoint_profiles']=profiles

alerts=pd.read_csv(V2/'evaluation/extra_trees_historical_comparison_alerts.csv',parse_dates=['timestamp'])
context=[]
for a in alerts.itertuples():
    old=products[(products.label_available_at<=a.timestamp)&products.label_valid.eq(1)]
    last=old.iloc[-1]
    next_event=events[events.event_at>a.timestamp].iloc[0] if (events.event_at>a.timestamp).any() else None
    event_before=events[events.event_at<=a.timestamp].iloc[-1] if (events.event_at<=a.timestamp).any() else None
    ids=np.flatnonzero(samples.timestamp.eq(a.timestamp))
    item=dict(timestamp=a.timestamp,is_true=bool(a.event_in_300s),risk=a.predicted_risk,
        last_product_type=last.failure_type,last_voltage=last.voltage_mean,last_feedrate=last.feedrate_mean,
        last_pressure=last.pressure_mean,last_product_age_s=(a.timestamp-last.label_available_at).total_seconds(),
        next_event_s=(next_event.event_at-a.timestamp).total_seconds() if next_event is not None else None,
        previous_event_s=(a.timestamp-event_before.event_at).total_seconds() if event_before is not None else None,
        operation_age_s=float(getfeature('observed_operation_hours')[ids[0]]*3600) if len(ids) else None)
    context.append(item)
pd.DataFrame(context).to_csv(OUT/'selected_model_alert_context.csv',index=False,encoding='utf-8-sig')
result['alerts']=dict(total=len(context),false_count=sum(not r['is_true'] for r in context),
    false_with_last_known_bad_product=sum(not r['is_true'] and r['last_product_type']!='none' for r in context),
    false_with_next_event_within_15min=sum(not r['is_true'] and r['next_event_s'] is not None and r['next_event_s']<=900 for r in context),
    context=context)
event_records=pd.read_csv(V2/'evaluation/extra_trees_historical_common_events.csv')
event_records['detected']=event_records.alert_at.notna()
event_records['before']=pd.to_numeric(event_records.lead_to_product_start_s,errors='coerce')>0
result['alerts']['by_type']={kind:dict(events=len(q),detected=int(q.detected.sum()),before_product=int(q.before.sum())) for kind,q in event_records.groupby('type')}
prediction=np.load(V2/'evaluation/extra_trees_historical_comparison.npz')
result['risk_score_distribution']={key:np.quantile(prediction[key],[0,.1,.5,.9,1]).tolist() for key in ['risk','calibrated_risk']}

# Recompute precursor summaries for all 62 events, keeping failure types separate.
precursors=[]
for e in events.itertuples():
    target=products.loc[products.product_id.eq(e.first_product_id)].iloc[0]
    q=products[(products.label_available_at<e.first_product_start)&products.label_valid.eq(1)&
        (products.end>=e.first_product_start-pd.Timedelta(minutes=30))]
    q=q[(abs(q.current_median-target.current_median)<=10)&(abs(q.argon_median-target.argon_median)<=1)&(abs(q.nitrogen_median-target.nitrogen_median)<=.5)]
    q=q.tail(5)
    precursors.append(dict(event=e.event_id,type=e.first_types,first_product_start=e.first_product_start,
        target_voltage=target.voltage_mean,target_feedrate=target.feedrate_mean,target_pressure=target.pressure_mean,
        predecessor_count=len(q),predecessor_failures=int(q.failure.sum()),
        prev_voltage=float(q.voltage_mean.median()) if len(q) else None,
        prev_feed=float(q.feedrate_mean.median()) if len(q) else None,
        prev_pressure=float(q.pressure_mean.median()) if len(q) else None,
        voltage_drop=float(target.voltage_mean-q.voltage_mean.median()) if len(q) else None))
pd.DataFrame(precursors).to_csv(OUT/'event_precursors.csv',index=False,encoding='utf-8-sig')
result['precursor_summary']={kind:dict(events=len(q),with_3_predecessors=int(q.predecessor_count.ge(3).sum()),
    median_voltage_drop=float(q.voltage_drop.median()),predecessor_all_normal=int(q.predecessor_failures.eq(0).sum()))
    for kind,q in pd.DataFrame(precursors).groupby('type')}

print('Reading independent reference device CSVs...',flush=True)
ref_frames=[]
paths=sorted((SOURCE/'Voltage Sample').glob('*.csv'))
for i,path in enumerate(paths):
    d=core.parse_csv(path.read_bytes())
    d['source_id']='REF_'+path.name;d['source_row']=np.arange(len(d))+2;d['timestamp_conflict']=False
    ref_frames.append(d)
reference=pd.concat(ref_frames,ignore_index=True).sort_values('timestamp').reset_index(drop=True)
del ref_frames
assert not reference.timestamp.duplicated().any()
# Use reference's own stations and feed only to identify comparable sprayed voltage.
# Do not apply main-device two-product stopping logic or append these data to training.
ref_products=pd.DataFrame(core.extract_products(reference,cfg)).sort_values('start').reset_index(drop=True)
ref_products.to_csv(OUT/'reference_device_product_voltage.csv',index=False,encoding='utf-8-sig')
rp=ref_products[(ref_products.current_median>100)&ref_products.voltage_mean.notna()]
terminal=rp[(rp.start>='2026-10-27 12:00')&(rp.start<'2026-10-27 16:00')]
lastlow=terminal[(terminal.start<'2026-10-27 14:14')&terminal.voltage_mean.lt(51)]
result['reference']=dict(device='separate device; descriptive voltage reference only',files=len(paths),rows=len(reference),
    product_intervals=len(ref_products),terminal_product_intervals=len(terminal),
    terminal_low_products=lastlow[['start','end','voltage_mean']].to_dict('records'),
    hour_voltage={str(hour):dict(products=len(q),median_voltage=float(q.voltage_mean.median()),min_voltage=float(q.voltage_mean.min()))
                  for hour,q in terminal.groupby(terminal.start.dt.hour)},
    workday_voltage={str(day):dict(products=len(q),median_voltage=float(q.voltage_mean.median())) for day,q in rp.groupby(rp.start.dt.date)})

plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False
fig,axes=plt.subplots(2,1,figsize=(12,7),layout='constrained')
axes[0].scatter(rp.end,rp.voltage_mean,s=2,alpha=.35)
axes[0].axhline(51,color='firebrick',linestyle='--');axes[0].set(title='另一台设备：有效喷涂产品电压，独立参考',ylabel='产品均值 V')
for _,segment in terminal.groupby(terminal.end.diff().dt.total_seconds().gt(300).cumsum()):
    axes[1].plot(segment.end,segment.voltage_mean,marker='.',linewidth=.7)
axes[1].axhline(51,color='firebrick',linestyle='--')
axes[1].axvspan(pd.Timestamp('2026-10-27 14:13:43'),pd.Timestamp('2026-10-27 14:50:00'),color='gray',alpha=.15,label='低电流停止区间；不等于精确换枪时刻')
axes[1].set(title='10月27日：下降、间歇性低压、停止及恢复',ylabel='产品均值 V');axes[1].legend(fontsize=8)
fig.savefig(OUT/'reference_voltage.png',dpi=150);plt.close(fig)

fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
for ax,ep in zip(axes.flat,endpoints.itertuples()):
    target=products.loc[products.product_id.eq(ep.product_id)].iloc[0]
    q=products[(products.end>=ep.failure_lower-pd.Timedelta(hours=3))&(products.end<=ep.failure_upper+pd.Timedelta(minutes=10))&products.label_valid.eq(1)]
    match=(abs(q.current_median-target.current_median)<=10)&(abs(q.argon_median-target.argon_median)<=1)&(abs(q.nitrogen_median-target.nitrogen_median)<=.5)
    ax.scatter((q.end-ep.failure_lower).dt.total_seconds()/60,q.voltage_mean,s=10,color='lightgray',label='其他工况')
    q=q[match];ax.scatter((q.end-ep.failure_lower).dt.total_seconds()/60,q.voltage_mean,s=12,color='steelblue',label='近似同工况')
    ax.axhline(51,color='firebrick',linestyle='--');ax.axvline(0,color='black',linestyle=':')
    ax.set(title=f'{ep.candidate_id} {ep.failure_lower:%m-%d %H:%M} · 寿命标签纳入={ep.eligible}',xlabel='距首件低压产品开始 分钟',ylabel='产品均值 V')
axes.flat[0].legend(fontsize=8)
fig.savefig(OUT/'main_device_endpoints.png',dpi=150);plt.close(fig)

def clean(obj):
    if isinstance(obj,dict):return {str(k):clean(v) for k,v in obj.items()}
    if isinstance(obj,list):return [clean(v) for v in obj]
    if isinstance(obj,(pd.Timestamp,)):return obj.isoformat()
    if isinstance(obj,np.generic):return clean(obj.item())
    if isinstance(obj,float) and not np.isfinite(obj):return None
    return obj
result=clean(result)
(OUT/'audit_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['alerts','reference']},ensure_ascii=False,indent=2),flush=True)
print('ALERT SUMMARY',json.dumps({k:v for k,v in result['alerts'].items() if k!='context'},ensure_ascii=False))
print('REFERENCE',json.dumps(result['reference'],ensure_ascii=False))
