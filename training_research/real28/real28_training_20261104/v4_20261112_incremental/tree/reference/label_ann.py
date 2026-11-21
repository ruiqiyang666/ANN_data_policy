"""Offline, reviewable ANN product labels and voltage-reference evidence.

Sources are read-only. Main data and the other-machine voltage reference are
never concatenated. The 2026-11-02 revision applies site process confirmation separately from
retrospective replacement candidates. No forecasting model is trained.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import platform
import re
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

PHYSICAL = ['Argon','Nitrogen','AirJet','Silvent','Current','Power','Voltage',
 'Cooling_water_flow','Conductivity','Cooling_water_inlet_temp.','Cooling_water_outlet_temp.',
 'Feedrate_1','Disk_1','Carrier_gas_1','Feed_Factor_1','Press._Hopper_1','Stirrer_1','Weight_Hopper_1',
 'Feedrate_2','Disk_2','Carrier_gas_2','Feed_Factor_2','Press._Hopper_2','Stirrer_2','Weight_Hopper_2',
 'Station_1','Station_2','O2_Sensor']
ORIGINAL = ['Date-Time','Time_s',*PHYSICAL]
TIME_FORMAT = '%d %m %Y %H:%M:%S'
NATURAL = '自然电极寿命终结（现场工艺说明支持）'
MANUAL = '换枪候选（动作时刻未确认，不是失败原因）'
UNKNOWN = 'failure原因待核实'
OTHER = '其他指标failure（原因待核实）'


def write_csv(frame, path, raw=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path,index=False,encoding='utf-8-sig',sep=';' if raw else ',',
                 float_format='%.12g',date_format=TIME_FORMAT)


def dump(value, path):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')


def payload(root, row):
    if row['zip_member']:
        with zipfile.ZipFile(root/row['container']) as archive:
            return archive.read(row['zip_member'])
    return (root/row['container']).read_bytes()


def parse_csv(data, voltage_only=False, strings=False):
    options=dict(sep=';',encoding='utf-8-sig',skipinitialspace=True)
    if voltage_only: options['usecols']=lambda x:x.strip() in ['Date-Time','Voltage']
    if strings: options.update(dtype=str,keep_default_na=False)
    d=pd.read_csv(io.BytesIO(data),**options)
    d.columns=d.columns.str.strip()
    required=['Date-Time','Voltage'] if voltage_only else ORIGINAL
    missing=set(required)-set(d.columns)
    if missing: raise ValueError('Missing required fields: '+repr(sorted(missing)))
    d=d[required].copy()
    if strings:
        for column in d: d[column]=d[column].str.strip()
    else:
        for column in d:
            if column!='Date-Time':
                d[column]=pd.to_numeric(d[column],errors='coerce').replace([np.inf,-np.inf],np.nan)
    d['timestamp']=pd.to_datetime(d['Date-Time'],format=TIME_FORMAT,errors='coerce')
    return d


def inventory(root, cfg, out):
    sources=[]
    for folder in sorted(root.iterdir()):
        if folder.is_dir() and re.fullmatch(r'20\d{6}-20\d{6}',folder.name):
            sources.extend(dict(data_stream='main',container=p.relative_to(root).as_posix(),zip_member='') for p in sorted(folder.rglob('*.csv')))
        elif folder.is_file() and re.fullmatch(r'20\d{6}-20\d{6}\.zip',folder.name):
            with zipfile.ZipFile(folder) as archive:
                sources.extend(dict(data_stream='main',container=folder.name,zip_member=p.filename) for p in archive.infolist() if p.filename.lower().endswith('.csv'))
    refdir=root/cfg['reference_source_directory']
    if not refdir.is_dir():raise FileNotFoundError(refdir)
    sources.extend(dict(data_stream='voltage_reference',container=p.relative_to(root).as_posix(),zip_member='') for p in sorted(refdir.rglob('*.csv')))
    manifest=[]; seen={}
    for i,r in enumerate(sources,1):
        data=payload(root,r); d=parse_csv(data,voltage_only=r['data_stream']=='voltage_reference')
        sha=hashlib.sha256(data).hexdigest(); key=(r['data_stream'],sha)
        r.update(source_id=f'S{i:04d}',bytes=len(data),sha256=sha,rows=len(d),
                 time_start=d.timestamp.min(),time_end=d.timestamp.max(),
                 invalid_timestamps=int(d.timestamp.isna().sum()),
                 out_of_order_steps=int(d.timestamp.diff().dt.total_seconds().lt(0).sum()),
                 identical_bytes_to=seen.get(key,''))
        if d.timestamp.isna().any():raise ValueError('Invalid timestamp: '+r['container'])
        seen.setdefault(key,r['source_id']); manifest.append(r)
        if i%200==0: print('inventory',i,'/',len(sources),flush=True)
    m=pd.DataFrame(manifest)
    write_csv(m,out/'来源文件清单.csv')
    return m


def contiguous_runs(d, mask, maxgap=2):
    mask=mask.fillna(False)
    gap=d.timestamp.diff().dt.total_seconds()
    starts=mask & (~mask.shift(fill_value=False)|gap.gt(maxgap))
    group=starts.cumsum()
    for _,q in d[mask].groupby(group[mask],sort=False):yield q


def reference(root,m,cfg,out):
    frames=[]
    for r in m[(m.data_stream=='voltage_reference')&m.identical_bytes_to.eq('')].to_dict('records'):
        q=parse_csv(payload(root,r),voltage_only=True)
        q['source_id']=r['source_id']; q['source_row']=np.arange(len(q))+2;frames.append(q)
    d=pd.concat(frames,ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    if d.timestamp.duplicated().any():raise ValueError('Reference duplicate timestamps need review')
    floor=cfg['voltage_reference_band_floor_v']; high=d.Voltage.gt(floor)
    d['steady_band']=False; runs=[]
    for q in contiguous_runs(d,high):
        use=(q.timestamp-q.timestamp.iloc[0]).dt.total_seconds().ge(cfg['reference_startup_guard_seconds'])
        d.loc[q.index[use],'steady_band']=True
        runs.append(dict(start=q.timestamp.iloc[0],end=q.timestamp.iloc[-1],samples=len(q)))
    minute=d[d.steady_band].groupby(d.loc[d.steady_band,'timestamp'].dt.floor('min')).agg(
        samples=('Voltage','size'),voltage_mean=('Voltage','mean'),voltage_median=('Voltage','median'),voltage_std=('Voltage','std'))
    minute=minute[minute.samples>=30].reset_index()
    # Voltage-only per-minute markers are NOT official product failure labels.
    minute['reference_minute_below_51']=minute.voltage_mean.round(10).lt(cfg['voltage_failure_below_v'])
    candidates=[]
    for q in contiguous_runs(d,~high):
        start,end=q.timestamp.iloc[0],q.timestamp.iloc[-1]
        duration=(end-start).total_seconds()+1
        if duration<60:continue
        pre=minute[(minute.timestamp>=start-pd.Timedelta(minutes=60))&(minute.timestamp<start)]
        post=minute[(minute.timestamp>end)&(minute.timestamp<=end+pd.Timedelta(minutes=15))]
        low=pre[pre.reference_minute_below_51]
        if len(low)<cfg['minimum_reference_low_minutes'] or len(post)<2:continue
        # The low-voltage evidence must directly precede this stop. Otherwise a
        # later extra restart would incorrectly reuse an earlier cycle's failure.
        if (start-low.timestamp.max()).total_seconds()>300:continue
        before=pre.tail(5).voltage_median.median();after=post.head(5).voltage_median.median()
        if after-before<cfg['reset_minimum_voltage_gain_v']:continue
        candidates.append(dict(off_start=start,off_end=end,resume_at=end+pd.Timedelta(seconds=1),
            last_low_minute=low.timestamp.max(),first_low_minute=low.timestamp.min(),
            low_minutes=len(low),before_voltage=before,after_voltage=after))
    if not candidates:raise ValueError('No suitable terminal reference window; do not invent template')
    terminal=candidates[-1]
    templates=[]; template_rows=[]
    for offset in [0,2,5]:
        endpoint=terminal['last_low_minute']-pd.Timedelta(minutes=offset)
        q=minute[(minute.timestamp<=endpoint)&(minute.timestamp>=endpoint-pd.Timedelta(minutes=60))].tail(cfg['shape_maximum_valid_minutes'])
        if len(q)<cfg['shape_minimum_valid_minutes']:continue
        templates.append(q.voltage_median.to_numpy())
        t=q.copy();t['template_id']=f'REF{len(templates)}';template_rows.append(t)
    if not templates:raise ValueError('Insufficient reference history')
    info=dict(user_confirmed_natural_electrode_life_reference=True,device='independent_reference_machine',
        allowed_columns=['Date-Time','Voltage'],exact_replacement_time_confirmed=False,
        reference_cycle_start_confirmed=False,terminal_selection=terminal,
        template_count=len(templates),independent_natural_cycles=1,
        threshold_note='51V per-minute markers here are descriptive, not product labels',
        working_band_floor_v=floor,startup_guard_seconds=cfg['reference_startup_guard_seconds'],
        counts=dict(rows=len(d),voltage_min=d.Voltage.min(),voltage_max=d.Voltage.max(),
                    voltage_missing=int(d.Voltage.isna().sum()),between20and30=int(d.Voltage.ge(20).mul(d.Voltage.lt(30)).sum())))
    write_csv(minute,out/'参考样本'/'电压分钟走势.csv')
    write_csv(pd.concat(template_rows,ignore_index=True),out/'参考样本'/'自然失效比较模板.csv')
    dump(info,out/'参考样本'/'参考信息.json')
    # A compact excerpt rather than exporting the reference device's other channels.
    q=d[(d.timestamp>=terminal['off_start']-pd.Timedelta(hours=2))&
        (d.timestamp<=terminal['off_end']+pd.Timedelta(minutes=15))]
    write_csv(q[['Date-Time','Voltage','source_id','source_row']],out/'参考样本'/'27_Oct_2026_电压末期参考.csv',raw=True)
    return templates,info,minute


def load_day(root,m,day):
    frames=[]
    for r in m[(m.data_stream=='main')&m.identical_bytes_to.eq('')&m.time_start.dt.date.eq(day)&m.rows.gt(0)].to_dict('records'):
        d=parse_csv(payload(root,r));d['source_id']=r['source_id'];d['source_row']=np.arange(len(d))+2;frames.append(d)
    if not frames:return pd.DataFrame()
    d=pd.concat(frames,ignore_index=True).sort_values(['timestamp','source_id','source_row'],kind='stable')
    exact=d.duplicated(['timestamp',*PHYSICAL]);d=d[~exact].copy()
    d['timestamp_conflict']=d.timestamp.duplicated(keep=False)
    conflicts=d.loc[d.timestamp_conflict,['timestamp','source_id','source_row']].copy()
    d=d.drop_duplicates('timestamp').reset_index(drop=True)
    return d,dict(identical_rows_removed=int(exact.sum()),conflicting_rows=len(conflicts))


def failure_bits(voltage,feed,pressure,cfg):
    values=np.asarray([voltage,feed,pressure],dtype=float)
    if not np.isfinite(values).all():return None,None,None
    v,f,p=np.round(values,10)
    return int(v<cfg['voltage_failure_below_v']),int(f<cfg['feedrate_failure_below'] or f>cfg['feedrate_failure_above']),int(p<cfg['pressure_failure_below'] or p>cfg['pressure_failure_above'])


def extract_products(d,cfg,station_threshold=None,minimum=None):
    rows=[];threshold=cfg['station_threshold_rpm'] if station_threshold is None else station_threshold
    minimum=cfg['minimum_station_samples'] if minimum is None else minimum
    gap=d.timestamp.diff().dt.total_seconds(); maxgap=cfg['maximum_sampling_gap_seconds']
    for station in [1,2]:
        active=d[f'Station_{station}'].gt(threshold)
        for q in contiguous_runs(d,active,maxgap):
            if len(q)<minimum:continue
            left,right=int(q.index[0]),int(q.index[-1])
            start,end=q.timestamp.iloc[0],q.timestamp.iloc[-1]
            f1=q.Feedrate_1.gt(0);f2=q.Feedrate_2.gt(0)
            n1,n2=int(f1.sum()),int(f2.sum());hopper=1 if n1>n2 else 2 if n2>n1 else None
            record=dict(station=station,start=start,end=end,samples=len(q),
                coverage=len(q)/((end-start).total_seconds()+1),hopper=hopper,
                dual_feed_samples=int((f1&f2).sum()),other_station_running_samples=int(q[f'Station_{3-station}'].gt(threshold).sum()),
                left_boundary_observed=left>0 and gap.iloc[left]<=maxgap and not active.iloc[left-1],
                right_boundary_observed=right<len(d)-1 and gap.iloc[right+1]<=maxgap and not active.iloc[right+1],
                label_available_at=d.timestamp.iloc[right+1] if right<len(d)-1 and not active.iloc[right+1] and gap.iloc[right+1]<=maxgap else pd.NaT,
                conflict_samples=int(q.timestamp_conflict.sum()),
                source_start=q.source_id.iloc[0],source_row_start=int(q.source_row.iloc[0]),
                source_end=q.source_id.iloc[-1],source_row_end=int(q.source_row.iloc[-1]),
                voltage_mean=np.nan,feedrate_mean=np.nan,pressure_mean=np.nan,spray_samples=0,
                current_median=np.nan,argon_median=np.nan,nitrogen_median=np.nan,
                powered_fraction=np.nan,zero_feed_samples=np.nan,label_missing_values=0)
            if hopper:
                selected_feed=q[f'Feedrate_{hopper}']; z=q[selected_feed.gt(0)]
                record.update(spray_samples=len(z),voltage_mean=z.Voltage.mean(),
                    feedrate_mean=z[f'Feedrate_{hopper}'].mean(),pressure_mean=z[f'Press._Hopper_{hopper}'].mean(),
                    voltage_std=z.Voltage.std(),voltage_min=z.Voltage.min(),voltage_max=z.Voltage.max(),
                    current_median=z.Current.median(),argon_median=z.Argon.median(),nitrogen_median=z.Nitrogen.median(),
                    powered_fraction=((z.Current>cfg['powered_current_above_a'])&(z.Power>cfg['powered_power_above_kw'])).mean(),
                    zero_feed_samples=int(selected_feed.eq(0).sum()),
                    label_missing_values=int(q[[f'Feedrate_{hopper}']].isna().sum().sum()+z[['Voltage',f'Press._Hopper_{hopper}']].isna().sum().sum()),
                    station_voltage_mean=q.Voltage.mean(),station_feedrate_mean=selected_feed.mean(),station_pressure_mean=q[f'Press._Hopper_{hopper}'].mean(),
                    within_current_range=q.Current.max()-q.Current.min(),within_argon_range=q.Argon.max()-q.Argon.min(),within_nitrogen_range=q.Nitrogen.max()-q.Nitrogen.min())
            bits=failure_bits(record['voltage_mean'],record['feedrate_mean'],record['pressure_mean'],cfg)
            record.update(voltage_fail=bits[0],feedrate_fail=bits[1],hopper_pressure_fail=bits[2])
            reasons=[]
            if not hopper:reasons.append('unknown_active_hopper')
            if not record['left_boundary_observed'] or not record['right_boundary_observed']:reasons.append('unobserved_product_boundary')
            if record['coverage']<cfg['minimum_sampling_coverage']:reasons.append('sampling_coverage_below_minimum')
            if record['dual_feed_samples']:reasons.append('both_hoppers_feeding')
            if record['other_station_running_samples']:reasons.append('both_stations_running')
            if record['conflict_samples']:reasons.append('timestamp_conflict')
            if record['label_missing_values']:reasons.append('missing_rule_values')
            if any(x is None for x in bits):reasons.append('unknown_rule_mean')
            record['quality_flags']=';'.join(reasons);record['label_valid']=int(not reasons)
            record['failure']=int(any(bits)) if not reasons else None
            record['failure_type']='|'.join(name for name,flag in zip(['voltage','feedrate','hopper_pressure'],bits) if flag==1) if not reasons else 'unknown'
            if record['failure']==0:record['failure_type']='none'
            alternate=failure_bits(record.get('station_voltage_mean',np.nan),record.get('station_feedrate_mean',np.nan),record.get('station_pressure_mean',np.nan),cfg)
            record['averaging_scope_disagreement']=bits!=alternate
            rows.append(record)
    return rows


def main_products(root,m,cfg,out):
    rows=[];minute_parts=[];state_rows=[];gaps=[];quality=[];sensitivity=[];previous=None
    days=sorted(set(m.loc[(m.data_stream=='main')&m.rows.gt(0),'time_start'].dt.date))
    begin=pd.Timestamp(cfg['analysis_start']);finish=pd.Timestamp(cfg['analysis_end_exclusive'])
    for i,day in enumerate(days):
        if pd.Timestamp(day)>=finish:continue
        d,qa=load_day(root,m,day)
        if d.empty:continue
        nxt=None
        if i+1<len(days):
            nextfiles=m[m.time_start.dt.date.eq(days[i+1]) & m.data_stream.eq('main') & m.rows.gt(0)]
            if len(nextfiles):
                first_hour=nextfiles.time_start.min().hour
                nxt,_=load_day(root,m[m.time_start.dt.hour.eq(first_hour)],days[i+1])
        context=pd.concat([q for q in [previous,d,nxt] if q is not None and len(q)],ignore_index=True).sort_values('timestamp').reset_index(drop=True)
        current=[r for r in extract_products(context,cfg) if r['start'].date()==day and begin<=r['start']<finish]
        rows.extend(current)
        if pd.Timestamp(day)>=begin.floor('D'):
            for threshold,minimum,mg in [(50,15,1),(50,15,2),(100,15,2),(50,30,2)]:
                altcfg={**cfg,'maximum_sampling_gap_seconds':mg}
                pp=[r for r in extract_products(context,altcfg,threshold,minimum) if r['start'].date()==day and begin<=r['start']<finish]
                sensitivity.append(dict(day=str(day),station_threshold=threshold,minimum_samples=minimum,max_gap=mg,products=len(pp),valid_products=sum(r['label_valid'] for r in pp),voltage_failures=sum(r['voltage_fail']==1 and r['label_valid']==1 for r in pp),all_failures=sum(r['failure']==1 for r in pp)))
        previous=d[d.timestamp>=d.timestamp.iloc[-1].floor('h')].copy()
        powered=(d.Current>cfg['powered_current_above_a'])&(d.Power>cfg['powered_power_above_kw'])
        for q in contiguous_runs(d,powered):state_rows.append(dict(start=q.timestamp.iloc[0],end=q.timestamp.iloc[-1],powered=True))
        for q in contiguous_runs(d,~powered):state_rows.append(dict(start=q.timestamp.iloc[0],end=q.timestamp.iloc[-1],powered=False))
        good=powered & d.Voltage.gt(cfg['voltage_reference_band_floor_v']) & ~d.timestamp_conflict
        z=d[good]
        mm=z.groupby(z.timestamp.dt.floor('min')).agg(samples=('Voltage','size'),voltage_median=('Voltage','median'),voltage_mean=('Voltage','mean'),voltage_std=('Voltage','std'))
        minute_parts.append(mm[mm.samples>=30].reset_index())
        gap=d.timestamp.diff().dt.total_seconds()
        for j in np.flatnonzero(gap.gt(1).to_numpy()):gaps.append(dict(before=d.timestamp.iloc[j-1],after=d.timestamp.iloc[j],gap_seconds=gap.iloc[j]))
        quality.append(dict(day=str(day),rows=len(d),numeric_missing_cells=int(d[PHYSICAL].isna().sum().sum()),**qa))
        print('products',day,len(current),'total',len(rows),flush=True)
    p=pd.DataFrame(rows).sort_values(['start','station']).reset_index(drop=True)
    p['product_id']=[f'P{i:06d}' for i in range(1,len(p)+1)]
    runs=[]
    for r in sorted(state_rows,key=lambda x:x['start']):
        if runs and r['powered']==runs[-1]['powered'] and (r['start']-runs[-1]['end']).total_seconds()<=cfg['maximum_sampling_gap_seconds']:
            runs[-1]['end']=r['end']
        else:runs.append(r.copy())
    state=pd.DataFrame(runs);state['duration_seconds']=(state.end-state.start).dt.total_seconds()+1
    ep=state[state.powered].reset_index(drop=True)
    indices=np.searchsorted(ep.start.to_numpy(),p.start.to_numpy(),side='right')-1
    p['power_episode_id']=np.maximum(indices,0)+1
    p['power_episode_start']=ep.start.iloc[np.maximum(indices,0)].to_numpy()
    p['seconds_since_power_on']=(p.start-p.power_episode_start).dt.total_seconds()
    p['startup_review']=p.seconds_since_power_on.lt(cfg['startup_review_seconds'])
    p['production_control_transition']=(p.powered_fraction.lt(.98)|p.within_current_range.gt(cfg['recipe_current_change_a'])|p.within_argon_range.gt(cfg['recipe_argon_change_nlpm'])|p.within_nitrogen_range.gt(cfg['recipe_nitrogen_change_nlpm']))
    minute=pd.concat(minute_parts,ignore_index=True).drop_duplicates('timestamp').sort_values('timestamp')
    # Exclude only startup minutes from SHAPE evidence, never delete product failures.
    ei=np.searchsorted(ep.start.to_numpy(),minute.timestamp.to_numpy(),side='right')-1
    minute['seconds_since_power_on']=(minute.timestamp.to_numpy()-ep.start.iloc[np.maximum(ei,0)].to_numpy())/np.timedelta64(1,'s')
    minute['shape_eligible']=minute.seconds_since_power_on.ge(cfg['reference_startup_guard_seconds'])
    write_csv(pd.DataFrame(quality),out/'每日数据检查.csv');write_csv(pd.DataFrame(gaps),out/'日内采样缺口.csv')
    ss=pd.DataFrame(sensitivity).groupby(['station_threshold','minimum_samples','max_gap'],as_index=False)[['products','valid_products','voltage_failures','all_failures']].sum()
    write_csv(ss,out/'分段参数敏感性.csv')
    write_csv(state,out/'设备带电状态区间.csv');write_csv(minute,out/'主数据电压分钟走势.csv')
    return p,minute,state,quality


def normalize_shape(values,points=48):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    if len(a)<2:return None
    a=np.interp(np.linspace(0,len(a)-1,points),np.arange(len(a)),a)
    scale=max(float(np.quantile(a,.75)-np.quantile(a,.25)),.3)
    return (a-np.median(a))/scale


def constrained_dtw_rmse(a,b,radius):
    """Exact squared-cost DTW in a Sakoe-Chiba band, sqrt(cost/path length)."""
    n,m=len(a),len(b);cost=np.full((n+1,m+1),np.inf);length=np.zeros((n+1,m+1),dtype=int);cost[0,0]=0
    for i in range(1,n+1):
        for j in range(max(1,i-radius),min(m,i+radius)+1):
            options=[(cost[i-1,j-1],length[i-1,j-1]),(cost[i-1,j],length[i-1,j]),(cost[i,j-1],length[i,j-1])]
            value,steps=min(options,key=lambda x:x[0]);cost[i,j]=value+(a[i-1]-b[j-1])**2;length[i,j]=steps+1
    return float(np.sqrt(cost[n,m]/length[n,m])) if length[n,m] else float('inf')


def shape_evidence(minute,endpoint,templates,cfg):
    q=minute[minute.shape_eligible & (minute.timestamp+pd.Timedelta(minutes=1)).le(endpoint)&minute.timestamp.ge(endpoint-pd.Timedelta(minutes=cfg['shape_history_minutes']))].tail(cfg['shape_maximum_valid_minutes'])
    result=dict(shape_minutes=len(q),dtw_distance=np.nan,drop_v=np.nan,slope_v_per_minute=np.nan,shape_match=False)
    if len(q)<cfg['shape_minimum_valid_minutes']:return result
    a=q.voltage_median.to_numpy();x=(q.timestamp-q.timestamp.iloc[0]).dt.total_seconds().to_numpy()/60
    norm=normalize_shape(a,cfg['shape_resample_points'])
    distances=[constrained_dtw_rmse(norm,normalize_shape(t,cfg['shape_resample_points']),cfg['dtw_band_radius']) for t in templates]
    result.update(dtw_distance=min(distances),drop_v=float(np.median(a[:5])-np.median(a[-5:])),slope_v_per_minute=float(np.polyfit(x,a,1)[0]))
    result['shape_match']=bool(result['dtw_distance']<=cfg['dtw_maximum_normalized_rmse'] and result['drop_v']>=cfg['natural_minimum_drop_v'] and result['slope_v_per_minute']<=cfg['natural_maximum_slope_v_per_minute'])
    return result


def export_excerpts(root,m,p,e,cfg,out):
    windows=[]
    for r in e.to_dict('records'):
        if r['event_kind']=='product_failure':
            windows.append(dict(start=r['start']-pd.Timedelta(minutes=cfg['excerpt_before_minutes']),end=r['end']+pd.Timedelta(minutes=cfg['excerpt_after_minutes']),event_id=r['event_id']))
        else:
            related=p[p.product_id.isin(str(r['product_ids']).split(';'))]
            pre=related[related.end.lt(r['start'])];post=related[related.start.gt(r['end'])]
            lo=min(r['start']-pd.Timedelta(minutes=5),pre.start.min()) if len(pre) else r['start']-pd.Timedelta(minutes=5)
            hi=max(r['end']+pd.Timedelta(minutes=5),post.end.max()) if len(post) else r['end']+pd.Timedelta(minutes=5)
            windows.append(dict(start=lo,end=r['start']+pd.Timedelta(minutes=2),event_id=r['event_id']))
            windows.append(dict(start=r['end']-pd.Timedelta(minutes=2),end=hi,event_id=r['event_id']))
    attention=pd.DataFrame(windows)
    total=0;files=0;recheck=[]
    for r in m[(m.data_stream=='main')&m.identical_bytes_to.eq('')&m.rows.gt(0)].to_dict('records'):
        wins=attention[(attention.start<=r['time_end'])&(attention.end>=r['time_start'])]
        if wins.empty:continue
        d=parse_csv(payload(root,r),strings=True);d['source_row']=np.arange(len(d))+2
        keep=np.zeros(len(d),bool)
        for w in wins.to_dict('records'):keep|=d.timestamp.between(w['start'],w['end']).to_numpy()
        d=d[keep & d.timestamp.ge(cfg['analysis_start']) & d.timestamp.lt(cfg['analysis_end_exclusive'])].copy()
        if d.empty:continue
        d['product_id']='';d['failure']='';d['failure_type']='not_applicable';d['label_valid']=0
        d['voltage_fail']='';d['feedrate_fail']='';d['hopper_pressure_fail']=''
        d['origin_label']='非加工/启停或产品间隙';d['origin_evidence']='不在已识别完整产品区间，failure留空'
        d['attention_event_ids']='';d['review_event_labels']=''
        d['source_id']=r['source_id'];d['cause_confirmed']=0
        d['origin_label_uses_future_context']=1
        d['voltage_mean_product']='';d['feedrate_mean_product']='';d['pressure_mean_product']=''
        d['label_available_at']='';d['label_scope']='not_applicable';d['quality_flags']=''
        for name in ['failure','voltage_fail','feedrate_fail','hopper_pressure_fail',
                     'voltage_mean_product','feedrate_mean_product','pressure_mean_product']:
            d[name]=d[name].astype(object)
        candidates=p[p.start.le(d.timestamp.max())&p.end.ge(d.timestamp.min())]
        for q in candidates.to_dict('records'):
            hit=d.timestamp.between(q['start'],q['end'])
            if not hit.any():continue
            values={k:q[k] for k in ['product_id','failure','failure_type','label_valid','voltage_fail','feedrate_fail','hopper_pressure_fail','origin_label','origin_evidence','quality_flags']}
            values.update(voltage_mean_product=q['voltage_mean'],feedrate_mean_product=q['feedrate_mean'],pressure_mean_product=q['pressure_mean'],label_available_at=q['label_available_at'].strftime(TIME_FORMAT) if pd.notna(q['label_available_at']) else '',label_scope='retrospective_product_label')
            for key,value in values.items():d.loc[hit,key]='' if pd.isna(value) else value
        for w in wins.to_dict('records'):
            hit=d.timestamp.between(w['start'],w['end'])
            d.loc[hit,'attention_event_ids']=d.loc[hit,'attention_event_ids'].map(lambda s:';'.join(sorted(set(filter(None,[*s.split(';'),w['event_id']])))))
        for er in e[e.event_kind.eq('replacement_candidate')].to_dict('records'):
            hit=d.timestamp.between(er['start']-pd.Timedelta(minutes=5),er['end']+pd.Timedelta(minutes=5))
            d.loc[hit,'review_event_labels']=d.loc[hit,'review_event_labels'].map(lambda s:';'.join(filter(None,[s,er['origin_label']])))
        d=d.drop(columns='timestamp')
        container=Path(r['container'])
        group=container.stem if r['zip_member'] else container.parts[0]
        filename=Path(r['zip_member']).name if r['zip_member'] else container.name
        dest=out/'关注片段_28参数'/group/filename
        write_csv(d,dest,raw=True);total+=len(d);files+=1
        # Round-trip original fields are compared, not merely row counts.
        again=parse_csv(dest.read_bytes(),strings=True)
        assert again[ORIGINAL].equals(d[ORIGINAL].reset_index(drop=True)),f'Raw values changed: {dest}'
        recheck.append(dict(relative_file=dest.relative_to(out).as_posix(),rows=len(d),source_id=r['source_id'],all_original_30_fields_preserved=True))
    write_csv(pd.DataFrame(recheck),out/'导出文件核对.csv')
    return dict(excerpt_files=files,excerpt_rows=total)



if __name__=='__main__':
    from revise_labels import main
    main()
