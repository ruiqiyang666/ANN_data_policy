"""Evaluation helpers: future event matching, merged alerts, missing-aware metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
from modeling import TARGETS,seconds,remaining_quantiles

def probability_metrics(y,p):
    valid=np.isfinite(y)&np.isfinite(p);yy=y[valid];pp=p[valid]
    n=len(yy);positive=int(yy.sum());negative=n-positive
    result=dict(eligible=n,positives=positive,negatives=negative,pr_auc=None,roc_auc=None,
                brier=float(np.mean((yy-pp)**2)) if n else None,ece=None,
                note=None,reliability=[])
    if positive and negative:
        result['pr_auc']=float(average_precision_score(yy,pp))
        result['roc_auc']=float(roc_auc_score(yy,pp))
        result['prevalence']=positive/n
    else: result['note']='无法评估区分能力：没有正例或没有负例；Brier仍可反映可观察样本概率误差。'
    ece=0.
    for low in np.linspace(0,.9,10):
        ix=(pp>=low)&(pp<low+.1 if low<.9 else pp<=1)
        if ix.any():
            avg=float(pp[ix].mean());freq=float(yy[ix].mean())
            ece+=ix.sum()/max(n,1)*abs(avg-freq)
            result['reliability'].append(dict(lower=float(low),count=int(ix.sum()),predicted=avg,observed=freq))
    if n:result['ece']=float(ece)
    return result

def build_alarms(frame,probability,threshold,cooldown=1800,merge=600,confirm=2):
    """Two recent origins confirm; continuous warnings merge, then 30min cooldown."""
    alarms=[];now=seconds(frame.timestamp)
    for cycle,ix in frame.groupby('cycle_id',sort=False).indices.items():
        ix=np.sort(ix);consecutive=0;last_above=None;last_alarm=-10**15;last_below=-10**15;active=False
        for i in ix:
            at=int(now[i]);above=np.isfinite(probability[i]) and probability[i]>=threshold
            if above:
                consecutive=consecutive+1 if last_above is not None and at-last_above<=90 else 1
                last_above=at
                # Clear an episode after >=10min without a qualifying origin, including gaps.
                if active and at-last_signal>merge: active=False
                if consecutive>=confirm and not active and at-last_alarm>=cooldown:
                    alarms.append(dict(index=int(i),time=str(frame.timestamp.iloc[i]),cycle_id=cycle,probability=float(probability[i])))
                    active=True;last_alarm=at
                last_signal=at
            else:
                consecutive=0
                if active and at-last_signal>=merge: active=False
                last_below=at
    return alarms

def event_metrics(frame,probability,threshold,events,y60,operating_hours,window=3600):
    alarms=build_alarms(frame,probability,threshold)
    now=seconds(frame.timestamp);results=[]
    events=events.sort_values('time')
    # Each event is counted once; an alarm is credited to its first future endpoint.
    assigned={}
    for ai,a in enumerate(alarms):
        i=a['index'];e=events[events.cycle_id.eq(a['cycle_id'])]
        delta=seconds(e.time)-now[i]
        valid=(delta>0)&(delta<=window)&(delta<frame.observation_horizon_seconds.iloc[i])
        if valid.any():
            match=e.iloc[np.flatnonzero(valid)[0]]
            assigned[ai]=match.event_id
            a['event_id']=match.event_id;a['false_alarm']=False;a['unknown']=False
        else:
            a['event_id']=None
            a['false_alarm']=bool(np.isfinite(y60[i]) and y60[i]==0)
            a['unknown']=not a['false_alarm']
    for e in events.itertuples():
        matching=[a for k,a in enumerate(alarms) if assigned.get(k)==e.event_id]
        first=matching[0] if matching else None
        lead=(pd.Timestamp(e.time)-pd.Timestamp(first['time'])).total_seconds()/60 if first else None
        endpoint=int(pd.Timestamp(e.time).to_datetime64().astype('datetime64[s]').astype('int64'))
        opportunity=(frame.cycle_id.eq(e.cycle_id).to_numpy())&(endpoint-now>0)&(endpoint-now<=3600)
        opportunity &= (endpoint-now<frame.observation_horizon_seconds.to_numpy())
        max_lead=float(np.max((endpoint-now[opportunity])/60)) if opportunity.any() else None
        results.append(dict(event_id=e.event_id,time=str(e.time),cycle_id=e.cycle_id,evidence=e.evidence,
            lead_minutes=lead,hit15=lead is not None and lead>=15,hit30=lead is not None and lead>=30,
            missed=first is None,first_alarm=first['time'] if first else None,
            maximum_observable_lead_minutes=max_lead,
            note='没有60分钟内可观察的预测机会' if max_lead is None else None))
    total=len(results);false=sum(a['false_alarm'] for a in alarms);unknown=sum(a['unknown'] for a in alarms)
    metrics=dict(total_events=total,hit15=sum(r['hit15'] for r in results) if total else None,
        hit30=sum(r['hit30'] for r in results) if total else None,
        missed=sum(r['missed'] for r in results) if total else None,
        false_alarms=false,unknown_alarms=unknown,alarms=len(alarms),operating_hours=float(operating_hours),
        false_alarms_per_hour=float(false/max(operating_hours,1e-9)),threshold=float(threshold))
    return metrics,results,alarms

def choose_threshold(frame,p,events,y,hours,false_budget=.5):
    best=None
    # Fixed modest grid, entirely within validation.
    candidates=np.unique(np.r_[np.linspace(.02,.98,49),1.000001])
    for threshold in candidates:
        metrics,_,_=event_metrics(frame,p,float(threshold),events,y,hours)
        feasible=metrics['false_alarms_per_hour']<=false_budget
        key=(feasible,metrics['hit30'] or 0,metrics['hit15'] or 0,
             -(metrics['missed'] or 0),-metrics['false_alarms_per_hour'],float(threshold))
        if best is None or key>best[0]:best=(key,metrics)
    return best[1]

def endpoint_events(cycles,split,weak=False):
    flag='eol_sensor_observed' if weak else 'eol_strict_observed'
    q=cycles[cycles.split.eq(split)&cycles[flag].eq(1)&cycles.eol_endpoint.notna()]
    return pd.DataFrame([dict(event_id=('WEAK_' if weak else 'EOL_')+r.cycle_id,time=pd.Timestamp(r.eol_endpoint),
        cycle_id=r.cycle_id,evidence='sensor_only' if weak else 'site_or_register') for r in q.itertuples()],
        columns=['event_id','time','cycle_id','evidence'])

def remaining_metrics(frame,p,weak=False):
    col='weak_remaining_seconds' if weak else 'strict_remaining_seconds'
    truth=frame[col].to_numpy(float)/60
    eligible=(truth>0)&(truth<=60)&(truth*60<frame.observation_horizon_seconds.to_numpy(float))
    # Pick at most one origin per calendar minute to avoid presenting overlap as independent evidence.
    eligible &= frame.timestamp.dt.second.eq(0).to_numpy()
    actual=[];predicted=[];covered=[];provided=0;both=0
    for i in np.flatnonzero(eligible):
        q=remaining_quantiles(p[i,:3])
        if q['remaining_minutes_q50'] is not None:
            actual.append(truth[i]);predicted.append(q['remaining_minutes_q50']);provided+=1
        if q['remaining_minutes_q10'] is not None and q['remaining_minutes_q90'] is not None:
            covered.append(q['remaining_minutes_q10']<=truth[i]<=q['remaining_minutes_q90']);both+=1
    err=np.asarray(predicted)-np.asarray(actual)
    return dict(eligible_origins=int(eligible.sum()),median_provided=provided,
        mae_minutes=float(np.abs(err).mean()) if len(err) else None,
        bias_minutes=float(err.mean()) if len(err) else None,
        interval_coverage=float(np.mean(covered)) if covered else None,interval_provided=both,
        note='仅已观察终点前60分钟且可观察、每分钟原点；CDF未达到50%/90%时返回超出时域，不能强造精确剩余时间。')
