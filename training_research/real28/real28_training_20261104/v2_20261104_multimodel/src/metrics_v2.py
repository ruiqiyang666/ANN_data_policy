"""Event-aware ranking and explicitly exploratory alarm policies."""
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss

def eligible(samples):return samples.event_in_300s.to_numpy()|(samples.observed_seconds.to_numpy()>=300)

def ranking(samples,probability):
    ok=eligible(samples);s=samples.loc[ok];y=s.event_in_300s.to_numpy();p=np.asarray(probability)[ok]
    per=[];negative=~y
    for event in s.loc[y,'event_id'].unique():
        mask=negative|s.event_id.eq(event).to_numpy()
        per.append(float(average_precision_score(y[mask],p[mask])))
    return {'n':len(y),'positive_windows':int(y.sum()),'events':len(per),'prevalence':float(y.mean()),
            'average_precision':float(average_precision_score(y,p)) if y.any() else None,
            'event_macro_ap':float(np.mean(per)) if per else None,
            'roc_auc':float(roc_auc_score(y,p)) if y.any() and (~y).any() else None,
            'brier':float(brier_score_loss(y,p))}

def alarms(samples,probability,threshold,events=None):
    t=samples.timestamp.to_numpy(dtype='datetime64[s]').astype('int64');ok=eligible(samples)
    high=(np.asarray(probability)>=threshold)&ok
    rising=high&~np.r_[False,high[:-1]]
    rising|=high&np.r_[True,np.diff(t)>10]
    candidates=np.flatnonzero(rising);chosen=[];last=-10**18
    for i in candidates:
        if t[i]-last>=300:chosen.append(i);last=t[i]
    a=samples.iloc[chosen];good=a.event_in_300s.to_numpy();detected=set(a.loc[good,'event_id'])
    all_ids=set(samples.loc[samples.event_in_300s,'event_id']);hours=ok.sum()*10/3600
    out={'events':len(all_ids),'detected_events':len(detected),'missed_events':len(all_ids-detected),
         'event_recall':len(detected)/len(all_ids) if all_ids else None,
         'alerts':len(a),'false_alerts':int((~good).sum()),'observable_powered_hours':float(hours),
         'false_alerts_per_100_hours':float((~good).sum()/hours*100) if hours else None,'threshold':float(threshold)}
    records=[]
    if events is not None:
        for e in events[events.event_id.isin(all_ids)].itertuples():
            hit=a[a.event_id.eq(e.event_id)&a.event_in_300s]
            first=hit.timestamp.min() if len(hit) else None
            records.append({'event_id':e.event_id,'type':e.first_types,'event_at':e.event_at,
                            'product_start':e.first_product_start,'alert_at':first,
                            'lead_to_result_s':(e.event_at-first).total_seconds() if first is not None else None,
                            'lead_to_product_start_s':(e.first_product_start-first).total_seconds() if first is not None else None})
        out['detected_before_product_start']=sum(r['lead_to_product_start_s'] is not None and r['lead_to_product_start_s']>0 for r in records)
    return out,records,np.array(chosen,dtype=int)

def threshold_from_oof(samples,probability,budget=20):
    p=np.asarray(probability);grid=np.unique(np.r_[np.quantile(p,np.r_[np.linspace(.5,.95,40),np.linspace(.95,.99999,300)]),[0,.001,.003,.01,.03,.1,.3,.5,.9,1.000001]])
    records=[alarms(samples,p,float(th))[0] for th in grid]
    feasible=[r for r in records if r['false_alerts_per_100_hours']<=budget]
    best=max(feasible,key=lambda r:(r['event_recall'] or 0,-r['false_alerts_per_100_hours'],r['threshold']))
    return best,records
