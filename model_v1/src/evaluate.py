"""Evaluate frozen configurations once; export real event/case/dashboard artifacts."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
import joblib,torch
from modeling import TARGETS,RiskNet,summaries,margin_probabilities,apply_calibration,seconds
from train import infer
from evaluation import probability_metrics,endpoint_events,event_metrics,remaining_metrics
from prepare import window_target

ROOT=Path(__file__).resolve().parents[1]

def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,(np.integer,np.floating,np.bool_)):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,pd.Timestamp):return str(value)
    return value

def frozen_predictions(root,x,frame,cfg):
    bases=joblib.load(root/'artifacts/baselines.joblib');z=summaries(x)
    p=dict(event_rate=np.tile(bases['event_rate'],(len(frame),1)),margin=margin_probabilities(frame,x),
        logistic=np.column_stack([m.predict_proba(z)[:,1] if hasattr(m,'predict_proba') else np.repeat(m,len(frame)) for m in bases['logistic']]))
    for kind in ['tcn','gru']:
        net=RiskNet(kind,hidden=cfg['hidden'])
        net.load_state_dict(torch.load(root/f'artifacts/{kind}/best.pt',weights_only=False,map_location='cpu')['model'])
        p[kind]=infer(net,x)
    calibrators=joblib.load(root/'artifacts/calibrators.joblib')
    return {name:apply_calibration(a,calibrators[name]) for name,a in p.items()}

def product_predictions(frame,y,p,products):
    """At actual future start, score last already-frozen prediction <=5min old."""
    rows=[];now=seconds(frame.timestamp)
    for cycle,ix in frame.groupby('cycle_id',sort=False).indices.items():
        ix=np.sort(ix);q=products[products.cycle_id.eq(cycle)&products.supervised_product_eligible.eq(1)]
        starts=seconds(q.start)
        j=np.searchsorted(now[ix],starts,side='left')-1
        for product,position in zip(q.itertuples(),j):
            if position<0:continue
            i=ix[position]
            if (product.start-frame.timestamp.iloc[i]).total_seconds()>300:continue
            if frame.next_product_id.iloc[i]!=product.product_id:continue
            rows.append(dict(product_id=product.product_id,cycle_id=cycle,split=product.split,
                start=str(product.start),prediction_origin=str(frame.timestamp.iloc[i]),data_cutoff=str(frame.timestamp.iloc[i]),
                lead_seconds=(product.start-frame.timestamp.iloc[i]).total_seconds(),
                **{f'label_{name}':float(y[i,9+k]) for k,name in enumerate(['any','voltage','feedrate','pressure'])},
                **{f'risk_{name}':float(p[i,9+k]) for k,name in enumerate(['any','voltage','feedrate','pressure'])}))
    return pd.DataFrame(rows)

def precursors(frame,cycles):
    rows=[];details=[]
    train=frame[frame.split.eq('train')]
    now=seconds(frame.timestamp)
    for minutes in [60,30,15,5]:
        values=[]
        for c in cycles[cycles.split.eq('train')&cycles.eol_strict_observed.eq(1)].itertuples():
            target=pd.Timestamp(c.eol_endpoint)-pd.Timedelta(minutes=minutes)
            subset=frame[frame.cycle_id.eq(c.cycle_id)&frame.timestamp.le(target)&frame.timestamp.ge(target-pd.Timedelta(seconds=90))]
            v=float(subset.iloc[-1].Voltage) if len(subset) else None
            details.append(dict(cycle_id=c.cycle_id,minutes=minutes,voltage=v,origin=str(subset.iloc[-1].timestamp) if len(subset) else None))
            if v is not None:values.append(v)
        # Normal comparison is training-only, with observed strict endpoint >2h away or censored.
        normal=train[(train.strict_remaining_seconds.isna()|train.strict_remaining_seconds.gt(7200))&train.timestamp.dt.second.eq(0)]
        rows.append(dict(minutes=minutes,event_count=len(values),normal_count=len(normal),
            voltage_event=float(np.median(values)) if values else None,
            voltage_normal=float(normal.Voltage.median()),
            note='仅训练严格终点；预测时刻往前90秒内无加工读数则缺失，未用跨夜最近点替代。点数并非独立正常周期数。'))
    return rows,details

def make_case(frame,p,row,name,case_id,kind):
    when=pd.Timestamp(row['time']);cycle=row['cycle_id']
    if row.get('first_alarm'):
        origin=pd.Timestamp(row['first_alarm'])
    else:
        wanted=when-pd.Timedelta(minutes=30)
        available=frame[frame.cycle_id.eq(cycle)&frame.timestamp.le(wanted)&frame.timestamp.ge(wanted-pd.Timedelta(seconds=90))]
        if available.empty:
            available=frame[frame.cycle_id.eq(cycle)&frame.timestamp.lt(when)&frame.timestamp.ge(when-pd.Timedelta(minutes=60))]
        origin=pd.Timestamp(available.iloc[-1].timestamp) if len(available) else None
    if origin is None:
        return dict(id=case_id,title=f'{cycle} · 无可观察原点',kind=kind,split=row['split'],
            origin=None,cycle_id=cycle,model=name,note='终点前60分钟无已保存可观察原点，无法构造预测案例。',points=[],events=[])
    start=min(origin-pd.Timedelta(minutes=30),when-pd.Timedelta(minutes=75))
    q=frame[frame.cycle_id.eq(cycle)&frame.timestamp.ge(start)&frame.timestamp.le(when+pd.Timedelta(minutes=15))]
    points=[]
    for i,r in q.iterrows():
        points.append(dict(time=str(r.timestamp),voltage=float(r.Voltage),risk=float(p[i,2]),
                           prediction_origin=str(r.timestamp),data_cutoff=str(r.timestamp)))
    return dict(id=case_id,title=f"{cycle} · "+('误报' if kind=='false_alarm' else '弱证据末期' if row.get('evidence')=='sensor_only' else '已确认末期'),
        kind=kind,split=row['split'],origin=str(origin),cycle_id=cycle,model=name,
        note=('这是传感器末期线索，未取得当日换枪登记。' if row.get('evidence')=='sensor_only' else '')+
             '每条风险是对应时刻独立保存的预测；原点之后电压为后来实测。采样缺口留空。',
        points=points,events=[dict(time=str(when),label='报警原点' if kind=='false_alarm' else '第二件低压完成')])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,default=ROOT);args=ap.parse_args()
    root=args.project;art=root/'artifacts'
    cfg=json.loads((art/'frozen_config.json').read_text(encoding='utf-8'))
    frozen_hash=hashlib.file_digest((art/'frozen_config.json').open('rb'),'sha256').hexdigest()
    torch.set_num_threads(cfg['threads'])
    data=np.load(art/'dataset.npz');x=data['x'];y=data['y']
    frame=pd.read_csv(art/'anchors.csv.gz',parse_dates=['timestamp'])
    cycles=pd.read_csv(art/'cycles.csv',parse_dates=['eol_endpoint'])
    products=pd.read_csv(art/'products.csv.gz',parse_dates=['start','end'],low_memory=False)
    # Predictions reconstructed only from frozen weights/config/preprocessor, before reading test labels for metrics.
    predictions=frozen_predictions(root,x,frame,cfg)
    np.savez_compressed(art/'predictions_frozen.npz',**predictions)
    model_rows=[];all_events=[];all_alarms=[];cases=[];reports={};sensitivity=[]
    for name,p in predictions.items():
        scored_products=product_predictions(frame,y,p,products)
        scored_products['model']=name
        scored_products.to_csv(art/f'product_predictions_{name}.csv',index=False)
        history=json.loads((art/name/'history.json').read_text(encoding='utf-8')) if (art/name/'history.json').exists() else []
        row=dict(name=name,selected=name==cfg['selected_model'],history=history,metrics={})
        reports[name]={}
        for split in ['train','validation','test']:
            ix=np.flatnonzero(frame.split.eq(split));sf=frame.iloc[ix].reset_index(drop=True);sp=p[ix];sy=y[ix]
            hours=products.loc[products.split.eq(split)&products.supervised_product_eligible.eq(1),'spray_samples'].sum()/3600
            events=endpoint_events(cycles,split)
            metrics,results,alarms=event_metrics(sf,sp[:,2],cfg['thresholds'][name]['eol_60m'],events,sy[:,2],hours)
            metrics['strict_events']=len(events);metrics['weak_events']=len(endpoint_events(cycles,split,True))
            metrics['targets']=[dict(name=target,**probability_metrics(sy[:,j],sp[:,j])) for j,target in enumerate(TARGETS)]
            q=scored_products[scored_products.split.eq(split)]
            metrics['product_targets']=[dict(name='next_'+cause,**probability_metrics(q['label_'+cause].to_numpy(),q['risk_'+cause].to_numpy())) for cause in ['any','voltage','feedrate','pressure']]
            primary=metrics['product_targets'][0]
            metrics.update(pr_auc=primary['pr_auc'],brier=primary['brier'],ece=primary['ece'],
                reliability=primary['reliability'],pr_auc_note='每件一条，实际开工前<=5分钟的已保存下一件风险；无近期原点的产品未评分。')
            metrics['rul']=remaining_metrics(sf,sp)
            metrics['rul']['coverage']=metrics['rul']['interval_coverage']
            if split=='test':
                metrics['note']='严格寿命终点0个，命中率、严格RUL MAE与覆盖率无法评估；下一件电压风险可评分，送料/罐压无正例。'
            for e in results:
                e.update(model=name,split=split,case_id=f'{name}-{split}-{e["event_id"]}')
                all_events.append(e)
                cases.append(make_case(frame,p,e,name,e['case_id'],'miss' if e['missed'] else 'success'))
            for a in alarms:
                a.update(model=name,split=split,reason='eol_60m超过冻结阈值，两原点确认',case_id=None)
                all_alarms.append(a)
            # Weak test is a separate endpoint experiment using unchanged strict model/threshold.
            if split=='test':
                weak=endpoint_events(cycles,split,True)
                wy=window_target(sf.weak_remaining_seconds,sf.observation_horizon_seconds,3600)
                # Unknown reported-damage cycle remains unscored, not a weak negative.
                wy[sf.cycle_id.eq('L011')]=np.nan
                wm,wr,wa=event_metrics(sf,sp[:,2],cfg['thresholds'][name]['eol_60m'],weak,wy,hours)
                metrics['weak_endpoint_metrics']=wm;metrics['weak_rul']=remaining_metrics(sf,sp,True)
                for e in wr:
                    e.update(model=name,split=split,case_id=f'{name}-{split}-{e["event_id"]}')
                    all_events.append(e);cases.append(make_case(frame,p,e,name,e['case_id'],'weak'))
            # Same frozen adjacent-voltage task can be scored independently of natural life evidence.
            adjacent=pd.read_csv(art/'target_events.csv')
            adjacent=adjacent[adjacent.split.eq(split)&adjacent.kind.eq('adjacent_two_voltage')].copy()
            adjacent['time']=pd.to_datetime(adjacent.time,format='%d %m %Y %H:%M:%S');adjacent['evidence']='diagnostic'
            dm,dr,da=event_metrics(sf,sp[:,5],cfg['thresholds'][name]['adjacent_voltage_60m'],adjacent,sy[:,5],hours)
            metrics['adjacent_voltage_events']=dm
            for e in dr:e.update(model=name,split=split)
            pd.DataFrame(dr).to_csv(art/f'adjacent_events_{name}_{split}.csv',index=False)
            row['metrics'][split]=metrics;reports[name][split]=metrics
            # A bounded selection of actual false-alert cases, all remain in CSV.
            for a in [a for a in alarms if a['false_alarm']][:2]:
                a['case_id']=f'{name}-{split}-false-{a["index"]}'
                cases.append(make_case(frame,p,dict(a,first_alarm=a['time']),name,a['case_id'],'false_alarm'))
            if split=='validation' and name==cfg['selected_model']:
                # Costs unknown: show chosen threshold +-0.1 as descriptive sensitivity; never retune test.
                for th in np.clip([cfg['thresholds'][name]['eol_60m']-.1,cfg['thresholds'][name]['eol_60m'],cfg['thresholds'][name]['eol_60m']+.1],0,1.000001):
                    m,_,_=event_metrics(sf,sp[:,2],float(th),events,sy[:,2],hours)
                    sensitivity.append(dict(model=name,split=split,**m))
        model_rows.append(row)
    prec,prec_detail=precursors(frame,cycles)
    limits=['测试集严格确认寿命终点为0，无法给出严格寿命测试成绩。',
        '11/10只有1个传感器末期过程；弱证据结果不证明泛化能力。',
        '测试送料、罐压失败正例均为0；全量没有连续两件送料失败。',
        '罐压连续失败仅1个训练过程，模型没有独立罐压验证。',
        '模型实际读取过去30分钟带缺失网格；没有完整24小时→60分钟窗口。',
        '终点是相邻第二件低压完成，PLC批次边界未知，维护策略尚未优化。',
        '实时服务目前接标准事件或文件回放，尚未接现场通信。',
        '验证集仅2个严格终点，并重复用于校准、早停、模型和阈值选择，验证成绩有选择乐观偏差。']
    dashboard=dict(meta=dict(status='完成正式训练与冻结测试',trained_at=cfg['trained_at'],data_version='11_16数据处理',
        device='Booth2',selected_model=cfg['selected_model'],model_version=cfg['model_version'],history_minutes=30,
        compute='CPU · 4线程（正式训练）',
        scope_note='2026-09-12至11-14；CPU正式训练；原划分；测试冻结后评估。下一件按实际开工前的已保存原点评分。',limits=limits),
        models=model_rows,event_results=all_events,alarm_results=all_alarms,cases=cases,precursors=prec,
        sensitivity=sensitivity,protocol=dict(frozen_config_sha256=frozen_hash,evaluated_at=str(pd.Timestamp.now()),
            alarm_confirm_origins=2,maximum_confirm_gap_seconds=90,merge_seconds=600,cooldown_seconds=1800,
            operating_hours='严格可监督产品活动料斗正送料实际观测秒/3600；不含待机、未知与跨夜',
            history='31 calendar grid steps, one minute apart, 30min; backward age<=60s, missing masks',
            remaining_time='calendar minutes; restricted cumulative risk on 0/15/30/60min, not an optimal-change label'))
    for file,obj in [('dashboard.json',dashboard),('evaluation.json',reports),('precursor_details.json',prec_detail),('sensitivity.json',sensitivity)]:
        (art/file).write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (art/'dashboard_data.js').write_text('window.DASHBOARD_DATA = '+json.dumps(clean(dashboard),ensure_ascii=False,allow_nan=False)+';\n',encoding='utf-8')
    pd.DataFrame(all_events).to_csv(art/'event_results.csv',index=False)
    pd.DataFrame(all_alarms).to_csv(art/'alarm_results.csv',index=False)
    # Explicit ledger: forecast origin and cutoff never rewritten from future actual values.
    chosen=predictions[cfg['selected_model']]
    ledger=frame[['anchor_id','timestamp','cycle_id','split']].copy().rename(columns={'timestamp':'prediction_origin'})
    ledger['data_cutoff']=ledger.prediction_origin;ledger['model_version']=cfg['model_version']
    for j,target in enumerate(TARGETS):ledger[target]=chosen[:,j]
    ledger.to_csv(art/'prediction_ledger.csv.gz',index=False,compression='gzip')
    print(json.dumps(clean(dict(selected_model=cfg['selected_model'],validation=reports[cfg['selected_model']]['validation'],test=reports[cfg['selected_model']]['test'])),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
