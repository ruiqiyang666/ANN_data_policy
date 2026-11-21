"""Export the replay bundle the dashboard loads, and prove the dashboard maths match.

The bundle uses the fold-f6 fits (trained through 2026-10-23) so every replayed
prediction of 2026-10-27..11-01 is out-of-sample. The all-date final refits stay
in this run directory; no validation figure applies to them and they are not what
the presentation shows.

`panel_real28.py` next to this file is a byte copy of the dashboard's
simulation/pipeline/inference/real28.py. Every landmark of the week is scored
through it and compared with the predictions the training code saved.
"""
import sys
sys.dont_write_bytecode=True
import torch
import hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import sklearn
from common_v2 import ROOT,GRID,TYPES,FOLDS,BLIND_FOLD,NEW_PARTITION,FIELDS,load,dump,sha,time_seconds as sec
from metrics_v2 import eligible,alarms,threshold_from_oof,ranking
from train_v4 import calibrate,apply_calibration
import panel_real28

OUT=ROOT/'bundle'
NAMES={'Argon':('氩气流量','NLPM'),'Nitrogen':('氮气流量','NLPM'),'AirJet':('喷枪前端冷却气压','bar'),'Silvent':('转台侧冷却气压','bar'),
 'Current':('喷枪电流','A'),'Power':('喷枪功率','kW'),'Voltage':('喷枪电压','V'),'Cooling_water_flow':('冷却水流量','L/min'),
 'Conductivity':('冷却水电导率','μs（原标注，待确认）'),'Cooling_water_inlet_temp.':('冷却水入口温度','°C'),'Cooling_water_outlet_temp.':('冷却水出口温度','°C'),
 'Station_1':('转台 1 转速','rpm'),'Station_2':('转台 2 转速','rpm'),'O2_Sensor':('环境氧气','%')}
for i in (1,2):
    for key,name,unit in [('Feedrate','送料量','g/min'),('Disk','送粉盘设定','%'),('Carrier_gas','载气流量','NLPM'),('Feed_Factor','送料系数','—'),
                          ('Press._Hopper','料斗压力','mbar'),('Stirrer','搅拌器设定','%'),('Weight_Hopper','料斗重量','kg')]:NAMES[f'{key}_{i}']=(f'{name} {i}',unit)
# Fixed before export; what each one contains is measured below and written to the manifest.
SEGMENTS=[('normal_20261029','2026-10-29 09:30:00','2026-10-29 11:00:00','10月29日 09:30–11:00 常规生产'),
          ('voltage_onset_20261028','2026-10-28 07:55:00','2026-10-28 08:35:00','10月28日 07:55–08:35 低压出现前后'),
          ('voltage_events_20261101','2026-11-01 13:05:00','2026-11-01 13:40:00','11月1日 13:05–13:40 两次低压事件')]

def plain_transform(t):
    # dtypes are kept exactly as fitted (float32); see the dashboard transform().
    return {'center':np.asarray(t.center),'mean':np.asarray(t.mean),'scale':np.asarray(t.scale)}

def main():
    if OUT.exists():shutil.rmtree(OUT)
    (OUT/'models').mkdir(parents=True);(OUT/'replay').mkdir()
    protocol=load(ROOT/'protocol.json');names=load(ROOT/'data'/'features.json')['names'];assert len(names)==292
    blind_index=[f['name'] for f in FOLDS].index(BLIND_FOLD);earlier=[f['name'] for f in FOLDS[:blind_index]]
    reports={task:load(ROOT/'evaluation'/task/'v4_report.json') for task in TYPES+['life']};contract_tasks={};checks={}
    samples={task:pd.read_parquet(ROOT/'data'/task/'samples.parquet') for task in TYPES}
    ids=np.flatnonzero(samples['voltage'].partition.eq(NEW_PARTITION).to_numpy());features=np.load(ROOT/'data'/'features.npy',mmap_mode='r')
    for task in TYPES:
        algo=protocol['risk_selection'][task];source=joblib.load(ROOT/'models'/task/'risk'/algo/BLIND_FOLD/'bundle.joblib')
        if source['extra'].method is not None:raise AssertionError('selected algorithm must use the identity feature map')
        s=samples[task];parts=[dict(np.load(ROOT/'evaluation'/task/f'{algo}_{name}.npz')) for name in earlier if (ROOT/'evaluation'/task/f'{algo}_{name}.npz').exists()]
        oof_ids=np.concatenate([p['ids'] for p in parts]);oof=np.concatenate([p['risk'] for p in parts]);rows=s.iloc[oof_ids];ok=eligible(rows)
        cal=calibrate(oof[ok],rows.event_in_300s.to_numpy()[ok]);best,_=threshold_from_oof(rows,apply_calibration(cal,oof))
        recorded=reports[task]['folds'][BLIND_FOLD]['scores'][algo]['alarms']
        if abs(best['threshold']-recorded['threshold'])>1e-12:raise AssertionError(f'{task}: threshold differs from the recorded blind evaluation')
        head=source['aux']['time']
        eta={'kind':'constant','bin':int(head['constant'])} if isinstance(head,dict) else {'kind':'logistic','model':head,'classes':head.classes_.astype(int)}
        fit=reports[task]['folds'][BLIND_FOLD]['models'][algo]
        joblib.dump({'task':task,'algorithm':algo,**plain_transform(source['transform']),'model':source['model'],'calibration':cal,'threshold':float(best['threshold']),
                     'eta_head':eta,'trained_through':'2026-10-23','train_windows':fit['train_windows'],'train_events':fit['train_events'],
                     'sklearn':sklearn.__version__},OUT/'models'/f'{task}.joblib',compress=3)
        blind=reports[task]['folds'][BLIND_FOLD];score=blind['scores'][algo]
        contract_tasks[task]={'algorithm':algo,'threshold':float(best['threshold']),'calibration':cal['kind'],'threshold_from_folds':earlier,
            'train_windows':fit['train_windows'],'train_events':fit['train_events'],
            'blind_week':{'events':blind['validation_events'],'windows':blind['validation_windows'],'prevalence':score['ranking']['prevalence'],
                'average_precision':score['ranking']['average_precision'],'event_macro_ap':score['ranking']['event_macro_ap'],'roc_auc':score['ranking']['roc_auc'],
                'events_alarmed_in_advance':recorded['detected_events'],'false_alarms':recorded['false_alerts'],
                'false_alarms_per_100_hours':recorded['false_alerts_per_100_hours'],'observable_powered_hours':recorded['observable_powered_hours']}}
    life_name=protocol['life']['selected'].split(' ')[0]
    for task in ('hours','pieces'):
        source=joblib.load(ROOT/'models/life/life'/life_name/f'blind_new_week_{task}.joblib')
        if not source.get('stage_transform') or source['model']['distribution']!='lognormal':raise AssertionError('unexpected lifetime model')
        joblib.dump({'variant':life_name,'task':task,'beta':np.asarray(source['model']['beta'],dtype='float64'),'parameter':float(source['model']['parameter']),
                     'distribution':'lognormal',**plain_transform(source['transform']),'feature_indices':[int(i) for i in source['feature_indices']],
                     'feature_names':[names[i] for i in source['feature_indices']],'trained_through':'2026-10-23'},OUT/'models'/f'life_{task}.joblib',compress=3)
    # Replay rows: the unseen week exactly as the training grid saw it.
    t=np.load(GRID/'grid_t.npy',mmap_mode='r');x=np.load(GRID/'grid_x.npy',mmap_mode='r');seen=np.load(GRID/'grid_seen.npy',mmap_mode='r')
    block=np.load(GRID/'grid_block.npy',mmap_mode='r');part=np.load(GRID/'grid_part.npy',mmap_mode='r');rows=np.flatnonzero(np.asarray(part)==NEW_PARTITION)
    if not np.array_equal(rows,np.arange(rows[0],rows[-1]+1)):raise AssertionError('new partition is not one contiguous grid range')
    s=samples['voltage'].iloc[ids];ends=s.end_index.to_numpy();landmarks=sec(s.timestamp)
    np.save(OUT/'replay/raw_t.npy',np.asarray(t[rows],dtype='int64'));np.save(OUT/'replay/raw.npy',np.asarray(x[rows],dtype='float32'))
    np.save(OUT/'replay/raw_seen.npy',np.asarray(seen[rows],dtype=bool));np.save(OUT/'replay/raw_block.npy',np.asarray(block[rows],dtype='int64'))
    np.save(OUT/'replay/landmarks.npy',landmarks.astype('int64'));np.save(OUT/'replay/landmark_block.npy',np.asarray(block[ends],dtype='int64'))
    np.save(OUT/'replay/features.npy',np.asarray(features[ids],dtype='float32'))
    assert np.array_equal(np.asarray(t[ends]),landmarks) and np.all(np.diff(landmarks)>0)
    segments=[]
    for name,a,b,label in SEGMENTS:
        start,stop=int(pd.Timestamp(a).value//10**9),int(pd.Timestamp(b).value//10**9);inside=(landmarks>=start)&(landmarks<stop)
        count=int(((np.asarray(t[rows])>=start)&(np.asarray(t[rows])<stop)).sum())
        if count<600 or inside.sum()<30:raise AssertionError(f'segment {name} has too little recorded data: {count} rows, {int(inside.sum())} landmarks')
        segments.append({'name':name,'label':label,'start':start,'stop':stop,'rows':count,'landmarks':int(inside.sum())})
    # Manifest first, so the dashboard loader can open the directory for the parity run.
    dump({'version':'pending','field_keys':FIELDS,'field_names':{k:NAMES[k][0] for k in FIELDS},'field_units':{k:NAMES[k][1] for k in FIELDS},
          'feature_names':names,'segments':segments,'files':{}},OUT/'manifest.json')
    dump({'input_mode':'pending','replay_period':'pending','model_trained_through':'2026-10-23','model_name':'pending','limits':[],'unavailable_outputs':[]},OUT/'contract.json')
    panel=panel_real28.Real28Bundle(OUT);matrix=np.asarray(panel.features)
    for task in TYPES:
        algo=protocol['risk_selection'][task];saved=dict(np.load(ROOT/'evaluation'/task/f'{algo}_{BLIND_FOLD}.npz'))
        assert np.array_equal(saved['ids'],ids)
        raw,score,alarm,eta=panel.risk(task,matrix);policy=reports[task]['folds'][BLIND_FOLD]['scores'][algo]
        events=pd.read_parquet(ROOT/'data'/task/'events.parquet');result,_,_=alarms(samples[task].iloc[ids],score,panel.models[task]['threshold'],events)
        rank=ranking(samples[task].iloc[ids],score)
        checks[task]={'landmarks':len(ids),'max_abs_raw_risk_difference':float(np.abs(raw-saved['risk']).max()),'max_abs_eta_difference_s':float(np.abs(eta-saved['eta']).max()),
            'alarms_reproduced':{k:result[k]==policy['alarms'][k] for k in ['events','detected_events','alerts','false_alerts']},
            'average_precision_reproduced':rank['average_precision']==policy['ranking']['average_precision'] if rank['average_precision'] is not None else policy['ranking']['average_precision'] is None,
            'landmarks_above_threshold':int(alarm.sum())}
        for i,seg in enumerate(segments):
            inside=(landmarks>=seg['start'])&(landmarks<seg['stop']);rows_in=samples[task].iloc[ids[inside]]
            seg.setdefault('contents',{})[task]={'episodes_starting_within_300s_of_a_landmark':int(rows_in.loc[rows_in.event_in_300s,'event_id'].nunique()),
                'landmarks_above_threshold':int(alarm[inside].sum()),'max_score':float(score[inside].max())}
        if checks[task]['max_abs_raw_risk_difference']>1e-9 or not all(checks[task]['alarms_reproduced'].values()):raise AssertionError(f'{task}: dashboard maths differ from training: {checks[task]}')
    life=pd.read_parquet(ROOT/'data/life/samples.parquet');checks['life']={}
    for task in ('hours','pieces'):
        saved=dict(np.load(ROOT/'evaluation/life'/f'life_{life_name}_blind_new_week_{task}.npz'));position=np.searchsorted(ids,saved['ids'])
        assert np.array_equal(ids[position],saved['ids'])
        got=panel.lifetime(task,matrix[position]);scale=np.maximum(np.abs(saved['predictions']),1e-9)
        checks['life'][task]={'landmarks':len(position),'max_relative_difference':float((np.abs(got-saved['predictions'])/scale).max())}
        if checks['life'][task]['max_relative_difference']>1e-5:raise AssertionError(f'life {task}: dashboard maths differ from training')
    blind_life=reports['life']['folds']['blind_new_week']['methods'][life_name];constant=reports['life']['folds']['blind_new_week']['methods']['constant']
    def life_row(m):
        e=m['per_endpoint'][0] if m['per_endpoint'] else {}
        return {'weak_endpoints':m['weak_endpoints'],'windows':m['windows'],'median_to_weak_interval_hours':m['macro_interval_distance'],
                'near_15min_distance_hours':e.get('near_15min_mean_interval_distance'),'censored_windows_already_outlived_the_median_fraction':m['censor_median_below_observed_lower_fraction'],
                'median_prediction_quantiles_hours':m['median_prediction_quantiles']}
    contract={'model_name':'28 参数实测模型 v4（实验回放）','model_kind':'real28','run':protocol['run'],'production_ready':False,
        'input_mode':'recorded_replay','input_mode_text':'实测回放：重放 2026-10-27 至 2026-11-01 主设备记录的 28 项读数，不是实时设备数据，也不是代码生成的模拟输入',
        'replay_period':['2026-10-27','2026-11-01'],'model_trained_through':'2026-10-23',
        'why_this_fit':'回放使用训练截至 2026-10-23 的第 6 折模型，回放周对模型完全未见；截至 2026-11-01 的全量重拟合没有任何独立验证分数，不用于演示',
        'inference':'292 项因果特征由训练管线对每个 10 秒预测点离线计算并随包提供；看板在请求时用导出的模型现算分数与分位数，不在线重算特征',
        'prediction_interval_seconds':10,'causal_history':'120 秒序列历史，统计特征最长回看 2 小时并受 3 小时来源上限约束',
        'tasks':contract_tasks,'alarm_policy':protocol['alarm_policy'],
        'risk_calibration':'分数经早期折外预测的 sigmoid 校准（送料为恒等），没有独立校准集；三类分数相互独立，不能相加当作总异常概率',
        'lifetime':{'variant':life_name,'supervision':protocol['label_rules']['natural_failure_supervision'],
            'blind_week_selected':life_row(blind_life['hours']['metrics']),'blind_week_constant_baseline':life_row(constant['hours']['metrics']),
            'interval_note':'10%–90% 为模型分位，未经独立校准，不是寿命置信区间'},
        'limits':['寿命估计来自低压—停机—恢复的弱标签，没有逐次确认的自然失效；回放周内约 92% 的未失效窗口其中位估计已被实际继续运行时间超过，系统性偏低，不能作为换枪依据',
            f"电压预警在回放周 {contract_tasks['voltage']['blind_week']['events']} 个事件中提前命中 {contract_tasks['voltage']['blind_week']['events_alarmed_in_advance']} 个、误报 {contract_tasks['voltage']['blind_week']['false_alarms']} 次；一周单设备，样本量不足以给出统计结论",
            f"送料预警在回放周 {contract_tasks['feedrate']['blind_week']['events']} 个事件中提前命中 {contract_tasks['feedrate']['blind_week']['events_alarmed_in_advance']} 个，排序接近随机，目前没有可用预警能力",
            '料斗压力在回放周没有任何事件，召回能力无法评价；无误报不代表有效',
            '报警阈值按每 100 带电小时不超过 20 次误报的实验预算从早期折外预测选取，不是现场确认的容忍值',
            '回放周的新增标签完全由自动规则生成，没有公司复核'],
        'unavailable_outputs':[
            {'field':'gun_health','name':'喷枪健康度','reason':'没有健康度监督标签，未训练'},
            {'field':'tank_health','name':'料斗健康度','reason':'没有健康度监督标签，未训练'},
            {'field':'pressure_60s','name':'60 秒后料斗压力','reason':'该预测目标已按要求取消，不训练、不输出'},
            {'field':'pressure_trend','name':'压力趋势','reason':'随压力预测一并取消'},
            {'field':'anomaly_probability','name':'当前异常概率','reason':'没有独立的瞬时异常真值；当前状态显示最近完成产品的工艺规则判定'},
            {'field':'root_candidates','name':'原因候选','reason':'未实现参数贡献分析'},
            {'field':'neural','name':'神经元激活','reason':'所选模型为集成树与 AFT，不是神经网络'},
            {'field':'interval_nominal','name':'寿命区间覆盖率','reason':'分位区间未经校准，没有名义覆盖率'}],
        'parity_with_training':checks,'sklearn':sklearn.__version__,'numpy':np.__version__}
    dump(contract,OUT/'contract.json')
    files={p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    version='real28-v4-'+hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:12]
    manifest=load(OUT/'manifest.json');manifest.update(version=version,files=files,segments=segments,created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        protocol_sha256=sha(ROOT/'protocol.json'),landmarks=len(ids),raw_rows=int(len(rows)),replay_first=str(pd.Timestamp(int(t[rows[0]]),unit='s')),replay_last=str(pd.Timestamp(int(t[rows[-1]]),unit='s')),
        panel_adapter_sha256=sha(Path(__file__).with_name('panel_real28.py')))
    dump(manifest,OUT/'manifest.json')
    final=panel_real28.Real28Bundle(OUT);final.verify()
    print(json.dumps({'version':version,'files':len(files),'bytes':sum((OUT/n).stat().st_size for n in files),'checks':checks,'segments':segments},ensure_ascii=False,indent=1,default=str))

if __name__=='__main__':main()
