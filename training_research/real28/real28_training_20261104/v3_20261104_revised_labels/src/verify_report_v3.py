"""Verify real artifacts and isolation, then write the complete experiment report."""
import sys
sys.dont_write_bytecode=True
import torch
import time,json,os
import numpy as np
import pandas as pd
from common_v2 import ROOT,SHARED,V1,FIELDS,TYPES,load,dump,sha
from evaluate_v3 import RISK,freeze
from train_stage_life import VARIANTS

LABELS={'logistic':'逻辑回归','histgb':'直方图梯度提升树','extra_trees':'极随机树','random_forest':'随机森林','tcn':'TCN','gru':'GRU','patch_transformer':'Patch Transformer','tcn_ssl':'TCN＋自监督预训练',
        'constant':'常数分布基准','lognormal_aft':'对数正态 AFT','weibull_aft':'Weibull AFT','spline_aft':'样条 AFT','mlp_aft':'MLP AFT','logistic_hazard':'逻辑离散风险模型','boosted_hazard':'梯度提升风险模型','forest_hazard':'森林风险模型'}
for v in VARIANTS:LABELS[v]=f'4特征约束{v.split("_")[1]} AFT / L2={ {"001":.01,"01":.1,"1":1.}[v.split("_r")[1]] }'
def fmt(x,n=3):return '无可评价事件' if x is None else f'{x:.{n}f}'
def main():
    frozen=freeze();checks=[];fit_count=0;ssl_count=0;gpu=[]
    for task,names in RISK.items():
        report=load(ROOT/'evaluation'/task/'comparison.json')
        assert report['freeze_sha256']==sha(ROOT/'development_freeze.json')
        samples=pd.read_parquet(SHARED/task/'samples.parquet')
        for name in names:
            base=ROOT/'models'/task/'risk'/name;summary=load(base/'complete.json')
            assert summary['status']=='trained' and summary['history_period_accessed'] is False
            for label in ['f1','f2','f3','final']:
                fit=load(base/label/'fit.json');assert fit['train_windows']>0 and fit['evaluation_windows']>0
                assert sha(base/label/('model.pt' if name in ['tcn','gru','patch_transformer','tcn_ssl'] else 'model.joblib'))==fit['model_sha256']
                fit_count+=1
                if name in ['tcn','gru','patch_transformer','tcn_ssl']:
                    h=load(base/label/'epochs.json');assert len(h)==6 and all(np.isfinite(x['loss']) for x in h)
                    gpu.extend(h)
                if name=='tcn_ssl':
                    h=load(base/label/'self_supervised_epochs.json');assert len(h)==3 and h[-1]['training_windows_only']
                    assert (base/label/'self_supervised_encoder.pt').exists();ssl_count+=1
            chosen=max([load(ROOT/'models'/task/'risk'/n/'complete.json') for n in names],key=lambda x:(x['mean_fold_event_macro_ap'],x['pooled_oof']['average_precision']))['name']
            assert chosen==frozen['risk_choices'][task]
            for part in ['development','calibration_diagnostic','historical']:
                a=np.load(ROOT/'evaluation'/task/f'{name}_{part}.npz');ids=a['ids'];rows=samples.iloc[ids]
                assert np.isfinite(a['risk']).all() and np.isfinite(a['calibrated_risk']).all()
                assert np.all((a['risk']>=0)&(a['risk']<=1))
                assert np.all((a['calibrated_risk']>=0)&(a['calibrated_risk']<=1))
                np.testing.assert_allclose(a['time'].sum(1),1,atol=2e-6)
                assert np.all((a['eta']>=14.99)&(a['eta']<=285.01))
                events=report['algorithms'][name][part]['event_records']
                assert all(r['lead_to_product_start_s'] is None or r['lead_to_product_start_s']>0 for r in events)
                assert report['algorithms'][name][part]['alarms']['detected_events']<=report['algorithms'][name][part]['alarms']['events']
        checks.append(f'{task}: all actual fold/final models, finite probabilities, normalized conditional time distributions, pre-product event leads and training-only selection')
    life=load(ROOT/'evaluation/life/comparison.json')
    for v in frozen['life_summaries']:
        assert len(v['fits'])==6
        for fit in v['fits']:
            model=ROOT/'models/life/life'/v['name']/f'{fit["fit"]}_{fit["task"]}.joblib';assert model.exists();fit_count+=1
        for label in ['development','historical']:
            for task in ['hours','pieces']:
                a=np.load(ROOT/'evaluation/life'/f'{v["name"]}_{label}_{task}.npz');p=a['predictions']
                assert np.isfinite(p).all() and (p>=0).all() and (np.diff(p,axis=1)>=0).all()
                assert life['algorithms'][v['name']][f'{label}_{task}']['confirmed_natural_failure_error'] is None
    assert fit_count==140 and ssl_count==4
    checks.append('140 actual main fits plus 4 training-only self-supervised pretraining fits; lifetime quantiles finite and ordered')
    old=load(ROOT/'previous_artifacts_sha256.json')
    for path,digest in old.items():assert sha(path)==digest,path
    assert sha(SHARED/'features.npy')==frozen['features_sha256']
    assert sha(ROOT/'protocol.json')==frozen['protocol_sha256']
    for name,digest in frozen['training_source_sha256'].items():assert sha(ROOT/'src'/name)==digest,name
    if 'data_sha256' in frozen:
        for path,digest in frozen['data_sha256'].items():assert sha(ROOT/path)==digest,path
    else:
        # The evaluator was already imported when the extra label-manifest
        # field was added. Preserve that original freeze instead of backdating
        # or replacing it after history access.
        dump({'recorded_after_model_freeze':True,'freeze_not_modified':True,
              'data_sha256':{str(p.relative_to(ROOT)):sha(p) for p in SHARED.rglob('*') if p.is_file() and p.suffix in ['.parquet','.npz','.csv','.json']}},ROOT/'evaluation/input_data_hashes_after_training.json')
    checks.append(f'{len(old)} old source/model/report/provenance artifacts unchanged by SHA-256')
    assert load(ROOT/'evaluation/episode_policy_regression.json')['passed']
    checks.append('Six synthetic ongoing-episode and persistent-alarm regression checks passed')
    repair=load(ROOT/'evaluation/conditional_probability_repair.json');assert repair['regression_passed'] and not repair['risk_values_changed'] and not repair['model_weights_changed']
    checks.append('Subnormal conditional time probabilities normalized; risks, weights, frozen choice unchanged')
    latest=load(ROOT/'all_data_refits.json') if (ROOT/'all_data_refits.json').exists() else None
    extra_fits=latest['main_fits'] if latest else 0
    if latest:
        smoke=load(ROOT/'evaluation/full_data_prediction_smoke.json');assert smoke['passed'] and smoke['refit_manifest_sha256']==sha(ROOT/'all_data_refits.json')
        for task,v in latest['fits'].items():
            if v['status']=='skipped':continue
            assert v['validation_metrics'] is None and v['previous_validation_scores_do_not_apply_to_this_refit']
            assert v['last_timestamp'].startswith('2026-10-23')
            for name,digest in v['artifact_sha256'].items():assert sha(ROOT/'models/full_data'/task/name)==digest
        checks.append(f'{extra_fits} separate all-data refits through October23 verified; no historical validation claimed for these fits')
    p=pd.read_parquet(V1/'products.parquet');terminal=pd.read_parquet(SHARED/'natural_endpoint_evidence.parquet');s=pd.read_parquet(SHARED/'life/samples.parquet')
    sources=pd.read_parquet(V1/'sources.parquet');assert len(sources)==1505 and int(sources.rows.sum())==5402169
    assert not sources.path.str.contains('Voltage Sample',regex=False).any()
    checks.append('All 1505 main-device source files accounted for; other-device Voltage Sample absent')
    counts=s.loc[s.rul_valid&s.rul_event].groupby(['partition','rul_endpoint_id']).size().to_dict()
    # Boundary sensitivity is reported, never used to silently relabel products.
    boundary={}
    for task,column,limits,tol in [('feedrate','feedrate_mean',[9.1,11.1],.1),('hopper_pressure','pressure_mean',[100,250],.5)]:
        ev=pd.read_parquet(SHARED/task/'events.parquet');a=ev.merge(p[['product_id',column]],on='product_id',validate='one_to_one')
        a['near_boundary']=np.min(np.abs(a[column].to_numpy()[:,None]-np.array(limits)),axis=1)<=tol
        a.to_csv(ROOT/'evaluation'/f'{task}_boundary_sensitivity.csv',index=False,encoding='utf-8-sig')
        boundary[task]={'episodes':len(a),'first_failed_products_near_HMI_boundary':int(a.near_boundary.sum()),'tolerance':tol,'original_labels_preserved':True}
    audit={'passed':True,'checks':checks,'comparison_main_fits':fit_count,'all_data_main_fits':extra_fits,'main_fits':fit_count+extra_fits,'ssl_pretraining_fits':ssl_count,'distinct_risk_algorithms':7,'risk_initialization_variants':8,'base_life_model_methods':7,'learned_life_variants':13,
           'gpu_max_recorded_temperature_c':max(x['temperature_c'] for x in gpu),'gpu_max_tensor_mib':max(x['gpu_tensor_peak_mib'] for x in gpu),
           'original_artifacts_verified':len(old),'boundary_sensitivity':boundary,'verified_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    dump(audit,ROOT/'evaluation/verification_final.json')
    contract={'raw_parameters':FIELDS,'raw_parameter_count':28,'features':load(SHARED/'ready.json')['features'],'risk_models_by_type':frozen['risk_choices'],
          'risk_horizon_seconds':300,'risk_event_time':'abnormal product START','risk_tasks_are_independent':True,'probability_of_any_type':None,
          'conditional_eta':'per subtype, conditioned on an event within 300s; not a guaranteed countdown',
          'life_method':frozen['life_choice'],'natural_failure_confirmed':0,'production_remaining_hours':None,'production_remaining_pieces':None,
          'health_percent':None,'pressure_forecast':False,'automatic_replacement':False,'production_ready':False,'dashboard_modified':False,
          'research_life_quantiles':'weak-proxy model distribution quantiles, not calibrated natural-life confidence intervals'}
    dump(contract,ROOT/'models/output_contract.json')
    lines=['# 第三轮本机训练结果','',f'完成并核验：{audit["verified_at"]}。全部产物位于本独立目录；原始数据、前两轮模型、看板和服务器未修改。','',
        '## 可直接使用的结论','',
        f'已完成 **140 次模型比较拟合**，另做 **4 次自监督预训练**。电压预警比较7种算法、8个实现对照；送料和料斗压力各比较3种；寿命比较7种建模方法、13个实现/正则设置及常数基准。同家族变体不冒充不同算法。',
        '模型已训练不等于已证明能准确预测自然失效时间。本轮自然失效监督仍是低电压—停机—恢复的规则推定，个体确认自然失效为0；生产用剩余小时、件数和健康百分比仍不输出。',
        f'前向验证选择的寿命候选：**{LABELS.get(frozen["life_choice"],"无，未通过常数基准及外推检查")}**。这项选择只说明是否通过本轮弱标签实验门槛，不构成上线或精确寿命验证。','',
        '## 修订了什么','',
        '- 主设备1,505个CSV、5,402,169行、28参数；Voltage Sample不参与训练、标准化、校准和评估。',
        '- 单件异常沿用已有规则与公司明确纠正：喷涂均压<51 V、送料均值不在9.1—11.1 g/min、料斗均压不在100—250 mbar。公司主设备同批连续两件异常触发停机，第三件可能已开始；本轮预测第一件异常开始，不等到两件失败才算提前预警。',
        '- R009按公司说明记为普通等待，移除其疑似换枪删失边界。其他未确认的提前干预仍为右删失，不当自然失效。',
        '- R011有持续低压及停机恢复证据，保留为弱终点；取消按不足10小时一刀切排除。未声称它是已确认自然失效或真实新枪寿命。',
        '- R007把相关首次低压和最终连续低压之间保留为终点不确定区间，防止把中途一件边缘恢复切成无监督尾段。',
        '- 2小时因果历史跨普通等待保留；数据大空白仍截断，不补造真实枪龄。增加同工况相对电压峰值、下降量、波动和上升/下降阶段特征，共292项。',
        '- 电压、送料、压力分别建模。两件已观察正常产品才结束同类异常 episode，单纯过夜/断电不产生一次新失效。',
        '- 预警目标改为第一件异常产品开始前300秒；已开始加工后才报警不算成功提前预测。报警须连续两个10秒采样点超阈值，并有300秒冷却。',
        '- 线性/样条AFT使用系数平方和正则化，降低少数终点下的高维过拟合；仍完整报告外推异常，没有用截断预测美化误差。','',
        '## 验证与模型选择','',
        '日期隔离的比较模型训练截至9月29日；比较结束后另存的全量候选使用截至10月23日的标签，见下文。异常任务采用9月1、8、15日为切点的3折时间前向验证；训练结果可用时间保留3,900秒，验证特征历史隔离10,800秒。标准化、缺失补值、自监督预训练、条件时间头全部只拟合各折训练部分。',
        '寿命用可能周期整体隔离的两次前向验证；没有被确认的真实电极身份。选择先看弱终点对数区间距离，同时必须在临近终点距离、优化收敛和外推检查上通过常数基准门槛。',
        '10月4—7日为开发诊断，8—9日为校准诊断。10月10—23日前两轮已看过、也用于这次标签复核，**仅为已见历史诊断，不是新盲测**。模型和阈值先冻结，再读取本轮历史预测。',
        '9月前向折也用于研究和模型选择，因此是开发评估。本轮没有新增独立盲测数据，不能把所选模型的分数当作生产泛化保证。',
        '报警阈值按9月折外预测选择，以最多20次误报/100有效带电小时为实验预算；不是用户确认的生产容忍值，也不保证后续时段满足。',
        'AP表示排序表现，正例稀少时不能把它当准确率。表中历史召回只统计异常产品开始前成功报警的事件；不同类型概率独立，不能直接相加当总异常概率。','']
    for task,title in [('voltage','电压'),('feedrate','送料'),('hopper_pressure','料斗压力')]:
        rep=load(ROOT/'evaluation'/task/'comparison.json');chosen=frozen['risk_choices'][task]
        lines += [f'## {title}预警结果','',f'按9月前向验证选择：**{LABELS[chosen]}**。','',
                 '| 算法 | 9月折均事件AP | 10月10—23日提前命中/事件 | 历史误报次数 | 误报/100h | 历史窗口AP |',
                 '|---|---:|---:|---:|---:|---:|']
        for summary in frozen['risk_summaries'][task]:
            name=summary['name'];d=rep['algorithms'][name]['historical'];a=d['alarms'];mark=' **所选**' if name==chosen else ''
            lines.append(f'| {LABELS[name]}{mark} | {fmt(summary["mean_fold_event_macro_ap"],5)} | {a["detected_events"]}/{a["events"]} | {a["false_alerts"]} | {fmt(a["false_alerts_per_100_hours"],2)} | {fmt(d["ranking"]["average_precision"],5)} |')
        selected=rep['algorithms'][chosen]['historical'];a=selected['alarms'];records=selected['event_records'];hit=[r for r in records if r['alert_at'] is not None]
        lines+=['',f'所选模型历史有效观察 {a["observable_powered_hours"]:.2f} 小时，提前命中 {a["detected_events"]}/{a["events"]}，误报 {a["false_alerts"]} 次。']
        if hit:
            leads=[r['lead_to_product_start_s'] for r in hit];lines += [f'命中事件对第一件异常产品开始的提前量：{", ".join(str(int(x))+"秒" for x in leads)}。']
        if a['events']==0:lines+=['该历史时段没有此类事件，无法验证召回能力；不能因为没有漏报就认为效果好。']
        eta=selected.get('eta_mae_s_all_positive_windows')
        lines += [(f'条件ETA在所有未来确有事件窗口上的MAE：{eta:.1f} 秒；包含未报警窗口，不代表实际倒计时已准确。' if eta is not None else '该时段没有此类正例，未计算ETA误差。'),'']
    lines += ['## 剩余带电时间与件数','',
        '下表衡量预测中位数到弱标签区间的距离。它不是公司确认的自然失效误差；数值越小越好。寿命分支均训练小时、件数两个目标。','',
        '件数为观察到的完整加工产品数，不是保证可生产的合格件数。','',
        '| 方法 | 前向小时区间距离 h | 前向对数距离 | 前向临近15分钟距离 h | 10月历史小时距离 h | 历史件数距离 |',
        '|---|---:|---:|---:|---:|---:|']
    for v in frozen['life_summaries']:
        near=[e['near_15min_mean_interval_distance'] for f in v['fits'] if f['fit']!='final' and f['task']=='hours' for e in f['metrics']['per_endpoint'] if e['near_15min_mean_interval_distance'] is not None]
        h=life['algorithms'][v['name']]['historical_hours'];pieces=life['algorithms'][v['name']]['historical_pieces']
        lines.append(f'| {LABELS[v["name"]]} | {fmt(v["forward_cycle_macro_hour_distance"])} | {fmt(v["forward_cycle_macro_log_distance"])} | {fmt(float(np.mean(near)))} | {fmt(h["macro_interval_distance"])} | {fmt(pieces["macro_interval_distance"],1)} |')
    lines += ['', '分布的10%/50%/90%分位数不是已校准的自然寿命置信区间。某些风险模型尾部外推很大，保留在原始结果中并通过门槛排除，不能拿来给出精确换枪倒计时。','',
              'AFT使用区间/右删失似然；离散风险方法对跨多个时间箱的宽终点区间仅使用已存活下界，因此这类弱事件的有效监督更少。各模型保存了此项计数。','',
              '## 标签支持数量','', '| 分区 | 弱终点 | 有效事件窗口 |', '|---|---|---:|']
    for (part,endpoint),n in counts.items():lines.append(f'| {part} | {endpoint} | {n} |')
    lines += ['', '0为9月训练，1为10月4—7日开发诊断，3为10月10—23日已见历史。共有6个弱终点，事件窗口虽多但高度重叠，不能当成数万个独立电极失效样本。完整终点区间见 data/natural_endpoint_evidence.csv。']
    if latest:
        lines += ['', '## 截至10月23日的全量再拟合候选','',
                  f'模型和比较结果冻结后，另做{extra_fits}次主拟合；本轮合计{fit_count+extra_fits}次主拟合，另有4次自监督预训练。候选保存在 models/full_data。上述所有验证数字对应日期隔离的比较模型，**不适用于这些已学习全量数据的候选**。',
                  '风险再拟合输出是未重新验证校准的分数；寿命仍是弱终点模型。它们为后续新数据观察准备，不自动接入生产看板。','',
                  '| 任务 | 所选算法 | 全量训练事件/弱终点 | 拟合窗口 |', '|---|---|---:|---:|']
        for task,v in latest['fits'].items():
            if v['status']=='skipped':lines.append(f'| {task} | 未通过开发门槛，跳过 | — | — |')
            else:lines.append(f'| {task} | {LABELS[v["algorithm"]]} | {v["event_or_weak_endpoint_count"]} | {v["train_windows"]} |')
    lines += ['', '## 为什么仍有局限','',
        '自然失效与提前人为换枪没有逐次确认；日期隔离的寿命比较模型仅用3个弱终点训练、2个周期做前向评估，全量再拟合候选才使用全部6个弱终点。首折电压预警甚至只有1个训练事件。主设备临近故障有的在几分钟内骤降，有的提前出现间歇低压，无法靠统一斜率推算数小时。',
        '送料和压力有独立工艺原因，HMI每秒采样与PLC更快采样在边界附近可能不同。本轮保留既有公司纠正与未知标记，不把其他边界产品擅自改成正常。边界敏感性清单单独保存。',
        '本轮与前两轮的目标时间、事件合并方式、分类型任务、报警策略和标签都发生变化，分数不能直接当同一基准下的改善幅度。已见历史上表现最好的算法没有反过来取代前向验证选择。',
        '这些结果用于离线研究；本轮未修改看板、未上传服务器、未启用自动换枪或健康百分比，也没有训练已取消的未来压力预测。','',
        '## 本机执行与核验','',
        f'本机 Core Ultra 9 275HX、约32GB内存、RTX 5070 Ti Laptop 12GB。CPU任务限4线程（寿命2线程），神经网络用本机CUDA、TCN/Transformer用BF16、GRU用FP32。GPU进程分配上限为显存50%。',
        f'监督训练记录的最高GPU温度 {audit["gpu_max_recorded_temperature_c"]}°C，峰值PyTorch张量显存 {audit["gpu_max_tensor_mib"]:.1f} MiB；这不是整机全部显存占用。83°C暂停，87°C终止。',
        f'已核验140次主拟合、4次自监督预训练、全部模型文件和预测分布，以及{len(old)}个既有源码/模型/报告/原始缓存的SHA-256未改变。详见 evaluation/verification_final.json。','',
        '最后核验发现极低风险（约10^-39）下的条件时间分布受float32下溢影响，加总可能不足1。已在实际推理边界加入归一化保护，修复4个预测档案，原档另存；权重、风险分数、报警和冻结选择均不变。见 evaluation/conditional_probability_repair.json。','',
        '## 文件与复现','',
        '- protocol.json：预先声明的标签、切分、选择与资源约束。',
        '- development_freeze.json：读取本轮历史预测前冻结的选择和模型哈希。',
        '- data/：修订证据、因果特征、各任务标签；raw_values等大数组只读复用前轮缓存。',
        '- models/<任务>/：每种算法及每个前向折/最终拟合的产物；life/life下保存寿命方法。',
        '- evaluation/<任务>/comparison.json：全部算法、开发/校准/历史指标；事件与报警CSV含具体时间。',
        '- logs/：实际训练过程；training_processes.json记录首批进程退出状态。',
        '- models/output_contract.json：研究输出约定，production_ready=false。','',
        '```powershell',
        '# 原目录已存在时训练脚本会复用完成检查点；请保留本轮结果，不重建覆盖。',
        'python -B real28_training_20261104/v3_20261104_revised_labels/src/run_v3.py',
        '$env:TRAIN_TASK="life"',
        'python -B real28_training_20261104/v3_20261104_revised_labels/src/train_stage_life.py',
        '$env:TRAIN_TASK="voltage"',
        'python -B real28_training_20261104/v3_20261104_revised_labels/src/train_ssl.py',
        'foreach ($taskName in @("voltage","feedrate","hopper_pressure","life")) {',
        '  $env:TRAIN_TASK=$taskName',
        '  python -B real28_training_20261104/v3_20261104_revised_labels/src/evaluate_v3.py',
        '}',
        'python -B real28_training_20261104/v3_20261104_revised_labels/src/probability_guard.py',
        'python -B real28_training_20261104/v3_20261104_revised_labels/src/verify_report_v3.py',
        '# 比较核验完成后，另存截至10月23日的全量候选，再次核验报告。',
        'python -B real28_training_20261104/v3_20261104_revised_labels/src/refit_latest.py',
        'python -B real28_training_20261104/v3_20261104_revised_labels/src/verify_full_data.py',
        'python -B real28_training_20261104/v3_20261104_revised_labels/src/verify_report_v3.py',
        'Remove-Item Env:TRAIN_TASK',
        '```','']
    (ROOT/'训练结果.md').write_text('\n'.join(lines),encoding='utf-8')
    (ROOT/'README.md').write_text(f'# 第三轮修订标签后的多算法训练\n\n已训练并核验。完整结果见 [训练结果.md](训练结果.md)。\n\n140次比较拟合、{extra_fits}次全量再拟合、4次自监督预训练；7种风险算法、8个电压实现对照；7种寿命建模方法及13个实现设置和常数基准。原始数据、旧模型、看板和服务器未修改。当前产物为离线实验，不能据此保证准确自然寿命。\n',encoding='utf-8')
    dump({'stage':'complete','trained_main_fits':fit_count+extra_fits,'comparison_main_fits':140,'all_data_main_fits':extra_fits,'self_supervised_fits':4,'verification_passed':True,'production_ready':False,'completed_at':audit['verified_at']},ROOT/'execution_status.json')
    print(json.dumps(audit,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
