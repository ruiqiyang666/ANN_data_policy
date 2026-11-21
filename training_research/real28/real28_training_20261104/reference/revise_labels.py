"""2026-11-02 revision: process labels and retrospective replacement candidates.

No forecasting model is trained. A replacement candidate is a review window,
not a confirmed work order or an exact replacement timestamp.
"""
from pathlib import Path
import argparse, hashlib, json, shutil
import numpy as np
import pandas as pd
import label_ann as core

FMT = core.TIME_FORMAT
VOLTAGE_CAUSE = '自然电极寿命终结（现场工艺说明支持）'
FEED_CAUSE = '活动料斗送料均值超标'
PRESSURE_CAUSE = '活动料斗压力均值超标'
ACTION_NAMES = {
    'after_voltage_failure': '低压后换新（按现场规则推定，时间待核实）',
    'preventive_candidate': '电压正常时疑似提前换新（待核实）',
    'post_failure_recovery': '低压后恢复／疑似维护（停机原因待核实）',
}


def load(path, dates=()):
    d = pd.read_csv(path)
    for col in dates:
        if col in d: d[col] = pd.to_datetime(d[col], format=FMT, errors='coerce')
    return d


def classify_product(row):
    if row['label_valid'] != 1:
        return ('采样覆盖不足，暂不能判定', '采样完整性未达到暂定98%门槛；不是已确认的数值异常，不自动补点或改为正常。', 'unknown')
    if row['failure'] == 0:
        return ('正常加工', '三个产品平均值均未触发工艺失败条件。', 'not_applicable')
    labels, reasons = [], []
    if row['voltage_fail'] == 1:
        labels.append(VOLTAGE_CAUSE)
        reasons.append('现场已澄清有效喷涂产品的低电压属于电极自然寿命终结；人工提前换新不会造成电压提前降低。此标签不再受DTW形态相似度或是否在启动期的限制。')
    if row['feedrate_fail'] == 1:
        labels.append(FEED_CAUSE)
        reasons.append('活动料斗的产品平均送料超出9.1—11.1 g/min；触发指标已明确，粉末、管路或料斗等物理根因尚未逐件确认。')
    if row['hopper_pressure_fail'] == 1:
        labels.append(PRESSURE_CAUSE)
        reasons.append('活动料斗的产品平均压力超出100—250 mbar；触发指标已明确，具体物理根因尚未逐件确认。')
    level = 'site_process_rule' if row['voltage_fail'] == 1 else 'process_threshold_only'
    return ('；'.join(labels), ' '.join(reasons), level)


def derive_shutdown(p, max_gap=1800):
    p = p.sort_values(['start','station']).reset_index(drop=True).copy()
    p['derived_shutdown_trigger'] = pd.Series([0 if x == 1 else pd.NA for x in p.label_valid], dtype='Int64')
    p['consecutive_failure_count'] = 0
    run, previous = 0, None
    for ix, r in p.iterrows():
        if r.label_valid == 1 and r.failure == 1:
            run = run + 1 if previous is not None and (r.start-previous).total_seconds() <= max_gap else 1
            p.loc[ix,'consecutive_failure_count'] = run
            if run == 2: p.loc[ix,'derived_shutdown_trigger'] = 1
            previous = r.end
        else: run, previous = 0, None
    return p


def failure_events(p, cfg):
    groups = []
    for r in p[p.failure.eq(1)].to_dict('records'):
        if groups and (r['start']-groups[-1][-1]['end']).total_seconds() <= cfg['episode_join_gap_seconds'] and r['power_episode_id'] == groups[-1][-1]['power_episode_id']:
            groups[-1].append(r)
        else: groups.append([r])
    events = []
    for members in groups:
        first, last = members[0], members[-1]
        causes = list(dict.fromkeys(x['origin_label'] for x in members))
        events.append(dict(event_id=f'E{len(events)+1:03d}', event_kind='product_failure',
            start=first['start'], end=last['end'], event_time=first['end'],
            label_available_at=max(x['label_available_at'] for x in members), failure=1,
            failure_type='|'.join(k for k in ['voltage','feedrate','hopper_pressure'] if any(k in x['failure_type'].split('|') for x in members)),
            origin_label='；'.join(causes), origin_evidence='；'.join(dict.fromkeys(x['origin_evidence'] for x in members)),
            product_ids=';'.join(x['product_id'] for x in members), failed_product_count=len(members),
            voltage_failed_products=sum(x['voltage_fail']==1 for x in members), cause_confirmed=False,
            cause_basis='site_process_rule' if any(x['voltage_fail']==1 for x in members) else 'process_threshold_only',
            confirmed_action='', confirmed_action_time='', review_notes='', label_uses_future_context=True))
    return events


def assess_window(pre, post, cfg, basic_ok=True, immediate=True):
    """Explicit screening rules, deliberately separate from failure labels."""
    if not basic_ok or len(pre) < 2 or len(post) < 3:
        return 'not_candidate', '边界、时长或前后产品证据不足'
    if not pre.label_valid.eq(1).all() or not post.label_valid.eq(1).all():
        return 'not_candidate', '前后近邻产品含采样不足，不能跳过未知产品制造连续证据'
    if not post.voltage_fail.eq(0).all():
        return 'not_candidate', '复机后近邻产品仍有低压，未形成稳定的新电压基线'
    similar = (abs(post.current_median.median()-pre.current_median.median()) <= cfg['recipe_current_change_a'] and
        abs(post.argon_median.median()-pre.argon_median.median()) <= cfg['recipe_argon_change_nlpm'] and
        abs(post.nitrogen_median.median()-pre.nitrogen_median.median()) <= cfg['recipe_nitrogen_change_nlpm'])
    if not similar: return 'not_candidate', '电流或气体工况改变，不能将电压差直接解释为换新'
    gain = float(post.voltage_mean.median()-pre.voltage_mean.median())
    if immediate and pre.tail(2).voltage_fail.eq(1).all():
        return 'after_voltage_failure', '紧邻停机前两件电压失败、观察到停电—复机，复机后同工况产品电压正常；依现场说明推定低压后换新，不等于取得逐次工单'
    if pre.voltage_fail.eq(1).any() and gain >= cfg['replacement_post_failure_gain_v']:
        return 'post_failure_recovery', '停前近邻产品出现过低压、停后电压回升；最后一件或停机时机未满足直接推定条件，保留维护候选'
    if len(pre) >= cfg['replacement_minimum_healthy_products'] and pre.voltage_fail.eq(0).all() and gain >= cfg['replacement_preventive_gain_v']:
        return 'preventive_candidate', '停前电压仍正常、停后同工况电压基线上移；仅提示可能提前换新，普通冷／热机恢复等也可能产生此变化'
    return 'not_candidate', '普通停复机；未达到换新筛选条件，不表示可以排除实际更换'


def replacement_windows(p, state, cfg):
    audit, candidates = [], []
    s = state.sort_values('start').reset_index(drop=True)
    begin, end = pd.Timestamp(cfg['analysis_start']), pd.Timestamp(cfg['analysis_end_exclusive'])
    for ix, r in s[(~s.powered) & s.start.ge(begin) & s.end.lt(end)].iterrows():
        pre = p[p.end.lt(r.start) & p.end.ge(r.start-pd.Timedelta(minutes=cfg['replacement_context_minutes']))].tail(cfg['replacement_context_products'])
        post = p[p.start.gt(r.end) & p.start.le(r.end+pd.Timedelta(minutes=cfg['replacement_context_minutes']))].head(cfg['replacement_context_products'])
        adjacent = bool(ix>0 and ix+1<len(s) and s.iloc[ix-1].powered and s.iloc[ix+1].powered and
            (r.start-s.iloc[ix-1].end).total_seconds()<=cfg['maximum_sampling_gap_seconds'] and
            (s.iloc[ix+1].start-r.end).total_seconds()<=cfg['maximum_sampling_gap_seconds'])
        overlap = int((p.start.le(r.end)&p.end.ge(r.start)).sum())
        duration_ok = cfg['replacement_off_min_seconds'] <= r.duration_seconds <= cfg['replacement_off_max_seconds']
        immediate = bool(len(pre) and (r.start-pre.end.iloc[-1]).total_seconds()<=cfg['replacement_failure_stop_max_seconds'])
        kind, reason = assess_window(pre, post, cfg, adjacent and overlap==0 and duration_ok, immediate)
        if not adjacent: reason='停电／复机边界未被连续采样覆盖，不能用日志断档推定换枪'
        elif overlap: reason='停电区间与候选产品区间重叠，需先核对工况与产品边界'
        elif not duration_ok: reason='停电时长不在本版筛选范围内；过短或长停机均保留在审查表，不确认为换新'
        elif len(pre)<2 or len(post)<3: reason='一小时内前后产品不足；缺少足够可比较的生产电压'
        def median(frame, col): return float(frame[col].median()) if len(frame) else None
        before, after = median(pre,'voltage_mean'), median(post,'voltage_mean')
        entry = dict(window_id=f'W{len(audit)+1:03d}', candidate_id='', candidate_kind=kind,
            start=r.start, end=r.end, resume_at=r.end+pd.Timedelta(seconds=1),
            duration_seconds=float(r.duration_seconds), pre_count=len(pre), post_count=len(post),
            before_voltage_median=before, after_voltage_median=after,
            voltage_gain_v=after-before if before is not None and after is not None else None,
            before_voltage_min=float(pre.voltage_mean.min()) if len(pre) else None,
            after_voltage_min=float(post.voltage_mean.min()) if len(post) else None,
            voltage_low_before=bool(pre.voltage_fail.eq(1).any()),
            trailing_two_voltage_failures=bool(len(pre)>=2 and pre.tail(2).label_valid.eq(1).all() and pre.tail(2).voltage_fail.eq(1).all()),
            adjacent_power_boundaries=adjacent, product_overlap_count=overlap,
            all_context_valid=bool(len(pre) and len(post) and pre.label_valid.eq(1).all() and post.label_valid.eq(1).all()),
            current_delta=median(post,'current_median')-median(pre,'current_median') if len(pre) and len(post) else None,
            argon_delta=median(post,'argon_median')-median(pre,'argon_median') if len(pre) and len(post) else None,
            nitrogen_delta=median(post,'nitrogen_median')-median(pre,'nitrogen_median') if len(pre) and len(post) else None,
            pre_product_ids=';'.join(pre.product_id), post_product_ids=';'.join(post.product_id),
            product_ids=';'.join([*pre.product_id,*post.product_id]), reason=reason,
            evidence_available_at=post.label_available_at.max() if len(post) else pd.NaT,
            exact_replacement_time='', actual_action_confirmed=False,
            confirmation_basis='现场说明下的停机后换新推定' if kind=='after_voltage_failure' else '仅有历史信号线索',
            ignition_template_status='尚未收到公司点火测试范例，不声称已识别点火指纹')
        if kind != 'not_candidate':
            entry['candidate_id']=f'R{len(candidates)+1:03d}'
            candidates.append(entry.copy())
        audit.append(entry)
    return pd.DataFrame(audit), pd.DataFrame(candidates)


def annotate(p, state, cfg):
    p = derive_shutdown(p, cfg['derived_shutdown_max_gap_seconds'])
    p['previous_origin_label_v1'] = p.get('origin_label','')
    labels = [classify_product(r) for r in p.to_dict('records')]
    p['origin_label']=[x[0] for x in labels]; p['origin_evidence']=[x[1] for x in labels]
    p['cause_basis']=[x[2] for x in labels];p['cause_confirmed']=False
    p['attention_event_ids']='';p['replacement_candidate_ids']=''
    events = failure_events(p,cfg)
    for ev in events:
        p.loc[p.product_id.isin(ev['product_ids'].split(';')),'attention_event_ids']=ev['event_id']
    audit,candidates = replacement_windows(p,state,cfg)
    for r in candidates.to_dict('records'):
        ids=r['product_ids'].split(';');hit=p.product_id.isin(ids)
        p.loc[hit,'replacement_candidate_ids']=p.loc[hit,'replacement_candidate_ids'].map(lambda x:';'.join(filter(None,[x,r['candidate_id']])))
        p.loc[hit,'attention_event_ids']=p.loc[hit,'attention_event_ids'].map(lambda x:';'.join(filter(None,[x,r['candidate_id']])))
        events.append(dict(event_id=r['candidate_id'],event_kind='replacement_candidate',
            start=r['start'],end=r['end'],event_time=r['start'],resume_at=r['resume_at'],
            label_available_at=r['evidence_available_at'],failure=None,failure_type='not_applicable',
            origin_label=ACTION_NAMES[r['candidate_kind']],origin_evidence=r['reason'],
            product_ids=r['product_ids'],failed_product_count=None,voltage_failed_products=None,
            cause_confirmed=False,cause_basis=r['confirmation_basis'],confirmed_action='',confirmed_action_time='',
            review_notes='',label_uses_future_context=True,candidate_kind=r['candidate_kind'],
            before_voltage_median=r['before_voltage_median'],after_voltage_median=r['after_voltage_median'],voltage_gain_v=r['voltage_gain_v']))
    return p,pd.DataFrame(events),audit,candidates


def reference_product_check(root, m, cfg, out):
    refinfo=json.loads((out/'参考样本/参考信息.json').read_text('utf-8'))
    stop=pd.Timestamp(refinfo['terminal_selection']['off_start'])
    rows=[]
    # These channels identify reference-device product boundaries only. No other
    # reference-device sensor is used as a main-line example or failure cause.
    ref=m[m.data_stream.eq('voltage_reference')&m.time_start.le(stop+pd.Timedelta(hours=2))&m.time_end.ge(stop-pd.Timedelta(hours=2))]
    for r in ref.to_dict('records'):
        import io
        d=pd.read_csv(io.BytesIO(core.payload(root,r)),sep=';',encoding='utf-8-sig',skipinitialspace=True,
            usecols=lambda x:x.strip() in ['Date-Time','Voltage','Station_1','Station_2'])
        d.columns=d.columns.str.strip();d['timestamp']=pd.to_datetime(d['Date-Time'],format=FMT)
        for c in ['Voltage','Station_1','Station_2']:d[c]=pd.to_numeric(d[c],errors='coerce')
        d['source_id']=r['source_id'];d['source_row']=np.arange(len(d))+2;rows.append(d)
    d=pd.concat(rows,ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    if d.timestamp.duplicated().any():raise ValueError('Duplicate reference timestamps')
    products=[]
    for station in [1,2]:
        active=d[f'Station_{station}'].gt(cfg['station_threshold_rpm'])
        for q in core.contiguous_runs(d,active,cfg['maximum_sampling_gap_seconds']):
            if len(q)<cfg['minimum_station_samples']:continue
            left,right=int(q.index[0]),int(q.index[-1]);start,end=q.timestamp.iloc[0],q.timestamp.iloc[-1]
            coverage=len(q)/((end-start).total_seconds()+1)
            valid=bool(left>0 and right<len(d)-1 and not active.iloc[left-1] and not active.iloc[right+1] and
                (start-d.timestamp.iloc[left-1]).total_seconds()<=2 and (d.timestamp.iloc[right+1]-end).total_seconds()<=2 and
                coverage>=cfg['minimum_sampling_coverage'] and not q[f'Station_{3-station}'].gt(cfg['station_threshold_rpm']).any() and q.Voltage.notna().all())
            products.append(dict(start=start,end=end,station=station,samples=len(q),coverage=coverage,
                voltage_mean=float(q.Voltage.mean()),reference_voltage_fail=int(q.Voltage.mean()<51) if valid else None,
                label_valid=int(valid),source_start=q.source_id.iloc[0],source_row_start=int(q.source_row.iloc[0]),
                source_end=q.source_id.iloc[-1],source_row_end=int(q.source_row.iloc[-1]),
                averaging_scope='reference_station_active_voltage_only',data_stream='reference_device_only'))
    rp=pd.DataFrame(products).sort_values(['start','station']).reset_index(drop=True)
    rp['sample_id']=[f'REF-P{i+1:03d}' for i in range(len(rp))]
    run=0;rp['derived_voltage_two_consecutive_trigger']=0
    for ix,r in rp.iterrows():
        if r.label_valid==1 and r.reference_voltage_fail==1:
            run+=1
            if run==2:rp.loc[ix,'derived_voltage_two_consecutive_trigger']=1
        else:run=0
    core.write_csv(rp,out/'参考样本/末期产品核对.csv')
    refinfo.update(station_used_only_for_product_boundaries=True,
        reference_product_check_scope=dict(start=str(d.timestamp.min()),end=str(d.timestamp.max())),
        reference_product_intervals=len(rp),reference_low_voltage_intervals=int(rp.reference_voltage_fail.eq(1).sum()),
        reference_two_consecutive_rule_triggers=int(rp.derived_voltage_two_consecutive_trigger.sum()),
        reference_product_note='Station瞬时掉速可能拆开区间；本表是当前口径的候选区间，不当作新电极周期计数；规则触发不等于PLC实际停机')
    core.dump(refinfo,out/'参考样本/参考信息.json')
    return rp,refinfo


def verify_sources(root,m):
    count=0
    for r in m.to_dict('records'):
        if hashlib.sha256(core.payload(root,r)).hexdigest()!=r['sha256']:
            raise ValueError('Source changed since previous extraction: '+r['container'])
        count+=1
    return dict(source_files_or_zip_members_verified=count,all_source_hashes_match=True)


def run(root,base,out,cfg,verify_hashes=True):
    root,out=root.resolve(),out.resolve()
    if out==root or root.is_relative_to(out):raise ValueError('Output must not replace the raw data root or its parent')
    if base:
        base=base.resolve()
        if out==base or out.is_relative_to(base) or base.is_relative_to(out):raise ValueError('Keep the previous analysis separate from the new output')
    if out.exists() and any(out.iterdir()):raise FileExistsError('Choose an empty analysis directory to preserve existing results')
    out.mkdir(parents=True,exist_ok=True)
    if base:
        p=load(base/'产品级标签.csv',['start','end','label_available_at','power_episode_start'])
        m=load(base/'来源文件清单.csv',['time_start','time_end']).fillna({'zip_member':'','identical_bytes_to':''})
        state=load(base/'设备带电状态区间.csv',['start','end'])
        minute=load(base/'主数据电压分钟走势.csv',['timestamp'])
        for name in ['来源文件清单.csv','设备带电状态区间.csv','主数据电压分钟走势.csv','每日数据检查.csv','日内采样缺口.csv','分段参数敏感性.csv','规则来源核对.json']:
            if (base/name).exists():shutil.copy2(base/name,out/name)
        shutil.copytree(base/'参考样本',out/'参考样本',dirs_exist_ok=True)
        source_check=verify_sources(root,m) if verify_hashes else {'all_source_hashes_match':'not_checked'}
    else:
        m=core.inventory(root,cfg,out);core.reference(root,m,cfg,out)
        p,minute,state,_=core.main_products(root,m,cfg,out)
        source_check={'all_source_hashes_match':'fresh_inventory'}
    original=p.copy()
    p,e,audit,candidates=annotate(p,state,cfg)
    for col in ['failure','voltage_fail','feedrate_fail','hopper_pressure_fail']:
        assert p[col].fillna(-1).equals(original[col].fillna(-1)),col
        p[col]=p[col].astype('Int64')
    core.write_csv(p,out/'产品级标签.csv');core.write_csv(e,out/'关注事件索引.csv')
    core.write_csv(audit,out/'停复机全量审查.csv');core.write_csv(candidates,out/'换枪候选索引.csv')
    rp,refinfo=reference_product_check(root,m,cfg,out)
    exported=core.export_excerpts(root,m,p,e,cfg,out)
    unknown=p[p.label_valid.eq(0)].copy()
    unknown['expected_samples']=((unknown.end-unknown.start).dt.total_seconds()+1).astype(int)
    unknown['missing_samples']=unknown.expected_samples-unknown.samples
    core.write_csv(unknown,out/'131件采样核查.csv')
    summary=dict(version='2026-11-02-v2',analysis_start=cfg['analysis_start'],analysis_end_exclusive=cfg['analysis_end_exclusive'],
        products=len(p),label_valid_products=int(p.label_valid.eq(1).sum()),failed_products=int(p.failure.eq(1).sum()),
        normal_products=int(p.failure.eq(0).sum()),reason_counts={k:int(p.loc[p.label_valid.eq(1),k].sum()) for k in ['voltage_fail','feedrate_fail','hopper_pressure_fail']},
        failed_product_origin_counts=p[p.failure.eq(1)].origin_label.value_counts().to_dict(),
        failure_episode_counts=e[e.event_kind.eq('product_failure')].failure_type.value_counts().to_dict(),
        attention_events=len(e),replacement_counts=candidates.candidate_kind.value_counts().to_dict() if len(candidates) else {},
        audited_stop_windows=len(audit),replacement_candidates=len(candidates),
        actual_replacements_individually_confirmed=0,derived_shutdowns=int(p.derived_shutdown_trigger.eq(1).sum()),
        unknown_products=len(unknown),unknown_missing_sample_counts={str(k):int(v) for k,v in unknown.missing_samples.value_counts().items()},
        unknown_coverage_min=float(unknown.coverage.min()),unknown_coverage_max=float(unknown.coverage.max()),
        reference=refinfo,source_verification=source_check,
        raw_field_preservation='Original time, Time_s and 28 physical fields retained; no imputation or smoothing',
        model_training_performed=False,**exported)
    core.dump(summary,out/'汇总.json');core.dump(cfg,out/'使用的参数.json')
    print(json.dumps(summary,ensure_ascii=False,indent=2,default=str),flush=True)
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--base-analysis',type=Path)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    args=ap.parse_args();cfg=json.loads(args.config.read_text('utf-8'))
    run(args.root.resolve(),args.base_analysis.resolve() if args.base_analysis else None,args.output.resolve(),cfg)


if __name__=='__main__':main()
