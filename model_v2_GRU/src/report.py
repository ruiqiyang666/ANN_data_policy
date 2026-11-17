"""Build a self-contained Chinese review page from frozen V2 artifacts.

This module only presents saved results. It does not fit, calibrate, select or
change any model, decision threshold, prediction or event timestamp.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return str(value)
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def records(frame):
    return clean(frame.astype(object).where(pd.notna(frame), None).to_dict('records'))


def load_json(path, default=None):
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding='utf-8-sig'))


def load_csv(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def build_data(root):
    a = root / 'artifacts'
    evaluation = load_json(a / 'evaluation.json')
    frozen = load_json(a / 'frozen_config.json')
    if not evaluation or not frozen:
        raise FileNotFoundError('先完成冻结评估，再生成结果页。')
    selected = evaluation.get('selected_model', frozen.get('selected_model'))
    models = evaluation['models']
    if selected not in models:
        raise ValueError('冻结模型在评估结果中不存在。')
    event_results = load_csv(a / 'event_results.csv')
    alarm_results = load_csv(a / 'alarm_results.csv')
    anchors = load_csv(root / 'data' / 'anchors.csv.gz')
    predictions = load_csv(a / 'predictions.csv.gz')
    if anchors.empty or predictions.empty:
        raise ValueError('缺少真实原点或已保存预测，不能生成案例曲线。')
    if 'anchor_id' not in predictions:
        raise ValueError('预测必须含 anchor_id，不能按猜测行序与原点关联。')
    if predictions.anchor_id.duplicated().any() or anchors.anchor_id.duplicated().any():
        raise ValueError('原点编号重复。')
    columns = ['anchor_id'] + [name for name in models if name in predictions.columns]
    missing = set(models) - set(columns)
    if missing:
        raise ValueError(f'预测缺少模型评分列：{sorted(missing)}')
    anchors = anchors.merge(predictions[columns], on='anchor_id', how='left', validate='one_to_one')
    anchors['timestamp'] = pd.to_datetime(anchors.timestamp)
    events = load_csv(root / 'data' / 'events.csv')
    events['time'] = pd.to_datetime(events.time)
    cases = []
    for e in events.loc[events.is_eol.eq(1) & events.split.eq('test')].sort_values('time').itertuples():
        q = anchors.loc[anchors.cycle_id.eq(e.cycle_id) & anchors.timestamp.between(
            e.time - pd.Timedelta(minutes=90), e.time + pd.Timedelta(minutes=10))].sort_values('timestamp')
        ts = q.timestamp.to_numpy(dtype='datetime64[ms]').astype('int64')
        chart = dict(time=ts.tolist(), voltage=q.Voltage.round(4).tolist(),
                     scores={name: q[name].round(7).tolist() for name in models})
        cases.append(dict(event_id=e.event_id, cycle_id=e.cycle_id, time=str(e.time),
                          deadline=str(e.time-pd.Timedelta(minutes=20)), chart=clean(chart)))
    histories = {}
    for name in models:
        value = load_json(a / 'models' / name / 'history.json', [])
        if isinstance(value, dict):
            value = value.get('history', value.get('epochs', []))
        if value:
            histories[name] = value
    summary = load_csv(root / 'data' / 'split_summary.csv')
    execution = load_json(a / 'run_log.json')
    if not execution:
        execution = load_json(a / 'environment_local.json')
    return clean(dict(selected_model=selected, models=models, frozen=frozen,
                      events=records(event_results), alarms=records(alarm_results),
                      cases=cases, histories=histories, split_summary=records(summary),
                      execution=execution, generated_at=datetime.now().isoformat(timespec='seconds')))


def update_readme(root, data):
    """Keep the concise result section in the user's entry document in sync."""
    readme = root / 'README.md'
    if not readme.exists():
        return
    specs = data['frozen'].get('model_specs', [])
    specs = {s['name']: s for s in specs} if isinstance(specs, list) else specs

    def label(name):
        s = specs.get(name, {})
        kind = s.get('kind', name)
        if kind == 'logistic':
            return f"逻辑回归 C={s.get('C')}"
        if kind == 'logistic_voltage':
            return f"电压逻辑回归 C={s.get('C')}"
        if kind == 'hgb':
            return f"梯度提升树（深度 {s.get('max_depth')}）"
        if kind in ('tcn', 'gru'):
            return f"{kind.upper()}（{s.get('hidden')} 隐单元）"
        return '电压阈值基线' if kind == 'margin' else name

    def fmt(value, decimals=2):
        return '—' if value is None else f'{value:.{decimals}f}'

    chosen = data['selected_model']
    selected = data['models'][chosen]
    m = selected['test']
    vselected = selected['validation']
    lines = ['## 本次真实结果', '',
             f"按验证集的预先固定规则，最终选择 **{label(chosen)}**。测试 **{m['hit20']}/{m['total_events']}** 次达到至少提前 20 分钟，"
             f"晚报 {m['late']} 次、漏报 {m['missed']} 次；误报 {m['false_alarms']} 条，"
             f"误报频率 {fmt(m['false_alarms_per_hour'], 4)} 次/生产小时；另有 {m['unknown_alarms']} 条无法判定的报警、"
             f"{m['post_event_alarms']} 条事件发生后才发出的报警。这里的晚报只指事件前但不足20分钟，事件后报警另列；零误报不表示没有漏报。", '']
    lines += [f"选定原因来自验证集：{vselected['hit20']}/{vselected['total_events']} 次提前至少 20 分钟，"
              f"误报 {vselected['false_alarms']} 条，达标事件平均超前量 {fmt(vselected.get('mean_excess_minutes'))} 分钟。"
              '逻辑回归、TCN 与 GRU 的相同口径结果见下表；最终采用冻结选择，不按测试成绩另换方案。', '']
    if m['hit20'] < m['total_events']:
        lines += ['当前冻结模型还没有让三条测试事件全部达到要求，不能把它写成“已实现稳定提前 20 分钟”。下面保留未达标结果，未据测试再次调参。', '']
    lines += ['| 测试事件 | 失败开始 T | T−20 截止 | 首次有效报警 | 提前量/min | 多提前/min | 结果 |',
              '| --- | --- | --- | --- | ---: | ---: | --- |']
    events = [e for e in data['events'] if e.get('model') == chosen and e.get('split') == 'test']
    for e in events:
        result = '达标' if e.get('hit20') else '晚报' if e.get('late') else '漏报'
        short = lambda v: str(v)[5:19] if v else '无有效报警'
        lines.append(f"| {e['event_id']} | {short(e['time'])} | {short(e.get('deadline'))} | {short(e.get('first_alarm'))} | "
                     f"{fmt(e.get('lead_minutes'))} | {fmt(e.get('excess_minutes'))} | {result} |")
    lines += ['', '候选结果如下。最终选择使用验证成绩；测试表现用于报告，未反过来替换冻结模型。', '',
              '| 候选 | 验证达标 | 验证误报 | 测试达标 | 测试晚报/漏报 | 测试误报 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, values in data['models'].items():
        v, t = values['validation'], values['test']
        lines.append(f"| {label(name)}{'（选定）' if name == chosen else ''} | {v['hit20']}/{v['total_events']} | {v['false_alarms']} | "
                     f"{t['hit20']}/{t['total_events']} | {t['late']}/{t['missed']} | {t['false_alarms']} |")
    run = data.get('execution', {})
    lines += ['', f"本机初始训练与验证选型耗时 **{fmt(run.get('elapsed_seconds'))} 秒**；"
              f"测试前有限扩展候选的拟合与验证另耗时 **{fmt(run.get('extension_fit_seconds'))} 秒**。"
              f"采样训练原点 {run.get('sampled_training_origins', '见日志')} 个，"
              f"验证原点 {run.get('validation_origins', '见日志')} 个；使用 CPU，"
              f"PyTorch {run.get('torch', '见日志')}。最终封存时间 `{data['frozen'].get('sealed_at_utc', '见冻结配置')}`。", '']
    replay = load_json(root / 'artifacts/replay_verification.json')
    if replay.get('passed'):
        lines += [f"独立逐原点回放已核对 **{replay['origins']:,}** 个测试原点，评分最大绝对差 "
                  f"`{replay['max_absolute_score_error']:.3g}`，全部 {replay['emitted_alarms']} 条报警时刻一致。"
                  '26项因果、删失、20分钟边界和推断测试通过。本次未执行现场服务器训练或现场接通。', '']
    source = readme.read_text(encoding='utf-8')
    if '<!-- RESULTS_START -->' not in source or '<!-- RESULTS_END -->' not in source:
        # A manually reviewed delivery README is kept intact on report rebuilds.
        return
    before, rest = source.split('<!-- RESULTS_START -->', 1)
    _, after = rest.split('<!-- RESULTS_END -->', 1)
    readme.write_text(before + '<!-- RESULTS_START -->\n' + '\n'.join(lines) + '\n<!-- RESULTS_END -->' + after, encoding='utf-8')


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Booth2 · 第二版维护预警评估</title><link rel="icon" href="data:,">
<style>
:root{--navy:#172f45;--ink:#253c4d;--muted:#667987;--line:#d9e3e8;--bg:#f3f6f7;--teal:#087e8b;--green:#267454;--red:#b64640;--amber:#a56a18;--soft:#e9f2f4}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.7 "Microsoft YaHei",system-ui,sans-serif}header{background:var(--navy);color:white}.wrap{max-width:1280px;padding:30px 34px;margin:auto}.eyebrow{font-size:12px;letter-spacing:2px;color:#abd0d9}h1{font-size:31px;line-height:1.4;margin:8px 0 12px;font-weight:650}h2{font-size:22px;margin:0 0 8px}h3{font-size:17px;margin:12px 0}p{margin:8px 0}.sub{color:#c1d5df;max-width:950px}.headerline{display:flex;align-items:flex-start;justify-content:space-between;gap:20px}.tag{border:1px solid #69808e;border-radius:30px;padding:6px 14px;white-space:nowrap;font-size:12px;margin-top:10px}nav{display:flex;gap:22px;flex-wrap:wrap;border-top:1px solid #41596a;margin-top:22px;padding-top:16px}nav a{color:#d2e3eb;text-decoration:none;font-size:13px}a{color:var(--teal)}section{background:white;border:1px solid var(--line);border-radius:12px;margin:22px 0;padding:26px 28px}.intro{margin-top:0}.section-top{display:flex;justify-content:space-between;gap:20px;align-items:center}.muted,small{color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:21px 0 8px}.card{padding:17px 18px;border:1px solid var(--line);border-radius:8px;background:#fafcfc}.card .label{font-size:12px;color:var(--muted)}.card b{display:block;font-size:29px;line-height:1.6;font-weight:650}.card .foot{font-size:12px;color:var(--muted)}.status{border-radius:5px;display:inline-block;padding:2px 8px;font-size:12px;white-space:nowrap}.good{background:#e8f3ed;color:var(--green)}.bad{background:#faeceb;color:var(--red)}.warn{background:#fcf2df;color:var(--amber)}.neutral{background:#ecf1f4;color:#4f6879}.notice{border-left:3px solid var(--teal);background:#edf6f7;padding:13px 16px;margin:16px 0}.notice.warn{border-left-color:#c28d3b;background:#fbf5e9;color:var(--ink);font-size:14px}.toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:12px;margin:16px 0}.toolbar label{font-size:13px;color:var(--muted)}select,button{font:inherit;font-size:14px;border:1px solid #bbcbd3;border-radius:6px;background:white;padding:8px 11px;color:var(--ink)}button{cursor:pointer}button:hover{background:#edf6f7}button.active{background:var(--navy);border-color:var(--navy);color:white}select{max-width:100%}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th{text-align:left;background:#eef3f5;color:#506675;font-weight:500;white-space:nowrap}th,td{padding:11px 12px;border-bottom:1px solid #e2e9ec;vertical-align:top}td.num{font-variant-numeric:tabular-nums;white-space:nowrap}tr.selected{background:#f0f8f8}tr.chosen td:first-child{border-left:3px solid var(--teal)}.table-note{font-size:12px;color:var(--muted);margin-top:10px}.cases{display:flex;gap:8px;flex-wrap:wrap}.case-title{font-size:19px;margin:18px 0 10px}.detailgrid{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);border-radius:8px;margin:12px 0;overflow:hidden}.detailgrid>div{padding:14px 16px;border-right:1px solid var(--line)}.detailgrid>div:last-child{border:0}.detailgrid span{font-size:12px;color:var(--muted);display:block}.detailgrid b{display:block;margin-top:4px;font-variant-numeric:tabular-nums}.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:12px;color:var(--muted);margin:12px 0}.legend i{display:inline-block;width:15px;height:3px;vertical-align:middle;margin-right:5px}.plot{width:100%;height:auto;display:block;border:1px solid #e5ecef;border-radius:8px;touch-action:pan-y}.charttext{font-family:"Microsoft YaHei",system-ui,sans-serif;fill:#637886;font-size:12px}.chartdetail{min-height:26px;font-size:13px;color:#526a7a;margin-top:6px}.explain{font-size:13px;color:var(--muted);margin-top:9px}.two{display:grid;grid-template-columns:1fr 1fr;gap:24px}.two>div{min-width:0}.pillbox{font-size:12px;padding:7px 10px;background:#eef3f5;border-radius:5px}details{border:1px solid var(--line);border-radius:8px;padding:13px 16px;margin-top:16px}summary{cursor:pointer;font-size:14px}footer{padding:10px 0 24px;color:var(--muted);font-size:12px}.empty{padding:26px;text-align:center;color:var(--muted);border:1px dashed var(--line);border-radius:8px}.red{color:var(--red)}.green{color:var(--green)}.anchor{scroll-margin-top:15px}.linkrow{display:flex;gap:20px;flex-wrap:wrap;font-size:13px}.code{font-family:Consolas,monospace}button:focus-visible,select:focus-visible,a:focus-visible{outline:3px solid #47acbd;outline-offset:3px}@media(max-width:850px){.wrap{padding:20px 16px}.cards{grid-template-columns:repeat(2,1fr)}section{padding:20px 16px}.detailgrid{grid-template-columns:repeat(2,1fr)}.detailgrid>div{border-bottom:1px solid var(--line)}.two{grid-template-columns:1fr}.headerline{display:block}h1{font-size:26px}.tag{display:inline-block}.section-top{display:block}}@media print{nav,.toolbar,.cases,button{display:none}body{background:white}.wrap{max-width:none;padding:10px}section{break-inside:avoid}.scroll{overflow:visible}header{background:white;color:var(--ink)}.sub,.eyebrow{color:var(--muted)}}
#comparison table{min-width:900px}#comparison th:first-child{min-width:180px}
</style></head><body>
<header><div class="wrap"><div class="headerline"><div><div class="eyebrow">BOOTH2 / MAINTENANCE EARLY WARNING / V2</div><h1>电极维护：提前 20 分钟，做到了多少？</h1><p class="sub">以最终连续失败的第一件产品开始加工为终点。先检查是否至少提前 20 分钟，再比较多提前了多久，并保留完整生产中的误报。</p></div><span class="tag">历史留出评估 · 2026.11.17</span></div><nav><a href="#overview">评估结论</a><a href="#comparison">模型比较</a><a href="#cases">三条测试案例</a><a href="#alarms">误报与漏报</a><a href="#training">训练曲线与范围</a></nav></div></header>
<main class="wrap">
<section id="overview" class="anchor intro"><div class="section-top"><div><h2>验证集选定的方案</h2><p id="selection" class="muted"></p></div><span class="status neutral">阈值在测试前冻结</span></div><div class="cards" id="overview-cards"></div><div id="conclusion" class="notice"></div><p class="explain">测试覆盖 L008—L013 六个完整周期，含 10/28、11/1、11/10 三次电压末期。11/6 预防性更换作为正常／删失观察保留。三个事件曾被第一版实验部分查看，因此这里是扩大历史留出评估，尚不能证明对未来周期稳定有效。</p></section>
<section id="comparison" class="anchor"><h2>模型比较</h2><p class="muted">所有候选使用相同终点、完整周期划分和实际加工预测机会。页面切换只改变展示。</p><div class="toolbar"><label for="model">当前查看模型</label><select id="model"></select><span id="model-tag" class="status neutral"></span></div><div class="scroll"><table><thead><tr><th>模型</th><th>验证达标</th><th>验证误报 / h</th><th>测试达标</th><th>测试晚报 / 漏报</th><th>测试误报</th><th>测试误报 / h</th></tr></thead><tbody id="comparison-rows"></tbody></table></div><p class="table-note">达标：首次有效报警在 T−20 或更早，并按预先固定的 60 分钟事件关联范围计分。误报 / h 的分母为对应集合全部有效产品的生产小时。未知告警单列；过早报警不会在后续进入关联窗口时被重新包装为命中。</p></section>
<section id="cases" class="anchor"><h2>三条测试案例</h2><p class="muted">每条都展示首次有效报警，不挑事后最接近 20 分钟的一次。</p><div id="case-buttons" class="cases toolbar"></div><div id="case-title" class="case-title"></div><div id="case-details" class="detailgrid"></div><div id="case-note"></div><div class="legend"><span><i style="background:#087e8b"></i>电压实测</span><span><i style="background:#567eb0"></i>当时保存的模型评分</span><span><i style="background:#a56a18"></i>T−20 截止</span><span><i style="background:#267454"></i>首次有效报警</span><span><i style="background:#b64640"></i>连续失败开始 T</span></div><svg id="case-chart" class="plot" viewBox="0 0 1100 440" role="img" aria-label="选中事件的实测电压和历史风险评分"></svg><div id="chart-detail" class="chartdetail">在曲线上移动或点击，可查看该原点的实测电压和已保存评分。</div><p class="explain">电压线全部是实测数据，报警之后的电压是后续实测，不能视为模型预测出的电压曲线。评分来自各自原点已经可用的输入；缺少加工原点处断线。时间均为设备记录的日历时间。</p><details><summary>三次测试事件的完整时间表</summary><div class="scroll"><table><thead><tr><th>事件</th><th>失败开始 T</th><th>T−20 截止</th><th>首次有效报警</th><th>提前量</th><th>多提前</th><th>结论</th></tr></thead><tbody id="event-rows"></tbody></table></div></details></section>
<section id="alarms" class="anchor"><h2>误报与漏报一起看</h2><div id="alarm-summary" class="notice"></div><div class="scroll"><table><thead><tr><th>测试报警时间</th><th>周期</th><th>评分</th><th>归类</th><th>关联事件</th></tr></thead><tbody id="alarm-rows"></tbody></table></div><p class="explain">“无法判定”表示报警之后的完整观察不足，不能当作零误报；“事件发生后”不算提前预警。表中保留所有已发出的测试报警，包括成功、过早和重复提醒。</p></section>
<section id="training" class="anchor"><h2>训练曲线与可用证据</h2><div class="two"><div><h3>神经网络收敛过程</h3><div class="toolbar"><label for="history-model">模型</label><select id="history-model"></select></div><div class="legend"><span><i style="background:#087e8b"></i>训练损失</span><span><i style="background:#c38c3c"></i>验证损失</span></div><svg id="loss-chart" class="plot" viewBox="0 0 540 280" role="img" aria-label="训练和验证损失曲线"></svg><p class="explain">损失下降说明模型在拟合任务，是否满足维护要求仍以逐事件首次报警、误报和漏报为准。</p></div><div><h3>固定划分</h3><table><thead><tr><th>集合</th><th>完整周期</th><th>电压末期</th></tr></thead><tbody><tr><td>训练</td><td>L000—L004 · 5 个</td><td>5 次</td></tr><tr><td>验证</td><td>L005—L007 · 3 个</td><td>2 次</td></tr><tr><td>测试</td><td>L008—L013 · 6 个</td><td>3 次</td></tr></tbody></table><p class="explain">预处理仅在训练集拟合，验证用于选模型及阈值，测试前固定配置。同周期及重叠历史窗口不跨集合。少量事件限制了结论强度，增加测试样本本身不会让模型变准。</p><div class="notice warn">送料连续失败无足够独立正例；罐压连续失败仅有少量训练证据。不能据此声称送料／罐压预警有效，也不能把它们算作电极寿命成功案例。</div><p class="explain">本页展示本机训练与历史加工锚点回放结果。现场应用、数据服务器和 ANN 服务器的接口尚未接通。</p></div></div></section>
<footer><div class="linkrow"><a href="../README.md">训练、复现与结果说明</a><a href="../docs/deployment.md">现场部署步骤</a><a href="../artifacts/event_results.csv" download>逐事件明细 CSV</a><a href="../artifacts/alarm_results.csv" download>全部报警明细 CSV</a></div><p id="generated"></p></footer></main>
<script>
const DATA=__DATA__;
const $=id=>document.getElementById(id), esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(x,n=2)=>x==null||!Number.isFinite(Number(x))?'—':Number(x).toFixed(n);
const yes=x=>x===true||x===1||x==='True'||x==='true'||x==='1';
const metric=(x,keys,d=null)=>{for(const k of keys)if(x?.[k]!=null)return x[k];return d};
const names={margin:'电压阈值基线',rule:'电压阈值基线',threshold:'电压阈值基线',logistic:'逻辑回归',tcn:'小型 TCN',gru:'小型 GRU'};
const spec=name=>Array.isArray(DATA.frozen.model_specs)?DATA.frozen.model_specs.find(s=>s.name===name):DATA.frozen.model_specs?.[name];
const label=name=>{const s=spec(name);return s?.display_name||(s?.kind==='logistic'?`逻辑回归 · C=${s.C}`:s?.kind==='logistic_voltage'?`电压逻辑回归 · C=${s.C}`:s?.kind==='hgb'?`梯度提升树 · 深度 ${s.max_depth}`:s?.kind==='tcn'?`TCN · ${s.hidden} 隐单元`:s?.kind==='gru'?`GRU · ${s.hidden} 隐单元`:names[name]||name)};
const stat=(name,split)=>DATA.models[name]?.[split]||{}, count=(m,k)=>metric(m,[k],0), total=m=>metric(m,['total_events','events'],0), ratio=m=>`${count(m,'hit20')} / ${total(m)}`;
const date=s=>s?String(s).replace('T',' ').slice(0,19):'无有效报警', short=s=>s?String(s).replace('T',' ').slice(5,19):'无有效报警';
const epoch=s=>s?Date.parse(String(s).replace(' ','T').replace(/Z$/,'')+'Z'):null, clock=t=>new Date(t).toISOString().slice(11,19);
const threshold=name=>{const t=DATA.frozen.thresholds?.[name];return typeof t==='object'?t.threshold:t};
let model=DATA.selected_model,caseIndex=0;
const selectedTest=stat(DATA.selected_model,'test');
$('selection').textContent=`${label(DATA.selected_model)} · 由验证集比较后固定，测试结果单独报告。`;
const hits=count(selectedTest,'hit20'),nt=total(selectedTest),miss=count(selectedTest,'missed'),late=count(selectedTest,'late');
$('overview-cards').innerHTML=[['测试事件达标',`${hits} / ${nt}`,'首次报警至少提前 20 分钟'],['测试误报',count(selectedTest,'false_alarms'),'覆盖完整测试生产历史'],['误报 / 生产小时',fmt(selectedTest.false_alarms_per_hour,4),`${fmt(selectedTest.operating_hours)} 小时有效产品生产`],['晚报 / 漏报',`${late} / ${miss}`,'少于 20 分钟也计不达标']].map(([a,b,c])=>`<div class="card"><div class="label">${a}</div><b>${b}</b><div class="foot">${c}</div></div>`).join('');
const selectedEvents=DATA.events.filter(e=>e.model===DATA.selected_model&&e.split==='test');
const leads=selectedEvents.filter(e=>yes(e.hit20)).map(e=>Number(e.lead_minutes));
$('conclusion').textContent=hits===nt&&nt>0?`这次历史测试中，${nt} 次末期均达到至少提前 20 分钟；达标提前量为 ${leads.map(v=>fmt(v,2)).join('、')} 分钟。是否接近 20 分钟，请与下面的逐事件时间和误报一起判断。`:`这次历史测试有 ${hits}/${nt} 次达到至少提前 20 分钟，提前量为 ${leads.map(v=>fmt(v,2)).join('、')} 分钟；尚未实现全部事件稳定接近 20 分钟。${miss} 次事件未在发生前有效预警；另有 ${count(selectedTest,'post_event_alarms')} 条发生后报警，均不计达标。`;
if(hits<nt)$('conclusion').classList.add('warn');
$('model').innerHTML=Object.keys(DATA.models).map(n=>`<option value="${esc(n)}">${esc(label(n))}${n===DATA.selected_model?' · 验证选定':''}</option>`).join('');$('model').value=model;
$('case-buttons').innerHTML=DATA.cases.map((c,i)=>`<button data-case="${i}">${esc(c.time.slice(5,10).replace('-','/'))} · ${esc(c.cycle_id)}</button>`).join('');
const getEvent=c=>DATA.events.find(e=>e.model===model&&e.split==='test'&&e.event_id===c.event_id);
const eventStatus=e=>!e?'无记录':yes(e.hit20)?'<span class="status good">达标</span>':yes(e.late)?'<span class="status warn">晚报</span>':'<span class="status bad">漏报</span>';
function drawComparison(){
 $('model-tag').textContent=threshold(model)>1?'冻结策略：不报警（会漏报）':model===DATA.selected_model?'验证选定方案':'候选对照';
 $('comparison-rows').innerHTML=Object.keys(DATA.models).map(n=>{const v=stat(n,'validation'),t=stat(n,'test');return `<tr class="${n===model?'selected ':''}${n===DATA.selected_model?'chosen':''}"><td><b>${esc(label(n))}</b>${n===DATA.selected_model?'<br><small>验证选定</small>':''}</td><td class="num">${ratio(v)}</td><td class="num">${fmt(v.false_alarms_per_hour,4)}</td><td class="num">${ratio(t)}</td><td class="num">${count(t,'late')} / ${count(t,'missed')}</td><td class="num">${count(t,'false_alarms')}</td><td class="num">${fmt(t.false_alarms_per_hour,4)}</td></tr>`}).join('');
 $('event-rows').innerHTML=DATA.cases.map(c=>{const e=getEvent(c)||{};return `<tr><td>${esc(c.event_id)}</td><td class="num">${short(c.time)}</td><td class="num">${short(c.deadline)}</td><td class="num">${short(e.first_alarm)}</td><td class="num">${fmt(e.lead_minutes)} min</td><td class="num">${yes(e.hit20)?fmt(e.excess_minutes)+' min':'—'}</td><td>${eventStatus(e)}</td></tr>`}).join('');
 const t=stat(model,'test');$('alarm-summary').textContent=`${label(model)}：测试达标 ${ratio(t)}；事件前不足20分钟 ${count(t,'late')} 次，未在事件前有效预警 ${count(t,'missed')} 次；误报 ${count(t,'false_alarms')} 条，无法判定 ${count(t,'unknown_alarms')} 条，事件发生后报警 ${count(t,'post_event_alarms')} 条。`;
 const rows=DATA.alarms.filter(a=>a.model===model&&a.split==='test');
 $('alarm-rows').innerHTML=rows.length?rows.map(a=>{const cat=yes(a.post_event)?['事件发生后','neutral']:yes(a.false_alarm)?['误报','bad']:yes(a.unknown)?['无法判定','warn']:['关联末期事件','good'];return `<tr><td class="num">${date(a.time)}</td><td>${esc(a.cycle_id)}</td><td class="num">${fmt(a.probability??a.score,4)}</td><td><span class="status ${cat[1]}">${cat[0]}</span></td><td>${esc(a.matched_event??a.event_id??'—')}</td></tr>`}).join(''):'<tr><td colspan="5">未发出测试报警；是否漏报见上方事件表。</td></tr>';
}
function svgText(x,y,text,anchor='start',fill){return `<text x="${x}" y="${y}" text-anchor="${anchor}" class="charttext"${fill?` style="fill:${fill}"`:''}>${esc(text)}</text>`}
function linePath(times,vals,x,y,gap=120000){let d='',prev=null;for(let i=0;i<times.length;i++){const v=vals[i];if(v==null||!Number.isFinite(v)){prev=null;continue}d+=`${prev==null||times[i]-prev>gap?'M':'L'}${x(times[i]).toFixed(2)},${y(v).toFixed(2)} `;prev=times[i]}return d}
function drawCase(){
 const c=DATA.cases[caseIndex];if(!c)return;const e=getEvent(c)||{},ch=c.chart;
 document.querySelectorAll('[data-case]').forEach(b=>b.classList.toggle('active',Number(b.dataset.case)===caseIndex));
 $('case-title').innerHTML=`${esc(c.time.slice(0,10))} · ${esc(c.cycle_id)} &nbsp; ${eventStatus(e)}`;
 $('case-details').innerHTML=[['连续失败开始 T',date(c.time)],['最晚达标报警 T−20',date(c.deadline)],['首次有效报警',date(e.first_alarm)],['实际提前量',e.lead_minutes==null?'无有效报警':fmt(e.lead_minutes)+' 分钟']].map(([a,b])=>`<div><span>${a}</span><b>${b}</b></div>`).join('');
 let note=yes(e.hit20)?`比 20 分钟要求多提前 ${fmt(e.excess_minutes)} 分钟。`:yes(e.late)?`首次报警只提前 ${fmt(e.lead_minutes)} 分钟，未达到 20 分钟要求。`:'按冻结在线规则，没有可计分的提前报警。';
 const after=DATA.alarms.find(a=>a.model===model&&a.split==='test'&&a.cycle_id===c.cycle_id&&yes(a.post_event));
 if(!yes(e.hit20)&&!yes(e.late)&&after)note+=` 首条事件后报警为 ${date(after.time)}，比失败开始晚 ${fmt((epoch(after.time)-epoch(e.time))/60000)} 分钟，不计提前预警。`;
 if(e.earlier_alarm_count>0)note+=` 此事件之前本周期另有 ${e.earlier_alarm_count} 条更早报警，第一条在 ${date(e.all_episode_first)}，完整报警表予以保留。`;
 if(c.cycle_id==='L012')note+=' 11/10 截止前最近的既有加工原点为 13:04:00；不能补造 13:08:30 的加工预测。';
 $('case-note').innerHTML=`<p class="explain">${esc(note)}</p>`;
 const T=epoch(c.time),lo=T-90*60000,hi=T+10*60000,left=65,right=1050,w=right-left,x=t=>left+(t-lo)/(hi-lo)*w;
 const vv=ch.voltage.filter(v=>v!=null&&Number.isFinite(v));const vmin=Math.min(50,Math.floor(Math.min(...vv)-.3)),vmax=Math.max(55,Math.ceil(Math.max(...vv)+.3));
 const yv=v=>174-(v-vmin)/(vmax-vmin)*129,yp=v=>363-v*124;let s='';
 s+=`<rect x="${x(T-20*60000)}" y="30" width="${x(T)-x(T-20*60000)}" height="345" fill="#faf3e8"/>`;
 for(let v=vmin;v<=vmax;v++){s+=`<line x1="${left}" y1="${yv(v)}" x2="${right}" y2="${yv(v)}" stroke="#e7edef"/>`+svgText(left-10,yv(v)+4,v,'end')}
 for(const v of [0,.25,.5,.75,1])s+=`<line x1="${left}" y1="${yp(v)}" x2="${right}" y2="${yp(v)}" stroke="#e7edef"/>`+svgText(left-10,yp(v)+4,v,'end');
 s+=svgText(left,22,'实测电压 / V')+svgText(left,219,'保存的风险评分');
 for(let t=lo;t<=hi;t+=20*60000){s+=`<line x1="${x(t)}" y1="30" x2="${x(t)}" y2="375" stroke="#eef2f4"/>`+svgText(x(t),399,clock(t).slice(0,5),'middle')}
 s+=`<path d="${linePath(ch.time,ch.voltage,x,yv)}" fill="none" stroke="#087e8b" stroke-width="1.8"/><path d="${linePath(ch.time,ch.scores[model],x,yp)}" fill="none" stroke="#567eb0" stroke-width="1.8"/>`;
 const th=threshold(model);if(th!=null&&th>=0&&th<=1)s+=`<line x1="${left}" y1="${yp(th)}" x2="${right}" y2="${yp(th)}" stroke="#567eb0" stroke-dasharray="5 4" opacity=".7"/>`+svgText(right,yp(th)-5,`冻结阈值 ${fmt(th,3)}`,'end','#567eb0');
 const markers=[[T-20*60000,'T−20','#a56a18',434],[epoch(e.first_alarm),'首次报警','#267454',418],[T,'T','#b64640',434]];
 for(const [at,title,color,yy]of markers)if(at!=null&&at>=lo&&at<=hi)s+=`<line x1="${x(at)}" y1="30" x2="${x(at)}" y2="375" stroke="${color}" stroke-width="1.6" stroke-dasharray="5 4"/>`+svgText(x(at)+3,yy,title,'start',color);
 s+=`<line id="crosshair" x1="0" y1="30" x2="0" y2="375" stroke="#8297a4" stroke-dasharray="2 3" visibility="hidden"/><rect id="hover-area" x="${left}" y="30" width="${w}" height="345" fill="transparent"/>`;
 $('case-chart').innerHTML=s;
 const inspect=evt=>{const rect=$('case-chart').getBoundingClientRect(),px=(evt.clientX-rect.left)/rect.width*1100,t=lo+(px-left)/w*(hi-lo);let j=0;for(let i=1;i<ch.time.length;i++)if(Math.abs(ch.time[i]-t)<Math.abs(ch.time[j]-t))j=i;if(!ch.time.length)return;const l=$('crosshair');l.setAttribute('x1',x(ch.time[j]));l.setAttribute('x2',x(ch.time[j]));l.setAttribute('visibility','visible');$('chart-detail').textContent=`${clock(ch.time[j])} · 电压 ${fmt(ch.voltage[j],3)} V · 保存评分 ${fmt(ch.scores[model][j],5)} · 距 T ${fmt((T-ch.time[j])/60000,2)} 分钟`;};
 $('hover-area').addEventListener('pointermove',inspect);$('hover-area').addEventListener('click',inspect);
 $('chart-detail').textContent='在曲线上移动或点击，可查看该原点的实测电压和已保存评分。';
}
function drawLoss(){
 const name=$('history-model').value,h=DATA.histories[name]||[];if(!h.length){$('loss-chart').innerHTML=svgText(270,140,'该模型没有神经网络训练曲线','middle');return}
 const tr=h.map(r=>r.train_loss??r.loss),va=h.map(r=>r.val_loss??r.validation_loss),xx=h.map((r,i)=>r.epoch??i+1),all=[...tr,...va].filter(v=>v!=null&&Number.isFinite(v));
 const lo=Math.min(...all)*.95,hi=Math.max(...all)*1.05,x=t=>55+(t-Math.min(...xx))/Math.max(1,Math.max(...xx)-Math.min(...xx))*455,y=v=>230-(v-lo)/Math.max(.00001,hi-lo)*190;let s='';
 for(let i=0;i<5;i++){const v=lo+(hi-lo)*i/4;s+=`<line x1="55" y1="${y(v)}" x2="510" y2="${y(v)}" stroke="#e7edef"/>`+svgText(46,y(v)+4,fmt(v,3),'end')}
 for(const t of [...new Set([xx[0],xx[Math.floor(xx.length/2)],xx[xx.length-1]])])s+=svgText(x(t),254,t,'middle');
 s+=`<path d="${linePath(xx,tr,x,y,Infinity)}" fill="none" stroke="#087e8b" stroke-width="2"/><path d="${linePath(xx,va,x,y,Infinity)}" fill="none" stroke="#c38c3c" stroke-width="2"/>`+svgText(510,274,'epoch','end');$('loss-chart').innerHTML=s;
}
$('model').addEventListener('change',e=>{model=e.target.value;drawComparison();drawCase()});
$('case-buttons').addEventListener('click',e=>{if(e.target.dataset.case!=null){caseIndex=Number(e.target.dataset.case);drawCase()}});
$('history-model').innerHTML=Object.keys(DATA.histories).map(n=>`<option value="${esc(n)}">${esc(label(n))}</option>`).join('')||'<option>无神经网络曲线</option>';
$('history-model').addEventListener('change',drawLoss);
$('generated').textContent=`第二版冻结结果 · 页面生成 ${DATA.generated_at.replace('T',' ')} · 本机离线查看，无需联网。`;
drawComparison();drawCase();drawLoss();
</script></body></html>'''


def write_share_report(root, page):
    """Embed the four linked documents so this HTML can be sent by itself."""
    root = Path(root)
    attachments = [
        ('../README.md', root / 'README.md', 'text/plain', 'README.md'),
        ('../docs/deployment.md', root / 'docs/deployment.md', 'text/plain', 'deployment.md'),
        ('../artifacts/event_results.csv', root / 'artifacts/event_results.csv', 'text/csv', 'event_results.csv'),
        ('../artifacts/alarm_results.csv', root / 'artifacts/alarm_results.csv', 'text/csv', 'alarm_results.csv'),
    ]
    for href, path, mime, filename in attachments:
        encoded = base64.b64encode(path.read_bytes()).decode('ascii')
        # The original CSV links already have a bare download attribute.
        page = page.replace(f'href="{href}" download', f'href="{href}"')
        page = page.replace(f'href="{href}"',
                            f'href="data:{mime};charset=utf-8;base64,{encoded}" download="{filename}"')
    page = page.replace('本机离线查看，无需联网。', '单文件离线报告；可直接转发，无需联网或安装 Python。')
    page = page.replace('<footer>', '<footer><p class="muted">本文件可单独转发。图表与下列可下载附件均已包含在文件内；请下载后用 Edge、Chrome 等浏览器打开。</p>')
    target = root / 'model_v2_results.html'
    target.write_text(page, encoding='utf-8')
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--share-only', action='store_true',
                        help='Package the existing saved page without changing results or rebuilding the dashboard')
    args = parser.parse_args()
    if args.share_only:
        target = write_share_report(args.root, (args.root / 'web/index.html').read_text(encoding='utf-8'))
        print(json.dumps({'share_page': str(target)}, ensure_ascii=False))
        return
    data = build_data(args.root)
    serialized = json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('</', '<\\/')
    out = args.root / 'web'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'index.html').write_text(HTML.replace('__DATA__', serialized), encoding='utf-8')
    (args.root / '打开第二版结果.cmd').write_bytes(b'@echo off\r\nstart "" "%~dp0model_v2_results.html"\r\n')
    update_readme(args.root, data)
    write_share_report(args.root, (out / 'index.html').read_text(encoding='utf-8'))
    print(json.dumps({'page': str(out / 'index.html'), 'models': len(data['models']),
                      'test_cases': len(data['cases']), 'selected_model': data['selected_model']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
