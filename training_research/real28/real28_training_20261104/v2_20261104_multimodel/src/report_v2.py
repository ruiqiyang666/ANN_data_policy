"""Write the actual multi-model results, including unsuccessful experiments."""
import sys
sys.dont_write_bytecode=True
from common_v2 import ROOT,DATA,MODELS,EVAL,load,dump,sha,status

LABELS={'logistic':'逻辑回归','histgb':'梯度提升树','extra_trees':'极端随机树','random_forest':'随机森林','tcn':'TCN','gru':'GRU',
 'patch_transformer':'分块Transformer','spline_logistic':'非线性加法模型','rbf_logistic':'RBF核特征模型',
 'constant':'固定寿命对照','lognormal_aft':'对数正态AFT','weibull_aft':'Weibull AFT','spline_aft':'样条AFT','mlp_aft':'神经网络AFT',
 'logistic_hazard':'逻辑回归分段生存','boosted_hazard':'提升树分段生存','forest_hazard':'随机树分段生存'}
def f(x,n=3):return '不可评价' if x is None else f'{x:.{n}f}'

def main():
    result=load(EVAL/'comparison.json');frozen=load(MODELS/'development_freeze.json');verification=load(EVAL/'verification_final.json')
    assert verification['passed']
    selected=result['risk_choice'];risk=result['risk'][selected]['historical_comparison']['same_v1_landmarks'];a=risk['alarms']
    lines=['# 第二轮多模型训练结果','',
      f'2026年11月4日。已完成 **{result["trained_risk_algorithms"]}种异常预测方法、{result["trained_lifetime_algorithms_excluding_constant"]}种寿命方法**，另有1个固定寿命对照。按算法计数，没有把随机种子、小时/件数分支或回测折数重复计成新方法。', '',
      f'共完成 **{verification["main_fit_count"]}次主模型拟合**，包括各方法的前向回测和最终拟合。模型、代码、数据缓存、日志及评估均位于当前独立目录。第一轮模型和原看板未修改。', '',
      f'异常风险按训练时段三折回测预先选中 **{LABELS[selected]}（{selected}）**。10月10—23日相同预测时点的历史对照中，9个事件预警{a["detected_events"]}个，其中{a["detected_before_product_start"]}个早于首件异常产品开始；误报{a["false_alerts"]}次，每100有效带电小时约{a["false_alerts_per_100_hours"]:.2f}次。', '',
      ('寿命模型中有候选通过训练内固定寿命对照检查，但仍只有少量弱失效终点，不能据此自动安排换枪。' if result['life_choice'] else '**寿命方面，没有学习模型通过训练内固定寿命对照检查。当前没有选出可用于换枪决策的寿命模型。**'), '',
      '**10月10—23日已在第一轮查看，本轮只能称历史对照，不能称新的独立测试。所有第二轮方法选择与阈值在本次历史推理前已冻结；不会根据下表历史成绩更换主模型。**', '',
      '## 异常模型比较','',
      '| 方法 | 三折事件宏平均AP | 合并回测窗口AP | 回测预警/40事件 | 相同时点历史预警/9事件 | 历史误报/100小时 |',
      '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in frozen['risk_summaries']:
        h=result['risk'][r['name']]['historical_comparison']['same_v1_landmarks']['alarms'];o=r['development_alarm_policy']
        lines.append(f'| {LABELS[r["name"]]} | {f(r["mean_fold_event_macro_ap"])} | {f(r["pooled_oof"]["average_precision"])} | {o["detected_events"]}/40 | {h["detected_events"]}/9 | {f(h["false_alerts_per_100_hours"],2)} |')
    lines+=['',
      'AP为平均精确率，不是准确率。事件宏平均AP对每个事件分别与该折完整负例比较再求平均，最后各折等权；合并窗口AP使用不同折模型的全部回测预测，含模型训练规模变化，两者不能混为一项指标。', '',
      '回测预警阈值在这些开发预测上选择，因此回测预警/误报列是开发结果。阈值目标为每100有效带电小时最多20次误报，这是实验预算，不是用户已接受的上线标准。用上升沿触发并在300秒内抑制重复报警。历史成绩才是冻结政策在后期时段的对照表现，但该时段已经看过，仍不能用于宣称独立泛化。', '',
      f'第一轮主候选浅树在相同历史时点为1/9事件、7次误报；本轮预先选中的{LABELS[selected]}为{a["detected_events"]}/9、{a["false_alerts"]}次误报。数据特征与开发选择流程有所变化，且历史时段已知，这个比较只说明本次回放表现，不代表已得到可靠上线效果。', '',
      '## 寿命模型比较','',
      '| 方法 | 前向周期对数区间距离 | 前向周期小时区间距离 | 历史弱终点小时区间距离 | 历史末15分钟区间距离 |',
      '| --- | ---: | ---: | ---: | ---: |']
    for r in frozen['life_summaries']:
        h=result['life'][r['name']]['hours'];near=[q['near_15min_mean_interval_distance'] for q in h['per_endpoint'] if q['near_15min_mean_interval_distance'] is not None]
        n=sum(near)/len(near) if near else None
        lines.append(f'| {LABELS[r["name"]]} | {f(r["forward_cycle_macro_log_distance"])} | {f(r["forward_cycle_macro_hour_distance"])} | {f(h["macro_interval_distance"])} | {f(n)} |')
    lines+=['',
      '距离越小越好；“区间距离”是到弱失效标签上下界的距离，不是已确认自然失效的MAE。前向验证仅两个弱终点，历史仅一个弱终点。小时和件数分别训练，件数没有按固定产速从小时换算。提前更换、观察结束与不明维护仍按删失/截断处理，没有伪造精确自然寿命。', '',
      '本轮寿命训练按疑似周期及“事件/删失×观察距离”分层加权，普通预测点取60秒步长，临近终点保留10秒步长。这样改变了拟合关注点，因此设置了同样权重的固定寿命对照；单凭相较第一轮23小时常数的误差下降，不能证明学到了状态相关的退化规律。', '',
      '样条AFT在开发周期出现极端外推。所有候选都接受相同开发检查：需超过固定寿命对照，临近终点误差不能更差，且不能大量外推到数据支持范围之外。400小时仅是本轮外推审查阈值，不是已证实的电极物理寿命上限。模型分位范围没有独立覆盖率承诺，异常外推结果如实留在表中，没有裁到好看的范围。', '',
      '## 本次具体改动与试错','',
      '- 输入仍保留28个原始参数；短序列额外含质量掩码与带电/喷涂状态，共58个内部通道。历史特征共266项，区分带电、有效喷涂、两个活动料斗及同工况产品变化。',
      '- 停电至少60秒或数据长缺口后重新起算可观察运行段，历史统计保留覆盖信息；可观察段时间不冒充真实枪龄。',
      '- 异常风险使用前三个训练时段滚动回测，共40个可评价事件；不再仅依据两次电压异常选择三类异常的共同模型。仍明确这属于时间/事件隔离，不是已确认的独立电极身份验证。',
      '- 神经网络只优化未来事件的分段风险损失。类型估计用单独条件模型，避免已知产品状态等辅助任务主导编码器；ETA来自同一事件时间分布。CPU模型使用另训的条件时间分布，与其事件概率组成同一定义下的实验联合分布。',
      '- 先跑七种风险方法，前向结果仍弱后追加非线性加法与RBF核方法。六种寿命方法未超过常数对照后，再追加随机树分段生存模型。',
      '- 上端阈值搜索网格过粗的问题在开发回测中发现并修正，对所有方法统一用密集上端分位；误报预算与模型选择规则未改变，未查看历史结果调阈值。',
      '- GRU一次评估触发浮点概率略大于1的边界错误。已用10,004个边界案例复现并验证修复，复用已完成的权重继续评价，没有重复训练充数。', '',
      '## 数据与文件','',
      '| 分区 | 窗口 | 未来事件正窗口 | 可观察事件 | 有效寿命窗口 |',
      '| --- | ---: | ---: | ---: | ---: |']
    counts=load(DATA/'ready.json')['counts']
    for k,label in [('0','8月15日—9月29日训练'),('1','10月4—7日开发检查'),('2','10月8—9日校准诊断'),('3','10月10—23日已知历史对照')]:
        c=counts[k];lines.append(f'| {label} | {c["windows"]:,} | {c["positive_windows"]} | {c["events"]} | {c["life_windows"]:,} |')
    lines+=['',
      '三折回测训练终点分别为9月1日、9月8日、9月15日；留出后续一段。未来300秒目标与训练边界隔离，留出段采用保守7200秒历史间隔，包含已完成产品本身的持续时间。归一化、填补、条件模型都只在对应训练折拟合。7件未知产品（含1件没有结果可用时间）均截断未来负例观察。', '',
      '压力60秒预测仍未训练。健康度没有独立监督，不输出假分数；当前已完成产品结果与独立的即时异常检测能力分开。现有看板、旧模型和原始CSV未改动，所有新模型目前只供离线研究与回放。', '',
      '- [全部指标与逐方法结果](evaluation/comparison.json)',
      '- [训练后、历史比较前的冻结记录](models/development_freeze.json)',
      '- [验证记录](evaluation/verification_final.json)',
      '- [28参数输出契约](models/output_contract.json)',
      '- [初始实验协议](protocol.json)',
      '- [追加风险试验依据](additional_trials.json)',
      '- [追加寿命试验依据](additional_lifetime_trial.json)', '',
      f'各模型位于 `models/risk/<方法>/final/` 和 `models/life/<方法>/final_hours.joblib`、`final_pieces.joblib`。共核验{verification["retained_model_files"]}个权重/预处理/条件模型文件；未使用服务器或下载外部基础模型。', '',
      '本轮满足至少5种方法的训练要求，也保留了低分和失败的尝试。达到训练数量不等于达到生产可用性；上线还取决于预警误报、漏报、提前量，以及自然失效寿命的有效验证。', '']
    path=ROOT/'多模型训练结果.md';path.write_text('\n'.join(lines),encoding='utf-8')
    readme=ROOT/'README.md';readme.write_text('# 第二轮多模型训练\n\n'+f'已完成{result["trained_risk_algorithms"]}种异常预测方法和{result["trained_lifetime_algorithms_excluding_constant"]}种寿命方法，另有常数对照。详见[多模型训练结果.md](多模型训练结果.md)。\n\n'+f'风险主候选：{selected}；寿命通过开发对照的候选：{result["life_choice"] or "无"}。全部模型仅用于实验回放，未接入原看板。\n\n'+'10月10—23日是已知历史对照，不能称为新独立测试。原始数据、第一轮模型、旧看板保持原样。\n',encoding='utf-8')
    status('v2_complete',risk_methods=result['trained_risk_algorithms'],lifetime_methods=result['trained_lifetime_algorithms_excluding_constant'],
           risk_choice=selected,life_choice=result['life_choice'],production_ready=False,report=str(path))
    artifacts=[p for p in ROOT.rglob('*') if p.is_file() and (p.suffix in ['.py','.json','.joblib','.pt','.md']) and p.name not in ['artifact_manifest.json','execution_status.json']]
    dump({str(p.relative_to(ROOT)):sha(p) for p in artifacts},ROOT/'artifact_manifest.json')
    print(path)

if __name__=='__main__':main()
