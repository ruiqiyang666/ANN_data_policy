"""Create the human-readable report from actual frozen artifacts, never placeholders."""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
from common import ROOT, DATA, MODEL, EVAL, dump, sha


def read(path):
    return json.loads(path.read_text('utf-8'))


def fmt(value, digits=3):
    return '不可评价' if value is None else f'{value:.{digits}f}'


def main():
    final = read(EVAL / 'final_metrics.json')
    verified = read(EVAL / 'verification_final.json')
    integrity = read(EVAL / 'original_source_integrity.json')
    assert verified['passed'] and integrity['all_original_sources_unchanged']
    frozen = read(MODEL / 'pretest_freeze.json')
    data = read(DATA / 'prepared.json')
    life = read(MODEL / 'lifetime_fit.json')
    selected = final['primary_risk_model']
    best = final['risk_models'][selected]
    ss = read(MODEL / 'self_supervised_metrics.json')
    histories = {n: read(MODEL / f'{n}_metrics.json') for n in ['b0_tcn', 'b1_ssl_tcn', 'b1_ssl_tcn_seed43', 'b1_ssl_tcn_seed44']}
    elapsed = sum(e['seconds'] for e in ss['epochs']) + sum(e['seconds'] for m in histories.values() for e in m['epochs'])
    peak = max(e['peak_gpu_MiB'] for m in histories.values() for e in m['epochs'])
    lines = [
        '# 28参数模型首轮训练结果', '',
        '2026年11月4日；运行编号 `20261104_v1`。', '',
        '**首轮训练及冻结测试已完成，本机资源足够。模型目前用于实验回放，尚不具备可靠预测自然失效并自动安排换枪的验证依据。**', '',
        f'已训练逻辑回归、浅树、随机初始化TCN、自监督TCN及其两个复训种子，并分别拟合剩余带电小时与剩余件数的生存模型。风险主候选依据测试前的选择集成绩确定为 **{selected}**。没有按测试成绩重新挑模型。', '',
        '所有新文件集中在本文件夹。1,505个原始CSV逐文件哈希核对未变；未连接服务器，未修改旧模型或现有看板。', '',
        '## 实际数据与划分', '',
        f'- 原始数据：{data["source_files"]:,}个文件，{data["rows"]:,}行；只包含主设备8月15日至10月23日的28参数，Voltage Sample排除。',
        f'- 产品重算：{data["products"]:,}件，其中{data["valid_products"]:,}件标签有效；工艺异常{data["failed_products"]}件，合并为{data["events"]}个事件片段。',
        '- 异常产品类型：45件电压、39件仅送料、29件仅压力、1件同时送料和压力。7件未知保留未知。',
        '- 原131件缺点样本经现有资料规则复核后有效；P003507按公司确认修正；旧241件关注样本逐一映射，未因补入早期数据导致编号错配。', '',
        '| 分区 | 日期 | 异常训练/评估窗口 | 有预测窗口的事件 | 有效寿命窗口 | 弱寿命终点 |',
        '| --- | --- | ---: | ---: | ---: | ---: |',
        '| 训练 | 8月15日—9月29日 | 104,571 | 46 | 103,965 | 3 |',
        '| 模型选择 | 10月4—7日 | 9,445 | 2 | 9,343 | 1 |',
        '| 独立校准 | 原定10月8—9日 | 0 | 0 | 0 | 0 |',
        '| 冻结测试 | 10月10—23日 | 34,572 | 9 | 34,325 | 1 |', '',
        '疑似同一电极周期归入较后分区并排除较早数据，因此独立校准集为空；不能重新拿测试数据调概率或预警门槛。事件列仅计有可观察预测窗口的事件，全部原始62个事件并非都能构造有效监督。设备没有真实电极编号，所以该划分不能宣称是真实枪号隔离的独立验证。', '',
        '低压后停机恢复共筛出6个终点候选；其中1个因距前次疑似更换只有约3.18带电小时、存在早期不稳定可能而排除，剩5个仅作弱证据终点。没有把6次都称为确认自然失效。3个人为提前维护候选及2个其他恢复候选共5处截断寿命观察，没有把这些动作当寿命终点。', '',
        '## 异常预测结果', '',
        '| 模型 | 选择集AP | 测试AP | 测试ROC-AUC | 测试Brier |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for name in frozen['evaluated_models']:
        val = frozen['selection_scores'][name]
        test = final['risk_models'][name]['windows']
        lines.append(f'| {name} | {fmt(val["average_precision"])} | {fmt(test["average_precision"])} | {fmt(test["roc_auc"])} | {fmt(test["brier"], 5)} |')
    prevalence = best['windows']['prevalence']
    alarms = best['alarms']
    lines += ['',
        f'AP为平均精确率，评估正例很少时比“准确率”更有信息；测试正例窗口比例为{prevalence:.2%}。这些窗口高度重叠，不能当成同等数量的独立故障。', '',
        f'主候选 `{selected}` 采用测试前冻结的实验阈值 **{alarms["threshold"]:.6f}**，结果：', '',
        f'- 有可观察预测窗口的{alarms["events_with_observable_prediction_windows"]}个事件中，预警{alarms["events_alerted"]}个；在第一件异常产品开始前预警{alarms["events_alerted_before_product_start"]}个。',
        f'- 共{alarms["alert_episodes"]}次合并警报，其中{alarms["false_alert_episodes"]}次后续5分钟内没有目标新事件。',
        f'- 每100个有效可观察带电小时约 **{alarms["false_alerts_per_100_observable_powered_hours"]:.2f}次误报**；分母是按10秒有效预测点累计的{alarms["observable_powered_landmark_hours"]:.2f}小时，排除了预热、缺口与不完整观察。',
        f'- 逐事件结果：[事件表](evaluation/20261104_v1/{selected}_test_events.csv)；逐警报结果：[警报表](evaluation/20261104_v1/{selected}_test_alerts.csv)。', '',
        '阈值来自选择集完整负例的99.5%分位，选择集仅两个事件，属于固定实验规则，没有独立概率校准或现场误报预算验证。ROC-AUC高也不能代替预警召回和误报检查。', '',
        '未来异常的时间标签是首件异常产品结果首次可判定的时刻，不是物理故障真正开始时刻。事件表另列距产品开始的提前量；负值表示该产品已经开始后才报警。', '',
        '## 剩余寿命结果', '',
        '小时模型与件数模型分别拟合区间事件及右删失概率。主模型只用过去5分钟电压、电流均值；对照加入冻结TCN表示的一个主成分。正则化系数固定为10，按疑似周期等总权重，未用测试数据拟合或选参数。', '',
        '| AFT分支 | 测试弱终点数 | 到弱标签区间的平均距离 | 模型区间平均宽度 |',
        '| --- | ---: | ---: | ---: |',
    ]
    for name, tasks in final['lifetime'].items():
        for task, metric in tasks.items():
            entries = metric['per_weak_endpoint']
            unit = '带电小时' if task == 'hours' else '件'
            distance = sum(e['mean_distance_to_weak_interval'] for e in entries) / len(entries) if entries else None
            width = sum(e['mean_model_interval_width'] for e in entries) / len(entries) if entries else None
            lines.append(f'| {name}/{task} | {metric["weak_endpoints"]} | {fmt(distance)} {unit} | {fmt(width)} {unit} |')
    lines += ['',
        '上述“距离”只是到弱事件区间的距离，不是确认自然失效MAE。测试只有一个弱终点；10%—90%模型分位范围未经独立校准，不能叫作已保证覆盖率的寿命范围。右删失的提前退出没有伪造精确失效时间。', '',
        '寿命模型还存在实际预测偏差：选择集中距离弱终点不足两分钟的样本，简单AFT平均仍预测约22.8带电小时，加入冻结TCN表示后仍约22.6小时。两个版本都没有学出可用的临近失效倒计时，不能仅用“已训练完成”来描述其效果。', '',
        '因此本轮已得到能够计算寿命分布的实验模型，但不能据此证明“还有几小时就会自然失效”足够准确。增加服务器算力无法解决独立失效证据少、真实周期身份未知的问题。', '',
        '## 看板输出的实际状态', '',
        '| 输出 | 本轮状态 |',
        '| --- | --- |',
        '| 剩余带电小时、件数 | 已拟合弱证据生存模型；实验数值可回放，未达到自动换枪依据 |',
        '| 寿命范围 | 有模型分位范围；无独立覆盖率承诺 |',
        '| 未来5分钟概率 | 已训练并测试；概率未独立校准 |',
        '| 未来类型、条件ETA | TCN已训练；必须与同一TCN风险一起解释，不能把树概率与TCN倒计时拼成一个已校准分布 |',
        '| 当前异常 | 学习最近已完成产品结果，结果本身已在历史输入中；不是独立的瞬时故障真值 |',
        '| 健康度 | 无独立监督，未训练，输出空值及状态 |',
        '| 当前压力 | 保留实际测量，mbar转bar并区分两个物理料斗 |',
        '| 60秒后压力 | 已排除，未生成训练头或输出 |', '',
        '看板字段样例使用实际测试时点：[dashboard_payload_example.json](evaluation/20261104_v1/dashboard_payload_example.json)。实验估计单列，正式运行字段保留空值与原因。现有看板代码没有接入这些新模型；旧21参数加载器不能直接加载它们。', '',
        '## 本机实测与验证', '',
        f'- 自监督{len(ss["epochs"])}轮；初始内部验证重建损失{ss["epochs"][0]["validation_reconstruction_loss"]:.4f}，最佳{ss["best_loss"]:.4f}。重建变好不自动证明故障预测更好。',
        f'- 异常训练轮数：' + '；'.join(f'{n} {len(m["epochs"])}轮' for n, m in histories.items()) + '。',
        f'- 神经网络训练轮次累计约{elapsed / 60:.1f}分钟，不含数据准备、开发、寿命拟合与最终测试。',
        f'- PyTorch记录的已分配张量峰值约{peak:.1f} MiB；这不包含CUDA上下文和桌面等显存，不能当作整机显存占用。训练期间整机GPU显存观测约3.3 GiB，训练进程主存约2.4 GB。',
        '- 13项关键测试通过；因果取窗、分区隔离、删失及区间边界、报警合并、原始文件和最终冻结哈希已核对。验证结果见 [verification_final.json](evaluation/20261104_v1/verification_final.json)。',
        '- 两个额外种子只改变自监督TCN的微调随机性，共享同一个预训练编码器；不是三套独立预训练结果。',
        '- 本次模型保存最佳权重，没有保存优化器逐步恢复状态。环境从目录内venv入口引用既有系统包；精确版本见 [environment.json](models/20261104_v1/environment.json)。', '',
        '## 文件入口', '',
        '- [完整机器可读测试结果](evaluation/20261104_v1/final_metrics.json)',
        '- [测试前模型与阈值冻结记录](models/20261104_v1/pretest_freeze.json)',
        '- [模型与28参数输出契约](models/20261104_v1/output_contract.json)',
        '- [寿命训练及选择集评估](models/20261104_v1/lifetime_fit.json)',
        '- [执行差异及复现命令](执行口径与复现.md)', '',
        'PatchTST、Chronos和60秒窗口对照本轮未运行。选择集只有2个事件，继续在该集合搜索更多结构难以提供可信优劣证据。后续可以用现有资料做滚动时间回测与告警策略研究；不将公司补充标注设为前提，也不修改已冻结测试结果。', '',
    ]
    (ROOT / '首轮训练结果.md').write_text('\n'.join(lines), encoding='utf-8')
    readme = (ROOT / 'README.md').read_text('utf-8')
    start = readme.index('执行更新（2026年11月4日）')
    end = readme.index('\n\n', start)
    update = '执行更新（2026年11月4日）：**首轮训练、冻结测试和产物核验已完成**。结果见[首轮训练结果.md](首轮训练结果.md)。风险主候选为 `' + selected + '`，寿命仅弱证据拟合，模型目前供实验回放，未接入旧看板或用于自动换枪。全部新产物位于本目录。'
    (ROOT / 'README.md').write_text(readme[:start] + update + readme[end:], encoding='utf-8')
    artifact_paths = list((ROOT / 'src').glob('*.py')) + list((ROOT / 'tests').glob('*.py')) + list(MODEL.glob('*'))
    artifact_paths = [p for p in artifact_paths if p.is_file() and p.name != 'artifact_manifest.json']
    dump({str(p.relative_to(ROOT)): sha(p) for p in artifact_paths}, MODEL / 'artifact_manifest.json')
    print(str(ROOT / '首轮训练结果.md'))


if __name__ == '__main__':
    main()
