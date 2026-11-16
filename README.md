# ANN_data_policy：

目标：利用Booth2传感器提前预警产品失败，并研究在连续两件失败停机前提醒换枪。历史范围2026-09-12—11-14，使用11_16最新数据处理。Booth1参考未混入训练。

## 只看三个文件夹

| 文件夹 | 放了什么 |
|---|---|
| data_processing_11_16 | review.html为数据审阅报告；label_samples.csv为353条关注产品；数据处理代码.zip为本次清洗/标注代码。只训练时不用解压。 |
| training_data | 同事训练要读取的全量数据，文件说明见下表。 |
| model_v1 | 第一版训练/评估代码、配置、已训练模型，以及评估报告.md。当前维护方案为逻辑回归，TCN和GRU权重也保留。 |

根目录的.gitignore用于防止把虚拟环境和自动生成的实验结果误传，不需要手动修改。原始周CSV/ZIP未随包提供；完整重跑清洗时再另取原始资料及追溯依赖。

## 训练数据分别是什么

| 文件 | 用途 |
|---|---|
| products.csv | 全量产品、起止时间、最终标签与来源；查看标注首先看它。 |
| features_10s.csv.gz | 每个时点已经观测到的传感器输入素材。 |
| targets_10s.csv.gz | 同一anchor_id对应未来目标，空值不能填成正常。 |
| electrode_cycles.csv | 电极安装/观察段，以及已固定的train/validation/test划分。 |
| target_events.csv | 单件/相邻连续失败事件，供训练目标与评价使用。 |
| maintenance_events.csv | 已登记或现场确认的换枪记录，和产品失败事件分开。 |
| split_summary.csv | 各集合的产品和事件数量。 |

`.csv.gz`是压缩CSV，Python用pandas.read_csv直接读取，无需先解压。日期为“日 月 年 时:分:秒”。failure为1失败、0未失败、空白为排除；使用supervised_product_eligible=1的产品作严格时间监督。

全量13,181个候选：13,046件未失败、51件失败、84段排除。另有一件现场时间对不齐，不能用于严格时间监督。不要只用关注表label_samples.csv训练。

产品含正转、短暂换向与反转，不能从中间拆开。按完整加工的活动料斗实际点求均值：电压<51 V、送粉<9.1或>11.1 g/min、罐压<100或>250 mbar，任一触发即失败。低压不自动等于寿命终结；缺少历史批次，因此相邻两件失败不冒充PLC批次停机真值。

## 怎么开始训练

需要Python 3.13。下载整个仓库后，在model_v1文件夹打开终端：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_training.py --output ../experiments/trial_01
```

这一条会依次构建输入、训练和评价，结果存到新实验目录，不覆盖交付的第一版。下次试验把trial_01改为trial_02等新名称。Linux/macOS使用对应虚拟环境的python运行同一脚本即可。

训练后先看实验目录的三个结果：`artifacts/evaluation.json`（模型评分）、`artifacts/event_results.csv`（每个故障的提前量/漏报）、`artifacts/frozen_config.json`（最终模型与阈值）。其余是自动生成的运行文件，不需要手动整理上传。训练过程日志为prepare.log、train.log、evaluate.log。

想了解实现时，src/prepare.py构建输入与标签；modeling.py定义网络和输入；train.py训练与选型；evaluation.py计算指标；evaluate.py输出评分。configs/train.json是实验参数。artifacts/里现有文件是第一版模型必需的权重、预处理器和阈值，程序会自动读取。

## 现在的边界

划分按周期/时间固定，归一化与训练只看训练集，验证选模型和阈值。不能随机打散重叠窗口，也不能把最终产品均值、未来换枪信息和事后原因当作实时输入。

测试严格寿命终点目前为0，11/10只有1个弱证据过程，等待现场核实后另做新版本。送料和罐压测试无正例，不能宣称这些目标已经验证准确。第一版报告中的54.6分钟是一次告警的提前量，不是倒计时误差。已有测试结果已被查看，后续研究保持透明，并以新未来数据作进一步验收。

本包用于同事训练和评估，不是完整现场部署包；原有完整工程、实时服务、服务器运行记录和核查附件都保留在原位置。没有重训或改变原始数据、标签、模型权重。

GitHub上传时无需先建空文件夹：在仓库根目录Upload files，拖入本目录的三个文件夹和README.md，网页会按原目录结构创建。不要上传外层“ANN_data_policy_待上传”文件夹。本包可一次上传，不需要之前的三批。
