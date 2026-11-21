# 训练研究归档 · 2026-11-21

本分支补齐 Phase2 的训练源码、已保存权重、预处理器、冻结协议、逐折预测、评估报告和核验记录。原有 `model_v1`、`model_v2_GRU`、`training_data` 保留。看板应用源码继续在 Gitee `SparklShaw/ANN` 的 `feature/depuy-synthes-suite` 维护。

| 目录 | 内容 |
| --- | --- |
| `real28/real28_training_20261104` | 28 参数 v1、多方法 v2、修订标签 v3、增量 v4；成功保存的各算法、种子、折与辅助头权重，训练/标签代码和评估 |
| `real28/real28_training_20261112_neural` | 三路风险 MLP 源码、协议、比较；`server-snapshot` 补齐学校服务器的 6 折、全量候选及折外预测 |
| `real28/exports` | 历史树/神经网络与 2 秒包的权重及契约，不含原始回放输入 |
| `booth2/model-v3-20261118` | v3 电极预警与剩余加工寿命、研究迭代和 NumPy 导出包 |
| `booth2/model-v4-20261119` | v4 树/神经/融合寿命研究、保存的拟合对象与留出预测 |
| `booth2/model-v5-20261120` | 六个寿命候选、逐折评估、全量研究权重、重载核验与固定输入快照 |
| `booth2/model-v6-20261120` | 九个寿命候选、两个独立墙钟预警头、KM 基线及修正前备份、逐折与全量权重 |
| `booth2/electrode-gru-20261118` | 原仓库 GRU-16 的 NumPy 权重、预处理器、原点与导出核验 |
| `booth2/heldout-ledger` | v5/v6 留出预测展示台账，不是新数据模型推理权重 |
| `booth2/depuy-theme-preview` | 训练所需 NumPy 推理实现、导出器、v5/v6 可复用源码与学习循环；无 UI 页面/样式 |
| `legacy_simulated21` | 已退役的 21 参数模拟训练源码、已保存模型和评估，单独保留为历史实验 |
| `legacy_extra_weights`、`legacy_synthetic_workbench` | 旧工作区中未与上述模型重复的权重变体及模拟训练工作台源码，按哈希留档 |
| `booth2/warehouse` | 历史采集/清洗、数据仓库 schema 与验证代码，不含数据库/原始数据 |
| `workflow`、`docs` | 人工发布的连续学习流程模板、结果与 RSI 文档 |

## 获取权重

大型模型、数组、预测表与压缩文件采用 Git LFS。普通 GitHub “Download ZIP” 可能只包含指针；请用 Git 克隆并获取 LFS 对象：

```bash
git lfs install
git clone --branch research/training-models-20261121 https://github.com/ruiqiyang666/ANN_data_policy.git
cd ANN_data_policy
git lfs pull
python training_research/verify_bundle.py
```

`SOURCE_MANIFEST.json` 逐文件记录本地来源、源 SHA-256 与提交 SHA-256。唯一路径适配是 v3 `research/data.py` 的 `ANN_DATA_REPO`：默认读取本仓库根目录，原始文件保留于 `frozen-source/data.py`。历史训练产物中的源哈希仍指向当时的冻结源码，不改写历史记录。`BUNDLE_MANIFEST.json` 核验此次交付的全部文件。

## 重现与输入

原始公司 CSV、截图、完整 SQLite 仓库、虚拟环境、缓存和旧 UI 发布包未新增上传。28 参数大体积 `.npy` 输入缓存保留本地，其来源、大小和哈希见 `INPUTS_LOCAL_ONLY.json`；需要重跑这些历史训练时，按各版本源码重建缓存或提供同哈希输入。`real28/exports` 因未携带回放输入，属于权重/契约留档，不能直接作为完整看板部署包启动。

Booth2 的因果特征帧、产品/事件表与冻结评估函数作为训练快照保留；v5/v6 输入来源哈希在各自 provenance 中。各目录保留原复现命令与版本记录。不同代的 sklearn/PyTorch 环境分开创建，不将它们安装进看板运行环境。v5/v6 示例在仓库根目录运行，输出写新目录：

```bash
python training_research/booth2/model-v6-20261120/research/math_checks.py
python training_research/booth2/model-v5-20261120/research/train.py --data training_research/booth2/model-v5-20261120/data --out experiments/v5-new --jobs 8
python training_research/booth2/model-v5-20261120/research/finalize.py --data training_research/booth2/model-v5-20261120/data --run experiments/v5-new
python training_research/booth2/model-v6-20261120/research/train.py --data training_research/booth2/model-v5-20261120/data --out experiments/v6-new --jobs 6
```

学习循环仍是自动采集/候选训练、评估后人工发布；模板不启用任何 timer。已附训练所需历史索引/同步和模拟回归依赖；`learning_cycle_regression.py` 可以独立运行。涉及完整看板 HTTP 的 `electrode_regression.py` 仍需 Gitee 应用与完整回放输入。不能把本归档当作已接通现场的数据链路。

## 结果口径

28 参数风险网络和 GRU-16 是既有看板模型；v3/v4 是历史研究接入。v5/v6 未取得稳定全面改进，保留为研究候选；全量重拟合权重没有独立验证成绩。既有 LOCO/时间前向历史已被查看，不能称为新增盲测。RUL 是加工小时；独立预警采用墙钟时间，两种口径不混用。早期 21 参数模型仅基于模拟数据。

joblib/pickle 仅加载已核验哈希的可信研究产物。读取已有权重时使用相应版本环境；不要运行从未知来源上传的序列化模型。
