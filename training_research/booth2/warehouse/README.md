# Booth2 数据仓库（booth2-warehouse-1，2026-11-18）

把 Booth2 等离子喷涂的全部历史数据整理成一套结构：原始 1 Hz 读数、清洗后的产品/周期/事件标签、
模型输入原点与目标、模型登记与预测台账。同一套逻辑表有两种落地：

- **现在可用**：`db/booth2.sqlite`（Python 标准库即可读）+ `lake/`（Parquet）。全部历史已载入。
- **上线用**：`schema/postgres_timescale.sql`（PostgreSQL 16 + TimescaleDB）。**已写好，未执行，未启动任何数据库。**

原有目录（`Real-time Data`、`.runtime/instrument`、`results` 等）没有改动；这里是整理后的独立副本。

## 目录

| 路径 | 内容 |
|---|---|
| `db/booth2.sqlite` | 全量库，29 张表 + 7 个视图，约 819 MB |
| `lake/raw_1hz/date=YYYY-MM-DD/part-0.parquet` | 原始 1 Hz，按日分区，71 天 |
| `lake/curated/*.parquet` | 17 张清洗/标签/模型表 |
| `schema/postgres_timescale.sql` | 生产库 DDL（超表、连续聚合、追加式台账、保留/压缩建议） |
| `schema/compose.timescaledb.yaml` | TimescaleDB 容器配置，**未启动**；端口、口令需先确认 |
| `schema/sqlite_schema.sql` | 由 SQLite 库导出的实际表结构（自动生成） |
| `ingest/load_history.py` | 从源数据重建以上全部内容，结果确定（同输入同哈希） |
| `ingest/opcua_collector.py` | OPC UA 订阅采集器；**未部署，未连接过任何现场或 Kepware** |
| `ingest/tags.json`、`collector.example.json` | 32 个标签登记（28 实测 + 4 预留）、采集器示例配置 |
| `manifest.json`、`verify.py` | 每个文件的 sha256/行数；`python3 verify.py` 复核 32 项 |

## 时间口径（重要）

历史数据的时间是 **HMI 导出的设备本地时钟，没有时区记录，本库不补时区**。`ts` 列 = 把这个本地时钟
当作 UTC 数出来的秒数（与源项目 `anchor_id = 'A' + ts` 一致），文本列统一为 `YYYY-MM-DD HH:MM:SS`
（`training_data` 原为 `DD MM YYYY HH:MM:SS`）。实时数据走 OPC UA，时间戳是 UTC 时刻，存入单独的
`live_sample`（PostgreSQL 中为 `sample`，`timestamptz`）。**现场确认 HMI 时区之前，历史与实时不能按时间直接拼接。**

缺失一律为 NULL，没有补零、没有插值。

## 覆盖范围

| 数据 | 范围 | 说明 |
|---|---|---|
| 原始 1 Hz（28 路） | 2026-08-15 09:00:00 — 11-01 16:59:58 | 5,909,066 秒，1,660 个小时文件 |
| 清洗标签与原点 | 2026-09-12 07:14:20 — 11-14 14:01:10 | 仓库 `ruiqiyang666/ANN_data_policy` 提交 c49f6c9 |
| 两者重叠 | 09-12 — 11-01 | 63,259 个原点可回查原始秒级行，9 路读数逐点相同 |
| **只有清洗结果** | **11-02 — 11-14** | **17,976 个原点（L010 大部分、L011—L013 全部）没有原始 1 Hz** |

原始数据的三段长缺口：09-29 18:15 → 10-04 14:00、10-17 20:00 → 10-19 07:00、10-23 18:00 → 10-27 07:00；
另有 5,862 处单秒漏采（`ingest_gap`）。8/15—9/11 只有原始数据，没有标签。

## 表（SQLite 行数）

| 表 | 行数 | 含义 |
|---|---:|---|
| `asset` / `tag` | 1 / 32 | 设备与标签登记：字段名、单位、Kepware 名、预期 OPC UA 节点、扫描周期、死区 |
| `source_file` | 1,660 | 每个原始 CSV 的大小、sha256、行数、起止 |
| `sample_1hz` | 5,909,066 | 原始读数，宽表，一秒一行 |
| `sample_1min` / `raw_day` | 98,593 / 71 | 分钟均值、按日覆盖 |
| `ingest_gap` / `quality_event` | 5,879 / 51 | 缺口；不可解析单元格（50 个，置 NULL）与 1 个空文件 |
| `electrode_cycle` | 14 | 电极周期 L000—L013（v2，11/17 纠正后） |
| `maintenance_event` | 12 | 换枪登记/现场确认，时间是候选范围 |
| `product` | 13,181 | 全量产品：13,046 正常 / 51 失败 / 84 未判 |
| `target_event` | 12 | 连续失败事件，其中 10 个电极末期（EOL） |
| `anchor` / `target` | 81,235 / 81,235 | v2 加工原点（10 秒）与 4 个监督目标 |
| `anchor_feature_28` / `target_v1_legacy` | 85,179 / 85,179 | v1 原点：28 路当秒值 + 60 秒均值；v1 目标（含各类首次失败秒数、RUL） |
| `model_registry` | 12 | v2 的 12 个候选，选中 `gru_16` |
| `prediction` | 974,820 | 追加式预测台账（12 × 81,235），触发器禁止改/删 |
| `alarm` / `event_result` | 43 / 120 | v2 冻结评估的报警与逐事件结果 |
| `label_sample`、`product_failure_event_v1`、`electrode_cycle_v1_legacy`、`split_summary*` | 353 / 124 / 14 / 3+3 | 审阅子集与 v1 口径原样留存 |
| `live_sample`、`collector_heartbeat`、`ingest_event` | 0 | 实时采集表，已建好，空 |

视图：`v_cycle_summary`、`v_day_coverage`、`v_eol_event`、`v_anchor_supervised`、`v_prediction`、`v_tag_opcua`、`v_sample_1hz`。

```sql
SELECT * FROM v_cycle_summary;                       -- 每个电极周期：终点、产品数、原点数、是否有原始数据
SELECT * FROM v_day_coverage WHERE coverage != 'both';
SELECT ts_local, Voltage, Current FROM v_sample_1hz WHERE ts BETWEEN 1793172600 AND 1793176320;  -- 10/28 末期
```

## 两套来源不一致的地方（已分开保存，未合并）

v2 `model_v2_GRU/data` 为准；`training_data`（v1）保留为 `*_v1_legacy` 表或 `split_v1` / `source_*` 列。

1. **终点定义不同**：v2 的 `eol_endpoint` = 最终连续失败序列**第一件开始加工**；v1 = **第二件失败结束**，晚 153—157 秒。
   因此 `target_v1_legacy.rul_strict_seconds` 比 `anchor.remaining_seconds` 大同样的秒数。
2. **划分不同**：v1 为 8/3/3 个周期（L000—L007 训练），v2 为 5/3/6（L000—L004 训练、L005—L007 验证、L008—L013 测试）。
   38,648 个原点、6,287 件产品的划分标签因此不同。
3. **L012（11/10）**：v1 记为 `sensor_reset_unconfirmed`、无严格终点；v2 按 11/17 现场确认改为 `natural_eol`。v1 的 RUL 在 L012 全为空。
4. **删失口径**：v2 把预防性更换（L007、L010）和损坏更换（L011）记为右删失，v1 没有。
5. **原点数不同**：v1 有 85,179 个，v2 为 81,235 个；多出的 3,944 个全部在产品开始后 0—2 秒，v2 要求开工确认 ≥3 秒。
   共有原点的 9 路读数和 60 秒电压统计完全相同。
6. `electrode_cycle.failure_products` 只统计可监督产品：N000499（9/14，电压失败，现场时间对不齐）失败但不计入，全量失败 51 件、可监督 50 件。
7. 60 秒电压统计 = 含原点当秒在内的最近 60 个秒级行的均值与**样本标准差（ddof=1）**；用原始数据复算，99.99% 的原点一致。
8. `20261009/` 文件夹补上了主目录缺的 10-09 20:00 — 10-10 06:59（39,548 秒，其中有 377 个原点）；与主目录重叠的 48,162 秒读数完全相同，已去重。**服务器的 `Real-time Data` 没有这 11 个小时**，本库有。
9. 原始数据里有 108 秒电压 >100 V（最大 544 V）等离群读数，原样保留；没有一个落在原点上。

## 实时链路（设计，尚未接通）

```text
Kepware (OPC UA Server) ──订阅──> opcua_collector ──> 本地缓冲 spool ──> sample 超表
                                                                        │
                        换枪确认 ──> electrode_cycle            1 Hz 网格 + 加工确认
                                                                        ↓
              prediction（追加式）<── 模型推理 <── anchor（10 秒原点 + 60 秒统计）
                     ↓
                   alarm ──> 实时预测页
```

- 采集器先把每条通知写入本地 SQLite 缓冲，再转发入库；数据库中断或采集器重启都不丢数据，
  `(tag_id, source_timestamp)` 幂等，重发无害。坏质量读数存状态码、数值置 NULL。
- OPC UA 只上报变化。采集器每 10 秒写一条心跳，用来区分“数值没变”和“没采到”。
- 原点生成规则须与历史一致：电流 >100 A 且功率 >5 kW、转台 |转速| >50 rpm、开工 ≥3 秒、整 10 秒。
  只用“带电 + 转台”两条在历史上可召回 100% 的原点、精确率 92.8%，其余来自离线的产品资格判定，实时侧还需实现。
- 自测：`python ingest/opcua_collector.py --self-test`，在本机进程内起一个 asyncua 服务器，10 项全部通过
  （含模拟数据库中断后补发、坏质量、字符串标签、未知节点）。**这不是现场验证。** PostgreSQL 写入路径没有连过真实数据库。

## 上线前必须在现场确认

| 项目 | 现在的假设 |
|---|---|
| OPC UA 端点 | `opc.tcp://<Kepware 主机>:49320`，占位 |
| 命名空间与节点 | `ns=2;s=HA.Booth2.Plasma2.<标签>`；建议按命名空间 URI 解析，不写死索引 |
| 标签名 | `Press._Hopper_1/2` ↔ `Press_Hopper_1/2`（部署说明已给出）；两个水温标签去掉末尾的点是推测 |
| 安全策略与账号 | Basic256Sha256 + SignAndEncrypt + 只读账号，证书需在 Kepware 侧信任 |
| 时钟与时区 | HMI 历史时钟的时区、PLC/Kepware/服务器是否对时 |
| 扫描周期与死区 | 每个标签 1000 ms、死区 0；要确认是否整表同步刷新 |
| 质量码 | 现场实际出现哪些状态码，陈旧上限（现设 5 秒） |
| 换枪重置信号 | 由谁、用什么信号确认换枪（操作确认 / Gun_ID / Gun_Hours）；不能凭电压回升猜 |
| Gun_ID、Gun_Hours、Ignitions | 已预留标签；计数何时增加、何时清零 |

## 为什么生产库选 PostgreSQL + TimescaleDB

| 方案 | 优点 | 不足 |
|---|---|---|
| **PostgreSQL + TimescaleDB** | 时序与标签/产品/台账在同一个库里用 SQL 关联；超表压缩、连续聚合、保留策略现成；约束与触发器能保证幂等和台账只增 | 需要自己写采集器（已写） |
| Kepware DataLogger → ODBC | Kepware 自带，配置即可 | 另需授权；按触发整表写入，丢掉每个点的源时间戳和质量码；断线缓冲能力有限 |
| Kepware IoT Gateway → MQTT | 推送式，易于跨网段 | 多一个 Broker 和一个落库服务；仍需同样的库表设计 |
| InfluxDB | 时序写入和降采样很强 | 产品、周期、事件、台账这类关系数据要另放一个库，训练取数要跨库拼接 |

## 重建与复核

```bash
python ingest/load_history.py --repo <ANN_data_policy 克隆> --raw "<Real-time Data>" --raw-extra "<20261009>" --out .
python3 verify.py          # 标准库即可；--quick 跳过 Parquet 哈希
```

`Voltage Sample`（Booth1 参考）按要求未纳入。本库只读使用即可；需要新版本时重建到新目录，不要原地改。
