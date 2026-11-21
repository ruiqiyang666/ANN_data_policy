# 学校服务器服务模板（未启用）

对应 `docs/continuous-learning.md`。这些是 systemd **用户**服务，源码路径默认 `~/Zhang/Phase2-learning`，持久数据在 `~/Zhang/private/continuous_learning`。复制前需将已验证源码固定在该路径，并调整 Python 环境；历史索引导入现有项目依赖。

先配置 `~/.config/phase2-learning/history.env`、权限 600 的只读 key 文件和审核过的 `candidate-recipe.json`。训练 recipe 必须固定源码/镜像版本并产出评估报告，不包含部署操作。首次连接先执行 `collect --once`，对照源端水位、记录数、去重、重连；停止网页对同一库的同步写入。没有特征编译器和已确认标签时，周任务只报告等待。

模板经校验后再复制到 `~/.config/systemd/user/` 并执行 `systemctl --user daemon-reload`。确认上述前提后可启用 `phase2-history.service`、`phase2-history-snapshot.timer`、`phase2-candidate-train.timer`。不要启用模拟 API 或示例数据的服务。学校账号需已有 linger 才能在退出 SSH 后保持用户服务；由服务器管理员核对。用 `systemctl --user status`、`journalctl --user -u ...`、数据目录 `collector-status.json`、`gate.json` 与任务 `job.json` 检查进度。

本轮仅交付模板并测试逻辑，未向公司服务器发起采集，未安装 timers，未变更 18776。
