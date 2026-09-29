# Walking 低速干预训练前门禁记录（2026-09-30）

本轮只执行 `walking-low-speed-boost.json` 的训练前门禁，完成后按要求停止，未进入 50,000 次完整训练。8 个独立 run 使用 8 张 H20G，每卡一个 learner，W&B 使用已核对的 `oh-my-duck` online 项目。

## 结果

- MuJoCo × RSL-RL/SB3 的 4 个 run 在 `outputs/experiments/walking-low-speed-boost-20260930-02/` 完成 smoke、导出、CPU/BAM rehearsal、恢复检查、8192 环境容量检查和容量导出。
- Newton × RSL-RL/SB3 的 4 个 run 第一次被当前源码 fingerprint 对应的缺失 USD 阻止；用当前源码重新转换 `walk` 资产后，在 `outputs/experiments/walking-low-speed-boost-20260930-03-newton-gates/` 完成相同门禁。
- 4 个 MuJoCo 和 4 个 Newton run 的结构化状态均为 `prepared`。`smoke-rehearsal` 返回码 2 表示未训练策略的行为失败，属于允许的行为结果；smoke、恢复、容量和导出执行均返回 0。
- 没有启动 full training、周期预览或最终无推扰评估，因此不能从这次门禁推断低速命令已经改善，也没有产生可封装为 locomotion tool 的策略。

## 运行时修复

第一次尝试 `outputs/experiments/walking-low-speed-boost-20260930-01/` 的 8 个 run 都在导入 MuJoCo 时失败：主机系统 `libEGL.so.1.1.0` 为空。项目已有的 pinned OSMesa loader 没有传播给 campaign 子进程。提交 `b8c36b2` 将该 loader 路径和 NVIDIA EGL vendor 选择写入公共 headless 配置，并在 campaign worker 复制环境前调用；新增 headless/campaign 回归测试。

## 来源与边界

门禁使用源码提交 `b8c36b2`、官方 `microduck_rl@cfe1c2a`、`microduck@f0d934e`，资产转换使用 Isaac Sim asset environment，实际 solver 仍由后续 Newton 训练/评估阶段单独验证。当前结果只证明 pipeline gates 可执行；低速干预仍是待验证假设，控制组和 boost=1.0 组的完整训练、跨后端行为和 CPU/BAM 最终指标仍待后续独立运行。
