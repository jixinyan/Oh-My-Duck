# Walking 低速命令干预（2026-09-30）

## 依据

`docs/reports/rl-walking-command-deadzone-2026-09-30.md` 对保存的 MuJoCo/RSL Walking 导出做了受控命令扫描：`0.1 m/s` 的响应比例约 `0.3%`，而 `0.2–0.4 m/s` 已有 `0.717–0.781` 的前进响应。该结果指向低速命令的目标信号不足，不能证明所有命令都没有执行能力。

## 实现

新增显式干预 `low_speed_tracking_boost`，默认值为 `0`，不改变官方 task factory。非零值将官方 `track_linear_velocity` 包装为同一公式乘以一个命令条件因子：仅当线速度命令在 `0.01–0.2 m/s` 之间时增加权重，转向原地和零命令保持原值。干预上限为 `4.0`，并把参数写入训练 identity；恢复时不能切换干预。它只改变奖励信号，不改变命令范围、61/14 接口、PPO、BAM、Newton 或行为验收标准。

配对完整流程配置为 [`configs/experiments/walking-low-speed-boost.json`](../../configs/experiments/walking-low-speed-boost.json)，覆盖 Walking × MuJoCo/Newton × RSL-RL/SB3，控制组和 boost=1.0 组各 50,000 次更新、8,192 环境，并保留 smoke、导出、CPU/BAM rehearsal、恢复、周期视频和最终无推扰评估门禁。

## 当前状态

本轮已完成 8 个 run 的训练前门禁，但按最小目标在门禁后停止，没有启动完整训练。MuJoCo 四组在 `outputs/experiments/walking-low-speed-boost-20260930-02/` 准备完成；Newton 四组在补做当前 fingerprint 的 `walk` USD 后，于 `outputs/experiments/walking-low-speed-boost-20260930-03-newton-gates/` 准备完成。门禁中的未训练 CPU/BAM rehearsal 仍为行为失败，不能把它当作低速响应改善。只有完整配对结果改善 `0.1 m/s` 响应且不牺牲静止、转向、横向 RMSE、跨后端和 CPU/BAM 指标时，才考虑后续采用；详见[门禁记录](rl-walking-low-speed-gates-2026-09-30.md)。
