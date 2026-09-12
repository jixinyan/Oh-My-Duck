# RL 任务目录

更新于 2026-09-12；依据本项目 `configs/tasks.json`，不是联网推断的官方
最新任务集合。框架使用固定官方版本，任务/MDP/机器人资产由本项目拥有。

目前注册 **33 个任务入口、13 个任务家族**。默认配方可选择原生 RSL-RL
或原生 SB3；注册和可构建不代表已经训练出有效策略。完整验收仍聚焦
Flat Walking 与 Flat StandUp。只有这两个入口目前绑定了 Newton 后端、
专用行为评估协议和策略打包配置；其余入口当前绑定 MuJoCo，需要补齐各自
的训练/导出/行为/CPU 演练验收，才能列为可交付能力。

| 任务家族 | 代表任务 ID | 含地形/Backlash 的入口数 | 当前后端绑定 |
|---|---|---:|---|
| 平地/粗糙地面速度跟踪 | `Mjlab-Velocity-Flat-MicroDuck` | 4 | isaac-newton, mujoco |
| 站立姿态与速度控制 | `Mjlab-VelStand-Flat-MicroDuck` | 4 | mujoco |
| 从坐姿、俯卧、仰卧起身 | `Mjlab-StandUp-Flat-MicroDuck` | 4 | isaac-newton, mujoco |
| 按命令坐下与站起 | `Mjlab-SitStand-Flat-MicroDuck` | 4 | mujoco |
| 低头触地/拾取动作 | `Mjlab-GroundPick-Flat-MicroDuck` | 4 | mujoco |
| 踢球 | `Mjlab-BallKick-Flat-MicroDuck` | 2 | mujoco |
| 轮脚行走 | `Mjlab-Velocity-Flat-MicroDuck-Rollers` | 2 | mujoco |
| 轮脚摆动滑行 | `Mjlab-Velocity-Swizzle-MicroDuck` | 2 | mujoco |
| 轮脚蹲伏 | `Mjlab-RollerCrouch-Flat-MicroDuck` | 2 | mujoco |
| 轮脚斜坡 | `Mjlab-RollerSlope-Flat-MicroDuck` | 2 | mujoco |
| 轮脚起身 | `Mjlab-RollerStandUp-Flat-MicroDuck` | 1 | mujoco |
| 旋转动作 | `Mjlab-Spin-Flat-MicroDuck` | 1 | mujoco |
| 翻滚动作 | `Mjlab-Roulade-Flat-MicroDuck` | 1 | mujoco |

Backlash 是同一任务的关节回差变体，不是新的技能。轮脚任务需要对应的
轮脚模型；无真机阶段只能声明仿真结果。`motor_testbench` 是诊断模块，
不在这 33 个训练入口内。

## 当前能证明什么

- Newton/RSL Flat StandUp 最终策略在 seed=42 的 Newton、MuJoCo 和 CPU/BAM
  四姿态测试均通过；最终策略的多 seed 稳定性正在安排诊断。
- 其余代表组合尚未完成有效策略验收。SB3 的数值更新稳定性修复不等于
  起身/行走已学成，Walking 后期还有行为退步。
- 新任务的合理后续顺序：先把已有代表任务的学习问题定位，再考虑
  SitStand、VelStand 等共享基本姿态控制的任务，之后才扩展 GroundPick、
  BallKick、Roulade 或轮脚任务。此顺序是基于现有代码依赖的开发建议，
  不是新增全部任务训练的承诺。

## 加入任务的接口

1. 在 `src/oh_my_duck/rl/tasks/<family>/` 中组织 `environment.py` 与 `ppo.py`，
   复用共享机器人/BAM/MDP；新奖励和 actor 配置直接在拥有的源码中修改。
2. 在 `configs/tasks.json` 注册环境工厂、原生 PPO 配置、runner 与后端
   runtime；SB3 actor 配置通过 `policy_configs` 明确提供。
3. 补齐 task-specific `evaluation` 与 `policy_package`，显式绑定 Newton
   时验证 Newton 物理与任务语义；未绑定后端应明确拒绝，不能回退到别的后端。
4. 通过 64 环境/5 更新、奖励与状态审计、归一化导出数值对齐、原生恢复、
   行为视频及 CPU/BAM 演练，之后再安排完整训练并更新成熟度。

官方 61D actor / 14 servo / HOME / 50 Hz 合约不随任务变化。在线 Harness
和硬件接入仍在后续 scope，本轮不扩展。
