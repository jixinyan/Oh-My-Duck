# 官方 RL 实现核对与训练预览修复（2026-09-26）

## Flat Walking 与 Flat StandUp 源码

本仓库固定的官方 `microduck_rl` 版本为 `29e887ecfbf5d37144759e5a9f8a176dfb83d547`，见 `configs/upstream.json`。直接读取该 Git commit，并与当前文件比较：

- `src/oh_my_duck/rl/tasks/walking/environment.py` 和官方 `microduck_velocity_env_cfg.py` 的任务配置保持一致。文件差异为项目模块路径、顶部说明，以及将 `MicroduckRlCfg` 移至 `src/oh_my_duck/rl/tasks/walking/ppo.py`。
- `src/oh_my_duck/rl/tasks/stand_up/environment.py` 和官方 `microduck_standup_env_cfg.py` 的任务配置保持一致。文件差异为项目模块路径、顶部说明，以及将 `MicroduckStandUpRlCfg` 移至 `src/oh_my_duck/rl/tasks/stand_up/ppo.py`。
- `src/oh_my_duck/robotics/microduck/actuators/friction_dr_bam.py` 与固定版本的官方 `friction_dr_bam.py` 文件内容相同。
- 两份移出的 PPO 配置仍使用 24 个环境步长、5 个学习轮次、4 个 minibatch、`adaptive` 学习率、0.01 目标 KL，以及启用 actor 和 critic 的观察归一化。配置值与固定版本相同。

官方 `develop` 的 `cb70b792312d559a4da09064d92009079671815f` 相对固定版本，在上述两份 Flat 任务文件、BAM、export、`pyproject.toml` 和 `uv.lock` 中没有文件差异。官方 MDP 文件新增的内容主要服务于 VelStand 与 rough terrain；Flat 任务仍使用原有配置。当前源码比较没有发现能够解释 Flat 任务学习失败的任务配置差异。此前的短程运行、重置状态和模型比较见 `docs/reports/official-baseline-audit-2026-09-08.md`；这些检查没有证明完整训练轨迹相同。

## 训练算法边界

MuJoCo/RSL 训练入口 `src/oh_my_duck/rl/learners/rsl_rl/train.py` 使用 `RslRlVecEnvWrapper`、任务自带的 `MicroduckOnPolicyRunner` 与原生 `runner.learn(..., init_at_random_ep_len=True)`。当前检查确认入口调用路径；完整训练行为仍需使用相同训练预算和行为评估验证。

SB3 是项目扩展。`src/oh_my_duck/rl/learners/sb3/train.py` 使用原生 SB3 PPO，actor 与 critic 分别读取对应观察组，并启用 SB3 `VecNormalize`；它以先前 rollout 记录的近似 KL 调整下一轮学习率。RSL 的学习率调整发生在 minibatch 更新中，两者的学习轨迹没有数值相同的保证。当前代码的 0.01 左右近似 KL 只能说明更新幅度，不能证明 Walking 与 StandUp 行为已经学会。

Isaac/Newton 通过 `src/oh_my_duck/rl/backends/isaac_newton/task_binding/environment.py` 保留官方任务管理器，并使用 Newton 的物理求解器。`src/oh_my_duck/rl/backends/isaac_newton/bam_actuator.py` 调用同一份 BAM 实现；`task_binding/sensors.py` 和 `task_binding/simulation.py` 提供接触、传感器与状态适配。这些适配层需要真实 GPU 的行为和物理比较才能判断训练等效性，本次本地源码检查没有得出等效结论。

## 独立训练预览

`src/oh_my_duck/rl/experiments/preview.py` 现在为每个训练 worker 在自身运行目录创建 `preview-campaign/campaign.json`，写入对应任务与 `run_output`。预览进程根据这份元数据读取该 worker 的 `result.json`，并继续从完整的 SB3 bundle 或 RSL checkpoint 选择预览来源。同一父目录的独立 worker 各自读取自身元数据；普通 campaign CLI 仍读取原有的 campaign 文件。

真实子进程测试覆盖两个独立 worker 共用父目录、父目录存在普通 campaign 文件、普通 campaign CLI。另一个测试用 Gymnasium Pendulum 和原生 SB3 PPO 完成一次训练更新，再用 `model.save` 与 `VecNormalize.save` 创建可读取的 checkpoint，确认独立运行目录的定位。运行 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .envs/review/bin/python -m pytest tests/test_rl_preview.py -q -k 'independent_worker or campaign_preview_cli' --basetemp .cache/preview-pytest-20260926-d`，新增 3 项测试通过；`git diff --check` 通过。测试没有运行 GPU 训练，也没有产生新的 Walking 或 StandUp 行为结论。
