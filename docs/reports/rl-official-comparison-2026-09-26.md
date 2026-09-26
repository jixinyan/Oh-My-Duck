# 官方 RL 实现核对与训练预览修复（2026-09-26）

## Flat Walking 与 Flat StandUp 源码

本仓库固定的官方 `microduck_rl` 版本为 `29e887ecfbf5d37144759e5a9f8a176dfb83d547`，见 `configs/upstream.json`。直接读取该 Git commit，并与当前文件比较：

- `src/oh_my_duck/rl/tasks/walking/environment.py` 和官方 `microduck_velocity_env_cfg.py` 的任务配置保持一致。文件差异为项目模块路径、顶部说明，以及将 `MicroduckRlCfg` 移至 `src/oh_my_duck/rl/tasks/walking/ppo.py`。
- `src/oh_my_duck/rl/tasks/stand_up/environment.py` 和官方 `microduck_standup_env_cfg.py` 的任务配置保持一致。文件差异为项目模块路径、顶部说明，以及将 `MicroduckStandUpRlCfg` 移至 `src/oh_my_duck/rl/tasks/stand_up/ppo.py`。
- `src/oh_my_duck/robotics/microduck/actuators/friction_dr_bam.py` 与固定版本的官方 `friction_dr_bam.py` 文件内容相同。
- 两份移出的 PPO 配置仍使用 24 个环境步长、5 个学习轮次、4 个 minibatch、`adaptive` 学习率、0.01 目标 KL，以及启用 actor 和 critic 的观察归一化。配置值与固定版本相同。

官方 `develop` 的 `cb70b792312d559a4da09064d92009079671815f` 相对固定版本，在上述两份 Flat 任务文件、BAM、export、`pyproject.toml` 和 `uv.lock` 中没有文件差异。官方 MDP 文件新增的内容主要服务于 VelStand 与 rough terrain；Flat 任务仍使用原有配置。当前源码比较没有发现能够解释 Flat 任务学习失败的任务配置差异。此前的短程运行、重置状态和模型比较见 `docs/reports/official-baseline-audit-2026-09-08.md`；这些检查没有证明完整训练轨迹相同。

| 核查项目 | 官方固定版本 | 当前项目 | 已确认的意义 |
| --- | --- | --- | --- |
| MuJoCo 训练环境依赖 | MuJoCo 3.10.0、mujoco-warp 3.8.1、Warp 1.12.0、Torch 2.9.1；aarch64 锁定 cu129 构建 | 同版本；当前 x86_64 锁文件包含 Torch cu130 构建 | 基础包版本相同；两份 CUDA 构建属于不同平台，不能据此认定相同平台存在版本差异。 |
| Isaac/Newton 训练环境依赖 | 官方 Flat 任务运行于 MuJoCo 环境 | MuJoCo 3.8.0、mujoco-warp 3.8.0.3、Warp 1.13.0、Torch 2.10.0+cu130 | 求解器与依赖版本均发生改变，尚无完整训练等效性结论。 |
| 每次训练的环境数量 | 官方说明中的默认训练命令为 4096 | `jd-reproduction-20260923.json` 为 8192 | 每次更新采集的样本数量不同；尚未证明它导致行为问题。 |
| SB3 初始学习率 | 官方使用原生 RSL-RL PPO，初始学习率 0.001 | 实验配置中的 SB3 初始学习率 0.0001 | 两种 PPO 实现的更新路径及初始学习率均不同，不能直接推断学习效果。 |

上述包版本读取自官方与项目的锁文件。官方最近公开的 Walking `441tzs6d@3750` 和 StandUp `69u48n8l@9750` 可作为训练产物标识；当前账号读取这两组 W&B 运行均得到 403，无法取得对应训练参数和原生 checkpoint。官方 runtime 中已发布的 ONNX 已有历史验证记录。机器人 runtime 最新 `a9ec4b2079ef8ee7904014089c885bb07d57d63c` 增加了 API 2 的 LSTM 模型支持，仍保留 API 1 的 61 维观察、14 个 servo、50 Hz 接口；现有 feedforward 策略的接口无需因此改变。

用 Python `ast` 解析固定版本的 `tasks/mdp.py`、当前 `rl/mdp/*.py`，以及两份 Flat 任务配置。Flat Walking 的直接 `microduck_mdp` 引用在官方和当前源码均为同一组 22 个名称；将模块迁移引入的局部 import 规范化后，22 个定义的 AST 全部相同。Flat StandUp 的引用均为同一组 32 个名称；其中 31 个定义的 AST 相同。剩余的 `set_random_ground_state` 从官方直接写入 `qpos`、`qvel`，改为当前的 Entity root/joint 写入；先前在真实 MuJoCo 环境中核查过相同重置状态与随机数消耗，见上述基线审查报告。每个函数的行号、AST SHA-256、源码差异与规范化结果保存在 `.cache/rl-audit-20260926/mdp_ast_compare.json`。该文件位于忽略目录，仅作为本次本地诊断数据。

## 训练算法边界

MuJoCo/RSL 训练入口 `src/oh_my_duck/rl/learners/rsl_rl/train.py` 使用 `RslRlVecEnvWrapper`、任务自带的 `MicroduckOnPolicyRunner` 与原生 `runner.learn(..., init_at_random_ep_len=True)`。当前检查确认入口调用路径；完整训练行为仍需使用相同训练预算和行为评估验证。

SB3 是项目扩展。`src/oh_my_duck/rl/learners/sb3/train.py` 使用原生 SB3 PPO，actor 与 critic 分别读取对应观察组，并启用 SB3 `VecNormalize`；它以先前 rollout 记录的近似 KL 调整下一轮学习率。RSL 的学习率调整发生在 minibatch 更新中，两者的学习轨迹没有数值相同的保证。当前代码的 0.01 左右近似 KL 只能说明更新幅度，不能证明 Walking 与 StandUp 行为已经学会。

Isaac/Newton 通过 `src/oh_my_duck/rl/backends/isaac_newton/task_binding/environment.py` 保留官方任务管理器，并使用 Newton 的物理求解器。`src/oh_my_duck/rl/backends/isaac_newton/bam_actuator.py` 调用同一份 BAM 实现；`task_binding/sensors.py` 和 `task_binding/simulation.py` 提供接触、传感器与状态适配。这些适配层需要真实 GPU 的行为和物理比较才能判断训练等效性，本次本地源码检查没有得出等效结论。

Newton 任务类继承固定版本 `mjlab==1.3.0` 的 `ManagerBasedRlEnv.step`、`reset` 和 `_reset_idx`。控制步骤仍按任务 `cfg.decimation` 执行，每个子步骤调用 action、scene 写入、simulation step、scene update；随后增加 episode 计数、计算 termination、按 `step_dt` 计算 reward、重置结束的环境、刷新状态和传感器、计算观察。`step_dt` 使用任务中的物理时间步长乘以 `cfg.decimation`，Flat 配置对应 0.02 秒。Newton 的 `NewtonSimulation.step` 调用原生 Newton 求解器，每个子步骤更新原生 scene，项目 `NewtonScene` 继承 mjlab 的 sensor update。此次静态检查没有发现上述步骤顺序和 reward 时间系数发生改变；接触与传感器数值仍需要真实 GPU 比较。

## 独立训练预览

`src/oh_my_duck/rl/experiments/preview.py` 现在为每个训练 worker 在自身运行目录创建 `preview-campaign/campaign.json`，写入对应任务与 `run_output`。预览进程根据这份元数据读取该 worker 的 `result.json`，并继续从完整的 SB3 bundle 或 RSL checkpoint 选择预览来源。同一父目录的独立 worker 各自读取自身元数据；普通 campaign CLI 仍读取原有的 campaign 文件。

真实子进程测试覆盖两个独立 worker 共用父目录、父目录存在普通 campaign 文件、普通 campaign CLI。另一个测试用 Gymnasium Pendulum 和原生 SB3 PPO 完成一次训练更新，再用 `model.save` 与 `VecNormalize.save` 创建可读取的 checkpoint，确认独立运行目录的定位。运行 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .envs/review/bin/python -m pytest tests/test_rl_preview.py -q -k 'independent_worker or campaign_preview_cli' --basetemp .cache/preview-pytest-20260926-d`，新增 3 项测试通过；`git diff --check` 通过。测试没有运行 GPU 训练，也没有产生新的 Walking 或 StandUp 行为结论。

## 原生 RSL-RL 恢复计数

原生 RSL-RL 的 `model_5000.pt` 将最近一次更新标签保存为 `iter=5000`；同一 checkpoint 的 `infos.env_state.common_step_counter=120024`，对应已经完成 `5001` 次、每次 24 个环境步长的更新。seed42 与 seed43 的真实完整训练 checkpoint 均符合该值。恢复预算按环境计数器计算为 `15000-5001=9999` 次，并将下一次更新标签设置为 `5001`。当前账号读取官方 W&B 运行受 403 限制；这个恢复修复依据项目自身的原生 checkpoint 与 RSL-RL 保存流程。

历史 seed42 `resume-check/model_4.pt` 的标签为 `4`，环境计数器为 `144`，已经完成 `6` 次更新。它包含一次重复使用标签的历史恢复，因此恢复计算允许已完成次数超过标签加一，并以环境计数器确定下一次更新标签。`recovery.validate_checkpoint` 对真实 seed42、seed43 `model_5000.pt` 与当前恢复配置完成 SHA-256、来源阶段、原生进度核查，两组均返回剩余 `9999` 次更新。

在 `jd_B300` 的 GPU1，使用从 `4253ee7` 复制的独立源码目录和本次修复文件，以真实 seed42 `resume-check/model_4.pt` 连续运行两次 Isaac/Newton StandUp 训练，每次 64 个环境、一次 PPO 更新，`WANDB_MODE=offline`。两次训练均使用 `--backend isaac-newton --env.scene.num-envs 64 --agent.max-iterations 1 --agent.seed 42 --agent.save-interval 1 --agent.upload-model False --agent.resume True --gpu-ids '[1]'`；第一次指定 `--agent.load-run source-resume-check --agent.load-checkpoint model_4.pt --agent.run-name audit-native-resume-pass1`，第二次指定 `--agent.load-run 2026-09-27_02-48-09_audit-native-resume-pass1 --agent.load-checkpoint model_6.pt --agent.run-name audit-native-resume-pass2`。两次训练进程均以退出码 0 完成。

版本管理中的 `scripts/verify_native_rsl_resume.py` 读取原始 checkpoint 与两次实际运行目录。其实际执行命令为 `.envs/mujoco/bin/python scripts/verify_native_rsl_resume.py --source-checkpoint logs/rsl_rl/microduck_stand/source-resume-check/model_4.pt --first-run logs/rsl_rl/microduck_stand/2026-09-27_02-48-09_audit-native-resume-pass1 --second-run logs/rsl_rl/microduck_stand/2026-09-27_02-50-38_audit-native-resume-pass2`，工作目录为独立源码目录，`OMD_PROJECT_ROOT` 与 `PYTHONPATH` 指向同一目录。脚本以退出码 0 完成，结果如下：

| checkpoint | 保存标签 | 已完成更新 | 环境计数器 | optimizer step |
| --- | ---: | ---: | ---: | ---: |
| 历史 `model_4.pt` | 4 | 6 | 144 | 120 |
| 第一次恢复 `model_6.pt` | 6 | 7 | 168 | 140 |
| 第二次恢复 `model_7.pt` | 7 | 8 | 192 | 160 |

脚本还核对两次运行的 `run.json` 恢复来源、保存的更新次数，以及 actor、critic 两组观察归一化计数持续增加。这个短程验证确认恢复时的标签、预算计算、环境计数器和原生训练状态接续；完整 15000 次训练与最终行为仍由正式恢复运行验证。
