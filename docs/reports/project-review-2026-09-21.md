# 项目检查 · 2026-09-21

## 检查范围

检查的源码版本为 `cb3c994`。本次阅读 README、开发说明、当前状态、服务器迁移说明、近期训练报告，以及训练调度、检查结果复用、checkpoint 恢复、策略来源检查、行为评分和应用接口的相关实现。

本次执行环境为 macOS，使用 Python 3.12.12 运行 CPU 检查。GPU 训练和仿真结果依据仓库中的历史报告；本次没有重新执行这些实验。旧服务器的 checkpoint、视频和原始轨迹没有包含在当前仓库中。

## 已确认的问题

### P1：Walking 评分允许持续运动通过停止场景

位置：`src/oh_my_duck/rl/evaluation/protocols.py`，`EvaluationProtocol.score`，第 28–53 行。

评分针对非零命令计算响应比例，停止命令只参与整个测试过程的 RMSE。当前 RMSE 上限无法充分约束停止时的运动，也无法充分约束前进时的转向。

对现有评分函数执行数学反例：保持身体高度 0.115 m、倾角 0，整个 14 秒过程的速度始终为 `vx=0.06 m/s`、`vy=0`、`yaw_rate=0.25 rad/s`。无论命令要求前进、停止还是转向，输入速度均保持不变。

函数返回：

| 字段 | 结果 |
|---|---|
| `success` | `true` |
| `twist_rmse` | `[0.05503245795502379, 0.0, 0.25]` |
| 前进响应比例 | 0.6 |
| 转向响应比例 | 0.5 |

这项检查验证评分逻辑，输入没有来自机器人运行。它证明持续运动可以通过当前评分，不表示已有策略产生了这种轨迹。

影响：同一个评分函数用于原生仿真评估和 CPU/BAM 回放，停止失败的策略可能被判定为通过。

修复要求：为每个命令阶段定义独立指标，明确转换期间允许的响应时间、停止阶段的速度与转速限制，以及前进或转向阶段其他方向的运动限制。各项指标必须进入最终通过判定。

### P1：复用训练前检查时，没有检查 `rl/mdp` 的源码变化

位置：`src/oh_my_duck/rl/experiments/preparation.py`，`CRITICAL`，第 8–12 行；使用位置为第 23、50 行。

`validate_preparation` 和 `reuse_smoke` 使用 `CRITICAL` 限定 Git 差异检查范围。当前列表没有包含 `src/oh_my_duck/rl/mdp`。该目录中的 13 个受版本管理的文件均未被覆盖，其中包括观测、奖励、课程、reset 事件和终止条件实现。

例如，只修改 `rl/mdp/observations.py`，同时保留相同的任务配置，当前源码差异检查仍允许复用旧检查结果。`worker.py` 在复用通过后会继续启动完整训练。

影响：训练关键行为已经改变时，旧的启动检查仍可能被当作当前版本的有效依据。

修复要求：检查范围包含完整的训练与评估依赖，并验证仅修改观测或奖励实现时，旧检查结果会被拒绝。

### P2：SB3 恢复时可能加载另一个 checkpoint

位置：`src/oh_my_duck/rl/experiments/worker.py`，第 75–77 行；恢复调用位于第 134–140 行。

`validate_checkpoint` 根据配置指定目录中的 `model.zip` 和 `run.json` 检查 hash、进度并计算剩余更新次数。`train` 收到该目录后，只要其中存在 `checkpoints/step_*`，就会改用最后一个周期保存目录；SB3 分支没有使用传入的 `resume_checkpoint`。

触发条件：恢复配置指向一个完整训练输出目录，该目录同时保存最终 `model.zip` 和周期 checkpoint。

例如，指定文件完成 100 次更新，最后一个周期 checkpoint 完成 90 次更新，目标为 200 次更新。恢复校验计算还需 100 次更新，训练却从 90 次开始，最终只有 190 次，同时使用了与已校验文件不同的模型和 normalizer。

这个问题通过调用路径与保存逻辑确认；本次没有运行完整 GPU 恢复流程。配置直接指向没有嵌套 `checkpoints` 的周期保存目录时，不触发这一分支。

修复要求：恢复训练使用已经校验的准确路径，加载后核对模型、normalizer、时间步数和剩余预算。

## 当前项目进展

| 部分 | 仓库提供的证据 |
|---|---|
| RL 工程 | MuJoCo / Newton 与 RSL-RL / SB3 的训练、恢复、导出、回放和打包路径已存在；历史报告记录相关运行结果 |
| Walking | 最近报告记录 231 次 checkpoint 预览，完整行为验收通过数量为零；这些预览使用同一个评估 seed |
| StandUp | Newton RSL 的已保存策略在两个原生后端 seed 42 均通过四种姿态；CPU/BAM 的完整姿态测试在 17 个 seed 中通过 14 个 |
| 机器人应用 | `ToolCatalog` 和 JSONL 记录器已有实现；执行后端、技能运行、Harness、语音和感知主要为接口定义 |
| 服务器迁移 | 源码、配置和依赖锁文件已进入仓库；历史训练文件保存在旧文件系统 |
| 真机 | 文档记录设备尚未可用，真机验收尚未完成 |

进展依据：`docs/implementation-status.md`、`docs/server-migration-2026-09-21.md`、`docs/reports/rl-walking-policy-assessment-2026-09-14.md` 和 `docs/reports/rl-causal-replay-2026-09-13.md`。

## 本次执行的检查

- `tests/test_protocol.py`、`tests/rl/sb3/test_checkpoint_progress.py`、`tests/rl/tasks/test_evaluation_protocols.py`、`tests/rl/tasks/test_motion_diagnostics.py`：共 16 项测试通过。
- `omd tasks` 与 `omd frameworks`：执行成功。
- `CRITICAL` 与受版本管理的 MDP 文件路径比较：确认 13 个文件均未覆盖。
- Walking 评分数学反例：确认持续运动被判定通过。
- logo：PNG 完整解码通过，尺寸为 1254 × 1254；复制前后的 SHA-256 一致，README 引用指向项目内文件。
- `git diff --check`：通过。

本次项目修改包括 README 中的 logo、原始 PNG 文件和本检查报告。以上三个代码问题保持待修复状态。
