# 训练检查与恢复验证 · 2026-09-21

## Walking 验收标准

`EvaluationProtocol.score` 使用 `scoring_version=3`。14 秒测试包含静止、前进、停止、转向、停止五个阶段，控制频率为 50 Hz。

每个阶段允许 0.5 秒转换时间，其后检查所有连续 0.5 秒窗口，窗口每次前进一个时间步，覆盖阶段末尾。阶段必须具有至少一个完整评分窗口。

| 检查项目 | 通过条件 |
|---|---|
| 前进或转向 | 每个窗口的平均速度与对应非零命令之比位于 `[0.5, 1.5]` |
| 运动阶段的零命令方向 | 每个窗口的平均速度绝对值不超过 `vx=0.02 m/s`、`vy=0.02 m/s`、`yaw_rate=0.1 rad/s` |
| 静止与停止 | 每个窗口三个方向的 RMS 分别不超过 `0.02 m/s`、`0.02 m/s`、`0.1 rad/s` |
| 全程跟踪 | 三个方向的 RMSE 分别不超过 `0.1 m/s`、`0.1 m/s`、`0.5 rad/s` |
| 身体姿态 | 全程高度至少 `0.065 m`，倾角不超过 `π/3` |
| 完整性 | 运行完成、全部阶段通过；输入尺寸一致且全部数值有限 |

这些数值定义项目的行为验收条件，尚未经过真机测量。运动阶段按窗口平均值判断持续运动，停止阶段使用 RMS 计入往复运动。全程 RMSE 和身体姿态检查包含转换期间。

评分输出保存阶段起止时间步、实际评分起点、命令、窗口数量、各方向指标和通过结果；`command_response` 保存转换时间之后的阶段平均响应，最终判定使用全部窗口。

## 训练前检查复用

`validate_preparation` 与 `reuse_smoke` 共用 `validate_critical_inputs`，检查范围为：

- `src/oh_my_duck` 全部源码，包括 MDP、机器人配置、训练、评估和公共基础模块。
- `environments` 中的环境配置与依赖锁文件。
- `configs` 全部配置，以及 `pyproject.toml`、`omd.py`。

检查从原检查记录的 `source_commit` 比较到当前工作目录，同时检查未被 Git 跟踪且未被忽略的文件。范围内已经提交、暂存、尚未暂存、新增和删除的变化均要求重新执行检查。仅修改 `docs` 允许复用。缺失来源 commit 时立即报告错误。

## SB3 恢复流程

`resume.run` 指定包含 `model.zip`、`vecnormalize.pkl`、`run.json` 的准确目录，`resume.checkpoint` 必须为 `model.zip`。最终训练目录和具体的 `checkpoints/step_N` 目录均可直接指定。

Campaign 校验和训练入口共用文件 hash 检查。训练入口加载该目录的原生模型与 normalizer，通过 `restore_progress` 核对模型时间步数与 metadata，并恢复课程进度。训练结束时检查实际时间步数等于起始值加上声明的训练预算，随后保存输出。

CPU 集成测试在 Gymnasium `Pendulum-v1` 中运行原生 PPO，产生实际模型、optimizer 与 VecNormalize 数据：

| 指定目录 | 恢复时间步数 | 继续更新次数 | 最终时间步数 |
|---|---:|---:|---:|
| 最终训练目录 | 72 | 2 | 120 |
| 明确指定的周期目录 | 48 | 3 | 120 |

每次更新采集 24 个时间步。最终模型与周期模型同时存在，测试通过 Campaign 使用的参数构造函数选择目录，然后实际加载并继续训练。加载后的模型参数、optimizer 数据、normalizer 统计量和更新次数均经过断言检查。文件缺失、文件 hash 不符、checkpoint 名称错误或 metadata 进度不符时报告错误。

## 已执行验证

环境：macOS、Python 3.12.12、PyTorch 2.9.1、Stable-Baselines3 2.7.1。PyTorch 和 SB3 版本与项目 MuJoCo 环境声明一致。

以下测试共 **53 项通过**：

- `tests/rl/tasks/test_evaluation_protocols.py`：Walking 数学输入，包括恒定运动、停止时移动、其他方向持续运动、超速、转换时间、持续响应、停止振荡、末尾样本和无效数据；同时检查 StandUp 的持续姿态条件。
- `tests/test_rl_preparation_inputs.py`：在实际 Git 仓库中验证来源版本与文件变化检查。
- `tests/rl/sb3/test_resume_selection.py`：原生 PPO 保存、加载、继续训练及错误输入检查。
- `tests/rl/sb3/test_checkpoint_progress.py`：课程进度恢复。
- `tests/rl/tasks/test_motion_diagnostics.py` 和 `tests/test_protocol.py`：相关诊断与接口检查。

评分数学输入用于检查判定逻辑。Pendulum 用于检查原生 SB3 保存与恢复。本次没有运行 Microduck GPU 仿真、完整训练、ONNX 导出或真机测试；当前环境没有历史训练 checkpoint 和原始轨迹，历史策略尚未按版本 3 重新评分。
