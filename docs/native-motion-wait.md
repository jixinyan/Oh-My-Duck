# 原生运动等待与进度

`microduck.wait_for_motion` 等待当前原生动作，并读取真实物理进度。
工具由固定 EDH 的 ActionGate、policy Task 和设备确认状态提供执行身份。

| 参数 | 范围 | 默认值 |
| --- | --- | --- |
| `max_wait_s` | 1–90 的整数 | 90 秒 |

`max_wait_s` 限制等待动作完成的窗口。窗口结束后读取当前传感器；
完整调用还包含传感器读取与通信耗时。非法数值和未知字段在调用位置拒绝。

```json
{
  "max_wait_s": 1
}
```

返回值包含 `episode_id`、`sequence`、`body_position_m`、`body_twist`、
`metric_motion`、`motion_guard`、`goal_check`、停止样本和原生剩余预算。
`execution` 保留执行身份、generation、boundary、当前状态和设备确认状态。

```json
{
  "motion_wait": {
    "max_wait_s": 1,
    "boundary_confirmed": false
  }
}
```

| 状态 | Agent 操作 |
| --- | --- |
| `boundary_confirmed=false` | 读取实际进度，继续监测或等待当前 execution |
| 确认的 `paused` | 检查物理停止样本、目标和剩余预算，准备下一段动作 |
| 确认的 `ended` | 读取结束原因，按原生任务流程继续正式检查或新执行 |

`boundary_confirmed` 来自 `paused` 或 `ended` 与 `device_confirmed`。
物理停止通过速度和 `stopped_samples` 检查。需要静止的运动准备工具要求至少五个停止样本。

进度读取等待当前控制步的原生发布和 controller 更新完成，随后读取同一物理 sequence
的身体位置、`metric_motion`、停止样本与 `goal_check`。等待窗口结束后，该同步读取
包含当前控制步完成和传感器通信所需的时间。

Agent 在 `execution.start` 或 `execution.resume` 后调用该工具，
检查返回状态并继续自己的监测过程。每个确认边界的原始测量均保留在 trace 中。
取消等待调用保留当前 policy Task；用户停止动作通过 Harness 原生任务停止接口执行，
并读取设备确认边界。实际 policy 与清理 Task 的执行错误继续传播。

## CPU 验证

锁定 CPU 环境、固定 EDH、官方策略目录与 OSMesa 准备完成后执行：

```sh
CUDA_VISIBLE_DEVICES='' python -m oh_my_duck.infrastructure.headless \
  --mujoco-renderer osmesa -- python -m oh_my_duck validate motion-wait \
  --catalog "$POLICY_DIRECTORY" --edh-source "$HARNESS_DIRECTORY" \
  --output outputs/acceptance/motion-wait-new
```

输出目录使用新的路径，保存真实事件、动作、测量、工具结果、相机 PNG 和资源释放状态。
验证覆盖短时窗口、取消等待、90 秒窗口结束后的继续执行、目标接近、
暂停与终止后的重复读取、七项非法参数拒绝和实际停止保持。

## 完整模型任务与记录检查

[`apartment-model-target.json`](../configs/simulation-demo/apartment-model-target.json) 指导 Agent
在每次执行或继续动作后使用 `max_wait_s=1` 查询进度，当前动作继续时使用
`max_wait_s=90` 等待确认边界。目标检测使用 YOLO26；仿真 RGBD 提供几何距离。

导出实际任务后执行公开检查：

```sh
omd validate replay outputs/model-target/replay \
  --expected-verdict passed --expected-stop-reason policy_stop \
  --require-stop-progress --require-target-tracking \
  --require-model-perception --require-motion-wait
```

`--require-motion-wait` 要求同时存在继续执行和确认边界的工具结果，
检查原始模型调用参数、对应的完成结果、同一物理 sequence 的测量、
执行身份及原生 `boundary_at`。边界时间使用原生记录中的发生时间。

固定源码的实际 Luna 任务通过独立 Verifier，保存十条等待结果、三次原始模型 RGBD
观测、连续目标查询、测得的接近、停止和资源释放。该范围见
[模型等待任务记录](reports/model-motion-wait-2026-10-13.md)。
独立安装后的完整任务、严格回放、全部服务关闭与 MP4 见
[完整运行记录](reports/native-model-task-lifecycle-2026-10-13.md)。

实现位置：

| 内容 | 源码 |
| --- | --- |
| 原生等待与进度 | `src/oh_my_duck/integrations/edh/session.py` |
| Agent tool Schema | `integrations/edh/server.mjs` |
| Planner 工具指导 | `integrations/edh/roles/planner.md` |
| 实际 CPU 验证 | `src/oh_my_duck/validation/harness/motion_wait.py` |
| 完整模型任务等待记录 | `src/oh_my_duck/validation/harness/motion_wait_records.py` |

固定源码的真实 CPU 动作、独立 ONNX、完整模型任务和 MP4 复核见
[验证记录](reports/model-motion-wait-2026-10-13.md)。
