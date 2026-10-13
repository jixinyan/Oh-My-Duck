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

实现位置：

| 内容 | 源码 |
| --- | --- |
| 原生等待与进度 | `src/oh_my_duck/integrations/edh/session.py` |
| Agent tool Schema | `integrations/edh/server.mjs` |
| Planner 工具指导 | `integrations/edh/roles/planner.md` |
| 实际 CPU 验证 | `src/oh_my_duck/validation/harness/motion_wait.py` |

固定源码的真实 CPU 运行与独立 ONNX、物理记录和图像复核见
[验证记录](reports/motion-wait-progress-2026-10-13.md)。
