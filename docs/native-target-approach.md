# 可见目标接近

`microduck.approach_object` 将当前 RGBD 观测中的目标转换为原生 locomotion 请求。
上层 Harness 负责选择目标、调用工具和管理任务；工具根据实际机器人位置控制距离、
bearing、制动和停止。原生 ActionGate、执行预算、ToF、外部接触与运动进度检查持续生效。

| 参数 | 含义 | 范围 |
| --- | --- | --- |
| `target_id` | 当前 `inspect_scene` 或 `observe` 返回的目标身份 | 1–256 个字符 |
| `standoff_m` | 机器人到所观察 world 表面点的水平距离 | 0.2–2.0 米，必填 |
| `speed_m_s` | locomotion 前向或后退命令的速度幅度 | 0.1–0.4 m/s，默认 0.4 |
| `max_age_ms` | 准备请求时允许的观测时间 | 100–60000 毫秒，默认 60000 |

调用需要确认的暂停边界、五个测得的停止样本，以及当前 episode 和物理 sequence
的有效 RGBD 目标。准备之后通过 `execution.resume` 执行，再使用
`microduck.wait_for_motion` 等待暂停边界，读取 `microduck.observe` 或
`microduck.task_progress` 中的 `metric_motion`。

```json
{
  "target_id": "ball_0",
  "standoff_m": 0.4
}
```

每个请求保留原始观测、目标、表面点、起始位置、请求身份和测量结果。
模型来源的请求同时保留原始 `models` SHA256，回放时与捕获的检测结果核查。
完成要求为水平距离误差不超过 0.05 米，并取得五个实际停止样本。
重复读取同一个 sequence 不会增加停止计数。该动作在执行期间使用捕获的固定表面点；
目标移动后的接近需要重新观察并准备新的请求。`follow_object` 的持续跟踪具有独立实现要求。

感知来源由当前场景公布。`source=models` 使用配置的感知服务，
`source=simulator_ground_truth` 使用仿真的原生实例几何；记录保留真实来源。
`standoff_m` 测量表面点，场景任务可以使用物体中心、目标区域或 waypoint，
正式任务结果由原生目标检查和独立 Verifier 给出。

## 场景与运行

公共场景为 [`apartment-target-approach.json`](../configs/simulation-demo/apartment-target-approach.json)：
官方公寓中的 `ball_0`，起始位置 `(-3.65, 0)`，目标中心半径 0.5 米、保持五个控制步；
接近指令要求距离所观察表面点 0.4 米，预算为 1000 个控制步和 300 秒。

按照 [Harness 准备说明](harness-native-integration.md) 准备固定的 EDH、官方 policy 与 Linux CPU 环境。
模型入口从调用环境读取 `EDH_MODEL`、`EDH_MODEL_API`、`EDH_MODEL_BASE_URL` 和
`EDH_MODEL_API_KEY`，也支持 `omd harness --provider-config` 的 TOML 配置。

```bash
omd harness \
  --scene-config configs/simulation-demo/apartment-target-approach.json \
  --mujoco-renderer osmesa \
  --model gpt-6-luna --model-api responses \
  --reasoning-effort high --max-output-tokens 8192
```

SSH worker 使用同一场景及固定来源，通过 `--worker-host`、`--worker-root`、
`--worker-python`、`--worker-edh-source` 和 `--worker-policy-dir` 指定实际运行环境。
CPU worker 的 `CUDA_VISIBLE_DEVICES` 为空。原生任务目录中的 scenario 为
`navigate-official-apartment-ball-approach`。

对原生导出的任务记录运行：

```bash
omd validate replay outputs/target-approach/replay \
  --expected-verdict passed --expected-stop-reason policy_stop \
  --require-stop-progress --require-target-approach
```

检查程序读取原始工具记录、观测身份、表面点、实际位置、停止边界和正式 verdict，
独立重算目标距离与误差。每个接纳的目标请求都需要完成记录；图像依据原始 attachment
hash 和 PNG 解码检查。原始 RGBD 像素的独立检查由 [感知验证](perception-navigation.md) 定义。

## 模型检测驱动的任务

[`apartment-model-target.json`](../configs/simulation-demo/apartment-model-target.json) 使用
`source=models` 和 `prompt=objects`，让上层 Agent 选择当前检测中的有效小型目标。
该场景起始位置为 `(-1.6, -0.5, π)`，目标为 `ball_0` 中心 0.5 米范围内保持五个控制步，
预算为 1000 个控制步和 600 秒。检测标签保持模型原始输出；RGBD 距离使用仿真测量。

在拥有 YOLO26 权重的独立 CPU perception 环境中运行：

```bash
CUDA_VISIBLE_DEVICES= python -m oh_my_duck.perception.service \
  --device cpu --yolo checkpoints/perception/yolo26s.pt \
  --port 8784 --record-inspections --output outputs/model-target-inspections
```

在同一主机运行 CPU worker，场景的 `perception_endpoint` 指向该服务。
上层 Harness 可以通过 [SSH worker 参数](harness-native-integration.md) 在另一台主机运行。

```bash
omd harness \
  --scene-config configs/simulation-demo/apartment-model-target.json \
  --mujoco-renderer osmesa \
  --model gpt-6-luna --model-api responses \
  --reasoning-effort high --max-output-tokens 8192
```

原生 scenario 为 `navigate-official-apartment-model-target`。导出完成的实际任务后执行：

```bash
omd validate replay outputs/model-target/replay \
  --expected-verdict passed --expected-stop-reason policy_stop \
  --require-stop-progress --require-target-tracking --require-model-perception
```

`--require-model-perception` 要求每个接近请求使用原始模型检测、相同模型 SHA256 和有效
RGBD 几何，并具有测得的完成和停止记录。`--require-target-tracking` 检查同一目标在不同
物理 sequence 上的当前查询。该场景的实际 Luna 任务、原始像素复核与 MP4 见
[模型目标任务记录](reports/model-target-task-2026-10-13.md)。

## 源码

| 部分 | 位置 |
| --- | --- |
| 目标距离与 bearing 控制 | `src/oh_my_duck/robotics/microduck/target_motion.py` |
| 原生会话、观测身份与暂停边界 | `src/oh_my_duck/integrations/edh/session.py` |
| 工具 Schema 与 planner 权限 | `integrations/edh/server.mjs`、`integrations/edh/roles/planner.md` |
| 原始记录中的距离与停止复核 | `src/oh_my_duck/validation/harness/target_approach.py` |
| 原生导出、正式 verdict 与图像检查 | `src/oh_my_duck/validation/harness/replay.py` |

CPU MuJoCo/BAM 与 Isaac/Newton/BAM 共享控制接口。当前执行条件下，Newton 的目标接近
GPU 验收保持待验证状态；真机控制需要可用的 Microduck。
