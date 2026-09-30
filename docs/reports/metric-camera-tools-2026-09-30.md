# 距离、角度工具与相机验证 · 2026-09-30

`microduck.walk(distance_m)` 与 `microduck.rotate(angle_deg)` 经实际 Newton/BAM、官方 `alpha_walking` 和原生 EDH ActionGate 执行。head RGB 使用官方相机 site 的前向与上向，观察相机提供机器人和 Office 环境的实际画面。完整参数、状态与调用条件见[工具接口](../metric-policy-tools.md)。

## 相机与场景

场景为 `nvidia-office-6.0`，USD 经 Newton 导入，保留 3646 个环境碰撞几何。head camera 使用官方 `head_camera` site 的 +X 前向、+Z 上向，转换到 OpenGL 的 −Z 前向、+Y 上向。CPU MuJoCo 与 Newton 共用 optical pose 计算。

Newton pinhole 射线的起点位于 0.01 米近裁剪平面，光心、方向和场景几何保持原始物理位置。观察相机的位置为身体位置加 `(2.0, -2.0, 1.4)` 米，观察目标位于身体上方 0.35 米，分辨率为 1280×720。head RGB 的相机位姿与亮度统计随 `read_sensor(head_rgb)` 返回，图片来自当前物理状态。

实际 GPU 相机检查使用 RGB 和原生 instance segmentation；场景形状包括桌子、墙面、灯具与门窗等，数量表示可见几何形状。

| 相机 | RGB 平均值 / 标准差 | 非地面场景像素 | 可见场景形状 | 机器人像素 |
|---|---|---:|---:|---:|
| head RGB | 56.449835 / 49.524902 | 41089 | 46 | 0 |
| observer | 93.071006 / 36.953553 | 79483 | 12 | 5114 |

`scripts/accept_isaac_camera.py` 同时检查实际射线近裁剪位置、相机前向/上向、图像变化、场景几何和机器人像素。CPU `scripts/accept_observer_camera.py` 另完成 100 个真实 MuJoCo/BAM 控制步与相机检查；CPU 距离/角度行为尚未验收。

## 实际运动与停止

工具从调用时的实际身体位置和 yaw 建立目标，使用当前物理 odometry 更新命令。动作通过原生 `execution.resume` 接纳，内部每段最多 100 个控制步；障碍、接触、停滞、lease 和预算限制继续由原生执行路径检查。目标制动后需要五个实际停止样本，距离误差要求不超过 0.05 米，角度误差要求不超过 5 度。

| 请求 | 实际测量 | 停止后的目标误差 | 转向平移 | 结果 |
|---|---:|---:|---:|---|
| 前进 0.4 m | 0.381051 m | 0.019061 m | — | complete |
| 前进 1.0 m | 0.985332 m | 0.022182 m | — | complete |
| 转向 +45° | +43.730295° | 1.269705° | 0.205673 m | complete |
| 转向 −45° | −40.912333° | 4.087667° | 0.239017 m | complete |
| 转向 +270° | +267.804383° | 2.195617° | 0.430718 m | complete |

前进测量值为初始朝向上的有符号位移，目标误差为停止后与目标坐标的 XY 距离。转向使用官方 walking policy 执行带平移的动作，`translation_xy_m` 返回实际平移。+270° 的最终 wrapped yaw 为 −1.581243 rad，unwrapped yaw 为 4.701942 rad，累计角度经过 ±180° 边界。

三组真实 worker 检查分别完成 359/1436、607/2428、933/3732 个控制步/物理子步，均保持直立、取得五个停止样本并以原生 `policy_stop` 结束，外部非地面接触累计为零。本次没有触发角度误差修正分支。

参数允许有符号 0.1–10 米、10–360 度；速度允许 0.1–0.4 米/秒、10–55 度/秒。当前行为证据覆盖表格中的 Office 单一 seed，请求范围的其余距离、角度、速度与后退行为仍需实际评估。其他官方 policy 通过统一选择/接续工具提供，物体效果需要分别验证；`roller` 与 `crouch` 要求对应机器人模型。

## 原生模型 Demo

真实 Astra/high 通过固定 EDH 原生 SessionEnvironment、EmbodiedBackend、ActionGate 与独立 Verifier，在 `jd_B300` GPU 4 的 Newton Office 调用传感器、官方 `velstand` / `alpha_walking`、`microduck.walk(distance_m=0.45, speed_m_s=0.4)` 与原生执行工具。距离目标停止后完成零命令控制，独立 Verifier 检查当前身体位置、直立与保持步数，Planner 完成计划并调用 `tasks.finish`。

| 项目 | 记录 |
|---|---|
| run | `8ae4b995-62b1-44a2-ad58-9a183c8ae964` |
| session | `9105bfe2-d1d0-446d-927a-a7ae053f945a` |
| execution | `6517769a-e8db-4977-876c-c83a3961e32c` |
| 最终确认边界 | `3eff0ab1-d368-497c-bd2c-8ad2774d7584` |
| 独立 verdict | `3956a823-b91c-48e1-a1d7-d29bd724edb5`，`passed` |
| run 状态 | `succeeded` |
| 正式目标误差 / 阈值 | 0.0584946 m / 0.15 m |
| 直立目标保持 / 要求 | 126 / 5 个控制步 |
| 控制步 / 物理子步 | 313 / 1252 |
| 外部非地面接触累计 | 0 |
| 原始事件 / observer / head 观测 | 782 / 62 / 2 |
| 停止进度审计 | 4 项通过，含 metric motion 目标、停止与实际命令身份 |
| 相机源分辨率 | observer 1280×720 |
| MP4 | 1920×1080，10 fps，603 帧，60.3 秒 |

MP4 保留公开 Planner 文字、计划、工具参数与测量反馈、正式 verdict 和实际机器人相机帧。等待片段按 12 倍墙钟速度压缩；相邻 observer 帧的播放时间至少等于实际物理时间间隔，6.1 秒物理运动保持该时间要求。模型内部 reasoning 没有导出。

产物为 `outputs/demos/oh-my-duck-office-metric-camera-20260930-verified.mp4`，SHA256 为 `ac8a119c4e28d65df8d275a9c2fe3418863fc5689a262427ba79d04d7e00dc6b`。检查覆盖全部相机源文件 hash、全部编码帧文字边界、时间序列和 FFmpeg 完整解码；图像内容检查使用实际 RGB 统计与原生 segmentation。当前没有人工视觉审查记录。

## 来源与复现

- EDH：`8a5e685b22d032207f53db20454f0992a4ad60fd`。
- 官方 ONNX：`1b56c396825c052a4e26e95cf2b8d8298af9e9b4`，manifest/hash 验证后加载。
- Newton 源码副本：`/home/jixin/workspace/code/Oh-My-Duck/.job-sources/metric-camera-20260930-07`；运行后保持不可变。
- GPU 环境：Isaac Lab 6.1.14、Newton 1.2.1、Warp 1.13、MuJoCo 3.8、Torch 2.10 cu130；SolverMuJoCo/MuJoCo-Warp、官方 BAM、50 Hz、每次四个 0.005 秒子步。
- 相机证据：`.cache/metric-camera/camera-result-06.json`；远程副本 `metric-camera-20260930-06/outputs/camera-acceptance-20260930-06/`。
- 正向/+45°：`.cache/metric-camera/metric-result-07.json`；远程 `metric-camera-20260930-07/outputs/metric-tools-20260930-07/`。
- 1 米/−45°：`.cache/metric-camera/metric-result-06.json`；远程 `metric-camera-20260930-06/outputs/metric-tools-20260930-06/`。
- +270°：`.cache/metric-camera/metric-result-07-wrap.json`；远程 `metric-camera-20260930-07/outputs/metric-tools-20260930-07-wrap/`。
- 原生模型导出：`.cache/metric-camera/replay-07/`，包含原始事件、终态与 PNG。

模型 Demo 与正向/+45°、+270° 使用相同 worker 源码；1 米/−45° 使用相同 `MetricMotion` 源码。当前本地与远程 source-07 的源码 hash 逐项一致：

| 文件 | SHA256 |
|---|---|
| `integrations/edh_native.py` | `9dcb434b1b8d8c0589462f7261180f747752bcb013d6afd555d84d0a1fc88467` |
| `robotics/microduck/metric_motion.py` | `f667c514e9b49176df01fe3d60113deea8cf0745d8847e2ed389e91dfc547d48` |
| `robotics/microduck/sim_sensors.py` | `9e8df810124aacb76011c9257dd8d92ff1bd2b48c7df203c8defe6fb56c5bd68` |
| `robotics/microduck/isaac_official.py` | `811aa1e18d5983428aa1c3850f1077b31fffb4d302767d3d7ccf42b9ea18f9ba` |
| `robotics/microduck/simulation.py` | `6100aa7b2d47fe3bc7e349b9e3c7d5667faeb4dbecbbfa576b1c0fe11f069e7f` |

实际行为检查在远程源码副本运行，分配前检查 GPU 使用情况，明确指定当前空闲设备：

```sh
PYTHONPATH=src:.cache/edh/8a5e685b22d032207f53db20454f0992a4ad60fd/harness/physical-runtime/src CUDA_VISIBLE_DEVICES="$IDLE_GPU_INDEX" \
  /home/jixin/workspace/code/Oh-My-Duck/.envs/isaac-newton/bin/python scripts/accept_metric_policy_tools.py \
  --scene-config configs/simulation-demo/office.json --catalog "$OFFICIAL_POLICY_DIR" \
  --distance 0.4 --angle 270 --output "$NEW_OUTPUT_DIRECTORY"
```

原始模型记录的正式检查与视频导出在本机锁定环境运行：

```sh
.cache/cpu-apartment-locked-venv/bin/python scripts/accept_harness_replay.py \
  .cache/metric-camera/replay-07 --expected-verdict passed --expected-stop-reason policy_stop \
  --require-stop-progress --require-metric-tools
.cache/cpu-apartment-locked-venv/bin/python scripts/render_microduck_run.py \
  --export .cache/metric-camera/replay-07 --output "$NEW_MP4_PATH" \
  --review-dir "$NEW_REVIEW_DIRECTORY" --fps 10 --wall-speed 12
```

会话返回 `state=closed`、`resources=released`，实际测试 worker 全部退出，GPU 4、5 各恢复至 1 MiB。RL 保持停止。当前验证范围为 Newton Office 单一 seed 的短距离工具及模型闭环；多场景长距离导航、物体效果和真机执行继续待完成。
