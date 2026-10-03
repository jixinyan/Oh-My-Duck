# 距离与角度 policy tools

原生 Harness 的 Microduck 工具使用官方 ONNX policy。`walk` 与 `rotate` 把距离和角度转为有界 policy command，通过实际位姿控制运动与制动。工具准备动作后，原生 `execution.resume` 接纳后续动作；原生 ActionGate 保留暂停、中断、任务 lease 与预算检查。

| 工具 | 参数 | 物理执行与测量 |
|---|---|---|
| `microduck.walk` | `distance_m`；可选 `speed_m_s` | 标准脚模型选择 `alpha_walking`，轮滑模型选择 `roller`；沿调用时的身体朝向前进或后退，检查停止后的目标距离 |
| `microduck.rotate` | `angle_deg`；可选 `angular_speed_deg_s` | 选择当前模型对应的 locomotion policy；累计实际 yaw，检查停止后的角度误差，返回 `translation_xy_m` |
| `microduck.select_policy` | `policy_name` | 在确认暂停且具有五个实际停止样本时选择 manifest policy |
| `microduck.transition_policy` | `policy_name` | 在原生 episodic 结束边界接续 policy，保留身体位姿、速度和上一动作 |
| `microduck.read_sensor` | `sensor` | 读取 `head_rgb`、`tof`、`imu`、`joint_state` 或 `odometry` |
| `microduck.task_progress` | 无参数 | 返回运动目标、测量值、误差、停止样本、接触证据与原生 execution 边界 |

`distance_m` 的绝对值为 0.1–10 米，正数前进、负数后退。`speed_m_s` 为 0.1–0.4 米/秒，默认 0.4。`angle_deg` 的绝对值为 10–360 度，正数绕世界 +Z 逆时针、负数顺时针。`angular_speed_deg_s` 为 10–55 度/秒，默认 45。速度参数表示传入 policy 的命令，实际动作按 odometry 测量。

调用示例：`microduck.walk({"distance_m": 1.0})`、`microduck.rotate({"angle_deg": -45})`。动作准备要求当前原生 execution 已确认暂停，并具有至少五个实际停止样本。调用 `execution.resume` 后，控制器在每个实际控制步检查目标；内部命令每段最多 100 个控制步。到达目标后执行零 twist，取得五个停止样本后暂停。停止后的目标距离误差需要不超过 0.05 米，角度误差需要不超过 5 度。

`task_progress.metric_motion` 包含 `requested`、`measured`、`unit`、`error`、`tolerance`、`phase`、`completed` 和原生 `sequence`。转向同时返回实际平移距离。障碍或停滞产生 `blocked`，停止后超出误差要求产生 `failed`，用户中断产生 `interrupted`。只有经过测量的目标与停止条件同时满足时，才返回 `completed=true`。目标动作结束后，`finish_policy` 仍检查确认边界与零命令停止证据；正式任务由独立 Verifier 判定。

官方 catalog 保留十个 policy 的文件 hash、命令编码与执行条件。当前 `allcollisions` 机器人通过统一工具接口接入八个 policy；`roller` 与 `crouch` 要求对应的 roller 机器人模型。拾取、踢球、翻滚等物体效果需要各自的物理检查。距离和角度行为验证范围见[验证记录](reports/metric-camera-tools-2026-09-30.md)。

2026-10-03 的标准脚模型实际检查覆盖 0.5 米与 45°：停止后的距离误差为 0.024196 米，角度误差为 0.522914°，均具有五个停止样本。转向同时产生 0.208189 米平移。控制器使用实际速度估计停止距离，每个控制步更新命令，目标停止后最多进行三次距离或角度修正。停滞检查分别记录运动与制动阶段，运动阶段的命令更新保留连续测量历史。见[验收记录](reports/end-to-end-2026-10-03.md)。

同日的 Hospital 轮滑模型检查覆盖 0.5 米与 45°：停止后的距离误差为 0.018337 米，角度误差为 3.893902°，均取得五个停止样本，转向平移为 0.011624 米。两个动作通过实际 Newton/BAM 和原生 ActionGate，外部障碍接触累计为零。轮滑控制使用当前位姿与速度，转向考虑零命令之后的机身角度变化。该记录覆盖单个 seed 与正向指令；其他距离、方向、速度、表面和场景需要各自测量。

两个仿真后端共用工具与控制器，odometry 分别来自当前 MuJoCo 或 Newton 物理状态。真机后端需要提供同单位的位姿、速度与停止证据，目前没有真机行为验证。

head RGB 使用官方 `head_camera` site 的 +X 前向与 +Z 上向生成 OpenGL optical frame。CPU MuJoCo 使用原生相机裁剪；Newton 在 pinhole 射线中应用 0.01 米近裁剪平面。Newton 观察相机跟随机器人，位置为身体位置加 `(2.0, -2.0, 1.4)` 米，观察目标位于身体上方 0.35 米。RGB 保留实际渲染值，原生 segmentation 提供可见机器人与场景几何的证据。
