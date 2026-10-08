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
| `microduck.observe` | 可选 `prompt` 与 `source`，同时提供 | 在确认暂停边界返回 head RGB、ToF、IMU、servo、odometry、进度和剩余执行时间；感知来源为 `simulator_ground_truth` 或 `models` |
| `microduck.wait_for_motion` | 无参数 | 等待当前原生运动确认暂停或结束，返回实际进度与剩余执行时间 |

`observe` 的传感器使用同一个 episode 和 sequence。附带感知时检查图像来源与该物理边界一致；ToF 读取可供 MotionGuard 的当前暂停边界检查使用。CPU 公寓提供当前帧的原生实例几何与射线距离；`scene_info.perception_sources` 公布当前配置可以调用的感知来源。`wait_for_motion` 最多等待 90 秒，调用超时向调用者报告，运动状态仍由原生 execution 管理。已经确认的边界可以再次读取，读取不会推进物理时间。进度中的 `execution.remaining_wall_time_s` 来自原生预算；Planner 应保留终点停止、`finish_policy`、独立 Verifier 与 `tasks.finish` 所需时间。

`distance_m` 的绝对值为 0.1–10 米，正数前进、负数后退。`speed_m_s` 为 0.1–0.4 米/秒，默认 0.4。`angle_deg` 的绝对值为 10–360 度，正数绕世界 +Z 逆时针、负数顺时针。`angular_speed_deg_s` 为 10–55 度/秒，默认 45。速度参数表示传入 policy 的命令，实际动作按 odometry 测量。

原生工具的参数 schema 提供速度默认值、单位、最小动作幅度和完成条件。导航使用默认速度，通过请求距离、角度与当前观测控制每段运动范围；其他速度需要在当前场景测量 policy 响应。标准脚模型转向同时使用 0.2 米/秒前向和 0.25 米/秒侧向命令，因此需要根据障碍位置预留运动空间，并在转向结束后重新测量位置。转向速度较低时，完成请求角度可能需要更多时间与平移距离。轮滑模型的转向命令为零前向、零侧向及请求 yaw 速度，完成后仍检查实际平移与停止状态。

调用示例：`microduck.walk({"distance_m": 1.0})`、`microduck.rotate({"angle_deg": -45})`。动作准备要求当前原生 execution 已确认暂停，并具有至少五个实际停止样本。调用 `execution.resume` 后，控制器在每个实际控制步检查目标；内部命令每段最多 100 个控制步。到达目标后执行零 twist，取得五个停止样本后暂停。停止后的目标距离误差需要不超过 0.05 米，角度误差需要不超过 5 度。

准备中的距离或角度动作保留其 `request_id`。下一次 `walk` 或 `rotate` 要求当前动作已经完成、失败、受到阻碍或被明确中断。准备动作后调用 `execution.resume` 执行该请求，随后读取确认暂停边界。主动修改动作时使用 `set_command` 明确设置新命令；原生执行仍要求调用 `execution.resume`。重复动作被拒绝时，原请求、command 和物理边界保持一致。

`omd validate metric-admission` 在 CPU MuJoCo/BAM 的 Mesa 软件渲染环境中检查准备请求、实际距离和角度执行、明确修改命令与资源释放。调用者的等待超时保留原生执行，由原生预算控制执行期限；需要停止动作时使用原生停止接口。

`task_progress`、`observe` 和 `wait_for_motion` 返回当前 sequence 的 `goal_check`，
包含原生目标范围、upright 要求及实际保持计数。计数由物理控制步累计，重复读取保留
相同计数；正式任务结果由独立 Verifier 判定。重试保留当前位姿、速度、时间及 policy
状态，新执行准备 75 个零 twist 控制步，明确准备的新命令保留参数和控制步数。
`omd validate execution-retry` 验证连续执行、目标信息读取、独立 ONNX 复核和资源释放，
见[原生重试验证](reports/native-execution-retry-2026-10-08.md)。

`task_progress.metric_motion` 包含 `requested`、`measured`、`unit`、`error`、`tolerance`、`phase`、`completed` 和原生 `sequence`。转向同时返回实际平移距离。MotionGuard 的障碍或运动停滞检查产生 `blocked`；目标进度不足返回 `failed` 和 `reason="metric_progress_stalled"`；停止后超出误差要求产生 `failed`，用户中断产生 `interrupted`。只有经过测量的目标与停止条件同时满足时，才返回 `completed=true`。目标动作结束后，`finish_policy` 仍检查确认边界与零命令停止证据；正式任务由独立 Verifier 判定。

官方 catalog 保留十个 policy 的文件 hash、命令编码与执行条件。当前 `allcollisions` 机器人通过统一工具接口接入八个 policy；`roller` 与 `crouch` 要求对应的 roller 机器人模型。拾取、踢球、翻滚等物体效果需要各自的物理检查。距离和角度行为验证范围见[验证记录](reports/metric-camera-tools-2026-09-30.md)。

2026-10-03 的标准脚模型实际检查覆盖 0.5 米与 45°：停止后的距离误差为 0.024196 米，角度误差为 0.522914°，均具有五个停止样本。转向同时产生 0.208189 米平移。控制器使用实际速度估计停止距离，每个控制步更新命令，目标停止后最多进行三次距离或角度修正。停滞检查分别记录运动与制动阶段，运动阶段的命令更新保留连续测量历史。见[验收记录](reports/end-to-end-2026-10-03.md)。

同日的 Hospital 轮滑模型检查覆盖 0.5 米与 45°：停止后的距离误差为 0.018337 米，角度误差为 3.893902°，均取得五个停止样本，转向平移为 0.011624 米。两个动作通过实际 Newton/BAM 和原生 ActionGate，外部障碍接触累计为零。轮滑控制使用当前位姿与速度，转向考虑零命令之后的机身角度变化。该记录覆盖单个 seed 与正向指令；其他距离、方向、速度、表面和场景需要各自测量。

2026-10-07 的 Hospital 轮滑矩阵通过三个独立会话，覆盖 +0.5 米、−0.5 米、+1.0 米与 ±45°，最大误差为 0.049210 米与 3.746165°。五次动作确认直立停止、零外部接触和资源释放，原始样本通过独立复核。轮滑转向保持请求的 angular command，并使用实际制动角度进行有界修正。三个场景的九个会话使用相同的控制器源码，完成十五次动作检查。见[批量验收记录](reports/metric-controller-acceptance-2026-10-07.md)。

两个仿真后端共用工具与控制器，odometry 分别来自当前 MuJoCo 或 Newton 物理状态。真机后端需要提供同单位的位姿、速度与停止证据，目前没有真机行为验证。

2026-10-07 的标准脚控制器在 CPU 公寓与 Newton Office 各三个独立会话中通过 +0.5 米、−0.5 米、+1.0 米与 ±45°检查。CPU 最大位置/角度误差为 0.035112 米/3.777766°，Office 为 0.021602 米/4.952898°；十次动作均确认直立停止，外部障碍接触累计为零。行走按完整目标位置修正横向距离与朝向，转向根据实际制动角度完成最多三次修正。当物理状态已经停止且原始目标误差满足要求时，控制器进入零命令制动，并重新测量最终误差与连续停止样本。50 个控制步内目标进度不足时报告失败并确认停止；修正阶段的进度使用当前制动补偿目标，最终误差使用原始请求。参数、来源与检查范围见[批量验收记录](reports/metric-controller-acceptance-2026-10-07.md)。

head RGB 使用官方 `head_camera` site 的 +X 前向与 +Z 上向生成 OpenGL optical frame。CPU MuJoCo 使用原生相机裁剪；Newton 在 pinhole 射线中应用 0.01 米近裁剪平面。Newton 观察相机跟随机器人，位置为身体位置加 `(2.0, -2.0, 1.4)` 米，观察目标位于身体上方 0.35 米。RGB 保留实际渲染值，原生 segmentation 提供可见机器人与场景几何的证据。
