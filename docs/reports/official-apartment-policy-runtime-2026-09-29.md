# 官方策略与公寓 CPU 仿真验收记录

## 来源与执行条件

策略仓库为 `pollen-robotics/microduck-policies`，本次使用的 `main` 提交为 `1b56c396825c052a4e26e95cf2b8d8298af9e9b4`。`manifest.json` 的 SHA-256 为 `622048c2c23ea58942023f66fd16b189a875fd169e88d85beb16ebbe63b20c94`。所有十个 ONNX 文件的 SHA-256、schema 2、model API 1、`float32[1,61]` 输入、`float32[1,14]` 输出、14 个关节顺序、61 个观察量顺序、HOME、50 Hz 控制频率和 CPU ONNX Runtime 推理均通过 `scripts/validate_official_policies.py` 检查。完整逐文件检查结果保存在 `.cache/official-policies/validation.json`；下载文件位于 `.cache/official-policies/1b56c396825c052a4e26e95cf2b8d8298af9e9b4/`。

公寓源文件为 `src/oh_my_duck/robotics/microduck/microduck/scene_apartment.xml`，来自官方 `microduck_rl@cfe1c2adcceb55f6b6e369c888b31c6873175c55`，SHA-256 为 `f89da842d8972db257cb9a1a7c1804e21e1d0c01cc3388d6de6ac6fa1060f6e2`。该文件包括 `robot_allcollisions.xml` 与 `apartment.xml`；外墙覆盖 x 方向 8 m、y 方向 6 m，保留六间房、家具、楼梯、停靠点、四个物体和三个球。编译后的模型含 222 个 geom，其中 138 个具有接触类型，14 个关节执行器。`scene.xml` 使用 groundcontact 机器人，`scene_walk.xml` 使用 walk 机器人；官方训练配置中 `velstand` 使用 groundcontact，`alpha_walking` 使用 walk。公寓使用 allcollisions，策略部署时保留其全部几何与碰撞。

锁定 CPU 环境位于 `environments/cpu-apartment/pyproject.toml` 和同目录的 `uv.lock`。本次通过锁文件在全新 `.cache/cpu-apartment-locked-venv` 安装 MuJoCo 3.8.0、固定提交 `62bd8ce12154340be97e06f7f41a0ca8f116d967` 的 BAM、ONNX Runtime 1.30.0、ONNX 1.23.1、Pillow 12.3.0、mjlab 1.3.0、torch 2.9.1、Warp 1.12.0、jsonschema 4.25.1 和 websockets 17.0.1。`mjlab` 的 `viser` 依赖声明限制 `websockets<17`，EDH physical-runtime 使用 17.0.1；环境文件为该项声明设置显式 `uv` override。CPU 物理接口、BAM/mjlab 导入、官方模型校验、命令槽位与原生 GT/停止接口均已在锁定环境运行；`viser` 模块导入也已检查，交互式 viewer 未纳入本次验收。物理步长为 0.005 s，每次策略动作执行四个物理步。BAM 参数为 `kp_fw=200`、`vin=7.4 V`、`vin_drop_gain=0.1`、`vin_min=6.0 V`。官方训练配置的动作延迟为 3–6 个物理步，即 15–30 ms；`mjlab` 在每个物理子步调用执行器的 `apply_delay`。官方 CPU rehearsal 默认动作延迟为零，本次公寓执行采用相同设置。训练期间的电压、摩擦和观察噪声随机化没有在固定条件的 CPU 验收中采样。

## 控制与原生观察

`CpuMujocoBamBackend` 在创建它的唯一线程执行 `reset_episode`、`observe_control`、`set_command`、`infer_policy`、`apply_policy_action`、`check_goal`、`select_policy` 和 `list_policies`。`infer_policy` 返回 61 维观察量和 14 维原始关节 offset，不推进物理时间；`apply_policy_action` 仅执行该推理 ticket 指定的 offset，并在每个 0.005 s 子步前检查 `should_stop`。返回值包含 episode、连续 sequence、仿真时间、已执行动作、物理步数、位置、姿态和传感器读数。完整动作产生四个物理步；动作被中途停止时保留已执行步数，该 request_id 不能再次执行。动作开始前收到停止信号时，BAM target 保持原值。

13 维命令依次为 twist `[vx, vy, yaw]`、head `[neck_pitch, head_pitch, head_yaw, head_roll]`、body `[x, y, z, roll, pitch, yaw]`。单位分别为 m/s、rad/s、rad 和 m/rad。官方 Walking 任务范围为 vx ±0.4 m/s、vy ±0.3 m/s、yaw ±1.0 rad/s；head 逐项范围为 ±1.1、±1.1、±1.4、±0.31 rad；body 逐项范围为 ±0.02、±0.02、±0.03 m 与三个 ±π/6 rad。`alpha_walking` 和 `velstand` 支持 twist/head/body；`alpha_stand` 支持 head/body；`sitstand` 支持 posture/head，第一维姿态标记为坐下 1、站立 0，body 槽位为零；`ground_pick` 和 `crouch` 仅接受由 manifest 产生的 phase，head/body 槽位为零；`roller` 支持 twist，head/body 槽位为零；`roulade` 与两个 kick 策略的 13 维命令全部为零。`set_command` 在收到当前策略不支持的非零命令时直接拒绝。以上槽位依据官方任务观察量配置和官方 CPU `PolicyInference` 命令逻辑；实际模型命令响应另行记录。catalog 完整显示 manifest 的 `command`、`ramp_s`、`unwind_s`、`slot`、`entry_pose` 和 `chain`；其中 `sitstand` 的两个时间参数为 2.0 s 与 1.0 s，`velstand` 的 slot 为 `walk` 且 entry_pose 为 `standing`，`roulade` 的 chain 为 true。`ActionSpec` 的 14 个 offset 界限由原始 position actuator 的每关节 `[-10,10]` rad target 范围、HOME 和 manifest action scale 计算，允许策略正常输出 HOME 之外的 target。公寓无被动轮关节，`roller` 与 `crouch` 在选择阶段拒绝执行。

head_camera 由 MuJoCo 原生相机渲染为 320×240 RGB PNG。已保存的 `.cache/official-apartment-full-3/head_camera_initial.png` 可由 Pillow 正常解码，像素最小值 4、最大值 217、方差 3222.91。ToF 由场景 `tof` site 的 8×8 MuJoCo 射线产生，距离单位为 mm；状态 5 表示有效命中，255 表示未发现目标。该次最后一帧含 63 个有效命中和 1 个未发现目标，未发现目标的距离不参与障碍距离判断。IMU、关节位置与速度、base 世界位置、body/world 速度、controller target 和 HOME 均由当前 MuJoCo/BAM 状态提供。

`capture_observer()` 在物理 owner 线程读取当前状态，通过独立 MuJoCo `Renderer` 和 `MjvCamera` 渲染跟随 `trunk_base` 的第三视角。相机距离 0.5 m、azimuth 45°、elevation −45°，输出 640×480 PNG、episode、sequence、仿真时间和相机参数；`close()` 释放该 renderer。它不改变公寓碰撞、策略观察量或 head_camera。`.cache/observer-camera-07/result.json` 包含初始及执行 100 个控制步后的原生 segmentation：机器人可见像素分别为 34,081 和 32,130，公寓 geom 可见像素分别为 273,119 和 275,070；两帧均包含 trunk、neck、左右大腿及踝关节，PNG 由 Pillow 验证尺寸与像素变化。head_camera 仍为 320×240。真实成功任务终点 `(0.746734, 0.682829, 0.116144) m` 的静态墙体射线在该相机参数下通畅，结果位于 `.cache/demo-review/observer-ray-success-8e00607b-b.json`。在该终点进行的 16 种显式初始化朝向诊断均显示 trunk 与 neck，最少 30,006 个机器人像素，结果位于 `.cache/observer-yaw-success-01/result.json`；任务实际终点朝向没有保存在原始导出中。

房间目标采用场景中具名墙壁 geom 的原生位置，要求机器人中心进入房间边界内侧 0.2 m。停靠目标采用 `dock_site` 的原生位置与 0.35 m 水平距离；物体目标采用相应 MuJoCo body 的位置与任务指定距离。所有导航目标还要求 trunk 高度至少 0.09 m、倾斜角不超过 25°，并连续保持任务指定的实际控制步数。重复读取观察量或调用 `check_goal` 不增加保持计数。独立接口验收保存在 `.cache/official-backend-contract/result.json`：连续站立保持 50 步、坐下高度 0.06105 m 时房间判定为未完成、物体与停靠点距离读取、重复 request receipt、动作前停止以及两个物理子步后的撤销均通过检查。

## 策略行为记录

`scripts/accept_official_apartment.py` 使用 0.5–1.5 的前进响应比例、真实 body-frame 速度和连续 0.5 s 时间窗口检查运动命令；停步使用官方 `EvaluationProtocol` 的速度阈值。所有测试均保留原始场景几何。此前的 `.cache/official-apartment-full-2/result.json` 保持原样；当前验收使用独立输出目录。

- `velstand`：公寓中 0.1、0.2、0.3、0.4 m/s 命令各持续 5 s，官方评分的平均前进响应比例依次约为 0.0015、0.0009、0.0795、0.3263。四项运动阶段均未达到 0.5；停步窗口符合阈值。0.3 m/s 的 2 s 测试执行 250 次动作、1000 个原生物理步，实际 x 位移 0.01143 m，相对命令位移 0.6 m 的比例为 0.0191；随后 2 s 停步阶段反向位移 0.01114 m。该结果保存在 `.cache/official-apartment-full-3/result.json`。
- `alpha_walking`：公寓中相同四个速度、5 s 运动阶段的平均前进响应比例依次约为 0.0016、0.0015、0.3574、0.2956，均未达到 0.5；停步窗口符合阈值。0.3 m/s 的世界 x 位移为 0.4680 m，相对命令位移 1.5 m 的比例为 0.3120。body-frame 速度采用官方 CPU 控制台所用的 freejoint 世界线速度与 root quaternion 旋转计算，世界 x 位移仅作为轨迹证据。四项原始数据保存在 `.cache/official-apartment-alpha-01-5s-v2/`、`alpha-02-5s-v2/`、`alpha-03-5s-v2/` 和 `official-apartment-alpha_walking-04-5s/`。
- `alpha_stand`：站立起点维持 100 步，最低 trunk 高度 0.11655 m，最大倾斜 0.00881 rad。另一轮将 head_pitch 命令增加 0.2 rad、body_z 增加 0.01 m，保持 2 s 后归零 2 s；末 0.5 s 平均 head_pitch 从 0.56294 回到 0.46621 rad，trunk 高度从 0.11870 回到 0.11676 m。观察到 head/body 指令响应；公寓内的跌倒恢复能力仍未验证。
- `sitstand`：姿态标记 1 保持 2 s，末端高度 0.06105 m；标记 0 保持 2 s，末端高度 0.11598 m。两个阶段完成连续原生动作，坐下阶段未触发通用高度终止。
- `ground_pick`：phase 命令执行全部 2.8 s，随后 `infer_policy` 返回 `complete=true` 且没有 action。没有物体抓取成功判定。
- `roulade`：13 维零命令执行全部 1.0 s，最大倾斜 2.96845 rad；到达 manifest 时间后返回 `complete=true` 且没有 action。末端仍在运动，完整翻滚与接续站立没有完成判定。
- `kick_left`、`kick_right`：各执行全部 0.5 s，随后返回 `complete=true` 且没有 action。球体接触与击球结果没有完成判定。
- `roller`、`crouch`：公寓模型没有轮关节，两项在首次策略动作之前收到明确拒绝。

各策略的原始记录位于 `.cache/official-mode-*/result.json`。锁定环境重新执行了 `sitstand`、`ground_pick`、`roulade` 与 `kick_left`，对应记录位于 `.cache/official-mode-locked-*/result.json`，每项均执行 manifest 对应的完整时长或姿态转换。`list_policies()` 为全部十项报告文件、SHA-256、manifest 类型、命令编码与有效槽位、timing、进入姿态、机器人模式、公寓可执行状态和上述行为状态。锁定环境中的命令校验记录位于 `.cache/official-command-slots/locked-result.json`，原生接口验收记录位于 `.cache/official-backend-contract/locked-result.json`，十个模型的文件与模型检查记录位于 `.cache/official-policies/locked-validation.json`。ONNX 可推理与行为验证分别记录。

## `omd sim` JSONL 接口验收

`scripts/accept_omd_sim_jsonl.py` 在锁定 CPU 环境启动真实 `omd.py sim --backend cpu-mujoco-bam` 子进程，加载官方 `alpha_walking.onnx`，经标准输入发送 JSONL 请求并读取标准输出响应。该 CLI 路径当前使用 `scene.xml` groundcontact 和持续推进的 CPU MuJoCo/BAM 控制线程。`get_capabilities` 报告 `move_for`；`read_sensor` 的 IMU 与 14 个关节通道有效；`run_skill` 执行 0.3 m/s 前进命令 20 个实际控制步，`get_task_status` 报告 20/20 步及停止确认；第二项任务经 `cancel_task` 返回 cancelled；`stop_motion` 返回确认；输入结束后进程以状态码 0 关闭。原始响应、策略动作与位置数据位于 `.cache/omd-sim-jsonl-alpha-03/result.json`，物理控制日志位于同目录 `service-stderr.log`。

这次短任务在命令阶段的世界 x 位移约 0.01263 m，命令对应的位移为 0.12 m。`TaskStatus.SUCCEEDED` 当前表示指定命令作用了完整控制步且停止获得确认；该状态不表示运动速度达到命令值。独立的 0.5 前进响应比例门禁仍按上文结果判定。CLI 使用的 `body_twist` 已按官方 CPU 控制台方式改为 freejoint 世界线速度经 root quaternion 转换后的机身速度。

## 重复运行

在仓库根目录运行，临时目录固定为 `.cache/tmp`。`uv sync --locked` 使用提交中的 CPU 环境配置，且将虚拟环境放在忽略目录。

```bash
TMPDIR="$PWD/.cache/tmp" UV_CACHE_DIR="$PWD/.cache/uv" UV_PROJECT_ENVIRONMENT="$PWD/.cache/cpu-apartment-locked-venv" uv sync --project environments/cpu-apartment --locked --no-dev
TMPDIR="$PWD/.cache/tmp" HF_HOME="$PWD/.cache/huggingface" PYTHONPATH=src .cache/cpu-apartment-locked-venv/bin/python scripts/fetch_official_policies.py
TMPDIR="$PWD/.cache/tmp" PYTHONPATH=src .cache/cpu-apartment-locked-venv/bin/python scripts/validate_official_policies.py --directory .cache/official-policies/1b56c396825c052a4e26e95cf2b8d8298af9e9b4 --output .cache/official-policies/locked-validation.json
TMPDIR="$PWD/.cache/tmp" PYTHONPATH=src .cache/cpu-apartment-locked-venv/bin/python scripts/accept_official_command_slots.py --output .cache/official-command-slots/locked-result.json
TMPDIR="$PWD/.cache/tmp" PYTHONPATH=src .cache/cpu-apartment-locked-venv/bin/python scripts/accept_official_backend_contract.py --output .cache/official-backend-contract/locked-result.json
TMPDIR="$PWD/.cache/tmp" PYTHONPATH=src .cache/cpu-apartment-locked-venv/bin/python scripts/accept_observer_camera.py --output .cache/observer-camera-recheck
TMPDIR="$PWD/.cache/tmp" PYTHONPATH=src .cache/cpu-apartment-locked-venv/bin/python scripts/accept_omd_sim_jsonl.py --output .cache/omd-sim-jsonl-recheck
TMPDIR="$PWD/.cache/tmp" PYTHONPATH=src .cache/cpu-apartment-locked-venv/bin/python scripts/accept_official_apartment.py --policy alpha_walking --forward-speed 0.3 --move-ticks 250 --output .cache/official-apartment-recheck-alpha-03
TMPDIR="$PWD/.cache/tmp" PYTHONPATH=src .cache/cpu-apartment-locked-venv/bin/python scripts/accept_official_policy_modes.py --policy sitstand --output .cache/official-mode-recheck-sitstand
```

`accept_official_apartment.py` 在行为门禁未通过时返回状态码 2，并保存完整 `result.json` 与 PNG。独立目录保留每次运行的原始记录。本次未运行 GPU 或 Newton 仿真；Newton 训练环境与源码未更改。
