# 原生 EDH 仿真部署

`omd harness` 启动固定版本 EDH `8a5e685b22d032207f53db20454f0992a4ad60fd` 的 `startServer`。OMD 的 `integrations/edh/` 定义团队、工具与部署；`oh_my_duck.integrations.edh_native` 使用原生 `NativeWorkerSession`、`NativeActionDevice`、ActionGate 与独立 Verifier。CPU MuJoCo/BAM 公寓的官方 ONNX 策略目录固定为 `pollen-robotics/microduck-policies@1b56c396825c052a4e26e95cf2b8d8298af9e9b4`。EDH 源码及其工作目录保持独立，模型凭证仅从私有配置或进程环境读取。

`omd setup-harness` 使用已有 EDH Git 仓库中的固定提交建立独立源码副本。
需要 Node.js 22.19.0 或更新版本、npm、Git 和 uv。执行时从 SDK 的
`packageManager` 使用 pnpm 11.19.0，通过 `--frozen-lockfile` 安装全部工作区依赖。
命令检查源码清洁状态、原生 SDK 导出与 physical Schema，按固定 SHA256 获取
manifest 和十项官方 ONNX，安装 `environments/cpu-apartment/uv.lock` 声明的 CPU 环境。
Python SDK 由固定源码的 `harness/physical-runtime` 安装；其 jsonschema 与
websockets 依赖使用 CPU 环境中的锁定版本。安装检查逐个比较 SDK Python
文件与固定源码，并读取原生 physical Schema。
凭证和模型服务设置在后续启动时提供。

原生上下文设置位于仓库根目录 `integrations/edh/model-context.mjs` 的
`nativeContextManagement`。部署声明 `32768` tokens 的上下文容量，模型
输出预算通过 `--max-output-tokens` 设置。原生 SDK 使用额外的
`4096`-token headroom；compaction 的 `thresholdRatio` 为 `0.7`，
`retainRatio` 为 `0.15`，摘要输出上限为 `8192` tokens。
模型输出预算为 `8192` 时，原生 SDK 计算的触发阈值为 `20480`，
保留预算为 `3686` tokens。图像历史由原生 `VisualHistory` 管理，
最多保留十二张图像；原始事件与 attachment 来源继续保存。

Linux 无界面 CPU 仿真通过 `--mujoco-renderer osmesa` 明确选择软件渲染。
Ubuntu 22.04 amd64 的固定 OSMesa 可以使用已有准备命令安装：

```sh
CUDA_VISIBLE_DEVICES='' python omd.py setup --backend mujoco --software-renderer --skip-env
CUDA_VISIBLE_DEVICES='' .cache/cpu-apartment-locked-venv/bin/omd harness \
  --provider-config "$PRIVATE_PROVIDER_CONFIG" --mujoco-renderer osmesa
```

启动入口设置 MuJoCo 与 PyOpenGL 的 renderer，并将已准备的本地动态库路径传递给
新进程。使用系统 OSMesa 时，由系统提供对应动态库及其依赖。
`--mujoco-renderer` 同样适用于 SSH CPU worker，设置在实际仿真主机上生效。
SSH worker 使用同一 `infrastructure/headless.py` 入口启动原生 runtime；
该选项要求 CPU MuJoCo 场景。

SSH CPU worker 明确设置 `CUDA_VISIBLE_DEVICES=''`；该设置在远程进程中生效。
会话关闭后要求原生资源释放、worker 退出和所属服务正常结束，
验证记录见[CPU worker 检查](reports/native-cpu-ssh-2026-10-11.md)。
Mac 承担模型连接的长时间检查可以使用 `caffeinate -is` 包含实际启动命令，
系统唤醒要求随该命令结束而释放。
两个 seed 的完整 Luna CPU 导航、原始事件与图像、正式 Verifier 和资源释放
已通过独立检查，见[重复任务验收](reports/native-cpu-navigation-2026-10-11.md)。

```sh
python omd.py setup-harness --edh-repository "$EDH_ROOT" \
  --output outputs/setup/harness-new
```

`EDH_ROOT` 指向包含固定提交的已有 EDH Git 仓库。默认 SDK、策略和 Python
路径与 `omd harness` 相同。`--edh-source`、`--policy-dir` 和
`--cpu-environment` 可以明确选择独立目录。输出目录必须为新的目录；
`result.json`、各阶段日志、下载原件和所属进程退出记录保存在该目录。
现有源码和策略必须通过来源检查。仿真 worker 已在其他主机准备时，
`--skip-cpu-environment` 仅执行 Node 与策略准备，记录明确保留 CPU 环境状态。

锁定 CPU 环境可以通过同一公开入口检查：

```sh
CUDA_VISIBLE_DEVICES='' .cache/cpu-apartment-locked-venv/bin/omd doctor \
  --runtime cpu-apartment \
  --edh-source .cache/edh/8a5e685b22d032207f53db20454f0992a4ad60fd \
  --catalog .cache/official-policies/1b56c396825c052a4e26e95cf2b8d8298af9e9b4
```

官方来源与 SHA256 位于 `robotics/microduck/policy_sources.py`，该模块只包含来源资料。
`official_policies.py` 检查 manifest、图结构、joint/HOME、61/14 维度及实际 CPU 推理。
`integrations/native_runtime.py` 由准备与启动共同检查固定 EDH HEAD 和源码清洁状态。
依赖、SDK 导入和推理检查具有各自范围；正式任务通过实际模型与独立 Verifier 验收。

有私有 Codex provider TOML 时运行：

```sh
PYTHONPATH=src .cache/cpu-apartment-locked-venv/bin/python omd.py harness --provider-config "$PRIVATE_PROVIDER_CONFIG"
```

已部署 OpenAI-compatible 图像与 tools 模型时，可以通过进程环境直接连接。无认证的本地服务无需设置 `EDH_MODEL_API_KEY`：

```sh
EDH_MODEL_BASE_URL=http://127.0.0.1:8000/v1 EDH_MODEL=your-vision-tool-model PYTHONPATH=src .cache/cpu-apartment-locked-venv/bin/python omd.py harness
```

浏览器打开 CLI 输出的本地地址。EDH 原生控制台同页显示 agent 工具记录与真实相机图像；侧栏可选择已保存的任务，事件与图像保留原生 run/evidence 身份。选择 `official-apartment-office` 建立会话，再选择 `navigate-office`。新会话从固定 seed 与 corridor 出生位置开始；同一会话的后续任务沿用实际物理状态。Planner 可读取公开公寓连接关系、RGB、ToF、IMU、关节与里程计，设置官方策略命令，并在真实停止确认后切换策略。目标坐标与成功检查留在环境及独立 Verifier。

正常结束时，Planner 在已确认暂停边界设置有界零 `twist` 命令，经原生 `execution.resume` 实际执行至少五个控制步，再读取 `task_progress` 的身体速度与停止状态；公寓导航验收使用 100 个零命令控制步。随后从当前确认边界读取 `execution_id`、`generation`、`boundary_id`，调用 `finish_policy`。该工具由物理 owner 确认当前零命令、已执行零命令控制步、连续停止样本和身体速度，原生 ActionGate 再以 `policy_stop` 结束执行。EDH 主机分配独立 Verifier 读取新鲜物理观测与目标状态。真实成功需要该正式结果支持。取消通过 EDH 原生任务接口执行，并以确认边界及后续动作计数检查。原生工具支持 perpetual、scripted 和 episodic policy；CPU 验证覆盖 `kick_left` 的时长边界与显式接续 `alpha_stand`，动作效果需要对应的物理证据。

## 场景配置

`--scene-config` 使用共用 JSON Schema，Python CLI 与 Node
部署通过相同 JSON Schema 检查参数。Schema 随安装包发布；源码文件为
`src/oh_my_duck/integrations/edh/scene.schema.json`，Python 校验位于
`scene_configuration.py`，Node 校验位于 `integrations/edh/scene-configuration.mjs`。

| 后端 | 机器人与资源 | 出生位置 | 目标 |
| --- | --- | --- | --- |
| `cpu-mujoco-bam` | 官方公寓，`allcollisions` | `x_m`、`y_m`、`yaw_rad` | `room`、`dock`、`object` |
| `isaac-newton` | USD 与来源资料，标准脚或轮滑模型 | `x_m`、`y_m`、`z_m`、`yaw_rad` | `point`，可附加有序 `waypoints` |

配置声明 `scene_id`、`task_instruction` 和 `budget`。数值必须有限，控制预算至少为
5 次且必须为整数，`hold_ticks` 为 1–500 的整数。CPU 公寓使用原生房间与物体名称，配置中
选择 `device` 时必须为 `cpu`。Newton 内部设备使用 `cuda:0`；远程分配通过
`--worker-cuda-device` 明确选择物理 GPU。当前 GPU 与 RL 保持停止。

使用已准备的固定 EDH、官方 policy 和私有 provider 配置启动 CPU 场景：

```sh
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .cache/cpu-apartment-locked-venv/bin/python omd.py harness \
  --provider-config "$PRIVATE_PROVIDER_CONFIG" \
  --scene-config configs/simulation-demo/apartment-metric.json
```

此配置的 profile 为 `official-microduck-apartment`，scenario 为
`navigate-official-microduck-apartment`。`--policy-registry` 可以同时指定训练包。
各 policy 的学习行为、每项导航任务和 GPU 执行分别进行验收。
7 份现有配置、23 组无效配置、完整 CLI/Node 启动及安装资源检查见
[配置验证](reports/native-scene-configuration-2026-10-08.md)。

## 模型连接检查

CLI 从 provider TOML 的 `wire_api` 选择固定 EDH 的 Responses 或 Chat Completions adapter。`--model-api` 可以明确选择接口；`--model` 可以明确选择模型。`--reasoning-effort` 使用 adapter 声明的能力和原生调用参数，服务拒绝请求时立即终止。模型 HTTP 请求使用项目客户端名称 `Oh-My-Duck/0.1`，凭证保留在进程内存。

模型连接读取 `HTTP_PROXY`、`HTTPS_PROXY`、`NO_PROXY` 及对应的小写环境变量。
代理连接由固定依赖 Undici 的 `EnvHttpProxyAgent` 管理；服务结束时关闭所属
dispatcher，并恢复原有 dispatcher。实际模型检查与正式任务使用相同连接配置。

`--data-dir` 中的 `model-transport.jsonl` 保存模型请求身份、HTTP 状态、Responses 结束状态、服务错误和 token 用量。记录不包含认证 header、请求内容或模型文字。SSE 使用固定 EDH 依赖的 `eventsource-parser` 读取；模型数据原样进入原生 adapter，服务错误立即终止当前请求。

Responses function tools 明确使用 `strict: false`，保留原生 SDK 的可选参数
和完整 Schema。SDK 负责接纳参数、工具权限和执行；关闭服务时恢复模型传输
资源。实际 Luna 报告读取和独立安装检查见[工具调用记录](reports/native-tool-parameters-2026-10-11.md)。

在分配 GPU 仿真前，使用相同的私有配置检查实际模型请求：

```sh
PYTHONPATH=src .cache/cpu-apartment-locked-venv/bin/python omd.py harness \
  --provider-config "$PRIVATE_PROVIDER_CONFIG" --reasoning-effort high --check-model
```

该检查必须收到实际文本与正常结束事件；检查成功后退出。正式任务仍需完成图像、工具、物理执行与独立 Verifier 验收。

## 远程 GPU 仿真

Node Harness 可以运行在本机，Python native worker 通过 SSH 运行在 NVIDIA 主机。远程主机需要完整的 OMD 源码副本、锁定的仿真环境、固定 EDH Python runtime 与 schema、官方 policy，以及当前场景资源。远程端通过 stdout 发送原生结构化通信，日志使用 stderr；SSH 转发当前会话的控制端口。关闭会话时，原生 worker 确认释放物理设备，SSH 进程随之结束。

运行前检查 GPU 使用情况，将 `IDLE_GPU_INDEX` 设置为当前空闲设备。`WORKER_ROOT` 指向准备好的源码副本，`REMOTE_PYTHON` 指向该主机的 Isaac Newton 环境：

```sh
PYTHONPATH=src .cache/cpu-apartment-locked-venv/bin/python omd.py harness \
  --provider-config "$PRIVATE_PROVIDER_CONFIG" --reasoning-effort high \
  --worker-host jd_B300 --worker-root "$WORKER_ROOT" \
  --worker-python "$REMOTE_PYTHON" --worker-edh-source "$REMOTE_EDH_SOURCE" \
  --worker-policy-dir "$REMOTE_POLICY_DIR" --worker-cuda-device "$IDLE_GPU_INDEX" \
  --scene-config configs/simulation-demo/office.json --port 4338 \
  --data-dir .cache/harness-office-demo
```

场景配置中的相对资源路径以远程 `WORKER_ROOT` 为基准。CUDA 设备由 `--worker-cuda-device` 指定，远程 worker 内部使用 `cuda:0`。固定 EDH Node 运行时继续使用本机完整源码与依赖，远程端执行相同 revision 的 Python physical runtime。

## Agentic MP4

距离与角度指令使用 `microduck.walk(distance_m)`、`microduck.rotate(angle_deg)`；运动结束与测量结果见[工具接口](metric-policy-tools.md)。Newton Office 的实际工具检查、相机来源和原生模型 Demo 见[验证记录](reports/metric-camera-tools-2026-09-30.md)。录制距离工具任务时，为下方 `accept_harness_replay.py` 命令增加 `--require-metric-tools`；该选项检查实际模型调用、odometry 目标、停止误差、五个停止样本与当前命令身份。

`scripts/record_harness_demo.py` 通过原生 HTTP API 创建会话、提交任务、读取进展、导出终态和关闭会话。导出包含完整事件与 SHA256 检查后的原始 PNG。`scripts/render_microduck_run.py` 将相机画面、Planner 公开文字、计划、工具反馈与正式 verdict 同步到 MP4，支持 CPU 的 640×480 和 Isaac 的 1280×720 相机。等待时间可以压缩，运动片段保留物理时间下限。

```sh
.cache/cpu-apartment-locked-venv/bin/python scripts/record_harness_demo.py open \
  --profile nvidia-office-6.0 --session .cache/office-demo/session.json
.cache/cpu-apartment-locked-venv/bin/python scripts/record_harness_demo.py task \
  --session .cache/office-demo/session.json --scenario navigate-nvidia-office-6.0 \
  --instruction configs/simulation-demo/office-agentic-instruction.md \
  --output .cache/office-demo/task.json
.cache/cpu-apartment-locked-venv/bin/python scripts/record_harness_demo.py export \
  --run-id "$RUN_ID" --output ".cache/demo/$RUN_ID" --data-directory .cache/harness-office-demo
.cache/cpu-apartment-locked-venv/bin/python scripts/render_microduck_run.py \
  --export ".cache/demo/$RUN_ID" --output outputs/demos/office-agentic.mp4 --wall-speed 8
.cache/cpu-apartment-locked-venv/bin/python scripts/accept_harness_replay.py \
  ".cache/demo/$RUN_ID" --expected-verdict passed --expected-stop-reason policy_stop \
  --require-stop-progress
.cache/cpu-apartment-locked-venv/bin/python scripts/record_harness_demo.py close \
  --session .cache/office-demo/session.json
```

`RUN_ID` 使用任务提交返回的身份。`scripts/accept_harness_replay.py` 检查原始模型工具调用、实际位移、ActionGate 终态、独立 verdict、对应相机文件和物理子步。完成导出后，关闭会话并核查远程 worker 退出。

`--data-directory` 使用服务启动时的 `--data-dir`。当前导出包含 `read_sensor(head_rgb)` 与 `inspect_scene` 的原生 tool image attachment，保存原始 PNG、SHA256、字节数与物理帧身份。感知与多阶段导航的运行参数见[感知工具](perception-navigation.md)。

`microduck.task_progress` 返回 `stopped_samples` 与 `required_stopped_samples`，样本由物理控制步累计。`--require-stop-progress` 检查相同 episode、sequence 与 policy 的重复读取计数，以及 `finish_policy` 对应的连续停止样本。独立 Verifier 正式通过后，Planner 将原生目标项目设为 `done`，使用 `last_verdict_ref` 引用当前正式 verdict，再调用 `tasks.finish`。视频分别显示正式 verdict 和 run 终态。

真实 Astra 会话已完成图像及多种状态传感器读取、官方 14 关节动作、有界命令、暂停、策略切换与恢复。独立新会话的 office 任务从固定 corridor 出生位置开始，执行 760 个控制步，外部障碍接触累计 0；100 个零命令控制步后 `finish_policy` 确认物理停止，独立 Verifier 正式判定 `goal_reached=true`，run 状态为 `succeeded`。运动段按完成的物理控制步计数，靠近障碍、外部接触及实际停滞可以请求原生 Gate 暂停；这些阻碍情形由独立真实 worker 验证，本次 Astra 任务的暂停原因均为命令段结束。会话事件保存在所选 `--data-dir`，原始导出保存在 `.cache/demo/<run-id>/`。固定 EDH 自带 `scripts/export-run-replay.js`；完成的 run 可以使用以下命令导出完整原生事件、对应 PNG、视频与网页回放：

```sh
TMPDIR="$PWD/.cache/tmp" node .cache/edh/8a5e685b22d032207f53db20454f0992a4ad60fd/scripts/export-run-replay.js --base-url http://127.0.0.1:4318 --run-id "$RUN_ID" --output ".cache/demo/$RUN_ID" --camera observer_follow.png
```

第三视角摄像头以独立 operator frame 身份保存；Planner 仍使用头部 RGB 与状态传感器。控制台同页显示 agent 工具记录与最新实际相机帧，导出网页支持历史回放。视频时间以物理 `simulation_time_s` 为准。
