# 原生 EDH 仿真部署

`omd harness` 启动固定版本 EDH `8a5e685b22d032207f53db20454f0992a4ad60fd` 的 `startServer`。OMD 的 `integrations/edh/` 定义团队、工具与部署；`oh_my_duck.integrations.edh_native` 使用原生 `NativeWorkerSession`、`NativeActionDevice`、ActionGate 与独立 Verifier。CPU MuJoCo/BAM 公寓的官方 ONNX 策略目录固定为 `pollen-robotics/microduck-policies@1b56c396825c052a4e26e95cf2b8d8298af9e9b4`。EDH 源码及其工作目录保持独立，模型凭证仅从私有配置或进程环境读取。

在 OMD 根目录建立忽略 Git 的缓存目录，准备固定 EDH 工作目录与 Node 依赖：

```sh
mkdir -p .cache/tmp .cache/uv .cache/edh
git -C "$EDH_ROOT" worktree add --detach "$PWD/.cache/edh/8a5e685b22d032207f53db20454f0992a4ad60fd" 8a5e685b22d032207f53db20454f0992a4ad60fd
cd .cache/edh/8a5e685b22d032207f53db20454f0992a4ad60fd
TMPDIR="$PWD/../../tmp" pnpm install --frozen-lockfile
```

`EDH_ROOT` 指向已有 EDH Git 仓库。CPU 环境使用 `environments/cpu-apartment/uv.lock`，在 OMD 根目录执行：

```sh
TMPDIR="$PWD/.cache/tmp" UV_CACHE_DIR="$PWD/.cache/uv" UV_PROJECT_ENVIRONMENT="$PWD/.cache/cpu-apartment-locked-venv" uv sync --project environments/cpu-apartment --locked --no-dev
```

官方策略文件按照 `src/oh_my_duck/robotics/microduck/official_policies.py` 的固定 revision、文件名和 SHA256 准备于 `.cache/official-policies/1b56c396825c052a4e26e95cf2b8d8298af9e9b4/`。启动前 CLI 检查 EDH HEAD 与受 Git 管理源码的清洁状态；策略加载时检查官方 manifest 与 ONNX 文件。

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

## 模型连接检查

CLI 从 provider TOML 的 `wire_api` 选择固定 EDH 的 Responses 或 Chat Completions adapter。`--model-api` 可以明确选择接口；`--model` 可以明确选择模型。`--reasoning-effort` 使用 adapter 声明的能力和原生调用参数，服务拒绝请求时立即终止。模型 HTTP 请求使用项目客户端名称 `Oh-My-Duck/0.1`，凭证保留在进程内存。

`--data-dir` 中的 `model-transport.jsonl` 保存模型请求身份、HTTP 状态、Responses 结束状态、服务错误和 token 用量。记录不包含认证 header、请求内容或模型文字。SSE 使用固定 EDH 依赖的 `eventsource-parser` 读取；模型数据原样进入原生 adapter，服务错误立即终止当前请求。

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

`scripts/record_harness_demo.py` 通过原生 HTTP API 创建会话、提交任务、读取进展、导出终态和关闭会话。导出包含完整事件与 SHA256 检查后的原始 PNG。`scripts/render_microduck_run.py` 将相机画面、Planner 公开文字、计划、工具反馈与正式 verdict 同步到 MP4，支持 CPU 的 640×480 和 Isaac 的 1280×720 相机。等待时间可以压缩，运动片段保留物理时间下限。

```sh
.cache/cpu-apartment-locked-venv/bin/python scripts/record_harness_demo.py open \
  --profile nvidia-office-6.0 --session .cache/office-demo/session.json
.cache/cpu-apartment-locked-venv/bin/python scripts/record_harness_demo.py task \
  --session .cache/office-demo/session.json --scenario navigate-nvidia-office-6.0 \
  --instruction configs/simulation-demo/office-agentic-instruction.md \
  --output .cache/office-demo/task.json
.cache/cpu-apartment-locked-venv/bin/python scripts/record_harness_demo.py export \
  --run-id "$RUN_ID" --output ".cache/demo/$RUN_ID"
.cache/cpu-apartment-locked-venv/bin/python scripts/render_microduck_run.py \
  --export ".cache/demo/$RUN_ID" --output outputs/demos/office-agentic.mp4 --wall-speed 8
.cache/cpu-apartment-locked-venv/bin/python scripts/accept_harness_replay.py \
  ".cache/demo/$RUN_ID" --expected-verdict passed --expected-stop-reason policy_stop \
  --require-stop-progress
.cache/cpu-apartment-locked-venv/bin/python scripts/record_harness_demo.py close \
  --session .cache/office-demo/session.json
```

`RUN_ID` 使用任务提交返回的身份。`scripts/accept_harness_replay.py` 检查原始模型工具调用、实际位移、ActionGate 终态、独立 verdict、对应相机文件和物理子步。完成导出后，关闭会话并核查远程 worker 退出。

`microduck.task_progress` 返回 `stopped_samples` 与 `required_stopped_samples`，样本由物理控制步累计。`--require-stop-progress` 检查相同 episode、sequence 与 policy 的重复读取计数，以及 `finish_policy` 对应的连续停止样本。独立 Verifier 正式通过后，Planner 将原生目标项目设为 `done`，使用 `last_verdict_ref` 引用当前正式 verdict，再调用 `tasks.finish`。视频分别显示正式 verdict 和 run 终态。

真实 Astra 会话已完成图像及多种状态传感器读取、官方 14 关节动作、有界命令、暂停、策略切换与恢复。独立新会话的 office 任务从固定 corridor 出生位置开始，执行 760 个控制步，外部障碍接触累计 0；100 个零命令控制步后 `finish_policy` 确认物理停止，独立 Verifier 正式判定 `goal_reached=true`，run 状态为 `succeeded`。运动段按完成的物理控制步计数，靠近障碍、外部接触及实际停滞可以请求原生 Gate 暂停；这些阻碍情形由独立真实 worker 验证，本次 Astra 任务的暂停原因均为命令段结束。会话事件保存在所选 `--data-dir`，原始导出保存在 `.cache/demo/<run-id>/`。固定 EDH 自带 `scripts/export-run-replay.js`；完成的 run 可以使用以下命令导出完整原生事件、对应 PNG、视频与网页回放：

```sh
TMPDIR="$PWD/.cache/tmp" node .cache/edh/8a5e685b22d032207f53db20454f0992a4ad60fd/scripts/export-run-replay.js --base-url http://127.0.0.1:4318 --run-id "$RUN_ID" --output ".cache/demo/$RUN_ID" --camera observer_follow.png
```

第三视角摄像头以独立 operator frame 身份保存；Planner 仍使用头部 RGB 与状态传感器。控制台同页显示 agent 工具记录与最新实际相机帧，导出网页支持历史回放。视频时间以物理 `simulation_time_s` 为准。
