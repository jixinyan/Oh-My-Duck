# 原生 EDH 公寓部署

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

正常结束时，Planner 在已确认暂停边界设置有界零 `twist` 命令，经原生 `execution.resume` 实际执行至少五个控制步，再读取 `task_progress` 的身体速度与停止状态；公寓导航验收使用 100 个零命令控制步。随后从当前确认边界读取 `execution_id`、`generation`、`boundary_id`，调用 `finish_policy`。该工具由物理 owner 确认当前零命令、已执行零命令控制步、连续停止样本和身体速度，原生 ActionGate 再以 `policy_stop` 结束执行。EDH 主机分配独立 Verifier 读取新鲜物理观测与目标状态。真实成功需要该正式结果支持。取消通过 EDH 原生任务接口执行，并以确认边界及后续动作计数检查。当前原生 EDH 工具执行 perpetual 策略；目录中的 episodic 策略仍可查看，但其原生 EDH 执行未开放。

真实 Astra 会话已完成图像及多种状态传感器读取、官方 14 关节动作、有界命令、暂停、策略切换与恢复。独立新会话的 office 任务从固定 corridor 出生位置开始，执行 760 个控制步，外部障碍接触累计 0；100 个零命令控制步后 `finish_policy` 确认物理停止，独立 Verifier 正式判定 `goal_reached=true`，run 状态为 `succeeded`。运动段按完成的物理控制步计数，靠近障碍、外部接触及实际停滞可以请求原生 Gate 暂停；这些阻碍情形由独立真实 worker 验证，本次 Astra 任务的暂停原因均为命令段结束。会话事件保存在所选 `--data-dir`，原始导出保存在 `.cache/demo/<run-id>/`。固定 EDH 自带 `scripts/export-run-replay.js`；完成的 run 可以使用以下命令导出完整原生事件、对应 PNG、视频与网页回放：

```sh
TMPDIR="$PWD/.cache/tmp" node .cache/edh/8a5e685b22d032207f53db20454f0992a4ad60fd/scripts/export-run-replay.js --base-url http://127.0.0.1:4318 --run-id "$RUN_ID" --output ".cache/demo/$RUN_ID" --camera observer_follow.png
```

第三视角摄像头以独立 operator frame 身份保存；Planner 仍使用头部 RGB 与状态传感器。控制台同页显示 agent 工具记录与最新实际相机帧，导出网页支持历史回放。视频时间以物理 `simulation_time_s` 为准。
