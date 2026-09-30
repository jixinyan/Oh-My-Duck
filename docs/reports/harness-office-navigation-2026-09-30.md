# 官方公寓 office 导航验收

2026-09-30 的原生 EDH 会话 `5f727c5b-0188-4884-8e41-f08e8d990980` 使用真实 `gpt-6-astra` 模型、CPU MuJoCo/BAM 公寓和官方预训练 ONNX 策略。任务 `b2cf1b1d-6ce6-4eff-a039-8ce8aec64a5e` 从 seed `20260929`、出生位姿 `(0, 0, 0)` 开始，目标为原生 `office` 房间检查，要求连续满足 5 个控制步。初始目标状态未满足。EDH 固定源码为 `8a5e685b22d032207f53db20454f0992a4ad60fd`；策略目录为 `pollen-robotics/microduck-policies@1b56c396825c052a4e26e95cf2b8d8298af9e9b4`，实际选用 `alpha_walking` 的 ONNX SHA256 为 `e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c`。

Planner 调用原生工具读取头部 RGB 10 次、ToF 10 次、里程计 10 次、IMU 3 次、关节状态 3 次，并读取公开场景与策略目录。`velstand` 零命令执行 10 步与 50 步；随后选择 `alpha_walking`，按实时传感器和位置依次设置六段各 100 步的 `twist`：`[0.2,0.25,0]`、`[0.2,0.25,0.4]`、`[0.2,0.25,0]`、`[0.2,0.15,-0.15]`、`[0.2,0.1,0]`、`[0.2,0.05,-0.1]`。每段经过原生 ActionGate 批准后执行，完成时自动请求暂停。最后执行 `[0,0,0]` 100 步，累计 760 个控制步、3040 个 MuJoCo 物理子步。`task_progress` 在 0、10、60、160、260、360、460、560、660、760 控制步记录的位置从 `(0,0)` 移动至 `(0.71012,0.46377)` 米；每次读取的非地面外部接触累计数均为 0，该值由物理后端逐子步累计。此次任务未触发近障碍、外部接触或停滞暂停。

`finish_policy` 在执行结束边界确认当前命令为零、已执行 100 个零命令控制步、连续停止样本 87，身体速度为 `[-0.001387,-0.000270,-0.001027]`。原生 Gate 以 `policy_stop` 和设备确认状态结束 execution `b436601b-9b7d-4921-87de-b7b0521ae8b0`，边界 `bde1712d-340c-4297-abfc-a9c6df61479c`。独立 Verifier 的 `verification.check` 读取原生目标检查，得到 `goal_reached=true`；证据为目标范围内的世界坐标 `(0.7101244,0.4637708,0.1167998)` 米、姿态条件满足和连续达标 `108/5` 个控制步。`verification.completed` 的正式结果 `passed`、verdict `8d0bdf35-f28c-40ce-9fce-4ba52dd21679` 绑定同一 execution 与结束边界。Planner 调用原生 `tasks.finish`，`run.succeeded` 指向该 verdict。

原始导出位于 `.cache/demo/b2cf1b1d-6ce6-4eff-a039-8ce8aec64a5e/`：`source/run.json`、`source/events.json`、`frames.json`、`manifest.json`、152 张真实 `observer_follow.png` 第三视角帧及原生视频。1579 条完整事件保留工具、传感器、动作、暂停、执行结束及正式验证身份。`scripts/accept_harness_replay.py` 已通过：逐项核对有界命令与九个运动暂停的 run、execution、边界、请求编号及实际步数；同时核对原始事件顺序、第三视角帧 SHA256、物理时间、策略选择、实际位置变化、Gate 结束边界、独立 Verifier verdict 和 `run.succeeded` 的身份关系。原始 `source/events.json` SHA256 为 `8391ccf11c1d6a67405445e226ab8839b3d3bdb447be5eba8c06c9f3ec01f295`，`source/run.json` 为 `bccb91504a48a956bd95ce1a8d46b32baf1dd985eaa984fdc75a4eba69ea685c`。

从同一导出生成的演示文件为 `outputs/demos/oh-my-duck-office-navigation-20260930.mp4`，SHA256 为 `448577fbd53ddc96fb06ce055cd92ea0d7942ddcd467bb39d897f696f7abfc27`；同名 JSON 保存来源与时间验证，SHA256 为 `4b52546d7d842b7c297806d103bb6fc6a4486c740896943f0464e831fd0f2878`。成片采用 H.264、1920×1080、1735 帧、173.5 秒；全片解码与文字时间边界检查通过，运动画面对应 152 张原生第三视角帧。

从项目根目录准备 CPU 依赖并启动相同配置的原生服务；`PRIVATE_PROVIDER_CONFIG` 指向已授权的模型配置文件，凭证仅由启动进程读取。控制台选择 `official-apartment-office` 和 `navigate-office`，实际任务使用真实模型自行决策。

```sh
TMPDIR="$PWD/.cache/tmp" UV_CACHE_DIR="$PWD/.cache/uv" UV_PROJECT_ENVIRONMENT="$PWD/.cache/cpu-apartment-locked-venv" uv sync --project environments/cpu-apartment --locked --no-dev
TMPDIR="$PWD/.cache/tmp" PYTHONPATH=src .cache/cpu-apartment-locked-venv/bin/python omd.py harness --remote-provider-config "$PRIVATE_PROVIDER_CONFIG" --ssh-host jd_B300 --remote-python /home/jixin/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12 --port 4338 --data-dir .cache/harness-data-navigation-01
```

完成任务后以实际 run ID 导出原始记录，并检查正式结果：

```sh
TMPDIR="$PWD/.cache/tmp" node .cache/edh/8a5e685b22d032207f53db20454f0992a4ad60fd/scripts/export-run-replay.js --base-url http://127.0.0.1:4338 --run-id b2cf1b1d-6ce6-4eff-a039-8ce8aec64a5e --output .cache/demo/b2cf1b1d-6ce6-4eff-a039-8ce8aec64a5e --camera observer_follow.png
TMPDIR="$PWD/.cache/tmp" .cache/cpu-apartment-locked-venv/bin/python scripts/accept_harness_replay.py .cache/demo/b2cf1b1d-6ce6-4eff-a039-8ce8aec64a5e --expected-verdict passed --expected-stop-reason policy_stop
```

运行时 OMD 文件 SHA256：`edh_native.py` 为 `ff4d88ef9977c3ac1cbcc098afb89fa89405fba2f4950246358e65112f19a9cd`，`motion_guard.py` 为 `d1a5dd49c9352b547ad10529165a4d3a64725bfea3111c2e4a4a3b4f7e738f63`，`server.mjs` 为 `12af5fdbb48e5a4911f5a781d1c8c20c40875ca2708d7637ee9adfc78276b2e7`，`planner.md` 为 `4433ccee0209a17c1acbfdc66dd89fc59d20fdbfa1a675f12cecc4777a25ead9`，`simulation.py` 为 `70a8b647664e841c6b46b3964363470e6aa501d2c8d81bea6465660a82ee2715`，`official_policies.py` 为 `576d5b0372257a5e5f0f83b5bc52d3dd63787d1ca43e08a7b26f3efbabe44dd7`。OMD 物理模块提交为 `421d268`，有界命令与停止模块提交为 `985bbf9`。私有模型凭证仅由本机进程读取，没有写入导出。

独立真实 worker 测试在同一 CPU 物理后端验证其他边界：`.cache/motion-guard/forward-stop-01.json` 在前向 ToF 86 毫米处暂停，外部接触累计 0；`.cache/motion-guard/lateral-stall-01.json` 的纯侧向命令在 51 个行走控制步后触发实际停滞暂停；`.cache/motion-guard/forward-recovery-lease-02.json` 验证旧命令被拒绝、读取新鲜 ToF 后更改命令并恢复；`.cache/motion-guard/disconnect-during-pause-01.json` 验证断线与暂停交错后控制步数保持不变；`.cache/motion-guard/lateral-finish-boundary-01.json` 验证移动命令和未执行的零命令均无法通过 `finish_policy`，实际零命令控制及停止样本满足要求后才结束。所有这些记录使用真实 ONNX、MuJoCo/BAM 与原生 Gate。此次 Astra 导航通过范围为单一 CPU 场景、固定 seed 与出生状态；策略通用速度跟踪、其他 seed、Newton 交互和真机执行仍需分别验收。
