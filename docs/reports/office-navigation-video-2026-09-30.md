# Office 导航演示视频来源与编码验收

正式演示文件为 `outputs/demos/oh-my-duck-office-navigation-20260930.mp4`，来源是原生 EDH 任务 `b2cf1b1d-6ce6-4eff-a039-8ce8aec64a5e` 的官方导出 `.cache/demo/b2cf1b1d-6ce6-4eff-a039-8ce8aec64a5e/`。原始 `source/events.json` 保留全部 1579 条事件；视频依照实际时间顺序呈现截至首个 `run.succeeded` 事件的 1550 条事件，并在该正式终态停留供阅读。原始导出包含 152 张 `observer_follow.png` 第三视角图像，视频使用全部 152 张图像，保留各自的事件时间与仿真时间。

原生 ActionGate 批准并执行了 760 个控制步和 3040 个物理子步。最后 100 个控制步执行零命令，连续停止样本为 87；结束原因为 `policy_stop`。原生 `goal_reached=true`，结束位置为 `(0.7101244, 0.4637708, 0.1167998)` 米，目标条件保持 `108/5` 个控制步，非地面外部接触累计为零。独立 `verification.completed` 的结果为 `passed`，verdict 为 `8d0bdf35-f28c-40ce-9fce-4ba52dd21679`；原生任务终态为 `run.succeeded`。本次 `task_progress` 的九条已记录 `motion_guard.reason` 均为 `command_segment_complete`；另有一条初始进度没有该字段。本次任务没有触发近障碍、停滞或恢复暂停，画面只展示实际记录的动作段完成暂停与后续决策。

视频合并原始 observer 帧时间节点与公开文字、计划、正式判定的阅读停留节点。相邻节点的播放时长取 `max(wall_delta/8, adjacent_observer_simulation_delta)`；相邻原始 observer 帧之间的实际仿真时间至少以一倍时长播放，等待时段以八倍速度播放。调度记录包含 152 个 observer 节点、合计 168 个节点和 15 次阅读停留；observer 仿真时间从 `0.1` 秒至 `15.2` 秒，跨度为 `15.1` 秒，全部相邻 observer 区间的最小播放时长比例为 `1.0`。视频复用原始图像；文字只取原始公开事件与工具结果。公开 reasoning 事件数为零，因此画面没有补写模型内部思考。

成片编码为 H.264，分辨率 `1920×1080`，帧率 `10 fps`，总计 1735 帧、173.5 秒、5,753,965 字节。`ffprobe` 核对了编码、尺寸、帧数和时长；`ffmpeg` 对全片解码返回状态码零。渲染器对每一帧执行文字边界检查。五张检查帧保存在 `.cache/demo-review/oh-my-duck-office-navigation-20260930-{start,motion_early,motion_late,verifier,final}.png`；项目负责人已检查机器人全身、门口家具、时间顺序记录、传感器序号及正式判定的可读性。原生导出另通过 `scripts/accept_harness_replay.py --expected-verdict passed --expected-stop-reason policy_stop` 验收。

SHA256 来源记录：

- 视频 `outputs/demos/oh-my-duck-office-navigation-20260930.mp4`：`448577fbd53ddc96fb06ce055cd92ea0d7942ddcd467bb39d897f696f7abfc27`。
- 视频来源记录 `outputs/demos/oh-my-duck-office-navigation-20260930.json`：`4b52546d7d842b7c297806d103bb6fc6a4486c740896943f0464e831fd0f2878`。
- 来源清单 `.cache/demo/b2cf1b1d-6ce6-4eff-a039-8ce8aec64a5e/manifest.json`：`588449d1d97099e08898a8eb566f44a6dc889ccde58fe8600378bf37ac26fb65`。
- 原始任务 `.cache/demo/b2cf1b1d-6ce6-4eff-a039-8ce8aec64a5e/source/run.json`：`bccb91504a48a956bd95ce1a8d46b32baf1dd985eaa984fdc75a4eba69ea685c`。
- 原始事件 `.cache/demo/b2cf1b1d-6ce6-4eff-a039-8ce8aec64a5e/source/events.json`：`8391ccf11c1d6a67405445e226ab8839b3d3bdb447be5eba8c06c9f3ec01f295`。
- 物理后端 `src/oh_my_duck/robotics/backends/simulation.py`：`70a8b647664e841c6b46b3964363470e6aa501d2c8d81bea6465660a82ee2715`。
- 视频渲染器 `scripts/render_microduck_run.py`：`85b7aea4eacf411302324d7c17d978b83c9a8fc97444ee4974b3590f7e8bf547`。

先前演示文件 `outputs/demos/oh-my-duck-agentic-demo-20260930.mp4` 保持原路径和内容，SHA256 仍为 `1afbf52a295b526a222d1dfd3fb1405f8c3aee7ad7112d615dceeb0330929fb3`。

同名 `outputs/demos/oh-my-duck-office-navigation-20260930.json` 保存逐项来源、时间安排和每帧检查结果。使用锁定的 CPU 环境可从原始导出重复生成视频：

```sh
TMPDIR="$PWD/.cache/tmp" .cache/cpu-apartment-locked-venv/bin/python scripts/render_microduck_run.py --export .cache/demo/b2cf1b1d-6ce6-4eff-a039-8ce8aec64a5e --output outputs/demos/<new-output-name>.mp4 --wall-speed 8
```

重复生成时使用新的输出文件名，保留既有视频与原始记录。
