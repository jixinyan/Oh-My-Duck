# Agentic MicroDuck 演示来源与验收

演示视频为 `outputs/demos/oh-my-duck-agentic-demo-20260930.mp4`。同名 JSON 记录生成参数、来源文件的 SHA256、视频 SHA256、事件范围、终态时间与编码检查。原始导出保存在 `.cache/demo/6b8b5c45-2f8a-45fa-8d28-e95a7465c668/`，包含 `source/run.json`、`source/events.json`、`frames.json`、`manifest.json` 和 800 张 `observer_follow.png` 原生相机帧。

运行使用 CPU MuJoCo/BAM、官方 `scene_apartment.xml` 和固定官方 ONNX 策略。独立第三视角相机跟随 `trunk_base`，参数为距离 0.5 m、azimuth 45°、elevation −45°；画面为实际 MuJoCo 渲染的 640×480 PNG。相机源码 `src/oh_my_duck/robotics/backends/simulation.py` 的 SHA256 为 `61085ad9cf2a27598c03324362c464e959058a65a3c05963ae8004a87011d92a`。视频中的 Planner 文字、工具调用、计划、执行状态与正式 Verifier 判定均来自原始事件。没有公开文字的 `agent.output` 不占用 trace 卡片或阅读停留。

该次原生执行完成 4000 个经 ActionGate 接受的控制步、16000 个 MuJoCo 物理子步，终止原因为 `budget_exhausted`，执行设备确认停止。独立 Verifier 判定 `goal_reached=false`，最终机器人世界位置为 `(0.581692, 0.021770, 0.116736) m`，目标保持步数为 `0/5`。正式判定为 `failed`，Planner 随后提交 `tasks.abandon`，原生运行终态为 `failed`。这段视频保留这些结果。

官方导出含 5223 个原始事件。视频依事件时间顺序呈现前 5196 个事件，结束于 `2026-09-30T02:03:22.447Z` 的 `run.abandoned`，并为终态保留 3.5 秒阅读时间。终态之后的事件继续完整保存在 `source/events.json`。原始时间间隔按 4 倍速度播放；Planner、计划、正式判定和终态另有明确阅读停留。视频只使用运行中已经记录的 800 张 observer 帧，没有补造画面或内部推理。

最终视频为 H.264、1920×1080、10fps、1506 帧、150.6 秒、5,876,024 字节。生成时逐帧检查所有文字边界，`ffprobe` 确认编码、分辨率、帧率、帧数和时长。五张时间点检查帧位于 `.cache/demo-review/oh-my-duck-agentic-demo-20260930-{start,quarter,middle,threequarter,final}.png`，均经 Pillow 确认尺寸为 1920×1080 且像素存在变化。原始 `observer_follow` 在运行到 sequence 815 时已由人工检查，机器人全身与公寓场景清晰可见。

可在锁定 CPU 环境中从原始导出重新生成视频。`--output` 必须指向尚未存在的新文件：

```sh
TMPDIR="$PWD/.cache/tmp" PYTHONPATH=src .cache/cpu-apartment-locked-venv/bin/python scripts/render_microduck_run.py \
  --export .cache/demo/6b8b5c45-2f8a-45fa-8d28-e95a7465c668 \
  --output outputs/demos/oh-my-duck-agentic-demo-reproduced.mp4 \
  --review-dir .cache/demo-review
```
