# Walking 配对训练停止记录（2026-09-30）

用户要求停止后台 RL 训练。2026-09-30 01:10 UTC 核查时，两个 launch supervisor（PID 490891、495773）及 summary 进程（PID 504612）均已终止，八组 Walking 配对任务的 worker、learner、预览与评估进程均已消失。`nvidia-smi` 仅显示其他项目 `openso101-v2` 的 PID 569875、569941；该项目未被触碰。停止前八组任务均已进入运行阶段。

全部输出、checkpoint 与 W&B 文件保持原有内容。四组 SB3 任务的输出目录共保留六份 `model.zip`，其中两组 MuJoCo 的记录达到 step 393216000，两组 Newton 的记录达到 step 196608000。固定源码的 `logs/rsl_rl/velocity/` 对应 full run 目录中，MuJoCo control/boost 均保留 `model_2000.pt`，Newton control/boost 均保留 `model_1000.pt`。这些文件是已有定期快照；停止时尚在内存中的更新没有确认为最终 checkpoint。

本轮完整训练预算与最终行为验收均未完成。训练保持停止状态，不自动恢复。原始启动身份和各次 attempt 的输出目录见[Walking 启动记录](rl-walking-low-speed-launch-2026-09-29.md)。
