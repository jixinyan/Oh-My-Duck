# 项目进度 · 2026-09-29

记录日期采用客户端的 America/Chicago 日期。`jd_B300` 已保存的部分实验目录与报告使用 2026-09-30 标识；本文沿用这些产物的原始名称。核查时远端源码为 `12a4cd3`，工作目录没有未提交修改；八张 H20G 均显示 0 MiB、0% 使用率，没有匹配到训练或预览进程。GitHub 当前开放 Issues 为 0；未完成事项继续由功能状态和实验结果记录。

## 完整训练与行为结果

MuJoCo/RSL-RL Walking 控制组使用 8192 环境完成 50000 次 PPO 更新。最终 `model_49999.pt` 的原生环境计数器为 1200000，完整训练、导出与本地封装均返回 0。导出检查确认 44 个训练标量均为有限值，9 项惩罚符号正确。标准 MuJoCo/Newton sim2sim、CPU/BAM 演练和两种无推扰评估均完成执行，行为均未通过；49 份周期预览也全部记录为 `behavior_failed`。无推扰 MuJoCo 中，`0.1 m/s` 前进命令的响应比例为 0.00323，转向响应比例为 0.96141；五个阶段只有三个停止阶段通过。无推扰 Newton 的前进与转向响应比例分别为 0.00365、0.45516。训练来源为 `4253ee7`，结果保存在 `outputs/experiments/jd-walking-rsl-control-20260926/mujoco-rsl-rl-walking-official-seed42/`。

Newton/RSL-RL StandUp 的 seed 42、43 均从已完成 5001 次更新的原生 checkpoint 接续，最终各完成 15000 次更新。两份 `model_14999.pt` 的原生环境计数器均为 360000；完整训练、导出与本地封装均返回 0。两组各有 51 个训练标量通过有限值检查、9 项惩罚符号检查。两个 seed 的标准 MuJoCo/Newton sim2sim 均为站立和坐姿成功、俯卧和仰卧失败，即每个后端 2/4；两种无推扰评估同为 2/4。CPU/BAM 演练两组均仅站立成功，即 1/4。每组 10 份周期预览全部记录为 `behavior_failed`。恢复训练来源为 `eb8eb0e`，结果分别保存在 `outputs/experiments/jd-resume-20260926/isaac-newton-rsl-rl-standup-seed42/` 和对应的 `seed43/`。

上述三个 `result.json` 的最终状态均为 `behavior_failed`。训练、导出、封装与行为检查流程均已执行；训练完成和文件可导出仍需与策略行为验收分开记录。历史中断原因及源码比较见[中断训练诊断](rl-jd-interrupted-2026-09-26.md)与[官方实现核对](rl-official-comparison-2026-09-26.md)。

## 后续实验与项目能力

保存的 Walking 策略在受控命令扫描中，于 `0.2–0.4 m/s` 获得 0.717–0.781 的前进响应比例；所有扫描均未通过整体严格行为验收。`low_speed_tracking_boost` 保持默认值 0，控制组与 boost=1.0 组覆盖 MuJoCo/Newton、RSL-RL/SB3 的 8 个 run。远端四组 MuJoCo 与四组 Newton 产物均为 `prepared`，完成 smoke、导出、恢复、8192 环境容量与 CPU/BAM 训练前门禁；完整训练、最终无推扰评估和行为验收尚未执行。见[低速命令诊断](rl-walking-command-deadzone-2026-09-30.md)与[门禁记录](rl-walking-low-speed-gates-2026-09-30.md)。独立的 `Mjlab-ProtectiveFall-Flat-MicroDuck` 当前只有配置和 14-servo 编译证据，尚无训练策略。

Qwen3-ASR、Qwen3-TTS VoiceDesign、Base 文件合成及确认后跨进程复用同一音色版本已在目标 GPU 验证；麦克风采集、扬声器播放、播放中断和外部 Harness 语音链路仍待接入。确定性 Harness mock 已验证文本路由、工具参数检查、请求幂等、事件去重与取消边界；真实 EDH transport 仍待其接口稳定。尚未提交的 interactive simulation 草稿需要在真实后端核查速度命令、传感器状态、取消和停止确认，才能形成可调用机器人后端。感知数据接口与官方 61 维观察、14 个 servo、50 Hz 策略格式分别保留既有边界；项目目前没有可用真机，硬件验证尚未执行。语音与 Harness 证据见[文件语音验收](voice-validation-2026-09-26.md)、[Harness 审查](harness-integration-review-2026-09-30.md)和[官方上游复核](upstream-review-2026-09-30.md)。

当前训练主机没有运行中的 RL learner。已准备的 Walking boost 配对实验可以继续执行完整训练和标准最终验收，需同时比较低速前进、停止、转向、横向误差、跨后端和 CPU/BAM 结果。StandUp 需要针对俯卧和仰卧恢复开展因果诊断并取得可重复的行为证据。语音后续核查麦克风、扬声器和播放中断；interactive simulation 草稿后续接受真实后端验收。当前工作仅同步状态与确定问题范围，训练 GPU 保持空闲。只有达到对应行为标准的策略才能进入 locomotion tool 接入与硬件验证。

## 本次状态检查

远端三个完整训练的 `result.json`、`run.json`，最终 checkpoint 的原生计数，以及八个门禁 run 的状态均已读取；上述更新次数、行为结果和 `prepared` 数量与保存产物一致。本地更新后的 `configs/project.json` 通过 `python3 -m json.tool` 检查，`python3 omd.py status` 正常返回并显示最新训练、主机和行为验收状态；`git diff --check` 通过。此次检查没有启动训练或新的 GPU 评估。
