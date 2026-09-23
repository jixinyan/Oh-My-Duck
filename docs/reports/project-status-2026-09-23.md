# 项目状态 · 2026-09-23

## GitHub 与源码

本次查询 GitHub，开放 Issues 为 0，开放 Pull Requests 为 0。尚未完成的功能与验证记录在项目文档和配置中。

Walking 逐阶段评分版本 3、完整训练输入检查和 SB3 指定目录恢复已进入 main，CPU 验证记录见 [训练检查与恢复验证](project-review-2026-09-21.md)。源码入口会将项目路径传递给子进程，两项独立 Python 子进程测试通过。

## 尚未完成的工作

| 优先级 | 项目 | 已有证据 | 完成条件 |
|---|---|---|---|
| P1 | Walking 行为验收 | 历史 231 份周期预览没有完整通过；MuJoCo SB3 官方课程具有相对稳定的前进响应 | 使用版本 3 完成前进、转向、停止、无推扰诊断、跨后端与 CPU/BAM 验证 |
| P1 | StandUp 稳定性 | Newton RSL 最终策略在两个原生后端 seed 42 通过四种姿态；CPU/BAM 完整测试为 14/17 seed | 使用多个训练 seed 和评估 seed 验证，重点检查 face_down |
| P1 | 新服务器训练环境 | Git 源码已经迁移，旧 checkpoint、normalizer、日志和视频未迁入当前仓库 | 完成锁定环境安装、资产转换、GPU 启动、导出、恢复和回放检查 |
| P2 | SB3 StandUp 与 Newton SB3 Walking | 历史起身停留在部分姿态成功，行走前进响应不足 | 根据无推扰轨迹和奖励记录定位原因，验证训练改动的效果 |
| P2 | 仿真执行接口 | `RobotBackend`、`SkillRunner` 为 Protocol；`ToolCatalog` 已有实现 | 在仿真中执行速度命令、查询状态、取消任务并确认停止 |
| P2 | 语音交互 | ASR、TTS、音色设计和音色保存已有接口定义 | 接入实际模型，实现录音、转写、音色确认与持久保存、合成与播放取消 |
| P2 | 主动感知 | sensor 与目标记录接口已有定义 | 接入带时间戳、坐标系和有效性信息的实际传感器数据 |
| P3 | 真机与部署 | 官方导出格式与 CPU/BAM 回放路径已有实现；设备验收未完成 | 在 Microduck 上验证部署、停止、传感器和音频流程 |

历史策略与训练状态依据 [Walking 复盘](rl-walking-policy-assessment-2026-09-14.md) 和 [因果回放](rl-causal-replay-2026-09-13.md)。这些结果来自旧服务器。本次新训练需要独立记录结果。

外部 Harness 继续负责通用 agent loop、长期记忆与经验检索；本项目可以独立推进语音服务、机器人 tools 和仿真执行。JSONL 记录器已有实现，后续执行过程需要接入记录器。

## jd_B300 训练计划

远端仓库为 `/mnt/data/users/jixin/workspace/code/Oh-My-Duck`。GPU 查询显示八张 `NVIDIA H20G`，每张约 268.6 GiB 显存，compute capability 为 10.3，driver 为 580.105.08。首次检查时 GPU 3–6 没有计算进程。启动训练前重新检查 GPU 使用情况。

配置：[jd-reproduction-20260923.json](../../configs/experiments/jd-reproduction-20260923.json)。配置已经通过 CLI dry-run，每个 learner 使用 8192 个环境。

| 训练 | seed | PPO 更新预算 | 目的 |
|---|---:|---:|---|
| MuJoCo / SB3 / Walking 官方课程 | 42 | 50000 | 完成较有希望的行走配置，并使用逐阶段标准评估 |
| Newton / RSL-RL / StandUp | 42 | 15000 | 在新主机复现已有起身结果 |
| Newton / RSL-RL / StandUp | 43 | 15000 | 检查训练 seed 对起身结果的影响 |

每组执行 64 环境、5 次更新的启动检查，完成奖励、导出、CPU 回放、恢复及容量检查后继续完整预算。每 1000 次更新保存 checkpoint 并生成后台视频；完整训练之后执行最终评估。W&B 使用 offline 模式。

## 本次运行记录

源码入口验证提交为 `743b6fb`，训练配置提交为 `47f1d17`。远端已获取这两个提交。

环境安装记录位于远端 `outputs/setup-20260923/`。安装依赖与图形库的验证正在执行，训练尚未启动。

本地 macOS、Python 3.12.12、Torch 2.9.1、SB3 2.7.1 环境执行 55 项 CPU 测试，全部通过。测试范围为 Walking/StandUp 评分、训练输入来源、SB3 指定目录保存与恢复、课程进度、运动诊断、公共接口和源码入口子进程。GPU 行为验证需要在新主机执行。

OSMesa、libglapi、LLVM 15 与 libdrm 已安装到项目的忽略目录；动态库加载检查和项目渲染依赖检查通过。实际仿真视频验证仍待执行。大型 wheel 使用锁文件地址和 SHA-256 校验；所有安装尝试保留各自日志。
