# 项目状态 · 2026-09-23

## GitHub 与源码

本次查询 GitHub，开放 Issues 为 0，开放 Pull Requests 为 0。尚未完成的功能与验证记录在项目文档和配置中。

Walking 逐阶段评分版本 3、完整训练输入检查和 SB3 指定目录恢复已进入 main，CPU 验证记录见 [训练检查与恢复验证](project-review-2026-09-21.md)。源码入口会将项目路径传递给子进程，两项独立 Python 子进程测试通过。

## 尚未完成的工作

| 优先级 | 项目 | 已有证据 | 完成条件 |
|---|---|---|---|
| P1 | Walking 行为验收 | 历史 231 份周期预览没有完整通过；MuJoCo SB3 官方课程具有相对稳定的前进响应 | 使用版本 3 完成前进、转向、停止、无推扰诊断、跨后端与 CPU/BAM 验证 |
| P1 | StandUp 稳定性 | Newton RSL 最终策略在两个原生后端 seed 42 通过四种姿态；CPU/BAM 完整测试为 14/17 seed | 使用多个训练 seed 和评估 seed 验证，重点检查 face_down |
| P1 | 新服务器训练环境 | CUDA 13 训练环境与 MuJoCo GPU 检查通过；W&B 在线账号和项目已验证；Isaac 资产转换通过；Walking 启动检查与导出通过 | 完成回放、恢复和容量检查，并执行完整预算 |
| P2 | SB3 StandUp 与 Newton SB3 Walking | 历史起身停留在部分姿态成功，行走前进响应不足 | 根据无推扰轨迹和奖励记录定位原因，验证训练改动的效果 |
| P2 | 仿真执行接口 | `RobotBackend`、`SkillRunner` 为 Protocol；`ToolCatalog` 已有实现 | 在仿真中执行速度命令、查询状态、取消任务并确认停止 |
| P2 | 语音交互 | ASR、TTS、音色设计和音色保存已有接口定义 | 接入实际模型，实现录音、转写、音色确认与持久保存、合成与播放取消 |
| P2 | 主动感知 | sensor 与目标记录接口已有定义 | 接入带时间戳、坐标系和有效性信息的实际传感器数据 |
| P3 | 真机与部署 | 官方导出格式与 CPU/BAM 回放路径已有实现；设备验收未完成 | 在 Microduck 上验证部署、停止、传感器和音频流程 |

历史策略与训练状态依据 [Walking 复盘](rl-walking-policy-assessment-2026-09-14.md) 和 [因果回放](rl-causal-replay-2026-09-13.md)。这些结果来自旧服务器。本次新训练需要独立记录结果。

外部 Harness 继续负责通用 agent loop、长期记忆与经验检索；本项目可以独立推进语音服务、机器人 tools 和仿真执行。JSONL 记录器已有实现，后续执行过程需要接入记录器。

## jd_B300 训练计划

远端仓库为 `/mnt/data/users/jixin/workspace/code/Oh-My-Duck`。GPU 查询显示八张 `NVIDIA H20G`，每张约 268.6 GiB 显存，compute capability 为 10.3，driver 为 580.105.08。CUDA 13.0 GA 要求 Linux driver 至少为 580.65.06。训练启动时逐张检查使用情况并明确指定空闲设备；其他任务已占用的设备保持原状。

配置：[jd-reproduction-20260923.json](../../configs/experiments/jd-reproduction-20260923.json)。配置已经通过 CLI dry-run，每个 learner 使用 8192 个环境。

| 训练 | seed | PPO 更新预算 | 目的 |
|---|---:|---:|---|
| MuJoCo / SB3 / Walking 官方课程 | 42 | 50000 | 完成较有希望的行走配置，并使用逐阶段标准评估 |
| Newton / RSL-RL / StandUp | 42 | 15000 | 在新主机复现已有起身结果 |
| Newton / RSL-RL / StandUp | 43 | 15000 | 检查训练 seed 对起身结果的影响 |

每组执行 64 环境、5 次更新的启动检查，完成奖励、导出、CPU 回放、恢复及容量检查后继续完整预算。每 1000 次更新保存 checkpoint 并生成后台视频；完整训练之后执行最终评估。W&B 使用已核对账号的 online 模式，并保留本地记录。在线环境验证记录为 [jd-b300-cuda13-environment-validation-20260923](https://wandb.ai/jixinyan831-northwestern-university/oh-my-duck/runs/cxxhmj25)。

## 本次运行记录

源码入口验证提交为 `743b6fb`，训练配置提交为 `47f1d17`。远端已获取这两个提交。

环境安装记录位于远端 `outputs/setup-20260923/`。正式 `.envs/mujoco`、`.envs/mujoco-sb3`、`.envs/isaac-newton` 均通过 `uv sync --locked`。Linux x86_64 的 MuJoCo 环境使用 Torch 2.9.1+cu130、torchvision 0.24.1+cu130、MuJoCo 3.10.0、Warp 1.12.0；Newton 环境使用 Torch 2.10.0+cu130、torchvision 0.25.0+cu130、MuJoCo 3.8.0、Warp 1.13.0。MuJoCo 的 Linux aarch64 锁定官方 cu129 来源，当前没有该平台的运行验证。Isaac 资产环境已完成安装，`artifacts/environments/isaac-assets/setup.json` 记录 `installed_not_worker_validated`；`walk` 与 `groundcontact` 转换命令均成功，日志位于 `outputs/setup-20260923/assets-*-online-04.log`；Newton worker 验证尚待运行。

本地 macOS、Python 3.12.12、Torch 2.9.1 和远端 Linux、Python 3.12.13 分别执行同组 55 项 CPU 测试，全部通过。Linux 的 MuJoCo SB3 环境使用 Torch 2.9.1+cu130，Newton 环境使用 Torch 2.10.0+cu130；测试记录为 `outputs/setup-20260923/linux-cpu-tests-cu130-mujoco-03.log` 和 `linux-cpu-tests-cu130-newton-02.log`。测试范围为 Walking/StandUp 评分、训练输入来源、SB3 指定目录保存与恢复、课程进度、运动诊断、公共接口和源码入口子进程。

OSMesa、libglapi、LLVM 15、libdrm 与 GLVND 已安装到项目的忽略目录。GPU 6 上的正式 MuJoCo 环境通过 TorchScript quaternion 连续十次调用、Warp、MuJoCo 物理状态及 EGL 图像检查；图像读取使用零初始化缓冲区，并检查形状、数值和非空像素。证据为 `outputs/validation-20260923/mujoco-cu130-target-probe-03` 和 `mujoco-cu130-base-probe-04`。原生 RSL 与导出模块已成功导入；实际 policy 导出仍需训练 worker 完成对应阶段。Microduck 训练与视频检查仍待运行。大型 wheel 使用锁文件地址和 SHA-256 校验；安装尝试各自保留日志。

Walking attempt04 已在 GPU 6 启动，worker PID `310920`，不可变源码提交 `c083476f1e4738e7d34d522fbb3b51be2b14a204`。64 环境、5 次更新的启动检查与 policy 导出已完成，回放和后续检查继续进行。启动检查的 [W&B 在线记录](https://wandb.ai/jixinyan831-northwestern-university/oh-my-duck/runs/rufx00sm) 已完成上传。运行目录为 `outputs/experiments/jd-reproduction-20260923-04/mujoco-sb3-walking-official`；完整预算尚未开始。旧服务器的 checkpoint、normalizer、日志和视频尚未迁入；本次训练从头开始。已停止的运行产物保留在各自目录。

两组 Newton StandUp 的资产环境安装与 `walk`、`groundcontact` 转换已完成。后台等待流程 PID `318849` 使用日志 `outputs/setup-20260923/resource-wait-online-07.log`，等待空闲 GPU 后分别为 seed 42、43 指定不同设备并启动 worker。转换、检查或启动失败时，流程记录错误并终止，不自动重试训练。Newton learner 尚未启动；两个训练 seed 的结果仍需结合多个评估 seed 检查稳健性。
