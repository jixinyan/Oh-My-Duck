---
title: Agentic Microduck — 分步执行计划
version: 0.3
status: In progress
created: 2026-09-06
updated: 2026-10-08
tags:
  - microduck
  - implementation-plan
  - sim-to-real
---

# Agentic Microduck 执行计划

## 当前工作与执行条件

当前开发完成不需要 GPU 的实现、调试、安装验证和源码组织，准备需要 GPU 的完整验收。
GPU 验收与 RL 保持停止。后续获得执行授权时，同时最多使用一张设备 2–4，
要求没有 compute PID 且持续零 utilization。保留其他用户和项目的进程。
当前没有可用 Microduck，真机步骤保留独立验收要求。

| 部分 | 已有证据 | 接续工作 |
| --- | --- | --- |
| 原生 Harness | 实际 Luna 文字和录音 CPU 导航、独立 Verifier、停止与原始资料核验 | 扩大声明场景的完整任务覆盖，保持实际目标和原有预算 |
| Policy 工具 | 原生米制行走、角度转向、姿态、转换、重试及训练包 CPU 执行 | 当前源码 Newton 长导航、所有严格精度与学习行为 |
| 感知 | CPU RGBD 与几何复核，固定源码 SAM3.1 + YOLO26 服务记录 | 独立标注、识别准确率、距离误差与目标跟踪 |
| 语音 | 三个固定 Qwen 模型 CPU 推理、四段 WAV 与全部回读、36 项音色和参数测试 | 主观音质、指令语料、交互延迟、机载设备 |
| 资产与依赖 | 两种模型、Office/Hospital CPU 导入、官方 contact model 和发布准备 | 当前源码 GPU 求解、首次 kernel 初始化与渲染 |
| 安装与记录 | 434 个文件、三份许可证、26 个独立调用，原生事件与 agentic MP4 | 干净环境完整流程复现、可分发模型与资产说明 |

每项证据使用对应源码和配置，详细记录见[CPU 开发验证](reports/cpu-development-readiness-2026-10-08.md)、
[语音模块](reports/voice-audio-modules-2026-10-08.md)、[语音导航](reports/cpu-voice-navigation-2026-10-08.md)、
[当前状态](implementation-status.md)和[发布要求](release-readiness.md)。
完整产品范围由[Project Design](Agentic%20Microduck%20-%20Project%20Design%20v0.1.md)维护。

## 1. 步骤、依赖和完成依据

| 步骤 | 工作 | 主要依赖 | 完成依据 |
| --- | --- | --- | --- |
| 00 | 共享接口、模块、入口与配置 | 产品设计 | 实际导入、参数检查、源码定位与公开入口 |
| 01 | 官方审读与版本来源 | 官方仓库 | 阅读覆盖、实现路径、许可证与固定版本 |
| 02 | 官方训练 backend 与基线 | 01、可用计算资源 | 原生训练、恢复、导出、回放和行为记录 |
| 03 | Isaac/Newton 资产与环境 | 01–02 | 结构、参数、reset、实际 solver 与来源 |
| 04 | BAM 与接触 | 03 | 单关节、负载、摩擦、延迟、接触和随机化 |
| 05 | 观测、动作和任务语义 | 04 | 固定状态、官方 policy 与跨 backend 比较 |
| 06 | Walking/StandUp 训练 | 05、独立 GPU 执行授权 | 既定训练预算、行为标准、seed 和评估 |
| 07 | 官方导出与部署回放 | 06 | normalized ONNX、数值 parity、schema 2 和 CPU/BAM |
| 08 | 板载和真机控制 | 07、实际 Microduck | 板载周期、控制与真实任务 |
| 09 | Policy、技能和通用工具 | 可用 policy、05/07 | 参数、物理效果、停止与后端语义 |
| 10 | Native Harness 文本任务 | 09、固定 EDH、真实 provider | 模型工具调用、ActionGate、独立 Verifier |
| 11 | 显式录音与 ASR | 10、音频设备与 Qwen 环境 | 录音来源、实际转写、同一任务链路 |
| 12 | 声音设置、持久音色与 TTS | 10、Qwen 环境 | 试听确认、不可变版本、跨会话复用 |
| 13 | Microduck 音频与中断 | 11–12、实际设备 | 机载命令与反馈、播放停止和动作取消 |
| 14 | RGB、ToF 与主动感知 | 09、传感器或仿真 | 当前帧、有效性、几何、识别与距离 |
| 15 | 目标交互、经验与完整 Demo | 10–14 | 目标效果、独立结果、声音和完整记录 |
| 16 | 可复现开源交付 | 所声明能力的验收 | 干净安装、教程、来源、示例与分发检查 |

每个步骤保留完整要求。已有有效证据按源码、配置和功能核对后复用。
设备条件缺失时继续独立的仿真、CPU、语音和文档工作，设备行为在实际环境验收。

## 2. 各步骤的实现与通过条件

### 第 00 步：模块、接口与入口

维护 `RobotBackend`、`SkillRunner`、`ToolCatalog`、`ActivePerception`、`PolicyAdapter`、
`HarnessBridge`、语音协议、`EpisodeRecorder`、训练接口和 `ApplicationServices`。
依赖从具体实现指向共享类型；应用和公开 CLI 在导入时保持模型、Torch 与模拟器延迟加载。
工具注册验证完整 Draft 2020-12 Schema，调用验证有限参数和实际 handler。
原生任务接口提供 `open`、`submit`、`status`、`wait`、`stop`、`close`。

交付源码说明、依赖方向、公开命令、场景 Schema、配置和实际检查。
通过条件包括模块实际导入、错误参数就地拒绝、身份关联、sim/real 记录、明确的 Newton 执行路径。
入口与 source map 见[architecture](architecture.md)和[应用源码](../src/oh_my_duck/agentic/README.md)。

### 第 01 步：官方源码与来源

固定 `microduck_rl`、`microduck`、policy、Isaac、Newton、BAM 和 Harness 版本。
建立文件清单与阅读覆盖，资产、生成文件和外部依赖单独记录。
跟踪训练入口、模型、执行器、MDP、PPO、checkpoint 和 exporter；
跟踪运行时的 policy 加载、61 维观测、14 维动作、joint control、传感器和停止。
核查相机、ToF、双 IMU、采音和播放的实际调用，保留版本与设备条件。

交付源码路径、关节顺序、HOME、动作处理、BAM 时序、导出格式、
迁移差异、许可证及需要继续核查的项目。通过条件为每项关键行为能够指向具体实现。
配置位于 `configs/upstream.json`，来源保存在 `third_party/`。

### 第 02 步：官方 backend 与基线

使用独立锁定环境和固定官方来源，保留实际依赖与配置。
运行匹配的官方已发布 policy，并保存前进、转向、停止、不同初态的测量和视频。
训练检查覆盖 reset、有限状态和 reward、日志、checkpoint 保存与 native resume。
使用官方导出和 CPU MuJoCo/BAM 加载路径完成回放。
评估固定命令、初态、seed、阈值和统计方法，测量速度跟踪、跌倒、脚滑、吞吐与资源。

交付可重复命令、参考环境、policy 来源、导出、回放与行为结果。
通过条件为重启后重复执行相同链路，结果对应明确的源码、配置和权重。
代表性范围保持 Walking/StandUp，官方原始对照保持独立。

### 第 03 步：Isaac/Newton 环境与资产

使用 `rl/backends/isaac_newton/` 的实际环境、task binding 和 asset 入口。
固定兼容的 Isaac Lab、Newton、MuJoCo-Warp、Warp、Python 与 PPO 依赖。
导入官方资产，比较 geometry、joint、质量、惯量、坐标、foot、collision、armature 和 HOME。
转换保存完整输入、生成文件 SHA256 和 source fingerprint；明确复用检查 clean ancestor、
相同输入、实际依赖和全部生成文件。

CPU 检查执行两种模型、passive wheel joint、USD 材质与几何、Office/Hospital 外部场景。
GPU 验收在获得执行授权后检查实际 solver、物理步、初始化、reset 和所有必要 joint state。
交付锁定环境、可复现资产、实际参数比较及来源。
`omd validate model-assets` 的 CPU 范围与实际求解分别记录。

### 第 04 步：BAM、接触与并行状态

实现官方 BAM XL330 M6、负载和电压行为、摩擦、物理子步延迟、encoder 和 passive joint 映射。
明确引擎与 actuator 各自负责的摩擦和阻尼。
比较相同 joint 阶跃、负载、延迟及 foot contact，核查 solver 材质、mask 和 ground pair。
Newton manager 在 CUDA graph capture 之前编译官方接触数据并保留 solver 数据所有权。
各 world 使用独立随机化、摩擦和缓存；reset 清除旧执行状态，随机化保持非累积。

交付 actuator、接触配置、响应曲线、数值结果和差异依据。
通过条件为相同输入下满足声明的比较标准，并能够定位超出标准的物理和时序差异。
当前 CPU 数组与实际 GPU 求解分别提供证据。

### 第 05 步：观测、动作与任务

保持 61 actor observations、14 named servos、canonical HOME、50 Hz 和四个 0.005 秒物理子步。
核对 joint order、单位、action scaling/clipping、previous action、command encoding 和 reset 初值。
迁移官方 reward、termination、command sampling、noise、NaN 保护和 sensor-view semantics。
固定状态比较观测与 ONNX action，在两个 backend 使用相同 policy、battery 和初态。
actor 使用目标设备可获得的观测，评估与 critic 的世界真值明确归属。

交付任务 factories、观测动作数值比较、官方 policy 的跨 backend 行为与视频。
通过条件包含任务语义、计数、接触、停止和各场景表现。
差异定位到模型、执行器、时序、reset 或观测后修正并重新验证。

### 第 06 步：代表任务训练与有效行为

范围为 Walking/StandUp × MuJoCo/Isaac-Newton × RSL-RL/SB3。
训练使用原生 PPO，保留 framework checkpoint、normalizer、curriculum、critic 和恢复设置。
完成官方 64-env/5-iteration smoke、数值与 penalty-sign 检查、native resume 和导出门槛。
按照已提交 campaign 的环境数、seed、预算、保存周期与评估规则运行。
W&B 读取 `configs/training.json` 的 online 配置和明确的环境变量，所有 worker 记录最终模式。

行为评估使用预先声明的命令、初态、失败条件和多个 seed，保存学习曲线、失败样本和视频。
按相同 curriculum 阶段诊断持续退化；不同 reward 实验保持独立。
交付 checkpoint、完整配置、训练与恢复来源、终止状态及最终行为报告。
通过条件为实际达到任务标准。RL 当前停止，本步骤的 GPU 执行等待明确的恢复指令。

### 第 07 步：官方格式、数值一致性与回放

通过官方 `run_export` 和 runner export 输出包含 native normalization 的 ONNX。
用同一观测批次比较 native/Torch 与 ONNX，保存数值误差及 source/checkpoint 身份。
检查名称、形状、joint、command、robot variant、metadata 和官方 schema 2。
策略包使用官方 publisher 格式，保留训练、版本、SHA256、兼容与评估资料。
冻结 policy 在两种 backend 与 CPU MuJoCo/BAM 运行相同 battery。

交付完整 policy 包、numerical parity、轨迹、指标和视频。
格式、回放和行为满足相应标准后标记 `sim_validated`。
本地检查完成之后，公开上传使用独立的发布授权与分发条件。

### 第 08 步：板载与真机

需要实际 Microduck 和官方运行时访问。
保存已知可用 policy 与配置，提供文件恢复和设备恢复操作。
测量板载加载、推理输出、耗时、控制周期、电池与运行状态。
按官方流程验证站立、前进、转向、停止，再运行声明的任务 battery。
保存硬件、runtime、地面、配置、通信、传感器与失败证据。

交付部署说明、真实设备日志、行为结果、恢复操作和已验证版本。
通过条件为目标设备满足既定任务标准，具体 policy 标记 `hardware_validated`。
设备条件未满足时本步骤保持待验收，独立的仿真工作继续。

### 第 09 步：通用工具、policy 与技能

注册有来源和适用条件的官方 policy，附加训练包通过 `--policy-registry` 明确接入。
保持共同 joint ONNX 检查，距离/角度工具的 locomotion policy 明确配置。
工具支持 policy catalogue、选择与转换、bounded command、walk、rotate、sensor、progress 和 finish。
`walk` 参数使用米，`rotate` 使用度；按实际 odometry、原始误差和五个停止样本完成。
转向返回实际平移，姿态命令验证实际头部与身体响应。

检查重复请求、明确修改命令、调用者等待超时、任务重试、旧 generation、动作期限和通信中断。
长工具记录实际停止、episodic 时长、接续 policy、goal evidence 和资源释放。
仿真与真机共享单位、结果和取消语义，能力由后端实际提供。
交付工具 Schema、skill 声明、实际连续动作与取消证据，见[工具说明](metric-policy-tools.md)。

### 第 10 步：原生 Harness 与文字任务

固定 EDH 源码、原生 SDK、Node 部署、SessionEnvironment、EmbodiedBackend、ActionGate 和 Verifier。
`NativeTaskClient` 连接真实 server，模型凭证通过私有配置与环境变量传递。
工具直接使用原生 worker，真实传感器和官方 policy 驱动物理状态变化。
规划保留速度、距离、角度、方向、目标、剩余预算和原生状态。
验证暂停、恢复、policy 转换、取消、连续任务及保持物理状态的重试。
连续录音在同一原生 session 中提交，明确选择最多四项已结束任务作为历史引用。
检查新任务的历史指令、结果、正式 verdict、归属和实际模型 brief，
以及收尾状态、当前物理状态与资源关闭。Planner 使用原生
`skills.search` 和 `skills.load` 按需读取持久经验。

交付真实模型任务、正式 verdict、完整事件、执行身份、物理资料与资源关闭。
通过条件包含实际目标完成、独立 Verifier 和原始证据复核。
现有 CPU 文字和录音导航具有对应报告，当前源码的多场景长任务继续独立验收。
两项连续录音已通过 CPU 正式 Verifier 和独立复核，包含任务历史引用、
经验搜索、新的目标保持记录、两个固定音色反馈 WAV 与资源释放，
见[任务历史引用](reports/voice-task-context-2026-10-09.md)。
运行方式见[原生部署](harness-native-integration.md)。

### 第 11 步：显式录音与 ASR

使用明确的开始和结束录音，保存文件、采样率、通道、SHA256 与请求身份。
Qwen ASR 使用固定模型和锁定环境，设备参数与音频内容在调用位置检查。
将真实转写通过同一原生客户端提交，保留音频与模型来源。
空录音、非有限采样、无有效文字和服务错误在对应位置终止请求。
以实际指令语料测量文字、意图正确率、转写耗时与资源。

交付 ASR 服务、录音入口、原生任务接入与语料结果。
通过条件为真实音频、真实模型与实际任务可追溯，识别性能使用独立样本评估。
本机设备和机载设备分别验证，录音生成来源明确记录。

### 第 12 步：Setup、固定音色与 TTS

使用 VoiceDesign 生成描述对应的候选，用户试听并明确确认。
SQLite 保存不可变 VoiceProfile、参考 WAV、文字、模型 revision、SHA256 与活动选择。
日常 Base TTS 自动使用活动版本，确认请求与冲突并发写入接受一致性检查。
重启、不同会话和后端切换继续使用同一 persona 的声音。
未确认候选、修改失败和取消 setup 保持当前音色。

比较短句、长句与不同文本的声音一致性，测量首次与后续合成延迟及资源。
独立解码 WAV、检查真实 ASR 回读和数据库保持。
交付候选、确认、查询、合成、HTTP 接口和试听样本。
源码与当前检查见[语音模块验证](reports/voice-audio-modules-2026-10-08.md)及[声音资料](voice-profiles.md)。

### 第 13 步：机载音频与交互中断

需要实际 Microduck 麦克风、扬声器和运行时。
核查官方音频 worker 的采集所有权，复用共享数据或明确调度采集。
录音传到外部推理服务，固定音色 WAV 传回设备播放，保存传输与播放状态。
开始新录音停止当前播音；动作取消使用原生任务停止路径，等待实际设备确认。
参考交互入口包含连接、录音、文字、声音 setup 和明确停止。

在安静和舵机运动背景下测试指令、噪声、传输、完整延迟、播放中断及任务取消。
Episode 保存实际播放进度和终止原因。
交付机载适配、完整语音应用和设备评估。
通过条件为用户通过机载麦克风提交命令，小鸭完成动作并用固定声音反馈。

### 第 14 步：传感器、模型感知与主动观察

RGB、8×8 ToF、IMU、joint 和 odometry 保留时间、frame、有效性、calibration 与来源。
CPU 和 Newton 的公开感知接口使用当前图像、depth、segmentation 和真实 camera geometry。
`simulator_ground_truth` 与 `models` 来源明确选择，配置、metadata 和工具 Schema 一致。
模型路径使用 SAM3.1 分割、YOLO26 关联和实际 RGBD 距离，保留空目标与模型身份。
头部观察与图像关联同一 physical sequence；过期、缺失和无效测量具有明确结果。

独立核对 frame、SHA256、目标框、mask、bearing、几何距离和读期间物理状态。
模型服务与客户端从原始像素复算测量，SAM 保留源图像尺寸的二值 PNG mask。
使用 `omd validate perception` 执行真实模型调用及已保存资料复核，
当前 CPU 结果见[测量验证](reports/perception-measurements-2026-10-08.md)。
识别与距离误差使用独立标注，目标跟踪保留时间、身份、丢失与停止规则。
仿真噪声、聚合与延迟依据实际设备标定。
交付传感器、主动观察、模型服务、calibration 和测量资料，见[感知工具](perception-navigation.md)。

### 第 15 步：目标交互、experience 与 agentic Demo

实现受限场景的静态目标接近及移动目标跟随，各自声明目标有效性、距离、快速反馈和停止。
使用适用 policy，保持原始任务目标、预算、动作精度、ToF、接触和姿态要求。
多场景展示有来源的 locomotion、坐立、头部、伸头、轮滑及策略转换。
抓取和携带任务需要实际物体状态变化及独立目标检查。
图像输入独立 VLN/VLA 使用专门 action/observation 适配与评估。

Episode 记录语音、观察、工具、物理结果、声音版本、仿真/真机 domain 和源码。
将事实与证据交给 Harness 的记忆接口，重启后保留身份、声音和历史来源。
MP4 包含实际场景、head RGB、公开 agentic trace、工具参数、结果与音频。
编码后执行完整解码、帧数与时长、文字范围、图像 SHA256 和原始事件核验。

交付完整目标任务、正式 Verifier、原始资料、经验接口与可播放 Demo。
通过条件为每个结果可追溯到实际观察和执行，所属服务、会话和 worker 全部关闭。

### 第 16 步：开源安装、示例与扩展

维护安装、模型获取、配置、启动、训练、恢复、导出、部署与问题定位文档。
提供仿真、原生工具、声音 setup、语音任务、传感器、policy 和新技能的实际入口。
依赖环境保持锁定与分离，使用实际依赖验证对应功能。
wheel 和 source distribution 检查 source/resources、许可证、项目目录之外的 CLI 和原始资料复核。
从干净环境执行公开流程，核对模型、场景和资产的获取与分发条件。

交付 README、source map、教程、环境和版本矩阵、policy 来源、评估方法及已验收范围。
通过条件为他人能够按文档运行所声明的功能并找到扩展位置，真实设备支持对应具体设备证据。

## 3. 当前接续顺序

1. 完成 CPU 可执行路径、实际模型与原生任务、源码组织、接口文档和独立安装检查。
2. 核对需要 GPU 的源码、依赖、场景、policy、contact、模型和 release plan；保存准备报告。
3. 根据保留记录诊断 Office 长行走后顺时针响应和 Hospital 后退停滞，保持原有阈值及预算。
4. 在新的 GPU 执行授权之后，按六阶段计划串行验证当前源码求解、矩阵、连续动作和完整任务。
5. 根据正式任务与实际识别评估推进长导航、目标交互、VLN 和物体效果。
6. 在 RL 恢复指令之后完成代表性训练的有效行为与多 seed 验收。
7. 在实际 Microduck 到位后执行板载、传感器、音频与真机任务。
8. 按通过的功能核验干净安装、Demo、教程和分发，准备公开版本。

GPU 执行计划使用[完整 release campaign](runtime-release-acceptance.md)，
保持实际模型、官方 policy、原生 ActionGate、独立 Verifier、原始媒体与进程释放。
每个阶段保存实际 passed/failed，执行失败保留产物并在原因定位之后更新代码。

## 4. 每次迭代与验收记录

| 内容 | 保存要求 |
| --- | --- |
| 具体目标 | 明确行为、输入、场景和通过条件 |
| 源码 | 固定 Git revision、配置、dirty state 和实际导入来源 |
| 环境与资源 | 锁定依赖、模型 revision、CPU/GPU 分配和进程身份 |
| 实际执行 | 命令、事件、物理计数、传感器、task verdict 和媒体 |
| 独立核验 | 原始文件、SHA256、数值、目标和终止状态 |
| 资源关闭 | 会话、服务、worker、端口与所属进程退出 |
| 文档 | 设计、执行计划、implementation status、CLI maturity 和 source map |
| 交付 | feature branch 的相关检查通过后合入 main 并推送 |

所有实验使用独立输出目录。源码位于 `src/oh_my_duck/`，依赖锁位于 `environments/`，
配置位于 `configs/`，中间结果位于忽略的 `.cache/`，实际输出位于 `outputs/`。
失败、checkpoint 和原始证据保留。声明能力使用对应范围的实际运行结果。
