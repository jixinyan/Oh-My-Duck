# Implementation status

Updated: 2026-10-13.

## Execution conditions

GPU acceptance and RL remain stopped. Subsequent authorized GPU execution may use
at most one idle device from physical GPUs 2–4, with zero compute PIDs and sustained
zero utilization. Other users' workloads are preserved. Actual Microduck hardware
is unavailable; hardware acceptance remains pending.

The current work completes CPU implementation, actual execution, dependency and
installation checks, source organization and GPU preparation. Each result uses
its recorded revision, environment, model and configuration. The complete product
scope remains in [Project Design](Agentic%20Microduck%20-%20Project%20Design%20v0.1.md);
work requirements and dependencies are in the
[Execution Plan](Agentic%20Microduck%20-%20Execution%20Plan%20v0.1.md).

## CLI maturity matrix

| Component | Verified scope | Remaining acceptance |
| --- | --- | --- |
| Source organization | Owned code under `src/oh_my_duck`; native Node deployment under `integrations/edh`; capability source maps | Continue documenting new capability and adapter entry points |
| CLI/application | 原生服务、验证与记录、场景 Schema、policy package 与语音；457 个安装文件、三份 license、Mac 与 Linux 各 34 项 CLI 调用 | 声明的完整任务在独立环境中的执行检查 |
| RL 参数入口 | 共同的 `argparse` 模块、27 项检查、六项评估入口、20 项原生训练命令和八次实际 CLI 调用；[验证记录](reports/training-arguments-2026-10-11.md) | 原生物理与训练验收使用各自声明的任务标准 |
| 发布模型设置 | 共享模型与 API 参数；实际原生 Luna 调用；六阶段、四场景完整发布前检查；查询期限与失败记录 | 声明的实际 GPU 导航运行验收 |
| 原生准备入口 | Mac 与 Linux 公开准备、固定 SDK 与 pnpm、11 项实际下载、两台主机原生 Luna 调用；Linux 九阶段退出、28 个 SDK 文件、锁定 CPU 环境及新环境连续物理运行通过；[入口记录](reports/native-preparation-2026-10-11.md) | 声明场景覆盖与连续语音任务 |
| 原生 MuJoCo renderer | 显式 OSMesa；Linux 本地与 Mac→SSH 会话；300 次控制、1,200 次物理步骤；150 次独立 ONNX 与 RGB 复核；两台主机各 28 项独立安装调用；[运行记录](reports/native-renderer-2026-10-11.md) | 多场景完整模型导航 |
| 原生 context 设置 | 默认 headroom、摘要预算和图像历史；两台主机六项实际 SDK 预算检查；四项来源文件与各 28 项独立安装调用；[配置记录](reports/native-context-preparation-2026-10-11.md) | 多场景长任务与更广泛的重复执行统计 |
| 原生 Responses 工具参数 | SDK 参数 Schema、实际 Luna 可选参数调用与正式报告读取；传输关闭；两台主机各 28 项独立安装调用；[调用记录](reports/native-tool-parameters-2026-10-11.md) | 多场景长任务与更广泛的模型任务统计 |
| SSH CPU worker | 实际进程的 CUDA visibility 为空；关闭后进程数量归零、session 释放资源、服务退出状态为零；两台主机各 28 项独立安装调用；[会话记录](reports/native-cpu-ssh-2026-10-11.md) | 多场景完整模型导航 |
| SSH 服务连接 | 共用 Python 参数、明确 keepalive、端口转发和参数拒绝；实际 CPU 服务连续空闲 660 秒，HTTP 转发保持有效；SSH 与服务退出 0、端口关闭；121 项检查、两个主机独立安装与原始模型任务安装版回放通过；[连接记录](reports/ssh-idle-lifecycle-2026-10-13.md) | 完整模型任务中的服务管理、多场景长时间连接与网络中断统计 |
| 原生 CPU 重复导航 | 两个 seed 完整 Luna 任务成功，正式 Verifier 通过；1,411 次控制、5,644 次物理步骤、2,661 条事件、281 张 observer 图像和 37 条停止进度记录通过独立检查；两段 1080p agentic MP4 的 665 帧完整解码通过；[验收记录](reports/native-cpu-navigation-2026-10-11.md) | 多场景长导航、独立图像导航和更广泛的重复任务统计 |
| GPU 资源读取 | 公共 CSV 模块、实际设备及重复进程记录、31 项检查、占用拒绝及持续空闲检查 | 每次分配重新检查实际状态；当前 GPU 运行验收保持停止 |
| Core types, tools and recording | Complete Draft 2020-12 Schemas, finite arguments, actual handlers, request identity, JSONL/native events, original image hashes and MP4 | New tools require their actual implementation and relevant execution evidence |
| Native Harness | Fixed EDH, actual models, official policies, sensors, ActionGate and independent Verifier; CPU text/recorded-voice office tasks passed | Current-source multi-scene long navigation and repeated task statistics |
| Policy tools | Metric units, pose response, policy transitions, bounded commands, state-preserving retries, progress and stop evidence | All long-distance precision, current Newton controller and object effects |
| 原生运动等待 | 同一控制步的物理与 controller 进度；461 次动作 ONNX 误差为零、11 条等待与取消检查；完整 Luna 任务通过 Verifier，485 次控制、十条模型等待与九项记录拒绝；101 项检查、两个主机安装、157.5 秒 MP4 通过；[任务记录](reports/model-motion-wait-2026-10-13.md) | 多场景重复任务、Newton 与真机执行 |
| 可见目标接近 | Luna 使用 YOLO26 检测调用 `approach_object` 的正式 CPU 任务通过；496 次控制、1,984 次子步、0.67 毫米误差；原始 RGBD、来源 SHA256、停止与 Verifier 通过复核，160 秒 MP4 完整解码；80 项场景与导入检查、两个主机各 454 文件和 33 项命令通过；[模型任务](reports/model-target-task-2026-10-13.md)、[控制测量与中断](reports/target-approach-2026-10-12.md) | 多场景重复模型任务、Newton 与真机执行 |
| 连续目标跟踪 | 原生 `read_track`、移动物体与实际 YOLO26 关联通过；425 次控制与逐动作 ONNX 复核；正式模型感知任务在 sequence 85、95 查询同一目标并完成接近、停止与 Verifier；[连续关联](reports/target-tracking-2026-10-12.md)、[模型来源任务](reports/model-target-task-2026-10-13.md) | 自主跟随、物体交叉、长时间遮挡、Newton 与真机执行 |
| Policy registry | Ten official joint graphs and registered training packages; actual CPU Walking/StandUp package execution, complete episodic duration and continuation | Learned behavior, scene generalization and hardware |
| Perception | Current CPU RGBD geometry, actual CPU YOLO, original-pixel measurement checks, source-bound tracking and saved-data audit; fixed-source Newton sensors and SAM3.1 + YOLO26 records | Current-source SAM mask/inference GPU acceptance, independently labelled accuracy, crossings/long occlusion and calibration |
| Voice interaction | Shared audio module; three fixed Qwen CPU models, four decoded WAV files and four matching ASR texts; 36 profile/admission tests; confirmed profile/database preservation | Voice quality, command-corpus accuracy, interaction latency and Microduck devices |
| ASR 真实录音 | FLEURS 中文与英文全部 1,592 条；中文 CER 7.681%、英文 WER 5.306%；原始录音、全部 WAV 与指标复核通过；六项数据拒绝、445 个安装文件与 29 项公开命令通过；[测量记录](reports/asr-corpus-2026-10-12.md) | 机器人命令意图、噪声、完整交互延迟与机载麦克风 |
| ASR 公开流程 | 全部语料准备、真实 CPU 转写和独立复核；五次安装说明命令、七项输入拒绝；两个主机各 447 个安装文件、三份 license 与 31 项 CLI 调用；所属 ASR 进程归零；[流程记录](reports/public-asr-workflow-2026-10-12.md) | 机器人命令意图、噪声、交互延迟与设备评估 |
| Voice task | Actual Qwen ASR → OpenAI `gpt-6-luna` high → native tools → formal Verifier → confirmed-profile feedback; CPU interruption and resource release | Live hardware recording/playback and broader speech-task statistics |
| MuJoCo RL | Both representative tasks and both native PPO frameworks have training/resume/export/replay lifecycle evidence | Effective learned behavior and final representative acceptance |
| Isaac/Newton RL | Both representative tasks and both native PPO frameworks have training/resume/export/replay lifecycle evidence | Current-source GPU execution, effective behavior and final evaluation |
| Newton physics and assets | Actual recorded solver/BAM audits; current CPU asset/contact checks cover both robot variants and Office/Hospital colliders | Current-source GPU solver, first kernel initialization, rendering and runtime campaign |
| Export and local packages | Native normalization, ONNX parity, official schema 2, source/checkpoint identity and CPU/BAM replay | Learned task success, hardware loading and public distribution |
| Multi-GPU frameworks | Native RSL MuJoCo DDP has recorded evidence; SB3 uses native vector environments and independent runs | Newton DDP acceptance; execution follows current single-GPU resource restriction |
| Hardware | Official source, interfaces and requirements are recorded | Actual robot control, sensors, audio, deployment and physical tasks |

`omd status` reads [`configs/project.json`](../configs/project.json) and describes
software maturity. Online device capability discovery comes from the actual
backend. The [architecture source map](architecture.md#source-organization)
identifies each implementation.

## Actual CPU workflows and evidence

客户端的 PortAudio 与 libsndfile 加载检查已在 macOS 和 Ubuntu 22.04 通过；
Linux 独立安装检查通过 438 个文件和 27 项 CLI 调用。
记录见 [音频动态库检查](reports/native-audio-readiness-2026-10-09.md)。

| Capability | Recorded result | Evidence |
| --- | --- | --- |
| 发布模型与查询期限 | 117 项检查，实际原生 SDK 模型请求、六阶段四场景发布前检查；440 个安装文件；三十秒查询期限与中止记录 | [发布模型设置](reports/release-model-settings-2026-10-09.md) |
| GPU 资源读取 | 31 项实际记录检查、持续空闲检查、占用拒绝及测量线程退出；439 个安装文件、Mac 29 项与 Linux 27 项 CLI 调用 | [GPU 状态读取](reports/gpu-inventory-2026-10-09.md) |
| 连续语音与任务历史 | 两项正式任务通过；745 次控制、2,980 次物理步骤、1,784 项事件、166 张 PNG；历史引用、两次经验搜索、独立目标记录、固定音色反馈与资源释放；438 个安装文件、Mac 29 项和 Linux 27 项 CLI 检查通过 | [任务历史引用](reports/voice-task-context-2026-10-09.md) |
| Recorded-voice office navigation | Formal Verifier passed; 666 controls, 2664 substeps, 1323 events, 141 original images, confirmed-profile feedback, five process exits; fully decoded 174.54-second 1080p agentic MP4 | [Voice navigation](reports/cpu-voice-navigation-2026-10-08.md) |
| Luna text navigation | Formal Verifier passed; 569 controls, 2276 substeps, 1040 events, 113 observer frames, 12 stop-progress checks and fully decoded 1080p MP4 | [Text navigation](reports/cpu-luna-navigation-2026-10-08.md) |
| Voice interruption | Actual ASR/model/tools/feedback; 1400 controls, 5600 substeps, 2427 events, 296 original images, two WAV files, unchanged profile and full cleanup | [Interruption](reports/cpu-voice-task-2026-10-08.md) |
| Shared voice modules | 24 original definitions preserved; three actual Qwen CPU models; four WAV files and four matching ASR texts; 36 profile/admission tests; five process exits; 434 installed files and 26 calls | [Voice modules](reports/voice-audio-modules-2026-10-08.md) |
| ASR 全量语料与独立复核 | FLEURS 1,592 条人声录音；CPU Float32、八个 threads；七次复核进程按预期退出，测量进程数量归零 | [真实录音测量](reports/asr-corpus-2026-10-12.md) |
| ASR 公开安装与完整流程 | 三个阶段完成全部 1,592 条录音；五次安装说明命令、两个主机安装检查与四个所属进程退出通过 | [公开流程验收](reports/public-asr-workflow-2026-10-12.md) |
| Native retries and goal evidence | Three executions preserve physical state; 200 controls, 800 substeps, six rejected starts, three read-only checks and independent ONNX parity | [Execution retries](reports/native-execution-retry-2026-10-08.md) |
| Metric tool parameters | Actual Luna four-motion execution, declared units/default speeds and original error/stop requirements | [Tool parameters](reports/metric-tool-guidance-2026-10-08.md) |
| Metric admission and wait | Duplicate requests, explicit command replacement, caller deadlines, measured motion and physical resource closure; finite positive native waits | [Metric lifecycle](reports/metric-request-admission-2026-10-08.md), [wait admission](reports/native-wait-admission-2026-10-08.md) |
| Current CPU perception | Five rendered states, 1253 native geometry intersections, target distance/bearing and unchanged physics during reads | [CPU perception](reports/cpu-scene-perception-2026-10-08.md) |
| Model RGBD measurements | Five actual CPU YOLO frames/targets, independent distance/position/bearing checks, 90 rejected alterations, three process exits, 436 installed files and 27 CLI calls | [Measurement validation](reports/perception-measurements-2026-10-08.md) |
| 原生目标跟踪与模型查询 | 动态实例、实际 YOLO26 连续关联、425 次 ONNX 动作、十次原始 capture、八项查询拒绝、正式 Luna 任务与 975 帧完整视频解码通过 | [连续目标验收](reports/target-tracking-2026-10-12.md) |
| 模型动作等待与同步进度 | 61 次实际 Luna Responses、485 次控制、1,940 次子步、1,015 条事件、十条等待结果；原始 RGBD 与严格安装版回放通过；任务资源已释放，远程服务正常退出；服务 SSH 管理连接退出状态为 255 | [完整等待任务](reports/model-motion-wait-2026-10-13.md) |
| Configured perception sources | Shared Python/Node admission, native metadata/tool schema and two actual CPU sessions | [Source selection](reports/perception-source-capabilities-2026-10-08.md) |
| Training package tools | Actual Walking/StandUp packages; 725 controls, 145 frames, 87 stopped samples, full episodic duration and transition | [Registered packages](reports/policy-packages-2026-10-08.md) |
| Head/body commands | 575 controls, actual positive/negative pitch and body-height response, return, 115 camera frames and stopped-state audit | [Pose tools](reports/native-pose-2026-10-08.md) |
| Policy and motion evidence | Continuous five-motion sequence, 1208 independently recomputed ONNX actions with zero error, measured state and stopping | [Validation modules](reports/validation-modules-2026-10-08.md) |
| 新锁定 CPU 环境 | 九个准备阶段正常退出；300 次控制、1,200 次物理步骤、连续任务状态保留；150 次独立 ONNX 动作与 150 张原始 RGB 图像核查通过，资源全部释放 | [原生准备与执行](reports/native-preparation-2026-10-11.md) |
| Application and scene interfaces | Actual native session lifecycle, complete tool Schemas, common scene configuration and independent installation | [Application](reports/native-application-interfaces-2026-10-08.md), [scenes](reports/native-scene-configuration-2026-10-08.md) |
| Native campaign inputs | Integer counts, finite proportions, actual CLI admission and unchanged committed plans; Linux checks passed | [Training inputs](reports/campaign-input-admission-2026-10-08.md) |
| Newton assets and contacts | Both variants, complete converted inputs, OpenUSD provider, actual CPU contact configuration, solver tables and Office/Hospital geometry | [CPU readiness](reports/cpu-development-readiness-2026-10-08.md) |
| Execution preflight and lifecycle | Dependency failures before CUDA/Isaac initialization, fixed-source six-stage preparation, installed process deadlines/interruption and retained results | [Preflight](reports/newton-execution-preflight-2026-10-08.md), [process lifecycle](reports/installed-audit-lifecycle-2026-10-08.md) |

The [CPU readiness record](reports/cpu-development-readiness-2026-10-08.md) includes
the source, asset, dependency and installed-package reports for each capability.
Model recognition quality, subjective audio quality, learned-policy performance
and hardware require their corresponding evaluation.

## Recorded Newton tasks and pending runtime

Fixed-source records include the three-scene 15-motion metric matrix, Office
four-policy skills/navigation, Hospital roller ordered navigation and recorded
voice execution. They preserve actual physics, images, task verdicts, media and
resource release:

- [Metric matrix](reports/metric-controller-acceptance-2026-10-07.md).
- [Office skills](reports/office-skills-acceptance-2026-10-07.md).
- [Ordered navigation](reports/navigation-acceptance-2026-10-07.md).
- [Newton recorded voice](reports/end-to-end-2026-10-03.md).

Current-controller GPU revalidation remains pending. Office long-walk clockwise
response, Hospital backward progress, initialization reliability and formal
completion within the original budget require their own evidence. Retained
failed attempts remain in their original output directories and reports.
The [runtime observation report](reports/runtime-observation-acceptance-2026-10-07.md)
records the stopped campaign and cleanup.

Subsequent authorized execution uses the
[six-stage runtime release campaign](runtime-release-acceptance.md), preserving
source, model, policy, real sensors, ActionGate, original tolerances, task budgets,
independent Verifier, MP4 decoding and worker release.

## RL scope and remaining behavior

[`configs/training.json`](../configs/training.json) selects Flat Walking and Flat
StandUp, both backends and both native PPO frameworks: eight combinations.
The full task registry remains extensible. Native checkpoints, normalization,
critic and PPO semantics stay with their respective framework. W&B uses the
configured online mode and the verified account, with local artifacts retained.

The eight-combination lifecycle has recorded smoke, resume, export, package and
both-backend/CPU rehearsal evidence. Short smoke policies have failed behavior
gates. A published `alpha_stand` reference passed 64/64 CPU reset samples and four
cases in each native backend. Recorded Newton RSL StandUp success is single-seed;
complete robustness and representative acceptance remain pending.

Learned Walking, SB3 behavior, matched official/owned training, multiple seeds,
sim2sim and final behavior must satisfy the declared task criteria. Training and
preview/evaluation workers remain stopped, with checkpoints, logs, packages and
videos preserved.

Source and result references:
[frameworks](rl-frameworks.md), [representative acceptance](rl-reproduction.md),
[campaigns](rl-campaigns.md), [pipeline evidence](reports/rl-pipeline-acceptance.md),
[published StandUp](reports/published-policy-reference-2026-09-08.md),
[framework behavior](reports/rl-framework-status-2026-09-09.md) and
[Walking stop record](reports/rl-walking-user-stop-2026-09-30.md).

## Release requirements

Release acceptance includes clean installation, declared native workflows,
source/model/scene provenance, actual physical goals, fixed voice, original media,
independent audits and complete resource closure. Remaining product capabilities
include labelled perception accuracy, autonomous following, independent image-driven VLN/VLA,
object effects, learned-policy behavior and actual Microduck integration.
Code, assets, weights and drivers retain separate license and distribution records.

Use the [release guide](release-readiness.md) and
[execution plan](Agentic%20Microduck%20-%20Execution%20Plan%20v0.1.md) for exact
requirements. Experiment results and historical source identities stay in
`docs/reports/`; implementation details stay in capability source maps.
