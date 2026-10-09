# Implementation status

Updated: 2026-10-09.

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
| CLI/application | 原生服务、验证与记录、场景 Schema、policy package 与语音；438 个安装文件、三份 license、29 项 CLI 调用 | 完整工作流程在独立环境中的执行检查 |
| Core types, tools and recording | Complete Draft 2020-12 Schemas, finite arguments, actual handlers, request identity, JSONL/native events, original image hashes and MP4 | New tools require their actual implementation and relevant execution evidence |
| Native Harness | Fixed EDH, actual models, official policies, sensors, ActionGate and independent Verifier; CPU text/recorded-voice office tasks passed | Current-source multi-scene long navigation and repeated task statistics |
| Policy tools | Metric units, pose response, policy transitions, bounded commands, state-preserving retries, progress and stop evidence | All long-distance precision, current Newton controller and object effects |
| Policy registry | Ten official joint graphs and registered training packages; actual CPU Walking/StandUp package execution, complete episodic duration and continuation | Learned behavior, scene generalization and hardware |
| Perception | Current CPU RGBD geometry, actual CPU YOLO, original-pixel measurement checks and saved-data audit; fixed-source Newton sensors and SAM3.1 + YOLO26 records | Current-source SAM mask/inference GPU acceptance, independently labelled accuracy, tracking and calibration |
| Voice interaction | Shared audio module; three fixed Qwen CPU models, four decoded WAV files and four matching ASR texts; 36 profile/admission tests; confirmed profile/database preservation | Voice quality, command-corpus accuracy, interaction latency and Microduck devices |
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
| 连续语音与任务历史 | 两项正式任务通过；745 次控制、2,980 次物理步骤、1,784 项事件、166 张 PNG；历史引用、两次经验搜索、独立目标记录、固定音色反馈与资源释放；438 个安装文件、Mac 29 项和 Linux 27 项 CLI 检查通过 | [任务历史引用](reports/voice-task-context-2026-10-09.md) |
| Recorded-voice office navigation | Formal Verifier passed; 666 controls, 2664 substeps, 1323 events, 141 original images, confirmed-profile feedback, five process exits; fully decoded 174.54-second 1080p agentic MP4 | [Voice navigation](reports/cpu-voice-navigation-2026-10-08.md) |
| Luna text navigation | Formal Verifier passed; 569 controls, 2276 substeps, 1040 events, 113 observer frames, 12 stop-progress checks and fully decoded 1080p MP4 | [Text navigation](reports/cpu-luna-navigation-2026-10-08.md) |
| Voice interruption | Actual ASR/model/tools/feedback; 1400 controls, 5600 substeps, 2427 events, 296 original images, two WAV files, unchanged profile and full cleanup | [Interruption](reports/cpu-voice-task-2026-10-08.md) |
| Shared voice modules | 24 original definitions preserved; three actual Qwen CPU models; four WAV files and four matching ASR texts; 36 profile/admission tests; five process exits; 434 installed files and 26 calls | [Voice modules](reports/voice-audio-modules-2026-10-08.md) |
| Native retries and goal evidence | Three executions preserve physical state; 200 controls, 800 substeps, six rejected starts, three read-only checks and independent ONNX parity | [Execution retries](reports/native-execution-retry-2026-10-08.md) |
| Metric tool parameters | Actual Luna four-motion execution, declared units/default speeds and original error/stop requirements | [Tool parameters](reports/metric-tool-guidance-2026-10-08.md) |
| Metric admission and wait | Duplicate requests, explicit command replacement, caller deadlines, measured motion and physical resource closure; finite positive native waits | [Metric lifecycle](reports/metric-request-admission-2026-10-08.md), [wait admission](reports/native-wait-admission-2026-10-08.md) |
| Current CPU perception | Five rendered states, 1253 native geometry intersections, target distance/bearing and unchanged physics during reads | [CPU perception](reports/cpu-scene-perception-2026-10-08.md) |
| Model RGBD measurements | Five actual CPU YOLO frames/targets, independent distance/position/bearing checks, 90 rejected alterations, three process exits, 436 installed files and 27 CLI calls | [Measurement validation](reports/perception-measurements-2026-10-08.md) |
| Configured perception sources | Shared Python/Node admission, native metadata/tool schema and two actual CPU sessions | [Source selection](reports/perception-source-capabilities-2026-10-08.md) |
| Training package tools | Actual Walking/StandUp packages; 725 controls, 145 frames, 87 stopped samples, full episodic duration and transition | [Registered packages](reports/policy-packages-2026-10-08.md) |
| Head/body commands | 575 controls, actual positive/negative pitch and body-height response, return, 115 camera frames and stopped-state audit | [Pose tools](reports/native-pose-2026-10-08.md) |
| Policy and motion evidence | Continuous five-motion sequence, 1208 independently recomputed ONNX actions with zero error, measured state and stopping | [Validation modules](reports/validation-modules-2026-10-08.md) |
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
include labelled perception accuracy, tracking, independent image-driven VLN/VLA,
object effects, learned-policy behavior and actual Microduck integration.
Code, assets, weights and drivers retain separate license and distribution records.

Use the [release guide](release-readiness.md) and
[execution plan](Agentic%20Microduck%20-%20Execution%20Plan%20v0.1.md) for exact
requirements. Experiment results and historical source identities stay in
`docs/reports/`; implementation details stay in capability source maps.
