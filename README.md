<p align="center">
  <img src="docs/assets/oh-my-duck.png" alt="Oh My Duck project logo" width="320" />
</p>

# Oh My Duck 🦆

[![Status: active development](https://img.shields.io/badge/status-active_development-blue)](docs/implementation-status.md)
[![RL behavior: pending](https://img.shields.io/badge/RL_behavior-pending-orange)](docs/reports/rl-walking-low-speed-launch-2026-09-29.md)
[![CPU motion: verified](https://img.shields.io/badge/CPU_motion-verified-2ea44f)](docs/reports/offline-release-validation-2026-10-07.md)
[![CPU Luna navigation: verified](https://img.shields.io/badge/CPU_Luna_navigation-verified-2ea44f)](docs/reports/cpu-luna-navigation-2026-10-08.md)
[![Voice: native task verified](https://img.shields.io/badge/voice-native_task_verified-green)](docs/reports/cpu-voice-navigation-2026-10-08.md)
[![Release: acceptance in progress](https://img.shields.io/badge/release-acceptance_in_progress-blue)](docs/release-readiness.md)
[![Isaac Office: agentic navigation verified](https://img.shields.io/badge/Isaac_Office-agentic_navigation_verified-green)](docs/reports/metric-camera-tools-2026-09-30.md)
[![Isaac Hospital: roller and crouch verified](https://img.shields.io/badge/Isaac_Hospital-roller_and_crouch_verified-green)](docs/reports/multiskill-demos-2026-10-01.md)
[![Office policy tools: meters and degrees verified](https://img.shields.io/badge/Office_policy_tools-meters_and_degrees_verified-green)](docs/metric-policy-tools.md)
[![Metric matrix: 3 scenes, 15 motions passed](https://img.shields.io/badge/metric_matrix-3_scenes_15_motions_passed-green)](docs/reports/metric-controller-acceptance-2026-10-07.md)
[![Office: 4 policies passed](https://img.shields.io/badge/Office-4_policies_passed-green)](docs/reports/office-skills-acceptance-2026-10-07.md)
[![Hospital: ordered navigation passed](https://img.shields.io/badge/Hospital-ordered_navigation_passed-green)](docs/reports/navigation-acceptance-2026-10-07.md)
[![Office current route: completion pending](https://img.shields.io/badge/Office_current_route-completion_pending-orange)](docs/reports/navigation-acceptance-2026-10-07.md#office-有序导航测量)
[![Hardware validation: pending](https://img.shields.io/badge/hardware_validation-pending-orange)](docs/implementation-status.md)

**An interactive, extensible Microduck—with a persistent voice, shared experiences, and skills you can train.**

Oh My Duck is the implementation of **Agentic Microduck**: a robotics foundation that connects an external embodied AI harness to a small biped robot. It brings together simulation, locomotion training, robot skills, active perception, voice interaction, and evidence of what the duck actually did.

> “Find the red ball and walk up to it.”
>
> The duck looks around, locates the ball, approaches it, and replies in the voice you chose. You can ask what it sees or cancel the task. Observations and outcomes become a traceable episode that the external harness can use as a shared experience.

This is the target experience. The project is under active development; [implementation status](docs/implementation-status.md) distinguishes working components from planned capabilities. Physical-robot validation is deferred until hardware is available.

## Verified workflows

| Workflow | Evidence |
| --- | --- |
| 录音指令 → Qwen ASR → Luna → 官方 policy → Verifier → 固定音色反馈 | [CPU 语音导航与 agentic MP4](docs/reports/cpu-voice-navigation-2026-10-08.md) |
| 文字指令 → 原生 Harness → 办公室导航 | [Luna CPU 导航](docs/reports/cpu-luna-navigation-2026-10-08.md) |
| 米制前进、角度转向、读取传感器、停止与保留物理状态的重试 | [工具参数](docs/metric-policy-tools.md)、[原生重试](docs/reports/native-execution-retry-2026-10-08.md) |
| 原始 RGBD、可见目标、距离与 bearing | [CPU 感知](docs/reports/cpu-scene-perception-2026-10-08.md)、[感知来源](docs/reports/perception-source-capabilities-2026-10-08.md) |
| Walking/StandUp 训练包登记、执行与 policy 接续 | [Policy registry](docs/policy-registry.md)、[训练包验收](docs/reports/policy-packages-2026-10-08.md) |
| Qwen CPU 推理、音色持久保存与任务中断 | [模型与音色](docs/reports/qwen-cpu-validation-2026-10-08.md)、[执行中断](docs/reports/cpu-voice-task-2026-10-08.md) |
| CPU 原生执行、资产、依赖与独立安装 | [CPU 开发验收](docs/reports/cpu-development-readiness-2026-10-08.md) |

GPU 验收与 RL 当前保持停止。当前源码的 Newton 运行、多场景长导航、识别准确率、
学习行为与 Microduck 设备需要各自的验收；范围和执行要求见
[发布准备](docs/release-readiness.md)与[运行验收流程](docs/runtime-release-acceptance.md)。

## What the project provides

| Capability | Purpose |
|---|---|
| **Text and voice interaction** | Explicit recording → speech recognition → the same text/tool control path; spoken responses and playback interruption |
| **A persistent voice** | Describe, audition and confirm a voice once; reuse the versioned voice profile across sessions and simulator/robot changes |
| **Robot skills as tools** | Discover capabilities, run skills, query progress, cancel tasks and confirm actual stopping |
| **Active perception** | Read RGB, 8×8 ToF, IMU and robot state; move the head to obtain observations with timestamps and coordinate frames |
| **Simulation and real-robot adapters** | Preserve tool parameters, units and result semantics across execution backends |
| **Two training backends** | Retain official mjlab / MuJoCo-Warp training and add Isaac Lab with **Newton physics** |
| **Sim2sim and policy export** | Compare policies under the same scenarios; export compatible joint policies with normalization and deployment metadata |
| **Traceable experiences** | Record sensor evidence, commands, task results and voice events for replay and external harness memory |

The duck's name, voice and remembered experiences give it continuity. Learning new motion skills remains an explicit data, training, evaluation and deployment process; conversation alone does not update a locomotion policy.

## How it fits together

![Project architecture: interaction, external harness, robot tools, simulation and experience recording.](docs/diagrams/project-overview.svg)

**外部 Harness 负责 agent loop。** `omd harness` 使用固定版本
`8a5e685b22d032207f53db20454f0992a4ad60fd` 的 Embodied-DeepSeek-Harness
原生 SessionEnvironment、ActionGate 和独立 Verifier。Oh My Duck 提供语音、
机器人工具、执行适配、训练和原始证据；工具共享原生任务接口和完整 JSON Schema。
部署与模型连接方式见[原生 Harness 使用说明](docs/harness-native-integration.md)。

High-frequency joint control stays with the policy/runtime. The harness chooses tasks and can observe, interrupt or replan. Independent verification uses the resulting physical state.

`microduck.observe` returns head RGB, 64 ToF measurements, IMU, 14 servo measurements,
odometry, motion progress, goal evidence and the remaining execution budget at a
confirmed physical boundary. `microduck.wait_for_motion` waits for that boundary.
`microduck.walk(distance_m)` and `microduck.rotate(angle_deg)` use measured
odometry, braking and five stopped samples. Turns report their actual translation.
The [tool guide](docs/metric-policy-tools.md) defines units, speeds and completion
requirements; the [metric matrix](docs/reports/metric-controller-acceptance-2026-10-07.md)
records measurements and source pins.

External Isaac USD scenes enter through `omd harness --scene-config`. A native
Python worker can run on an explicitly selected remote GPU over SSH. Standard
feet use official `alpha_walking`; the roller model uses `roller` and exposes its
four passive wheel joints. `--policy-registry` also makes verified training
packages available to the same tools. See [policy registration](docs/policy-registry.md),
[scene configuration](docs/harness-native-integration.md#场景配置) and
[agentic MP4 production](docs/harness-native-integration.md#agentic-mp4).

![Actual Newton Office robot frame and head RGB with recorded Planner text, metric tool feedback and a passed independent verdict.](docs/assets/office-agentic/metric-demo.png)

Hospital 的官方 `roller` 完成东向、北向和西向三阶段导航，行走段端点位移累计
5.44 米，独立 Verifier 与原始记录复核通过；170 秒 MP4 包含实际场景、head RGB、
工具参数和公开 agentic trace。[导航证据](docs/reports/navigation-acceptance-2026-10-07.md)
记录固定场景的测量、原始图片和资源释放。

`microduck.inspect_scene(prompt, source)` provides frame-bound target boxes,
surface distance and bearing. The isolated SAM3.1 + YOLO26 service performs
text-prompt segmentation and box association; the explicit `simulator_ground_truth`
source uses native shape masks and calibrated ray distances. Results retain their
detection, mask and distance sources. See [perception tools](docs/perception-navigation.md).

Recorded Hospital and Office tasks demonstrate multiple official policies:
`roller` and `crouch` for locomotion and body motion; `sitstand`, `alpha_stand`,
`ground_pick` and `alpha_walking` for sitting, standing, head control, reaching,
recovery and navigation. Their videos combine actual scene cameras, head RGB,
tool parameters and independent verdicts. See the
[Office task](docs/reports/office-skills-acceptance-2026-10-07.md) and
[Hospital tasks](docs/reports/multiskill-demos-2026-10-01.md).

## Train with MuJoCo or Isaac / Newton

![RL pipeline: two simulation backends and two native PPO frameworks, normalized export, sim2sim and deployment rehearsal.](docs/diagrams/rl-pipeline.svg)

RL frameworks are a separate extension axis: RSL-RL and Stable-Baselines3 are the initial supported integrations, with explicit adapters for each supported simulation backend. Framework-native checkpoints and normalization remain part of the policy artifact; framework-independent evaluation enables comparison. See [RL framework choices and extension](docs/rl-frameworks.md).

Both training backends are part of the project scope. The official backend remains available after the Isaac migration. **Isaac uses Newton** with its MuJoCo-Warp solver.

Representative RL tasks are flat-ground Walking and StandUp. Task recipes, MDP functions, robot assets and actor/critic settings are maintained in the [RL and robotics modules](docs/architecture.md#source-organization); both backends build on this owned source. BAM actuator behavior, joint mapping, observation/action timing and normalization must match before comparing learning results. Compatible joint policies follow the official **61-observation / 14-action, 50 Hz** interface. Vision/navigation policies integrate through explicit action and observation adapters.

Task families have separate environment and PPO configurations under `rl/tasks/<family>/`; see the [RL source map](src/oh_my_duck/rl/README.md). [Training campaigns](docs/rl-campaigns.md) assign independent task/framework runs to GPUs, with explicit sharing and checkpoint recovery when needed.

Training and evaluation run headlessly. `omd preview` creates checkpoint videos
and a local gallery alongside numerical metrics. The training host is `jd_B300`;
W&B records online under the verified project account and keeps local artifacts.
Training is stopped, with checkpoints, exports and logs preserved. Learned-policy
behavior requires its own acceptance. See [training workflows](docs/rl-campaigns.md)
and [implementation status](docs/implementation-status.md).

## A voice and history that persist

Voice setup is a deliberate choice: **describe → generate → listen → confirm → save**. Daily TTS uses the active voice profile; restarting or switching execution backends does not silently choose a new voice.

Qwen3 ASR, VoiceDesign and Base TTS use confirmed voice profiles across processes.
`voice-task` connects recorded speech to the native Harness, model-selected policy
tools, independent Verifier and fixed-voice feedback. `voice-session` provides
device recording, playback, task submission and interruption. Actual CPU and
Newton records verify stopping and resource release; Mac audio devices have
separate checks. Microduck audio hardware requires its own acceptance.
See [voice interaction](docs/voice-interaction.md),
[CPU voice navigation](docs/reports/cpu-voice-navigation-2026-10-08.md),
[task interruption](docs/reports/cpu-voice-task-2026-10-08.md) and
[Newton voice acceptance](docs/reports/end-to-end-2026-10-03.md).

Episodes preserve what was heard, observed, requested, executed and spoken—including cancellation and partial playback. Simulation and real-world experiences remain labeled separately. The external harness decides how to summarize and retrieve this evidence.

## Built to extend

- Add or replace an external harness through the bridge.
- Register a new skill with explicit inputs, prerequisites, resources, success/failure and cancellation conditions.
- Add sensors with freshness, validity, units, coordinate frames and calibration references.
- Train a joint policy, add a velocity-level navigation policy, or adapt an action-sequence model.
- Run the same tool semantics in simulation and, once validated, on a physical duck.

The first scope is one robot, one external host and one active task. Complex whole-home navigation, continuous full-duplex listening, NFC interaction and general VLA training are later research directions.

## Explore the project

`omd harness --scene-config` accepts the official CPU apartment and Isaac/Newton
scenes. Python and Node validate the same packaged schema before native execution.
CPU configurations select a spawn and a room, dock or object goal; Newton
configurations provide USD provenance and a point or ordered route goal.
See [native configuration](docs/harness-native-integration.md#场景配置).

| Start with | What it explains |
|---|---|
| [Project Design](docs/Agentic%20Microduck%20-%20Project%20Design%20v0.1.md) | Full product scope, decisions, module boundaries and interfaces |
| [Execution Plan](docs/Agentic%20Microduck%20-%20Execution%20Plan%20v0.1.md) | Milestones, dependencies and acceptance criteria |
| [Getting started](docs/getting-started.md) | Setup, local training, multi-GPU jobs and headless video evaluation |
| [Architecture and extension guide](docs/architecture.md) | Full framework, module contracts, dependency direction and adapter extension points |
| [Development guide](docs/development.md) | Module layout, source pins, environments, Git and file management |
| [Implementation status](docs/implementation-status.md) | Current progress, measured evidence and remaining work |

Python modules live in `src/oh_my_duck/`. The native Node deployment lives in
`integrations/edh/`, and dependency locks live in `environments/`.

| Capability | Source |
| --- | --- |
| Training, simulators, PPO, export and evaluation | [rl/](src/oh_my_duck/rl/) |
| Native physical session, devices, tools and transport | [integrations/edh/](src/oh_my_duck/integrations/edh/) |
| Native Node deployment and Planner role | [integrations/edh/](integrations/edh/) |
| Application, tool and skill interfaces | [agentic source map](src/oh_my_duck/agentic/README.md) |
| Native task HTTP client | [native_client.py](src/oh_my_duck/integrations/native_client.py) |
| Robot models, motors and execution backends | [robotics/](src/oh_my_duck/robotics/) |
| Verified joint graphs and training packages | [robotics/policies/](src/oh_my_duck/robotics/policies/) |
| Physical records, policy and release audits | [validation/](src/oh_my_duck/validation/) |
| Perception services and frame validation | [perception/](src/oh_my_duck/perception/) |
| Qwen audio and confirmed voice profiles | [voice/](src/oh_my_duck/voice/) |
| Episode records and replay export | [experience/](src/oh_my_duck/experience/) |
| Shared types and configuration | [core/](src/oh_my_duck/core/) |
| Environments, processes and tracking | [infrastructure/](src/oh_my_duck/infrastructure/) |
| Public commands | [cli/](src/oh_my_duck/cli/) |


Use `omd harness` for the native agent deployment, `omd validate` for actual
acceptance and recorded-evidence audits, and `omd replay` for session management
and terminal run export. The same commands are available through
`python -m oh_my_duck`; `python omd.py --help` is the source entry point.
Voice inference uses the isolated ASR and TTS environments. Installed-runtime
checks and video commands are in the [release guide](docs/release-readiness.md).

## Upstream

Oh My Duck builds on [Microduck](https://github.com/pollen-robotics/microduck), [microduck_rl](https://github.com/pollen-robotics/microduck_rl), [BAM](https://github.com/Rhoban/bam), and [Isaac Lab](https://github.com/isaac-sim/IsaacLab) / [Newton](https://github.com/newton-physics/newton) integration.

Code, 3D assets, model weights and third-party drivers have separate license declarations. See [third-party notices](THIRD_PARTY_NOTICES.md). A distribution license for this project's original code has not yet been selected.
