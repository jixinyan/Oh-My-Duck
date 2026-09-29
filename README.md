<p align="center">
  <img src="docs/assets/oh-my-duck.png" alt="Oh My Duck project logo" width="320" />
</p>

# Oh My Duck 🦆

[![Status: active development](https://img.shields.io/badge/status-active_development-blue)](docs/implementation-status.md)
[![RL behavior: unmet](https://img.shields.io/badge/RL_behavior-unmet-orange)](docs/reports/project-status-2026-09-29.md)
[![Recorded CPU checks: 55 passed on 2026-09-23](https://img.shields.io/badge/recorded_CPU_checks-55_passed-2ea44f)](docs/reports/project-status-2026-09-23.md)
[![Voice: file inference verified](https://img.shields.io/badge/voice-file_inference_verified-green)](docs/reports/voice-validation-2026-09-26.md)
[![Hardware validation: pending](https://img.shields.io/badge/hardware_validation-pending-orange)](docs/implementation-status.md)

**An interactive, extensible Microduck—with a persistent voice, shared experiences, and skills you can train.**

Oh My Duck is the implementation of **Agentic Microduck**: a robotics foundation that connects an external embodied AI harness to a small biped robot. It brings together simulation, locomotion training, robot skills, active perception, voice interaction, and evidence of what the duck actually did.

> “Find the red ball and walk up to it.”
>
> The duck looks around, locates the ball, approaches it, and replies in the voice you chose. You can ask what it sees or cancel the task. Observations and outcomes become a traceable episode that the external harness can use as a shared experience.

This is the target experience. The project is under active development; [implementation status](docs/implementation-status.md) distinguishes working components from planned capabilities. Physical-robot validation is deferred until hardware is available.

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

**The external harness supplies the agent loop.** Oh My Duck supplies robot-specific tools, execution adapters, training, voice services and evidence. We do not build a second planner or long-term memory system inside the robot integration. The deterministic bridge mock now covers protocol development; the real Embodied DeepSeek Harness transport remains unavailable until its wire API is released.

High-frequency joint control stays with the policy/runtime. The harness chooses tasks and can observe, interrupt or replan; a model response is not itself evidence that a physical task succeeded.

## Train with MuJoCo or Isaac / Newton

![RL pipeline: two simulation backends and two native PPO frameworks, normalized export, sim2sim and deployment rehearsal.](docs/diagrams/rl-pipeline.svg)

RL frameworks are a separate extension axis: RSL-RL and Stable-Baselines3 are the initial supported integrations, with explicit adapters for each supported simulation backend. Framework-native checkpoints and normalization remain part of the policy artifact; framework-independent evaluation enables comparison. See [RL framework choices and extension](docs/rl-frameworks.md).

Both training backends are part of the project scope. The official backend remains available after the Isaac migration. **Isaac uses Newton**, initially targeting its MuJoCo-Warp solver; a PhysX substitution is not an equivalent backend.

Representative RL tasks are flat-ground Walking and StandUp. Task recipes, MDP functions, robot assets and actor/critic settings are maintained in the [RL and robotics modules](docs/architecture.md#source-organization); both backends build on this owned source. BAM actuator behavior, joint mapping, observation/action timing and normalization must match before comparing learning results. Compatible joint policies follow the official **61-observation / 14-action, 50 Hz** contract. Vision/navigation policies need their own adapters; an arbitrary VLA cannot be deployed by simply renaming its output.

Task families have separate environment and PPO configurations under `rl/tasks/<family>/`; see the [RL source map](src/oh_my_duck/rl/README.md). [Training campaigns](docs/rl-campaigns.md) assign independent task/framework runs to GPUs, with explicit sharing and checkpoint recovery when needed.

Training and evaluation run headlessly. `omd preview` creates checkpoint videos and a local gallery alongside numerical metrics. The training host is `jd_B300`; W&B records training under the verified project account and keeps local artifacts. The MuJoCo/RSL Walking run finished 50,000 updates, and both Newton/RSL StandUp runs finished 15,000. Native export and packaging completed, while final behavior gates failed: Walking has a measured low-speed command dead zone, and both StandUp policies fail prone and supine recovery. Eight paired Walking control/boost runs have passed preparation gates; their full training has not started. No RL learner is running now. Locomotion tools await policy behavior acceptance. See the [current project status](docs/reports/project-status-2026-09-29.md) and [implementation status](docs/implementation-status.md).

## A voice and history that persist

Voice setup is a deliberate choice: **describe → generate → listen → confirm → save**. Daily TTS uses the active voice profile; restarting or switching execution backends does not silently choose a new voice.

Qwen3 ASR, VoiceDesign and Base TTS have been verified on the target GPU with WAV files. The command-line flow generates a candidate, saves it after explicit confirmation, and reuses the same voice profile across processes. Microphone recording, playback control and Harness integration remain open. See [voice usage](docs/voice-profiles.md) and [GPU validation](docs/reports/voice-validation-2026-09-26.md).

Episodes preserve what was heard, observed, requested, executed and spoken—including cancellation and partial playback. Simulation and real-world experiences remain labeled separately. The external harness decides how to summarize and retrieve this evidence.

## Built to extend

- Add or replace an external harness through the bridge.
- Register a new skill with explicit inputs, prerequisites, resources, success/failure and cancellation conditions.
- Add sensors with freshness, validity, units, coordinate frames and calibration references.
- Train a joint policy, add a velocity-level navigation policy, or adapt an action-sequence model.
- Run the same tool semantics in simulation and, once validated, on a physical duck.

The first scope is one robot, one external host and one active task. Complex whole-home navigation, continuous full-duplex listening, NFC interaction and general VLA training are later research directions.

## Explore the project

| Start with | What it explains |
|---|---|
| [Project Design](docs/Agentic%20Microduck%20-%20Project%20Design%20v0.1.md) | Full product scope, decisions, module boundaries and interfaces |
| [Execution Plan](docs/Agentic%20Microduck%20-%20Execution%20Plan%20v0.1.md) | Milestones, dependencies and acceptance criteria |
| [Getting started](docs/getting-started.md) | Setup, local training, multi-GPU jobs and headless video evaluation |
| [Architecture and extension guide](docs/architecture.md) | Full framework, module contracts, dependency direction and adapter extension points |
| [Development guide](docs/development.md) | Module layout, source pins, environments, Git and file management |
| [Implementation status](docs/implementation-status.md) | Current progress, measured evidence and remaining work |

All first-party code lives in `src/oh_my_duck`, grouped by project capability: `rl`, `agentic`, `robotics`, `perception`, `voice`, and `experience`, with shared `core` and `infrastructure` modules. Dependency locks live in `environments/`.

```text
src/oh_my_duck/
├── rl/              tasks, rewards, simulators, PPO, export and evaluation
├── agentic/         external Harness, tools, skills and application assembly
├── robotics/        robot models, motors, execution and policy interfaces
├── perception/      perception interfaces
├── voice/           Qwen file inference and persistent voice profiles
├── experience/      episode records
├── core/            shared contracts
├── infrastructure/  environments, jobs and tracking
└── cli/             public commands
```


The public development entry point is `python omd.py --help`. Voice file inference runs through `python -m oh_my_duck.cli.voice` in the isolated ASR and TTS environments. Backend dependencies remain isolated.

## Upstream

Oh My Duck builds on [Microduck](https://github.com/pollen-robotics/microduck), [microduck_rl](https://github.com/pollen-robotics/microduck_rl), [BAM](https://github.com/Rhoban/bam), and [Isaac Lab](https://github.com/isaac-sim/IsaacLab) / [Newton](https://github.com/newton-physics/newton) integration.

Code, 3D assets, model weights and third-party drivers have separate license declarations. See [third-party notices](THIRD_PARTY_NOTICES.md). A distribution license for this project's original code has not yet been selected.
