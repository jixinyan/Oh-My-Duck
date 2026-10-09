# Getting started

The [README](../README.md) describes the whole project; [implementation status](implementation-status.md)
records measured progress. All first-party code is under `src/oh_my_duck`.

The [release readiness guide](release-readiness.md) covers recorded voice,
native agent execution, official policies and the required acceptance evidence.

## Prepare isolated dependencies

```bash
python omd.py --help
python omd.py setup --backend mujoco --rl-framework sb3
python omd.py setup --backend isaac-newton --rl-framework sb3
python omd.py tasks
python omd.py frameworks
```

Setup installs the maintained package with locked generic dependencies. It does not
change system Python. MuJoCo, Isaac/Newton and Isaac Sim conversion use separate
environments. See [Newton details](isaac-newton.md) for dependency and physics gates.

## Inspect CPU workflows

```bash
python omd.py status
python omd.py harness --help
python omd.py voice-task --help
python omd.py voice-session --help
python omd.py validate release-plan
```

The [native deployment guide](harness-native-integration.md) defines the locked
CPU apartment environment, real provider configuration, official policy tools
and terminal run export. The [voice guide](voice-interaction.md) defines separate
ASR/TTS environments, explicit recording, confirmed profiles and interruption.
Actual CPU task, model and installation evidence is listed in
[CPU readiness](reports/cpu-development-readiness-2026-10-08.md).

## Train and evaluate after GPU execution resumes

GPU acceptance and RL remain stopped. Subsequent authorized single-GPU execution
runs headlessly on the development host. Set `OMD_GPU_ID` to an audited idle
physical device from 2–4, with no compute PIDs and sustained zero utilization;
use at most one device and preserve other users' workloads. Every output directory
and run name must be new. W&B uses `configs/training.json` and the verified account;
`WANDB_MODE` can explicitly select the mode. RL requires a new instruction to resume.

```bash
CUDA_VISIBLE_DEVICES="${OMD_GPU_ID:?}" python omd.py assets --backend isaac-newton -- --model walk --accept-eula
CUDA_VISIBLE_DEVICES="${OMD_GPU_ID:?}" python omd.py assets --backend isaac-newton -- --model groundcontact --accept-eula
CUDA_VISIBLE_DEVICES="${OMD_GPU_ID:?}" python omd.py train --backend mujoco --rl-framework rsl-rl -- Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5 --agent.run-name walk_smoke
CUDA_VISIBLE_DEVICES="${OMD_GPU_ID:?}" python omd.py train --backend isaac-newton --rl-framework sb3 -- Mjlab-StandUp-Flat-MicroDuck --num-envs 64 --iterations 5 --output outputs/stand_smoke
CUDA_VISIBLE_DEVICES="${OMD_GPU_ID:?}" python omd.py export --backend isaac-newton --rl-framework sb3 -- --run outputs/stand_smoke --output outputs/stand_export
CUDA_VISIBLE_DEVICES="${OMD_GPU_ID:?}" python omd.py eval --backend isaac-newton -- --task Mjlab-StandUp-Flat-MicroDuck --policy outputs/stand_export/policy.onnx --output outputs/stand_eval --video
```

Commit source before execution. Jobs use an immutable checkout and explicitly
share environments and artifacts. Campaign operation and native framework
parallelism are documented in [training campaigns](rl-campaigns.md), under the
current resource restriction. RSL environment counts are per rank; SB3 uses
native vector environments and independent runs.

## Rehearse, compare and package

```bash
CUDA_VISIBLE_DEVICES='' python omd.py rehearsal -- --task Mjlab-StandUp-Flat-MicroDuck --policy outputs/stand_export/policy.onnx --output outputs/stand_cpu --video --mujoco-renderer osmesa
CUDA_VISIBLE_DEVICES="${OMD_GPU_ID:?}" python omd.py compare -- --task Mjlab-StandUp-Flat-MicroDuck --policy outputs/stand_export/policy.onnx --output outputs/stand_sim2sim --video --mujoco-renderer osmesa
CUDA_VISIBLE_DEVICES='' python omd.py package -- --onnx outputs/stand_export/policy.onnx --checkpoint outputs/stand_smoke/model.zip --output outputs/stand_package
```

These commands use task-registered evaluation/deployment profiles. Comparison runs
the same frozen policy and battery on both backends. CPU rehearsal uses the official
full groundcontact deployment scene and BAM. Local packaging checks that the ONNX
actually belongs to the supplied native checkpoint and uses official schema 2;
it has no upload mode.

Each evaluation records `result.json`, per-scenario trajectories and optional
1280×720 videos. A valid execution with failed behavior returns code 2; an
implementation/runtime error returns code 1. Automatic task resets are disabled.
Five iterations validate the pipeline, not a learned gait or recovery skill.

## Extend the source

See [architecture](architecture.md) for domain boundaries and [task extension](rl-task-extension.md)
for actors, rewards, new tasks, runtime factories, evaluation and packaging.
Generated data remains under ignored `.cache/`, `.envs/`, `artifacts/`, `outputs/`,
`logs/` and `wandb/`. Source, tests, locks, documentation and licenses are versioned.

MuJoCo video can explicitly use software rasterization on hosts with unreliable
EGL readback. Install the optional pinned local library on Ubuntu 22.04 amd64:

```bash
python omd.py setup --backend mujoco --software-renderer --skip-env
CUDA_VISIBLE_DEVICES='' python omd.py rehearsal -- --task Mjlab-Velocity-Flat-MicroDuck --policy /path/policy.onnx --output outputs/rehearsal-new --video --mujoco-renderer osmesa
CUDA_VISIBLE_DEVICES="${OMD_GPU_ID:?}" python omd.py compare -- --task Mjlab-Velocity-Flat-MicroDuck --policy /path/policy.onnx --output outputs/compare-new --video --mujoco-renderer osmesa
```

`--mujoco-renderer` also applies to task `eval`; Isaac still renders through
Newton. See [rendering validation](reports/rendering-validation.md) for the
host limitation, library provenance and video acceptance evidence.
