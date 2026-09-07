# Getting started

The [README](../README.md) describes the whole project; [implementation status](implementation-status.md)
records measured progress. All first-party code is under `src/oh_my_duck`.

## Prepare isolated dependencies

```bash
python omd.py --help
python omd.py setup --backend mujoco --rl-framework sb3
python omd.py setup --backend isaac-newton --rl-framework sb3
python omd.py assets --backend isaac-newton -- --model walk --accept-eula
python omd.py assets --backend isaac-newton -- --model groundcontact --accept-eula
python omd.py tasks
python omd.py frameworks
```

Setup installs the maintained package with locked generic dependencies. It does not
change system Python. MuJoCo, Isaac/Newton and Isaac Sim conversion use separate
environments. See [Newton details](isaac-newton.md) for dependency and physics gates.

## Run locally, submit multi-GPU experiments

Single-GPU development, training and evaluation run directly on this host. Select a
GPU explicitly with `CUDA_VISIBLE_DEVICES` when needed. Every output directory and
run name must be new; W&B stays offline.

```bash
CUDA_VISIBLE_DEVICES=0 python omd.py train --backend mujoco --rl-framework rsl-rl -- Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5 --agent.run-name walk_smoke
CUDA_VISIBLE_DEVICES=0 python omd.py train --backend isaac-newton --rl-framework sb3 -- Mjlab-StandUp-Flat-MicroDuck --num-envs 64 --iterations 5 --output outputs/stand_smoke
python omd.py export --backend isaac-newton --rl-framework sb3 -- --run outputs/stand_smoke --output outputs/stand_export
python omd.py eval --backend isaac-newton -- --task Mjlab-StandUp-Flat-MicroDuck --policy outputs/stand_export/policy.onnx --output outputs/stand_eval --video
```

Use the scheduler only for multi-GPU experiments. Commit the source first; jobs run
from an immutable Git worktree and explicitly share environments and artifacts.

```bash
python omd.py submit --name omd-walk-ddp-001 --gpus 2 -- python omd.py train --backend mujoco --rl-framework rsl-rl -- Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5 --agent.run-name walk_ddp --gpu-ids all
```

RSL environment counts are per rank. SB3 has native vector environments and
independent runs, not distributed gradient updates. Measure before scaling.

## Rehearse, compare and package

```bash
python omd.py rehearsal -- --task Mjlab-StandUp-Flat-MicroDuck --policy outputs/stand_export/policy.onnx --output outputs/stand_cpu --video
python omd.py compare -- --task Mjlab-StandUp-Flat-MicroDuck --policy outputs/stand_export/policy.onnx --output outputs/stand_sim2sim --video
python omd.py package -- --onnx outputs/stand_export/policy.onnx --checkpoint outputs/stand_smoke/model.zip --output outputs/stand_package
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
python omd.py rehearsal -- --task Mjlab-Velocity-Flat-MicroDuck --policy /path/policy.onnx --output outputs/rehearsal-new --video --mujoco-renderer osmesa
python omd.py compare -- --task Mjlab-Velocity-Flat-MicroDuck --policy /path/policy.onnx --output outputs/compare-new --video --mujoco-renderer osmesa
```

`--mujoco-renderer` also applies to task `eval`; Isaac still renders through
Newton. See [rendering validation](reports/rendering-validation.md) for the
host limitation, library provenance and video acceptance evidence.
