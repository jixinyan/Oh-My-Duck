# Isaac Lab / Newton integration

The backend runs maintained Walking and StandUp recipes with native Isaac Lab,
Newton's MuJoCo-Warp solver and the official BAM M6 motor model. Both native
RSL-RL and SB3 PPO use the same task runtime. Check [current measured status](reports/domain-refactor.md)
for the distinction between pipeline validation and learned behavior.

## Setup and local execution

```bash
python omd.py setup --backend isaac-newton --rl-framework sb3
python omd.py assets --backend isaac-newton -- --model walk --accept-eula
python omd.py assets --backend isaac-newton -- --model groundcontact --accept-eula
python omd.py train --backend isaac-newton --rl-framework rsl-rl -- Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5 --agent.run-name walk_smoke
python omd.py train --backend isaac-newton --rl-framework sb3 -- Mjlab-StandUp-Flat-MicroDuck --num-envs 64 --iterations 5 --output outputs/stand_smoke
python omd.py export --backend isaac-newton --rl-framework sb3 -- --run outputs/stand_smoke --output outputs/stand_export
python omd.py eval --backend isaac-newton -- --task Mjlab-StandUp-Flat-MicroDuck --policy outputs/stand_export/policy.onnx --output outputs/stand_replay --video
```

Run single-GPU commands directly on the development host. Set `CUDA_VISIBLE_DEVICES`
explicitly when choosing a device. Output directories must be new. Use the
scheduler only for multi-GPU experiments, for example native RSL-RL with
`--gpu-ids all` inside a two-GPU allocation. W&B stays offline on every rank.

The conversion command accepts the NVIDIA EULA for that process. The two commands build
walk and groundcontact assets with content hashes under `artifacts/isaac-newton/`.
Generated assets are ignored; owned MJCF/mesh source remains in `robotics/microduck`.

## Runtime and dependency boundaries

| Source / environment | Responsibility |
|---|---|
| `rl/backends/isaac_newton/task_binding/` | Native physics lifecycle, canonical state, sensors, collision tables, DR and audit |
| `rl/tasks/`, `rl/mdp/` | Maintained official recipes, reward and observation logic |
| `rl/learners/{rsl_rl,sb3}/` | Shared native PPO orchestration and artifacts |
| `rl/evaluation/task.py` | Task-specific metrics, explicit resets and native 1280×720 video |
| `.envs/isaac-newton` | Isaac Lab/Newton simulation and native learners |
| `.envs/isaac-assets` | Isolated Isaac Sim MJCF-to-USD conversion |
| `.envs/mujoco`, `.envs/mujoco-sb3` | Official metadata reference and normalized runner export |

Microduck code is owned source; only generic Isaac/Newton/mjlab/PPO dependencies
come from pinned external packages. Newton's graph-coloring approximation is
replaced with exact official static contact tables at the native manager's
solver-construction extension point, before graph capture. Actual compiled tables,
collision hulls, contact force signs, non-accumulating randomization and four BAM
calls per action are audited. The physics timestep remains 5 ms and policy rate 50 Hz.

Exports load the native checkpoint and normalization. They explicitly build the
MuJoCo reference for deployment metadata in its compatible environment; provenance
records the training backend separately. This does not switch the training physics.
Isaac's Torch/Warp pins remain separate; W&B/Tyro match the versions verified with
RSL 5.0.1. Fresh locked sync reproduces that dependency choice.

## Diagnostics and extension

`Omd-Microduck-PD-Diagnostic-v0` remains an explicit small diagnostic. Its old
probe/train/export paths are retained for asset and dependency debugging. A PD
checkpoint cannot certify a BAM task. Tasks absent from the Newton registry fail
explicitly. See [adding tasks and models](rl-task-extension.md).

Learned gait/recovery quality and sim2sim transfer require their own measured
acceptance. No physical robot has been tested and no policy has been uploaded.
