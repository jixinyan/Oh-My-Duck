# Representative RL reproduction

The acceptance scope is Flat Walking and Flat StandUp on MuJoCo/mjlab and Isaac
Lab/Newton, with native RSL-RL PPO and Stable-Baselines3 PPO. The 33 maintained
recipes are an extension inventory; only the two representative tasks are selected
for full acceptance. `configs/tasks.json` owns bindings and evaluation factories;
`configs/training.json` owns the selected scope and offline logging defaults.

## Validation stages

Each combination must pass configuration/physics checks, the official 64-env /
5-iteration smoke, finite rewards and weighted penalties, native checkpoint
continuation, normalized ONNX export, task-specific behavior and sim2sim replay.
See [measured evidence](reports/domain-refactor.md). The source refactor and both
Newton task bindings have now passed actual execution; a short checkpoint still
cannot establish learned behavior.

The common evaluator uses task-registered protocols: Walking tests hold, forward,
stop and turn commands; StandUp tests standing, sitting, face-down and face-up
starts. Automatic resets are disabled. Videos default to 1280×720. Success metrics
and terminated trajectories are preserved independently from training rewards.

```bash
python omd.py tasks
python omd.py train --backend mujoco --rl-framework rsl-rl -- Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5 --agent.run-name smoke
python omd.py train --backend isaac-newton --rl-framework sb3 -- Mjlab-StandUp-Flat-MicroDuck --num-envs 64 --iterations 5 --output outputs/smoke
python omd.py eval --backend isaac-newton -- --task Mjlab-StandUp-Flat-MicroDuck --policy outputs/export/policy.onnx --output outputs/replay --video
```

## Native PPO and throughput

Both backends use project-owned orchestration around the native RSL-RL learner and
torchrunx for distributed training. Environment counts are per rank; record total
batch size, seeds and iteration count. SB3 uses its native PPO and vectorized GPU
simulation. Concurrent independent SB3 runs do not imply distributed gradients.
There is no decoupled asynchronous actor/learner or policy-lag algorithm.

Single-GPU work runs locally; multi-GPU experiments use committed source snapshots
through `omd submit`. Measure collection/learning time and aggregate transitions
per second before choosing scale. Shared-host contention must be recorded.

## Artifacts and extension

W&B runs offline, with native rank-zero RSL logging and SB3 TensorBoard integration.
Never log in or sync to the saved account. Preserve checkpoints, normalizers,
source revisions, native reward audits, failed attempts and videos locally.

ONNX export uses the official runner path and deployment metadata reference;
SB3 bakes VecNormalize's mean, variance, epsilon and clipping. Numerical parity
and behavior are separate gates. See [task/actor/reward extension](rl-task-extension.md)
and [Newton integration](isaac-newton.md). Current work is on
`feat/domain-architecture`; no commits have been pushed.
