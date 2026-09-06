# Representative RL reproduction

The acceptance scope is **Flat Walking** and **Flat StandUp**, each on MuJoCo/mjlab and Isaac Lab/Newton with RSL-RL and Stable-Baselines3. Walking covers continuous command tracking and the walk collision model; StandUp covers recovery and the ground-contact model. The 33-task official registry is retained as an inventory for future extension, not a promise that every task has been reproduced.

`configs/training.json` owns the selected tasks and logging defaults. `python omd.py tasks` lists the representative tasks; `--all` shows the pinned registry inventory. Regenerate inventory with `.envs/mujoco/bin/python src/oh_my_duck/rl/tasks/catalog.py --output configs/official_tasks.json`. Registration and validation are separate: adding a name cannot make an unsupported Isaac task available.

## Acceptance matrix

| Official task | MuJoCo / RSL-RL | MuJoCo / SB3 | Isaac Newton / RSL-RL | Isaac Newton / SB3 |
|---|---|---|---|---|
| Mjlab-Velocity-Flat-MicroDuck | Earlier single/dual-GPU smoke and export passed; owned-source GPU rerun and learned behavior pending | Single-GPU smoke/resume/export passed; learned behavior pending | BAM/contact replay passed; full task pending | Full task pending |
| Mjlab-StandUp-Flat-MicroDuck | Earlier dual-GPU smoke/reward audit/export passed; owned-source rerun and behavior pending | Earlier smoke/reward audit/resume/export passed; owned-source rerun and behavior pending | Earlier ground-contact conversion passed; new source fingerprint and full task pending | Earlier ground-contact conversion passed; new source fingerprint and full task pending |

Every combination requires configuration/state/reward checks, the official 64-environment / 5-iteration smoke, native checkpoint continuation, normalized official-runner ONNX export, task-specific behavior measurements and recorded playback. Smoke checkpoints do not establish task reproduction. Cross-simulator replay must preserve commands, observation/servo ordering, HOME, BAM, timing and task-specific spawn semantics.

## Native PPO and training throughput

RSL-RL retains the original PPO and native distributed gradient synchronization. MuJoCo delegates to the pinned mjlab trainer and its torchrunx launcher. Isaac delegates to the pinned Isaac Lab distributed launcher. Environment counts are per rank; record both per-rank and total environments, rollout size, seeds and update count. More GPUs change the collected batch, so benchmark throughput and behavior instead of assuming linear speedup.

SB3 retains native PPO. Its vectorized GPU simulator and independent training jobs can run concurrently across GPUs; this is not distributed-gradient SB3. No decoupled asynchronous actor/learner or policy-lag algorithm is introduced. Separate learning FPS, collection FPS, end-to-end transitions per second and aggregate task throughput when comparing configurations.

## Offline W&B

W&B is a standard local experiment log and is **offline only**. The saved online account is not the user's account. No login, upload or `wandb sync` is part of this workflow. RSL-RL uses its native W&B writer; its launcher environment allowlist is extended to propagate offline mode to every rank. SB3 uses W&B's TensorBoard integration around its original PPO trainer. `configs/training.json` specifies the local project name; credentials never belong in configs or job commands.

Keep local W&B run data alongside checkpoints, normalization state, source revisions, reward audits and videos. Multi-rank runs log through the native rank-zero writer. A resumed training run should preserve its optimizer/normalizer and record the source checkpoint; a new local logging run may link to that source without pretending it is the same experiment.

## Implementation order

1. Verify representative MuJoCo tasks, native RSL-RL multi-GPU launch and offline W&B for both frameworks.
2. Migrate full official task semantics into a reusable Isaac binding: observations, commands, sensors, rewards, DR, curricula and NaN/reset handling. Keep task recipes in the project-owned factories under `training/microduck`; unsupported terms fail explicitly.
3. Validate Isaac training, resume and official-compatible export for both frameworks, then train and evaluate behavior in both simulators.
4. Benchmark environment/GPU configurations after correctness gates, and use the measured configuration for longer native-PPO training.

Local `main` includes merge `0ab535a`. Further work is on `feat/rl-task-reproduction`. Neither branch has been pushed by this workflow.

## Owned-source migration

Tasks and policy configuration are now maintained under [training/microduck](../training/microduck/README.md). Existing evidence predates this source switch; CPU inventory parity is not GPU acceptance. See [migration handoff](reports/owned-task-migration.md).
