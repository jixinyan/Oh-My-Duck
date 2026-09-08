# Official baseline reproduction audit — 2026-09-08

The official baseline is MuJoCo/mjlab plus native RSL-RL PPO. SB3 and Newton are
project extensions. Completed short pipelines and increased reward do not prove
reproduction of the official learned behavior. A matched original-versus-refactor
training and evaluation comparison has not yet been completed.

Read-only comparison against pinned `microduck_rl` commit `29e887e` found:

- The three top-level definitions in the two representative task files, their
  two extracted native PPO assignments, and `MicroduckOnPolicyRunner` have identical
  ASTs. All 71 top-level task assignments (35 Walking, 36 StandUp), including PPO
  configuration and feature toggles, match.
- Of 229 top-level MDP definitions, 228 match after removing only project-local
  import relocations. This is static evidence, not proof of runtime equivalence
  of global bindings or compiled simulator state.
- `set_random_ground_state` has a real implementation change: direct official
  qpos/qvel writes became Entity root/joint write calls to support both backends.
  Subsequent real-runtime comparison below verifies state writes and RNG use.
  This rewrite has not been identified as the cause of poor learning.
- The owned BAM actuator implementation matches upstream bytes. Core locked
  mjlab 1.3.0, MuJoCo 3.10.0, mujoco-warp 3.8.1, Warp 1.12.0, RSL-RL 5.0.1,
  Torch 2.9.1 and BAM revision `62bd8ce` align with the pinned official lock.
- Our RSL launcher is owned orchestration, not a byte-identical copy of the native
  launcher. Both use the same seed convention, vector wrapper, runner and native
  `learn(..., init_at_random_ep_len=True)` path; matching end-to-end state and
  learning traces still requires direct runtime comparison.

The current videos use a project-defined fixed-command/per-spawn evaluation with
zero head/body commands, disabled curriculum and explicit thresholds. They retain
play-mode settings such as frequent push events. This is useful task acceptance,
but it is not evidence from an identical official evaluation protocol. No change
to thresholds is justified merely to make a preview pass.

At the latest status review the retained MuJoCo RSL runs were around 1,700 Walking
and 2,200 StandUp iterations, below their full configured budgets. StandUp's spawn
curriculum still changes at 2,500 and later stages. This limits conclusions from
early checkpoints but does not excuse missing reproduction controls. The observed
SB3 fixed-rate KL overshoot concerns the extension, not the official RSL learner.

Required closure: matched pinned-original and owned MuJoCo/RSL runs with equal seed, environments,
training budget and evaluation. Preserve the current artifacts. All additional
GPU work remains restricted to GPU 7. Until that evidence exists, official learned
behavior reproduction must remain explicitly unverified.

Machine-readable static results and exact MDP differences:
`outputs/diagnostics/official-baseline-audit-0908-01/{audit.json,mdp-diff.txt}`.
The official checkout remains pristine; it was read as reference, not installed
as a replacement for owned task source.

## Active goal and runtime controls

The current priority is official learned-behavior reproduction for Walking and
StandUp in MuJoCo/native RSL-RL, then comparable results for Newton and native
SB3. Smoke success, rising return and source similarity are intermediate evidence.
Acceptance needs task metrics, representative videos and CPU/BAM deployment
rehearsal; no hardware claim or public upload is included.

An isolated official control was installed **offline with the original lock** from
an archive of commit `29e887e`, under `.envs/official-baseline`. Its source is in
`outputs/baselines/official-0908-01/source`; the archive hash and provenance are
recorded alongside it. The owned package is absent in this environment. This is
an experiment control, not a replacement dependency for project tasks. The cached
checkout was not modified.

Verified so far:

- The compiled StandUp models have **468 exactly equal array fields**. Runtime
  task/PPO config descriptions match except callable module relocation.
- In one real 64-world MuJoCo environment, the exact official reset function and
  owned Entity writes give **bitwise equal qpos, qvel and CUDA RNG consumption in
  15 cases**: all four spawn buckets plus a mixed bucket, each with all worlds,
  a reversed noncontiguous subset and an empty subset. Sitting noise, tilt and
  face-up roll noise are enabled; untouched worlds remain unchanged.
- Independent seed-42 processes produce equal initial qpos, qvel and 61-D actor
  observations. Critic contact forces differ slightly. An official-versus-official
  repeat also differs: first-step qvel maximum difference was 0.00851 versus
  0.00136 in the original/owned comparison. Therefore full trajectory bitwise
  equality is not a valid convergence requirement. This does not yet establish
  statistical behavioral equivalence.
- Both official and owned StandUp native PPO completed **64 environments / 5
  iterations**. The official run passed finite checks on **51 scalar tags**, all
  **9 penalty checks**, and original `run_export` plus ONNX schema/forward checks.
- Two existing reset regression tests pass in the locked MuJoCo/SB3 environment.
  The plain MuJoCo environment has no pytest installed; its attempted test command
  did not run tests. No dependency changes were made to running environments.

The first original export audit attempt failed before export because direct
execution placed the evaluation directory's `mujoco.py` ahead of the installed
MuJoCo package. Re-running with Python `-P` completed successfully in a separate
artifact path. The failed log remains preserved.

`experiments/baseline_probe.py` records model/config and 120 action steps per
curriculum stage (StandUp: 0, 600, 1500, 2500, 4000). These are diagnostic traces,
not training or a curriculum pacing experiment. The first diagnostic version
manually reset a subset without the complete sensor refresh; its initial full
reset/model comparisons remain useful, but its subset trace is superseded by v2,
which uses public `env.reset(env_ids=...)`. V2 matched controls and Walking controls
are running; no result from an unfinished control is claimed.

Evidence: `outputs/baselines/official-0908-01/`, especially
`probe-owned-standup-v2/result.json`, `export-official-standup-v2/result.json`,
and `remaining-controls-{plan,result}.json`. New GPU work uses only GPU 7 and
all W&B runs remain offline. Long-run learned behavior remains unverified.
