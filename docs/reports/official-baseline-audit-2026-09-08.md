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
  Matched state/observation parity must be demonstrated before treating this
  change as equivalent. No evidence yet identifies it as the cause of poor learning.
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

Required closure: state/observation comparison for the changed reset path, followed
by matched pinned-original and owned MuJoCo/RSL runs with equal seed, environments,
training budget and evaluation. Preserve the current artifacts. All additional
GPU work remains restricted to GPU 7. Until that evidence exists, official learned
behavior reproduction must remain explicitly unverified.

Machine-readable static results and exact MDP differences:
`outputs/diagnostics/official-baseline-audit-0908-01/{audit.json,mdp-diff.txt}`.
The official checkout remains pristine; it was read as reference, not installed
as a replacement for owned task source.
