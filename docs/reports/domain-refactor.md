# Domain architecture refactor — 2026-09-06

The source package is `src/oh_my_duck`. Simulator environments install that package,
including project-owned task recipes, MDP, models, motor code and policy export.
`environments/` contains isolated locks. `third_party/microduck_rl/RELOCATIONS.json`
preserves the relocation map alongside original source ancestry and license.

## Verified migration evidence

| Task / learner | Scheduled smoke | Resources | Result |
|---|---|---|---|
| Walking / RSL-RL | `omd-domain-walk-rsl-0906-02` / `9d56446a` | 2 GPUs, 64 envs per rank, 5 iterations | Passed |
| StandUp / RSL-RL | `omd-domain-stand-rsl-0906-02` / `b5694689` | 2 GPUs, 64 envs per rank, 5 iterations | Passed |
| Walking / SB3 | `omd-domain-walk-sb3-0906-01` / `19ef8b56` | 1 GPU, 64 envs, 5 rollouts | Passed |
| StandUp / SB3 | `omd-domain-stand-sb3-0906-01` / `c04c5e29` | 1 GPU, 64 envs, 5 rollouts | Passed |

Both native RSL reward/export audits passed (`e23d8ca9`, `f0aee1c8`). SB3 normalized
exports passed 32-sample parity including clipped outliers: Walking max error
9.536743e-7, StandUp 1.907349e-6. Both SB3 resumes advanced 7680 to 15360 timesteps
and verified 768 timeout snapshots. RSL resumes also succeeded (`3cfadc8e`, `f2f3829f`).

Evidence lives in `outputs/domain-*-export-0906-01`, `outputs/domain-*-sb3-resume-0906-01`
and `logs/rsl_rl/*/*domain*`. These are pipeline checks; learned behavior is unvalidated.
The first RSL launch attempts failed at CLI parsing (`--gpu-ids 0 1`); retries use
the native `--gpu-ids all` selection. Original failed artifacts remain preserved.

Walk and groundcontact USD rebuilds succeeded (`69b56846`, `9c473aef`). Static
source-to-USD matching covers all 5 walk and 11 groundcontact collision meshes.

The wheel build includes CLI, shared contracts, RL export, both robot resource
families and Newton's ground plane. No cached Microduck task package is installed.
CPU tests cover the 21 lightweight core cases, migrated MDP/manifest/SB3 behavior,
native Isaac diagnostic contracts, virtual sensor parity and permuted-joint resets.
The StandUp entity-API reset is bit-identical to its prior implementation under the
same seed on the canonical model; it also respects permuted solver joint indices.

## Newton task binding — still gated

The shared-task adapter uses native Isaac `ManagerBasedEnv` and Newton `SolverMuJoCo`.
It binds canonical entity state, native BAM, measured contacts, virtual sites/IMU,
flat terrain rays and per-world model randomization. Original task managers retain
observation delays, noise, rewards, terminations, curriculum and recorder ordering.
Neither representative task is yet registered as trainable on Newton.

GPU gates have identified missing native scene fields, empty-manager initialization,
and a missing terrain sensor frame count. Each failed run is retained under
`outputs/newton-task-gate-*`; the admission-lock timeout for attempt 02 is retained
separately. Fixes are under active validation. Full Newton smoke/resume/export/replay,
behavior and sim2sim acceptance remain open. A PD diagnostic is not task migration.

## Reproducible jobs and cleanup

New submissions run from committed detached worktrees in `.job-sources/<revision>`.
Source/configuration are fixed; generated outputs and dependency environments are
explicit shared links. Snapshots are ignored and are never edited during a run.

Obsolete training-package metadata will be removed after its replacements pass the
remaining relevant gates. Failed experiment evidence and licensing are retained.
