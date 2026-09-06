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
CPU tests cover the 22 lightweight core cases, migrated MDP/manifest/SB3 behavior,
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

## Local validation after execution-policy update

Single-GPU work now runs directly on the development host; multi-GPU experiments
still use immutable scheduled snapshots. Walking and StandUp both passed 64-world,
120-step Newton task execution (`outputs/newton-task-local-0906-01` and
`outputs/newton-stand-local-0906-01`), with finite 61D actor observations and rewards.
Walking also passed actual hull/mask, non-accumulating DR, native mass/COM mirror,
4 BAM calls per action, and weighted-penalty audits
(`outputs/newton-walk-audit-local-0906-01`).

The larger MuJoCo StandUp attempts exposed a wrong entity write method name.
The corrected official `write_root_link_pose_to_sim` API passed a fresh local
64-env/5-rollout SB3 run (`outputs/domain-stand-sb3-local-0906-01`). The interface
regression test now checks against the real Entity class; 56 task/SB3 tests pass.
Native contact-force parity also corrected the primary-to-secondary sign used by
the official contact sensor. Failed scale jobs are retained, not counted as passes.

## Exact contact compilation and native learner bindings

Newton graph coloring introduced extra StandUp contacts between source mask groups.
The project now extends Isaac's native MJWarp manager and recompiles only static
contact tables from official masks before CUDA graph capture. StandUp audit 06
passed all 11 hulls (maximum support error 9.64e-9 m), 66 candidate pairs and
10 explicit ground contacts, non-accumulating DR, native mass/COM synchronization,
4 BAM calls per 50 Hz action, finite state and nonpositive weighted penalties.
Evidence: `outputs/newton-stand-audit-local-0906-06/result.json`. Earlier failed
attempts are preserved, including incorrect audit field/name assumptions.

Both representative task bindings are now explicit in `configs/tasks.json`.
Newton/SB3 Walking completed 64 envs × 5 rollouts (7680 transitions), saved native
PPO and VecNormalize, and passed native reload. RSL 5.0.1 rejected W&B 0.29's
removed `start_method`; Isaac now locks the same W&B 0.24.0 and Tyro 1.0.5 as the
validated MuJoCo environment. Native PPO semantics are unchanged.

Normalized exports use the official MuJoCo metadata reference in its isolated
compatible environment. This is an explicit export reference, not a training
fallback. Export provenance records the checkpoint's training backend. Newton's
newer Warp cannot instantiate mjlab's old export-reference simulation directly;
that failed attempt is retained. Lifecycle and behavioral validation continue.

## Lifecycle verified; evaluation modules consolidated

Newton Walking and StandUp both completed native RSL and SB3 64-env/5-iteration
training, native resume and official normalized export. RSL exports audited 44/51
scalar tags and 9 penalties each. SB3 exports matched 32 inputs (including clipped
outliers) within 9.54e-7. Both SB3 timeout resumes recorded 768 snapshots and
advanced 7680 to 15360 transitions. Evidence uses `outputs/newton-*-local-0906-*`
and the `newton_*_local` / `newton_*_resume` native RSL run directories.

Task-registered evaluation, CPU BAM rehearsal, sim2sim comparison, local schema-2
packaging and source/checkpoint identity checks are now implemented under `rl/`
and `infrastructure/`. Newton native 1280×720 video was exercised: Walking failed
at 60 ticks; all four StandUp smoke-policy scenarios completed but failed the
sustained upright gate. These are honest behavioral failures, not trained skills.
MuJoCo task video produced a black frame; explicit camera tracking is included in
the consolidated validation batch. Failed frames and original traces are retained.

The next unified batch validates the assembled interfaces before longer training.
Current CPU checks: 24 lightweight tests, 59 task/SB3 tests. All first-party code
is under `src/oh_my_duck`; the obsolete tracked training packages and their stale
bytecode/metadata have been removed. Original licenses and experiment evidence
remain. The user confirmed module-batched implementation, then consolidated
validation, with early physics gates where later work depends on physical semantics.


## Architecture merge checkpoint, 2026-09-06

The user requested merging the implemented domain architecture before completing
runtime acceptance. Source has nine domain directories under `src/oh_my_duck`.
Eight obsolete directories contained only ignored Python bytecode; these were
removed from the working tree, along with generated egg metadata. Wheel package
rules exclude bytecode. README now shows the actual directory tree.

The immutable `5971dca` batch passed all eight training/resume combinations.
SB3 passed four exports and local packages; RSL passed MuJoCo Walking, while
three exports failed an elementwise stress gate on 100x random observations.
Ordinary samples passed. The revised audit keeps the nominal elementwise gate
and uses a per-action-vector infinity norm for stress samples, with both errors
and tolerances reported; five focused tests reject injected numerical errors.
This revision still needs actual checkpoint revalidation.

EGL frames can be black during MuJoCo task replay and deployment rehearsal.
Single-thread ONNX sessions passed one isolated render experiment but did not
resolve full replay failures, so they are only an inference resource choice.
Blank-frame rejection now also covers CPU rehearsal. Failed attempts remain in
`outputs/acceptance-20260906-{rsl-rl,sb3}-01` and thread/EGL diagnostic directories.
Newton DDP remains queued. No smoke checkpoint demonstrates learned behavior,
and no policy was uploaded. Validation continues after the local main merge.


## Post-merge export acceptance

Local main merge: `6303ca4`; follow-up branch: `feat/rl-pipeline-validation`.
At that immutable source, all four RSL checkpoints from the consolidated batch
passed native normalized export, numeric parity, reward/penalty audit, provenance
checks and schema-2 local packaging. Results are in
`outputs/postmerge-rsl-export-0906-01/result.json`. Together with the four SB3
exports/packages, this establishes eight lifecycle paths through packaging.
It does not establish learned behavior or a successful full replay battery.
Pre-merge tests: 24 lightweight + 64 task/SB3; local Markdown links checked.


## Completed single-GPU execution acceptance

All eight combinations completed the lifecycle and both-backend/CPU replay.
Sixty 720p videos and finite traces were checked; all short policies failed
behavior gates. SB3 curriculum state now survives consecutive resumes. See
[final evidence and remaining work](rl-pipeline-acceptance.md).
