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
which uses public `env.reset(env_ids=...)`. Both v2 task controls and all four
64-env/5-iteration native PPO controls have now completed.

Evidence: `outputs/baselines/official-0908-01/`, especially
`probe-owned-standup-v2/result.json`, `export-official-standup-v2/result.json`,
and `remaining-controls-{plan,result}.json`. New GPU work uses only GPU 7 and
all W&B runs remain offline. Long-run learned behavior remains unverified.


## Walking controls and a corrected behavior gate

Walking's 468 compiled arrays match exactly. Full and noncontiguous subset reset
qpos/qvel and actor/critic observations also match exactly. First-step reward is
identical; the largest qvel difference is 2.38e-7. StandUp v2 full and subset reset
qpos/qvel and actor observations match; critic contact-force differences remain.
All four original/owned × Walking/StandUp native PPO smoke runs completed.
Original Walking export passed 44 scalar and 9 penalty checks; owned StandUp
export passed 51 scalar and 9 penalty checks. Owned Walking export also passed 44 scalar and 9 penalty checks.
Runtime comparisons are in `runtime-comparison.json` under the evidence directory.

The pinned published `alpha_walking.onnx` is **not established as a successful gait
reference**. In the current CPU/BAM battery it stood almost stationary: commanded
forward 0.1 m/s yielded mean 0.000172 m/s, and commanded yaw 0.5 rad/s yielded
0.0163 rad/s. Total XY displacement was approximately [0.00256, 0.00098] m. Video
keyframes corroborate this. The 61-D input carries the correct twist in slots
48:51 (error <1.5e-9), with zero head/body command slots. Its metadata declares the
same observation order, servos and HOME convention; the cause of its behavior in
this runtime is not yet established. In task MuJoCo it also fails the old tracking
criterion. This is not evidence that every official policy behaves this way.

The CPU battery originally returned `passed`: global RMSE is diluted by idle
segments, and the old limits even allowed a completely stationary policy to pass.
That was a scoring defect. Walking scoring version 2 retains the prior fall and
tracking limits and additionally requires at least 50% signed mean response in
**each commanded motion segment**. This is a project acceptance guard against
non-response, not a change to official training/rewards or a claim that 50% alone
establishes gait quality. Real motion, command tracking and video review remain
required. Regression tests cover stationary, wrong-direction, missing-turn,
tracking and falling cases; six protocol/reset tests pass.

Original traces, videos and the false-positive result are preserved under
`published-walking-cpu/`. `rescored-v2.json` explicitly supersedes its behavior
classification with `behavior_failed`; forward/turn response fractions are 0.17%
and 3.25%. Existing immutable training/preview snapshots retain their old scorer:
use the current scorer on their saved traces before making new behavior claims.
No long-run policy has passed current behavior acceptance yet.


## Original full training and baseline resource priority

The isolated original StandUp control passed its 4096-environment / 5-iteration
capacity gate and scalar/penalty/export audit, then entered a **fresh 15,000-iteration
native PPO run**, seed 42, on GPU 7. Its earlier 64-env CPU/BAM battery completed
all four 8-second scenarios with finite state and video; all four short-checkpoint
behaviors failed. These are execution gates, not convergence evidence.

`outputs/baselines/official-standup-full-0908-01/{plan,launch,result}.json` records
exact commands, gate hashes, archive hash, pinned original commit and project
observer commit. The training process uses the isolated official package and
native CLI; owned tasks are absent from that environment. Full-run completion
alone leaves behavior explicitly pending.

An additional observer around the original runner's export checked 32 ordinary
and stress samples against its normalized native inference graph. Nominal maximum
absolute error was 1.43e-6; stress normalized error was 6.99e-7. Both pass the same
numerical tolerances used by the owned exporter. No training implementation was
changed; evidence is `capacity-numerical-export/result.json`.

First-update checkpoint comparisons also include an original-versus-original
control. Largest actor-state tensor differences were 0.01834 for two original
runs and 0.00814 for original versus owned; corresponding critic differences were
0.01749 and 0.00770. Checkpoint 0 is **after one PPO update**, not an initial-weight
snapshot. These comparisons expose the scale of run variability and do not prove
long-run learning equivalence. See `first-update-comparison.json`.

To prioritize the user's official-first goal, three live Newton learner process
groups (two RSL, one SB3 Walking) are now **suspended with SIGSTOP**, preserving
in-memory state as well as existing checkpoints. They were not terminated for
poor learning. Per-run `operator-pause.json` records the verified PID/group and
SIGCONT resumption mechanism; `/proc` state confirmed the suspension. Two owned
MuJoCo learners and the original StandUp control remain active. The three earlier
stopped SB3 attempts remain separate preserved failures/triage records. Raw campaign
worker status remains `running` for suspended live workers; the operator pause
records describe their actual state.

Measured 10-iteration mean times before/after priority adjustment were:

| Learner | Before | After |
|---|---:|---:|
| Original StandUp | 12.002 s | 7.829 s |
| Owned Walking | 10.723 s | 7.034 s |
| Owned StandUp | 11.525 s | 7.575 s |

This is about 1.5× faster baseline progress under current host contention, not an
isolated GPU benchmark. GPU 7 remains the only RL GPU; W&B remains offline.
Resuming suspended groups requires verifying their recorded command and PID
identity before SIGCONT. Do not launch duplicate replacements while they are live.

Existing Walking preview traces were rescored with version 2, without overwriting
their old reports. Latest completed preview forward/yaw response fractions were
0.205/1.007 (owned MuJoCo/RSL), 0.517/0.959 (Newton/RSL), and 0.118/0.994
(Newton/SB3). All still fail the complete behavior gate. These are specific saved
checkpoints, not assertions about the current unsaved training weights. Evidence:
`outputs/baselines/official-0908-01/preview-rescored-v2.json`.


## CPU BAM load-indexing correction — revalidation required

The trained owned Walking checkpoint 2000 also nearly stood still in CPU/BAM:
forward mean -0.000176 m/s and turn mean 0.01939 rad/s, despite nonzero commands.
StandUp checkpoint 2500 passed standing/sitting and failed face-down/face-up there.
Those traces are preserved in `outputs/baselines/trained-cpu-0908-01`.

Investigation found a concrete CPU controller defect in pinned BAM `62bd8ce`:
`MujocoController.update()` subtracts DOF-friction forces by matching `efc_id` to
**joint ids**. The robot's controlled joints are 1–14, while their DOFs and actual
FRICTION_DOF constraint ids are 6–19. The Warp BAM path already scatters these
forces by DOF. MuJoCo's own [3.10 constraint code](https://mujoco.readthedocs.io/en/3.10.0/_modules/mujoco_warp/_src/island.html)
also uses `dof_treeid[efc_id]` for FRICTION_DOF, versus a joint-to-DOF lookup for
joint-limit constraints. Incorrect subtraction contaminates the external motor
load used by BAM's load-dependent friction model. Its behavioral impact remains
an A/B measurement, not an assumed explanation of every failed policy.

The project extension `robotics/microduck/actuators/cpu_bam.py` retains the native
motor/sag update and recomputes only the friction fields with correctly indexed
loads before `mj_step`. The friction formula is the same stateless BAM function.
Reset also aligns the controller clock with the rewound simulation clock; the
current XL330 proportional controller does not use dt, so that timing correction
cannot explain Walking response. No dependency pin or training recipe changes.
CPU rehearsal reports now identify the controller and DOF indexing explicitly.

Three tests pass, including a real robot comparison against **MuJoCo's constraint
Jacobian projection**, independent of efc-id indexing. The old budget differs; the
corrected budget matches the projected load, while native motor torque is exactly
preserved. Reset clock/targets/sag and all official ground spawn states also pass.
Evidence: `tests/rl/tasks/test_cpu_bam.py` and
`outputs/baselines/trained-cpu-0908-01/cpu-bam-tests.log`.

Earlier CPU rehearsal behavior results are retained as historical observations
under the old controller and require revalidation. Their finite execution/video
checks do not certify matched motor-load semantics. Current full MuJoCo training
uses the unaffected Warp controller and continues while corrected CPU A/B runs
are prepared. Learned-behavior acceptance remains open.


## Corrected CPU A/B and interface controls

The corrected CPU controller replay is complete for owned Walking checkpoint
2000, owned StandUp checkpoint 2500, and the published alpha Walking policy.
Policy hashes and seed 42 are identical across each before/after pair; evidence:
`outputs/baselines/cpu-bam-dof-0908-01/{launch,comparison,video-check}.json`.
All six videos are 1280×720 at 25 FPS. Selected Walking frames at 4/12 seconds and
all four final StandUp frames were inspected: Walking remains essentially in
place, while prone/supine StandUp remains down. These are selected-frame checks,
not a claim of reviewing every video frame.

| Policy | Corrected CPU result |
|---|---|
| Owned Walking 2000 | Forward mean −0.000163 m/s for 0.1 m/s; yaw mean 0.01677 rad/s for 0.5 rad/s; fails |
| Published alpha Walking | Forward mean 0.000106 m/s; yaw mean 0.01541 rad/s for the same commands; fails |
| Owned StandUp 2500 | Standing/sitting pass; face-down and face-up fail (final tilt 0.522/1.150 rad) |

Thus the DOF correction fixes a measured load computation defect but **does not
restore the missing behavior in these samples**. The published policy's old
`success: true` in the preserved comparison input used scoring v1; its separately
rescored original trace also fails v2. Do not attribute that status change to the
physics correction. Full corrected CPU acceptance across eight combinations is
still outstanding, as is learned-behavior acceptance.

Further diagnostics kept the official task recipe unchanged:

- Twelve short command/scene cases compared groundcontact and walk scenes for
  published/owned policies. The two scenes produced identical trajectories for
  each case. A 0.3 m/s forward command mostly produced sideways velocity
  (−0.089/−0.114 m/s), not forward tracking. A 1 rad/s yaw command elicited about
  0.727 rad/s from the owned policy. This is a diagnostic command scan, not a
  relaxed acceptance protocol. Evidence: `outputs/baselines/cpu-command-scene-0908-01`.
- A temporary observer matched training's one-control-step joint-velocity delay.
  It did not restore forward response. It is not a production inference change.
  Evidence: `outputs/baselines/cpu-command-lag-0908-01`.
- At 32 equal recorded states with noise, delay and IMU randomization disabled
  only for the diagnostic, all 61 raw actor inputs matched to maximum error
  7.45e-9. With synthetic nonzero root velocity, actor error was 8.94e-8 and
  body-frame linear velocity error was 6.29e-8. This rejects a simple observation
  ordering/frame mismatch in these cases; it does not establish full temporal
  or physics equivalence. Evidence: `cpu-observation-parity-0908-02` and
  `cpu-velocity-parity-0908-01` under `outputs/baselines`.

The first observation diagnostic (`cpu-observation-parity-0908-01`) is invalid:
its observation manager reused cached observations and retained IMU misalignment.
Its measurement note explicitly excludes it. The corrected attempt refreshes
history and removes that diagnostic-only corruption; both attempts are preserved.

## Both original full controls active

Original Walking also passed 64-env/5-iteration smoke, finite scalar/penalty audit,
official export, 4096-env/5-iteration capacity, and native normalized export
parity (32 samples; ordinary error 1.19e-6, stress normalized error 8.99e-7).
Its initial CPU execution gate predates the load correction above. It is training
from scratch at seed 42, 4096 environments and the official 50,000-iteration
budget, alongside original StandUp's 15,000-iteration budget. Evidence:
`outputs/baselines/official-walking-full-0908-01/{plan,launch,result}.json`.

There are now **four active MuJoCo learners** on GPU 7: original and owned for
each representative task. The three Newton groups remain SIGSTOP-suspended;
the three previously stopped SB3 attempts remain separate. A brief single-learner
measurement restored every temporarily suspended MuJoCo group in its finalizer:
original StandUp averaged 9.046 s with four learners and 3.143 s alone. This is a
per-learner speed comparison, not a 2.88× aggregate training-throughput claim.
Evidence: `official-standup-full-0908-01/single-learner-measurement.json`.

The matched checkpoint-250 CPU comparison completed on the second attempt. Its first export
attempt failed before evaluation because selecting OSMesa in the isolated
original environment could not initialize OpenGL. The original training process
was unaffected. The new attempt uses its working headless EGL import and a new
output directory: `outputs/baselines/matched-iteration-250-0908-02`; the failed
`...-01` remains preserved. This early checkpoint control does not replace
long-run behavior evaluation.


At checkpoint 250, both original and owned Walking completed the 14-second CPU
battery but failed command tracking (forward means −0.0150/−0.00414 m/s; yaw means
0.0892/0.1485 rad/s). Both StandUp policies passed only the standing-start case;
sitting, face-down and face-up failed. The four export/scalar/penalty audits passed.
Original normalized export parity passed 32 samples per task (ordinary maximum
errors 1.31e-6 Walking / 2.09e-6 StandUp). These CPU diagnostics intentionally
omit rendering; later video acceptance remains required.

At matched training iterations 200–250, original/owned mean rewards were
95.67/96.22 for Walking and 34.01/33.43 for StandUp. Linear tracking weighted reward
was 0.989/0.979; StandUp standing-composite reward 1.537/1.468. Early behavior and
learning curves therefore provide no measured evidence that refactoring caused
the missing early skill; this is a single training seed and does **not** establish
long-run equivalence. Source/checkpoint hashes, all four corrected CPU traces and
export results are in `outputs/baselines/matched-iteration-250-0908-02`.

The 05:59 UTC live snapshot shows original Walking/StandUp at 301/385, with latest
50-record rewards 98.04/39.99 versus previous 95.95/37.59. Owned Walking/StandUp at
2443/2873 are nearly flat or slightly lower (114.08/40.40 versus 114.73/41.15).
Their main task terms are also approximately flat, so no new skill claim is made.
StandUp has just passed the 2500 recovery-spawn curriculum stage; a small window
change alone is not evidence of irreversible divergence. Full context is recorded
in `reward-comparison.json`; further checkpoint behavior determines continuation
or diagnosis under the user's instruction to stop unpromising runs.


## Ongoing checkpoint observation

The old preview watcher (`25fd4df`, scoring v1) was stopped between renders after
verifying it had no child process. Its gallery and attempts remain preserved.
The replacement uses immutable `0792f99`, GPU 7 and scoring v2, watching only the
two active owned MuJoCo runs. Native preview gallery:
`outputs/previews/shared-gpu7-0908-02/index.html`; launch/provenance record:
`outputs/previews/shared-gpu7-0908-02-launch.json`. These native task videos are
separate from the corrected CPU/BAM A/B videos above.

A lightweight milestone observer waits for original and owned checkpoints at
500, 1000, 1500, 2500, 4000 and 6000 iterations, then runs the same export/scalar/
penalty and corrected CPU diagnostics. It pins observer code to `0792f99`, uses
only GPU 7, preserves each milestone directory, and reports errors without
reusing or overwriting failed attempts. It does not restart or alter training.
Evidence: `outputs/baselines/matched-growth-0908-01/{launch,result}.json`.
These single-seed, no-video diagnostics measure learning progress; repeated-seed
native-backend and CPU video acceptance remain separate requirements.


## Current recovery triage and corrected CPU matrix

A paired 16-reset StandUp battery found sitting success falling from 13/16 at
2000/2250 to 10/16 at 2500 and 8/16 at 2750, with no prone/supine successes. The
owned MuJoCo StandUp learner is now SIGSTOP-paused at logged iteration 2985 for
diagnosis; original controls and owned Walking continue. A separate, gated native
PPO experiment follows the official recipe's own first recommendation: halve only
body angular-velocity penalty. See [recovery diagnosis](standup-recovery-diagnosis-2026-09-08.md)
for exact results, curriculum continuity, reward measurements, pause identity and
experimental limits. This supersedes the preceding four-active-learner snapshot.

Corrected CPU/BAM matrix revalidation is complete for the original eight preserved
acceptance policies. All eight executions pass finite-state/61/14 checks and all
20 videos pass 1280×720, 25 FPS and trace-matched frame counts. Selected final
Walking and face-up StandUp frames were inspected for each policy; they show falls
or remaining down. All eight still report `behavior_failed`, as short policies.
Evidence: `outputs/baselines/corrected-cpu-matrix-0908-01/result.json`. This closes
corrected CPU execution revalidation, not learned-task or hardware acceptance.


The scheduled checkpoint-500 comparison completed all four exports and corrected
CPU batteries. Both original/owned Walking fail tracking (forward mean
−0.00464/−0.01060 m/s, yaw 0.0831/0.2224 rad/s); both StandUp policies still pass
only the standing-start case. No long-run equivalence or learned recovery claim
follows. Evidence: `matched-growth-0908-01/iteration-500/result.json` under
`outputs/baselines`. The observer now waits for checkpoint 1000.


## Successful published StandUp reference

The separately downloaded, pinned official `alpha_stand` passes all 64 corrected
CPU reset samples and all four seed-42 CPU video cases, with final ground geometry
contacts on both feet only. It also passes all four native MuJoCo and all four Isaac/Newton scenarios; eight
native videos and finite 61/14 traces are checked. Its training checkpoint/settings are absent from
ONNX metadata, so this demonstrates compatible execution of a successful policy,
not our own training reproduction. See [published-policy reference](published-policy-reference-2026-09-08.md)
for the source/hash, videos, exact paired states, SVG trajectories and motion-cost
measurements. Published Walking still fails at runtime action scales 1.0 and 0.9.


## Subsequent official-history and checkpoint audit

See [official source history](official-source-history-2026-09-08.md) for verified
recipe configuration continuity, exact published-byte runtime commits, CAD/contact
geometry comparison and the failed full-task checkpoint-1500 batteries. The
replacement milestone observer uses the correct resume directories and advances
original tasks independently. Only the original learners continue; the angular
penalty diagnostic has finished without prone/supine recovery, and further tuning
is deferred. Earlier active-run/observer statements above are historical.
