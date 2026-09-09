# SB3 critic repair and RSL cross-simulator diagnosis

Follow-up to the [framework review](rl-framework-status-2026-09-09.md).
User requested SB3 repair, diagnosis of the weaker MuJoCo RSL run, and capability
validation of the Newton RSL policies including MuJoCo transfer.

## Critic interface implementation

Fresh SB3 training now defaults to separate official actor/critic observation
groups. Native SB3 PPO uses its DictRolloutBuffer and its existing timeout
bootstrapping. Independent VecNormalize statistics are saved for each group.
The policy keeps 61 actor inputs, 14 outputs and task-local network widths;
critic observations cannot enter actor hidden layers or its exported graph.
Both terminal groups are sampled before reset without advancing live delay
buffers or consuming the training RNG. Export still uses official run_export
and the runner export method, with actor normalization baked in.

`--critic-observations actor` preserves the former input for controlled studies.
Resume infers the recorded layout (legacy artifacts imply actor-only) and rejects
changing layout while restoring a checkpoint. Old exports remain supported.
PPO optimizer, value loss and learning-rate defaults are unchanged in this
isolated interface change. This does not claim that critic repair alone fixes
learning behavior.

CPU checks: 14 SB3 tests passed, including privileged-input/gradient isolation,
native DictRolloutBuffer training and normalized terminal-critic bootstrapping,
model/normalizer save-reload, asymmetric actor graph parity, legacy graph parity,
and real observation-manager delay/history preservation. Required native
64-env/5-update, export and CPU/BAM video gates are pending.

## Frozen-policy transfer experiment

`outputs/diagnostics/newton-transfer-0909-02` uses immutable source `668b661`:
Newton StandUp checkpoint 13000 and Walking 14000, plus final MuJoCo RSL StandUp.
Native backends test recovery at seeds 42/100/101; CPU/BAM tests recovery at 42
and 100–115. Seed 42 includes videos. The same ONNX bytes are checked in each
backend. This avoids conflating simulator transfer with another training run.

Attempt 01 failed before any rollout because the diagnostic driver globally
selected OSMesa before loading its optional library. Attempt 02 keeps EGL in
the parent and selects OSMesa only in the existing isolated MuJoCo video worker;
both attempts and their logs are retained. No training physics changed.
Results and remaining causal uncertainty will be recorded after completion.

## Native SB3 learning-rate feedback

A separate serializable callable schedule and rollout-boundary callback now
reduce/increase the learning rate from SB3's preceding logged approximate KL.
The native PPO update and early-stop implementation remain unchanged. Bounds
1e-5–1e-2 and factor 1.5 match the official RSL controller's bounds, but timing
and KL estimator differ: this is not numerically identical RSL PPO. Fresh runs
default to adaptive; resume preserves the saved mode. Explicit constant mode
supports matched comparisons, including legacy checkpoints. Both schedule state
and native optimizer learning rate survive save/reload in the tests.

16 SB3 CPU tests pass. A real checkpoint comparison of constant versus adaptive
updates is still required before claiming the observed KL instability is fixed.
The critic-only gates run from source `c897d10` and retain constant learning rate;
the new controller is a separate change, not silently added to those runs.

Transfer evidence so far: Newton StandUp 13000 in CPU/BAM passes standing,
sitting and supine at all 17 tested base seeds; prone passes only 3/17. Native
4/4 preview success is therefore not full transfer success. Native paired tests
use each backend's registered interpreter in attempt 03; attempt 02 incorrectly
attempted MuJoCo inside Newton's isolated dependency environment and failed
before a native rollout. CPU attempt 02 remains valid and continues independently.

## Initial episode phases and RSL diagnosis

The official RSL entry point calls `learn(init_at_random_ep_len=True)`. Fresh
SB3 runs now use the same one-time uniform episode-length initialization, so
thousands of worlds do not all time out on the same control step. Ordinary
subsequent resets still reset their counters normally. Native SB3 PPO and vector
stepping remain intact; this is not a decoupled actor implementation. Explicit
`--initial-episode-phase synchronized` supports comparison; legacy resume keeps
its previous setting. All three runtime choices are recorded in checkpoints.
17 SB3 CPU tests passed, including deterministic randomized initial phases.

The RSL source-668b661 reconstruction at 8192 environments/seed 42 finds exact
initial actor tensor hashes across backends and identical recorded solver
options. Saved full-run environment YAML is identical; agent YAML differs only
in run_name. CUDA RNG state before runner construction differs despite the same
seed; CPU RNG and initial actor weights match. This establishes different
exploration streams, not proof that randomness alone explains the final gap.
Evidence: `outputs/diagnostics/rsl-backend-initialization-0909-01`.

Newton StandUp 13000 transfers to native MuJoCo at seeds 42/100/101 with 4/4
successes each. MuJoCo can execute the complete behavior; its own final policy's
missing supine skill is a learning outcome, not an inability of the simulator to
represent it. CPU/BAM still exposes a prone-transfer gap. No official rewards,
curricula, robot parameters or RSL defaults have been changed.

## Verified update comparison and next full runs

The fixed-checkpoint comparison (`outputs/diagnostics/sb3-kl-feedback-0909-01`)
completed with identical old update-10000 model/normalizer, 8192 environments,
seed 42, synchronized initial phases and actor-only critic in both arms. Only
KL feedback differs; initial learning rate is 1e-4. Each arm executed 32 native
PPO rollouts. SB3 logs the preceding update, yielding 31 displayed KL samples.

| Metric | Constant | Adaptive |
|---|---:|---:|
| Mean logged KL | 0.07300 | 0.01196 |
| Last 16 mean KL | 0.05339 | 0.00858 |
| Maximum logged KL | 0.19825 | 0.03440 |
| Early-stop messages / 32 rollouts | 32 | 16 |
| Final learning rate | 1e-4 | 2.963e-5 |

This is evidence for improved update stability, not learned-skill recovery.
The critic-only four-combination gates completed from `c897d10`: all native
64-env/5-update runs, resumes, official-route normalized exports and CPU/BAM
videos executed. Smoke behavior failures are retained and expected.

`configs/experiments/sb3-official-critic.json` makes critic layout, learning-rate
mode and initial episode phase explicit at every worker stage. The new campaign
will rerun all gates from the combined committed implementation before fresh
full training. No old gate is reused across changed policy/model source. Policy
model sources are now included in preparation invalidation. Regression checks:
29 tests and 9 subtests passed for SB3, campaign options, recovery and scheduling.

## Combined recipe startup verification

All four `sb3-repair-0909-01` learners now produced full PPO updates after their
own fresh 64/5 smoke, normalized export, CPU video, native resume, 8192/5
capacity and export gates. Training snapshot: `bea3eb1`. GPU allocation is
0/2/4/5, explicitly shared with existing project runs; GPU 1/3 foreign workloads
are untouched. Native process checks confirm 8192 environments, official critic,
adaptive rate, randomized initial phases and offline W&B in all four learners.
`outputs/experiments/sb3-repair-0909-01/repair-runtime-audit.json` records this.
The preview worker saves snapshots starting at update 1000 in
`outputs/previews/sb3-repair-0909-01/index.html`. No convergence claim is made.

## Walking transfer and metric interpretation

The no-push intervention zeros only velocity-push amplitudes, leaving event
sampling and other play settings intact. Newton-trained Walking 14000 produces
0.0572 m/s in Newton and 0.0627 m/s in MuJoCo for a 0.1 m/s command; yaw response
is 92.3% and 91.1% respectively. Both miss the existing whole-episode lateral
RMSE gate (~0.124 vs 0.1 m/s). This primarily measures periodic sway: during the
forward segment mean lateral velocity is 0.00242/0.00169 m/s, and one-second
averaged lateral error RMS is 0.00979/0.00499 m/s. Do not label it persistent
sideways drift or silently relax the acceptance threshold. The MuJoCo-trained
checkpoint in the same diagnostic has only 1.7–1.8% forward response across
backends, so external pushes do not explain Newton's superior forward response.

CPU/BAM Walking 14000 stays upright but averages -0.00118 m/s forward and
0.3004 rad/s yaw. Its deployment forward transfer fails. Native recovery videos
show actual rise and hold; the same policy in CPU/BAM settles into a lean after
prone reset. Inspected frames: `newton-transfer-0909-03/recovery-transfer.jpg`.
Video gallery: `outputs/diagnostics/newton-transfer-0909-03/index.html`.
Evidence: `outputs/diagnostics/walking-no-push-0909-01/{result,segment-diagnosis}.json`.

Remaining work is learned convergence of the repaired SB3 runs and causal
isolation of the MuJoCo learning/CPU-transfer gaps. Native normalizer updates,
value/advantage loss conventions and timeout bootstrap conventions differ across
frameworks and remain explicit; native PPO has not been replaced with a shared
custom optimizer. A no-reward-change RSL seed control is the next diagnostic,
not another unverified reward ablation.

## Integration and additional seed control

The verified SB3 repair milestone is merged and pushed to main at `49b56de`.
Follow-up uses `feat/rl-transfer-validation`, created from that main revision.
`configs/experiments/mujoco-recovery-seed43.json` repeats only MuJoCo RSL StandUp
with seed 43, retaining 8192 environments and the official 15000-update budget.
It will pass fresh smoke/export/CPU-video/resume/capacity gates before training;
its purpose is to test exploration sensitivity, not to establish a backend-wide
success rate from one extra seed. GPU 1 became free and is allocated explicitly.


## Completed frozen-policy battery

`newton-transfer-0909-03/result.json` completed all 14 groups. Each StandUp
policy was evaluated from four poses at seeds 42, 100 and 101 in both backends;
Walking used seed 42 and the 700-tick hold/forward/stop/turn schedule. Policy
hashes were checked against the same frozen exports. Newton StandUp 13000 passed
12/12 in each backend. MuJoCo StandUp final passed 9/12 in each backend, failing
all three supine cases: its missing recovery skill follows the policy across
backends. This is evidence against an inability of MuJoCo to execute recovery,
not a proof of the cause of unsuccessful learning.

Standard native Walking retains external pushes and gives these measured
responses for Newton checkpoint 14000:

| Evaluation | Forward / command | Yaw / command | Lateral RMSE | Result |
|---|---:|---:|---:|---|
| Newton | 66.2% | 103.4% | 0.1263 m/s | Lateral gate fails |
| MuJoCo | 45.4% | 62.6% | 0.1239 m/s | Forward and lateral gates fail |

The no-push comparison above isolates the effect of perturbations; it does not
replace standard acceptance. The native success of StandUp and partial Walking
transfer do not close the separate CPU/BAM deployment gap. The gallery now
contains nine native/CPU clips, with contact sheets `recovery-transfer.jpg` and
`walking-transfer.jpg` beside it. Videos use seed 42; numerical multi-seed
results are retained in `summary.json` and the full result files.

The additional seed-43 campaign was launched from `1cb7738` on GPU 1. Its fresh
startup pipeline is in progress; full training must await successful completion
of every gate. The preview worker records checkpoints starting at update 1000
in `outputs/previews/mujoco-recovery-seed43-0909-01/index.html`. There is no new
seed-control behavior claim. Repaired SB3 learned convergence, seed sensitivity
and CPU/BAM transfer remain open; full-run diagnosis follows training completion.


## User-directed retirement of unpromising old runs

On 2026-09-09, four old Walking learners were intentionally stopped with
identity-checked SIGTERM after reviewing saved behavior evaluations. No reward
changes or new evaluation runs were introduced for this decision.

| Run | Recent completed forward responses / command | Last preserved checkpoint |
|---|---|---|
| Owned MuJoCo RSL, 8192 envs | 45%, 42%, 46%, 29% at updates 12000–15000; prior no-push diagnostic only ~1.7% | model_16000.pt |
| Old MuJoCo SB3, 8192 envs | 27%, 25%, 26%, 33%, 35% at updates 14000–18000 | update 18000 bundle |
| Old Newton SB3, 8192 envs | 14%, 17%, 12%, 27%, 26% at updates 6000–10000 | update 10000 bundle |
| Pinned original MuJoCo RSL, 4096 envs | 46%, 59%, 42%, 43%, 24% at updates 26000–30000; final evaluated yaw response also 24% | model_31000.pt |

The SB3 runs show some forward improvement, but remain below the 50% response
gate after repeated snapshots and have been superseded by the repaired recipe.
The original control also shows persistent lateral errors and recent behavior
regression. These are resource-retirement decisions, not claims that an entire
framework or the official task can never learn. The latest saved checkpoints
may be newer than the last completed behavior evaluations listed above.

All four learner exits were confirmed. Checkpoint hashes, process identity,
reviewed results and stop reasons are saved in `intentional-stop-20260909.json`
inside each run directory. The three owned runs are under
`outputs/experiments/fixed-8192-0908-01`; the original is under
`outputs/baselines/official-walking-full-0908-01`. Raw supervisor nonzero-exit
records remain intact and must be interpreted alongside these intentional-stop
records, rather than reported as unexplained crashes. No existing artifacts
were deleted. Pending preview/final-evaluation work can finish independently.

Six learners remain: Newton RSL Walking, four repaired SB3 combinations, and
MuJoCo RSL StandUp seed 43. All six processes were checked alive after stopping
the old runs. Newton RSL StandUp completed 15000 updates; its final native
four-pose preview passed, while final deployment acceptance remains open.
