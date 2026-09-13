# Frozen-policy diagnosis and causal follow-up — 2026-09-13

Job `omd-rl-diagnose-0912-02` succeeded on one node / four H800 GPUs. Platform
ID: `d0c14f4b-03f9-45cd-83ed-52d7559cd505`; immutable source: `4ffb970`.
All **49 cases executed**, with no execution errors. This does not mean 49
behavior passes. Raw evidence: `outputs/diagnostics/learning-0912-02/result.json`.

## Verified behavior

| Frozen policy | MuJoCo standard / training-stage | Newton standard / training-stage |
|---|---|---|
| Newton RSL final StandUp, iteration 14999 | 4/4 / 4/4 poses | 4/4 / 4/4 poses |
| MuJoCo RSL StandUp seed 43 | 2/4 / 2/4 poses | 2/4 / 2/4 poses |
| Repaired MuJoCo SB3 StandUp | 2/4 / 2/4 poses | 2/4 / 2/4 poses |
| Repaired Newton SB3 StandUp | 2/4 / 2/4 poses | 2/4 / 2/4 poses |
| Both repaired SB3 Walking, Newton RSL Walking 14000 and 22000 | All below acceptance | All below acceptance |

Native tests use seed 42. The training-stage profile freezes the official
curriculum at 360000 control steps, uses the training factory and zero push
amplitude. Multiple conditions change together; these cells are diagnostic,
not standard acceptance and not a single-factor causal attribution.

Final Newton RSL StandUp CPU/BAM rehearsal at seeds 42 and 100–115:
**standing 17/17, sitting 17/17, face_up 17/17, face_down 14/17**. Full four-pose
batteries pass at 14/17 seeds (65/68 individual poses). Prone failures are seeds
101, 105 and 112. This improves on the older iteration-13000 checkpoint's 3/17
prone result; different checkpoints must remain distinct. Hardware and broad
robustness acceptance remain unavailable/unresolved respectively.

## What the traces localize

Failed SB3 StandUp policies settle around 0.59 rad (34 degrees) trunk tilt on
both fallen starts. In matched MuJoCo final-stage zero-push replay, MuJoCo SB3's
prone terminal weighted reward rate is **7.915**, versus **11.220** for successful
Newton RSL. Height and leg-pose terms remain nearly saturated, while upright,
composite and body-tracking terms decline. This is a measurable suboptimal stable
pose retaining about 70.5% of successful reward, not a sign-error finding or proof
that any one regularizer caused failed exploration. Successful Newton policy
transfer shows that MuJoCo can execute the recovered behavior.

Walking needs more precise diagnosis than a single pass/fail flag:

- Newton RSL iteration 22000, native Newton training-stage/no pushes: forward
  segment mean vx **0.0583 m/s** for command 0.1, and yaw **0.5017 rad/s** for
  command 0.5. It has partial gait/turn response. Forward lateral instantaneous
  RMSE is **0.165 m/s**, but one-second-window mean RMSE is **0.012 m/s**:
  much of the lateral metric is oscillation, not sustained sideways drift.
- The same policy on MuJoCo under that profile barely walks forward
  (**0.00084 m/s**, approximately 1% response), although turn response remains
  near 99%. Walking transfer is consequently still unresolved even if lateral
  oscillation is interpreted differently. Existing acceptance is unchanged.
- Repaired SB3 Walking in either backend has approximately 1% forward and
  5–8% yaw response in training-stage/no-push conditions. Standard play sometimes
  produces much larger apparent response. Pushes are a plausible confound;
  changing play/training conditions simultaneously does not isolate them.

Retrospective segment diagnostics:
`outputs/diagnostics/learning-review-0913-01/motion-summary.json`.
New reward-recording Walking evaluations include segment means, instantaneous
RMSE, standard deviation and one-second-window mean RMSE. Windows do not cross
command changes; these fields do not alter success thresholds.

## Next scheduled iteration

`configs/experiments/push-causal-diagnostics.json` completes the other 32 cells
of the 2×2 factory/profile × push-amplitude matrix: standard/no pushes and
training-stage/normal pushes. Compare only the same frozen policy, backend,
seed and profile to isolate amplitude. Eight policies and both simulators remain
fixed. Four selected MuJoCo cases save 1280×720 videos of good/bad StandUp and
moving/stationary Walking. Original artifacts remain untouched.

The same single-node, four-GPU job then runs
`configs/experiments/newton-export-gates.json` with **`--prepare-only`**:
for both representative Newton RSL tasks, 64-env/5-update smoke, finite scalar
and penalty-sign audit, official export, CPU/BAM rehearsal/video, native resume,
and 8192-env/5-update capacity/export. These are short validation learners, not
new full training or resumed historical runs. No environment sweep is performed.

The export gate now requires the **actual periodic callback ONNX** and compares
65 deterministic inputs against a separate official `run_export` result. It
checks canonical servo metadata and explicit Newton/BAM metadata, and fails on
missing, stale or numerically different artifacts. A separate successful export
alone can no longer mask a broken Newton periodic callback in campaign gates.

Official task rewards, curricula, physics, actor layout and native PPO algorithms
remain unchanged. Reward/curriculum interventions will be separate experiments
once single-factor evidence supports them. W&B stays offline; no local-host GPU
fallback and no automatic failed-case retry. Submission status is recorded below
once accepted; a prepared plan is not execution evidence.

## Submission and validation record

`omd-rl-causal-0913-01` was accepted, queue ID `local-de066dd84a81`, initially
waiting for the dispatcher. Source snapshot: `e804d6ce9570e925d4f1d5e9a8b2f73ba3569eb3`.
One node / four GPUs, with 32 diagnostic cases followed by two preparation-only
pipelines. Outputs: `outputs/diagnostics/push-causal-0913-01` and
`outputs/diagnostics/newton-export-gates-0913-01`. Scheduler evidence is retained
in `outputs/jobs/omd-rl-causal-0913-01/`. GPU validation is pending execution;
no full training has restarted.

Across targeted CPU suites, 24 tests and nine subtests pass: motion/protocol/
frozen-curriculum checks, actual native normalized ONNX callback checks including
rejection of stale weights, and campaign/options validation. Both plans pass
CLI dry-run and all frozen policy hashes are verified. No acceptance threshold
or official recipe was changed.

Startup verification: scheduler now reports **Running**, platform ID
`c33528a5-a6ca-4fba-8484-5c5a0f96eba6`, one node / four H800 GPUs.
Displayed start time: `2026-09-13 10:28:16` (scheduler timezone).
Results remain pending; no ongoing assistant polling is scheduled.


## User-requested status check and scheduling correction

At this check, `omd-rl-causal-0913-01` is Running. All 32 paired replays
completed without execution errors. Walking preparation is complete; StandUp
has completed smoke, export, CPU rehearsal and native resume and is running
8192-env capacity validation. Both actual Newton smoke callback exports match
separate official-route exports on 65 inputs with maximum absolute error 0;
Walking's capacity callback also passes. CPU smoke rehearsal exit 2 is expected
unlearned behavior, not task acceptance. No full learner is running in this job.

The user requested complete training submissions instead of separate short
validation jobs. Future routine jobs combine required gates, automatic full
training after passing gates, and final evaluation/video. Existing verified gates
are reused when configuration-matched. This immutable, already-running
preparation-only job remains accurately labeled and does not become a full run
by changing the working tree. Failed gates still stop their affected run.
