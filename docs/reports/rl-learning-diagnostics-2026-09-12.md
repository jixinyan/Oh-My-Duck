# RL learning diagnosis — 2026-09-12

**Completed 2026-09-13:** all 49 cases executed; behavior remains partial. See
[results and causal follow-up](rl-causal-replay-2026-09-13.md). Queue statements
below are the September 12 historical record.

User instruction resumed iteration toward both representative tasks across both
simulation backends and both native PPO frameworks. New GPU work uses scheduled
single-node, multiple-GPU jobs; the development host's GPUs are occupied by other
work. Existing paused learners remain stopped. Task/reward/robot/PPO recipes have
not been changed without evidence, and no new learner has been launched.

## Scheduled frozen-policy matrix

`configs/experiments/learning-diagnostics.json` pins eight policy exports by
SHA-256: successful final Newton StandUp, failed MuJoCo seed-43 StandUp, both
repaired SB3 StandUp policies, early/late Newton Walking and both repaired SB3
Walking policies. The job has 49 cases:

- 32 native evaluations: eight policies × two backends × standard play versus
  frozen final training configuration with zero push amplitude. Standard cases
  retain existing acceptance. The training-profile intervention changes several
  configuration conditions and is a localization diagnostic, not proof of any
  individual cause. A later causal ablation must isolate one difference.
- 17 CPU/BAM evaluations of final Newton StandUp, at seed 42 and 100–115, to
  distinguish its established single-seed success from multi-seed robustness.

Evaluation now optionally records weighted reward rates, transition rewards,
joint position/velocity and base pose, alongside observations/actions and existing
behavior metrics. Frozen official curricula are applied through live managers
before forced protocol resets; they are then held fixed, avoiding curriculum
replacement of the selected pose. Diagnostic profiles and scaled pushes cannot
be mistaken for standard acceptance (`acceptance_eligible=false` and a diagnostic
status prefix). No acceptance thresholds were relaxed.

Submission attempts, both preserved:

| Attempt | Result |
|---|---|
| `omd-rl-diagnose-0912-01` | Submission failed: scheduler shared-lock timeout; no GPU execution |
| `omd-rl-diagnose-0912-02` | Accepted, queue `local-4a0ab8eebdb0`; waiting for project quota |

Attempt 02 requests one node with four GPUs and uses immutable source `4ffb970`.
Each allocated GPU runs independent cases through the backend's locked Python
environment. There are no PPO learners in this diagnostic job, no local GPU
fallback, no public upload and no automatic retry of failed cases. Output target:
`outputs/diagnostics/learning-0912-02`. Submission/log evidence:
`outputs/jobs/omd-rl-diagnose-0912-02/`.

## Confirmed export defect and repair

The completed Newton RSL training log repeatedly contained
`ONNX export failed ... index 0 is out of bounds for axis 0 with size 0`.
Native checkpoints survived and separate CPU official-route export succeeded,
but the periodic export's metadata stage was defective.

The official `get_base_metadata` obtains actuator IDs from the canonical robot
spec, then indexes `env.sim.mj_model.actuator_gainprm/biasprm`. Newton's solver
representation applies BAM through DOF forces and has no native actuator rows.
Its canonical MJCF reference still has the 14 actuator metadata rows. A CPU
regression reproduces the exact indexing error with the real compiled Microduck
spec and empty Newton-style actuator arrays.

The owned runner now keeps native Mjlab checkpoint persistence and the inherited
normalizer-aware ONNX exporter, then calls the official metadata function with a
read-only view of the matching canonical MJCF. It declares that the metadata is
nominal MJCF/BAM metadata and never swaps or mutates the running physics model.
The MuJoCo callback remains unchanged. Offline/no-upload behavior is retained.

Tests cover original metadata equivalence, the old Newton failure, 14 joint and
actuator entries, physical-model immutability, native iteration/curriculum/
optimizer checkpoint persistence, actual native RSL MLP normalization and ONNX
numerical parity. The save callback leaves actor weights/normalizer unchanged.
This fixes an export-flow defect; it is **not identified as the cause of poor
learning**. GPU integration of the changed callback remains to be exercised
before new full Newton training. The queued frozen-policy job needs no callback
change and keeps its earlier source snapshot.

## Historical rewards and remaining hypotheses

The preserved last training reward rates show failed policies can collect much
of the height/pose stack. For example, MuJoCo SB3 StandUp has leg-pose rate 1.90
and height rate 0.939, near successful Newton RSL's 1.899/0.966, while its standing
composite is 0.862 versus 1.214 and the saved recovery behavior fails both fallen
poses. These are different policy/distribution averages, not matched causal
measurements. They motivate per-pose reward/state traces; they do not justify a
blind reward change. Raw extracted rates are retained in
`outputs/diagnostics/learning-cpu-review-0912-01/historical-final-reward-rates.json`.

Still open: reward/local-optimum versus exploration/curriculum effects; native
framework normalization/value-target/loss differences; training versus evaluation
conditions; Walking regression; final StandUp CPU robustness. Effective learning
across all eight representative combinations has **not** been demonstrated.

## Validation and next step

Ten profile/protocol tests and three native export-metadata/callback tests pass
on CPU. Plan dry-run validates all policy hashes. CLI `diagnose` provides a
modular entry point; `tasks --all` and `docs/rl-task-catalog.md` document 33
registered entries in 13 families. Only Flat Walking/StandUp currently bind
Newton and full behavior/package profiles; additional tasks need their own gates.

Next, let the queued diagnostic execute when project quota permits, review
weighted per-pose failures and transfer robustness, isolate a causal difference,
then validate a minimal correction before allocating full training. A queued job
is not a completed diagnostic or a successful training run.
