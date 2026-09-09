# RL pause and debugging handoff — 2026-09-09

Training and background evaluation/preview workers are stopped at the user's
request. No automatic resume or new experiment was launched. All five remaining
learner exits and both live preview-worker exits were verified, and a subsequent
process scan found **zero project RL processes**. This is an operational pause
with retained checkpoints, not SIGSTOP retaining GPU memory.

This report supersedes earlier operational snapshots that describe active runs.
It does not replace the whole-project design or broaden the acceptance scope:
Flat Walking and Flat StandUp, both simulation backends, both native PPO frameworks.
No hardware or public policy publication is involved; W&B remains offline.

## Recoverable state

The last logged update can be newer than the last durable checkpoint. Resume
must use the recorded checkpoint, its native normalizer/optimizer state and its
immutable source snapshot; unsaved intervening updates are not recoverable.
Behavior results below are from the latest **completed saved-policy preview**,
not a new evaluation at the stop step. Interrupted previews are left intact and
must not be counted as completed assessments.

| Stopped learner | Last logged update | Durable checkpoint update | Latest completed preview |
|---|---:|---:|---|
| Newton / RSL / Walking | 22,990 | 22,000 | Forward 33.9%, yaw 107.7%; behavior gate failed |
| MuJoCo / SB3 / Walking | 12,278 | 12,000 | Forward 28.0%, yaw 97.2%; behavior gate failed |
| MuJoCo / SB3 / StandUp | 11,478 | 11,000 | 2/4 poses; prone/supine failed |
| Newton / SB3 / Walking | 4,563 | 4,000 | Forward 25.3%, yaw 78.9%; behavior gate failed |
| Newton / SB3 / StandUp | 6,054 | 6,000 | 2/4 poses; prone/supine failed |

All five used 8192 environments. Newton RSL source is `3070b71`; repaired SB3
source is `bea3eb1`. Checkpoint paths, SHA-256 hashes, process identities and raw
pre-stop stage records are in
`outputs/diagnostics/rl-pause-0909-01/stop-record.json`. Structured behavior and
completed-run results are in the adjacent `status-summary.json`. Existing raw
supervisor nonzero exits are preserved; this intentional stop is not an
unexplained training crash. Earlier four retired Walking runs and their
`intentional-stop-20260909.json` records are also retained.

## Verified outcomes and limits

- **Newton RSL StandUp:** completed 15000 updates. The final exported policy
  passed all four poses in native Newton, native MuJoCo and CPU/BAM rehearsal
  at evaluation seed 42. Export and local packaging also completed. This is a
  successful single-seed end-to-end result, not multi-seed deployment acceptance.
  The common final policy SHA-256 is
  `0333bb741bdd4536769c3052a3da456080a26c06401ba8a682d93bee2e9a51f1`.
- **Earlier Newton StandUp checkpoint 13000:** native 12/12 per backend, but CPU
  prone only 3/17. These results concern a different checkpoint and remain valid;
  the final policy's single-seed pass does not retroactively close that robustness
  gap. Repeated-seed verification of the final policy remains open.
- **MuJoCo RSL StandUp:** seed 42 final policy passed 3/4 at evaluation seed 42;
  seed 43 finished 15000 updates but passed only 2/4 in both native backends and
  CPU/BAM (prone/supine failed). One extra training seed did not solve recovery.
- **Repaired SB3:** independent official critics, randomized initial episode
  phases and KL feedback are implemented and tested. Latest logged approximate
  KL was about 0.01 in all four runs, but StandUp remained at 2/4 over repeated
  previews and Walking forward response stayed below the 50% gate. Update
  stability improved; effective task learning has not been demonstrated.
- **Newton RSL Walking:** previously learned forward/yaw responses, but standard
  previews at 20000/21000/22000 regressed to about 27%/23%/34% forward response.
  Lateral RMSE also fails. Earlier no-push success on individual response measures
  does not establish full walking acceptance. Best earlier checkpoints remain.

Final StandUp videos are under
`outputs/experiments/fixed-8192-0908-01/isaac-newton-rsl-rl-stand_up/`:
`sim2sim/mujoco`, `sim2sim/isaac-newton` and `rehearsal`, each with four pose clips.
Existing galleries remain viewable under `outputs/previews/`; they no longer
refresh because their workers are stopped. Numerical results were reviewed for
this handoff; no new video inspection or new rollout battery was performed.

## Next priority: diagnose ineffective learning

Proceed with investigation before allocating another long training budget.
The observations below motivate checks; they are not established root causes.

1. **Freeze a reproducible comparison.** Use the pinned official implementation,
   owned MuJoCo baseline, successful Newton StandUp final policy and failed
   policies with explicit source/checkpoint/normalizer hashes. Compare equivalent
   curriculum phases, control-step counts, reset cases and command schedules;
   total reward alone is insufficient. Reuse preserved evidence first.
2. **Separate policy learning from evaluation/export/physics differences.**
   Trace actor inputs, normalization, actions and joint/base trajectories from
   matched initial states. Check train-versus-play reward/sensor views, reset
   distributions, observation/action delay, BAM load/friction/voltage, contacts
   and solver timing. Rehearse the same exported bytes. Verify the successful
   final Newton StandUp policy across seeds before declaring CPU transfer robust.
3. **Explain the behavior plateaus.** Inspect reward components, weighted penalty
   signs, contacts, height/tilt clusters and pose-specific recovery trajectories.
   StandUp can gain height while retaining a large tilt; determine how the actual
   reward and termination logic score these cases. For Walking, inspect command
   tracking versus periodic sway and perturbations, and locate the regression
   relative to curriculum/DR stage transitions. Do not change acceptance limits
   to turn a failure into a pass.
4. **Audit framework learning differences.** Beyond the repaired critic/KL paths,
   compare native observation normalization, timeout bootstrapping, actor/critic
   value targets, advantage normalization, value clipping/loss, minibatch sizing
   and optimizer behavior. Preserve native SB3/RSL PPO and explicitly distinguish
   intentional framework semantics from adapter defects. Matching KL is not a
   learning-equivalence test.
5. **Only then run bounded causal checks.** Test one evidence-backed hypothesis
   at a time using the official smoke/audit/export/rehearsal gates. Do not change
   rewards, curriculum and simulator settings together. Keep failed cases and
   require behavior improvement against the frozen reference before proposing
   resumed full training. Default task/robot/BAM conventions stay official.

Immediate deliverable for the next development session: a prioritized causal
diagnosis with reproducible evidence and a minimal proposed correction. No new
training is authorized by this handoff itself; wait for the user's next work
instruction. The external Harness remains deferred.
