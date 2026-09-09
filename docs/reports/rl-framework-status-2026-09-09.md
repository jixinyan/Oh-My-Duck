# Representative RL behavior review — 2026-09-09

User-requested snapshot at approximately 02:27 UTC; intervention verified at 02:29 UTC.
Campaign `fixed-8192-0908-01`, immutable training source `3070b71`, preview source
`05c90ce`. All eight started with 8192 environments, 24-step rollouts, seed 42,
headless execution and offline W&B. No rewards or live learner settings changed.
This is a checkpoint review, not a completed eight-combination acceptance.

## Progress and saved behavior

Progress is native rollout updates; SB3 transitions are divided by 8192 × 24.
RSL final index 14999 represents completion of its 15000-update budget.
Evaluation snapshots can lag the live learner by up to a checkpoint interval.

| Backend / PPO | Task | Live update at snapshot / budget | Evaluated checkpoint | Observed behavior |
|---|---|---:|---:|---|
| MuJoCo / RSL-RL | Walking | 13829 / 50000, running | 13000 | Forward response 41.9%; full battery fails |
| MuJoCo / RSL-RL | StandUp | 15000 / 15000, completed | final | 3/4 poses; supine fails in native, Newton and CPU/BAM |
| MuJoCo / SB3 | Walking | 16547 / 50000, running | 16000 | Forward response 26.3%; full battery fails |
| MuJoCo / SB3 | StandUp | 15000 / 15000, completed | final | 0/4 in native, Newton and CPU/BAM |
| Newton / RSL-RL | Walking | 14278 / 50000, running | 14000 | Forward 66.2%, yaw 103.4%; lateral RMSE fails |
| Newton / RSL-RL | StandUp | 13672 / 15000, running | 13000 | 4/4 native poses; repeated success at earlier milestones |
| Newton / SB3 | Walking | 9634 / 50000, running | 9000 | Forward response 27.3%; full battery fails |
| Newton / SB3 | StandUp | 10209 / 15000 at snapshot; stopped shortly afterward | 10000 | 0/4 across all ten periodic checkpoints |

StandUp tests use standing, sitting, prone and supine resets, seed 42, eight
seconds per pose and a one-second final hold. Newton RSL passes the selected
6000, 9000, 10000, 12000 and 13000 checkpoints. Extracted video frames at
0, 0.5, 1, 2 and 6 seconds confirm actual prone/supine recovery by about one
second and an upright held posture afterward. SB3 comparison clips instead
settle into a forward lean or remain on the back. This supports a learned
Newton-native recovery skill; multi-seed robustness and this checkpoint's
sim2sim/CPU acceptance are still pending final evaluation.

Newton RSL Walking 14000 records 0.0662 m/s for a 0.1 m/s command and
0.517 rad/s for a 0.5 rad/s yaw command. Its lateral RMSE is 0.1263 m/s against
the project's 0.1 m/s threshold; the other recorded scalar gates pass. Frames
show changing leg poses and body heading. These are standard native play
rollouts, including the recipe's retained perturbations, not a new nominal,
no-push tracking test. Do not equate their response fractions with clean
deployment tracking or relax the gate to declare a pass.

The original, unrefactored 4096-env StandUp control also finished: its native
final preview passes standing/sitting, fails prone/supine (2/4). This excludes
an explanation based solely on switching frameworks. Original versus owned
runs have different batch sizes and training histories; this is not a matched
causal comparison. The original Walking control continues separately on GPU 7.

## Confirmed framework differences

Both learners use the migrated official environment/reward factories and
512/256/128 ELU actor/critic widths. Both actors retain the deployment 61D input
and 14 actions. Equal task definitions do not make the learner recipes equal.

| Aspect | RSL-RL official recipe | Current SB3 mapping |
|---|---|---|
| Critic input | 74D separate critic group | Same 61D input as actor |
| Critic-only information | Linear velocity, foot air time/contact/forces; separate sensor view | Not consumed |
| Learning rate | Native KL-adaptive, starting at 1e-3 | Explicit constant 1e-4 and native KL early stop |
| Value loss | Clipped loss enabled | Default `clip_range_vf=None` |
| Advantage normalization | Whole rollout by default | Per minibatch |
| Observation normalization | Native actor and critic normalizers | Shared actor-view VecNormalize, clip 100, reward normalization off |
| Initialization | Native RSL model initialization | Native SB3 orthogonal initialization |

Evidence: `src/oh_my_duck/rl/{models/sb3.py,learners/sb3/train.py,learners/sb3/environment.py}`,
actual RSL `params/agent.yaml` and printed 61/74 input-layer shapes, and installed
native PPO sources. SB3's configurable value clipping and KL limit are also
specified in its [official PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).
Clipping implementations need not be numerically identical even when enabled.

At matched update 9000, RSL StandUp has 3/4 native successes in MuJoCo and 4/4
in Newton; both SB3 variants have 0/4. Unequal wall-clock progress therefore
cannot explain the entire difference. The critic mismatch is a concrete gap in
our adaptation, not evidence that SB3 inherently cannot learn Microduck.
Its causal contribution has not yet been isolated from the other differences.

Recent 100 logged Newton SB3 StandUp KL values average 0.04114 against a 0.01
target; one snapshot reaches 0.14346. Logs repeatedly show native PPO early
stopping, including within the first epoch. Reducing the old learning rate to
1e-4 did not remove this instability. The policy's exploration standard
deviation is about 0.136. MuJoCo SB3 StandUp finishes with KL averaging 0.01145
but still 0/4 behavior, so KL alone is not a sufficient explanation. Its high
explained variance (~0.996) describes prediction of its current returns, not
successful recovery.

Both SB3 Walking runs currently have recent mean KL near 0.0094 and some
forward-response improvement; do not generalize the StandUp instability to all
SB3 runs. Total rewards can be similar despite different motion quality:
Newton Walking recent mean rewards are roughly 126.5 (RSL) and 127.6 (SB3),
while the saved forward response differs considerably. Task behavior remains
the acceptance criterion. Export parity and normalization checks passed, which
argues against a missing export normalizer as the immediate explanation, but
does not certify every adapter detail.

## Action and remaining diagnosis

Under the user's standing instruction to stop unpromising training, only the
Newton SB3 StandUp learner was terminated with SIGTERM at 02:28:41 UTC, after
checking its PID/start time/command and the complete update-10000 bundle.
All three checkpoint/normalizer/metadata hashes were rechecked unchanged.
The campaign records the worker exit as failed (full stage exit 241); this is
an intentional diagnostic stop, not an unexplained simulator crash or a retry.
Five new learners continue; the two completed MuJoCo StandUp runs are retained.
GPUs 1 and 3 are free in the verified resource snapshot. GPU 7 still hosts the
original Walking control. No new long experiment was launched.

Next diagnosis should preserve native PPO: first expose the official separate
critic observations through an SB3-compatible policy/buffer interface while
keeping actor/export 61D, then use controlled comparisons for normalization,
value loss and update magnitude. Test timeout bootstrapping and export parity
on that interface before launching another full run. Do not simultaneously
change rewards, curricula and optimizer settings or call a smoke a learned
policy. Keep the promising Newton RSL run to its existing final budget and let
its existing final export, sim2sim and CPU/BAM pipeline execute.

Evidence bundle: `outputs/diagnostics/framework-status-0909-01/` contains the
runtime snapshot, checkpoint history, TensorBoard scalar summaries, inspected
video contact sheets and `stop-newton-sb3-standup.json`.
Videos: `outputs/previews/fixed-8192-0908-01/index.html`.

Version-control handoff: the accumulated RL operations branch is integrated
into `main` at `6902943` before a fresh `fix/sb3-official-critic` branch. Merge checks passed:
26 tests and 9 subtests covering campaign options, preparation/recovery, GPU
staging, previews, scaling, CPU BAM, evaluation scoring and SB3 checkpointing.
No new SB3 critic implementation is included in this status review.
