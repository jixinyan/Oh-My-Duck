# Walking curriculum pacing experiment — 2026-09-13

## Completed job and corrected conclusions

`omd-rl-causal-0913-01` **Succeeded**, platform ID
`c33528a5-a6ca-4fba-8484-5c5a0f96eba6`, immutable source `e804d6c`.
Scheduler display: 10:28:16–10:44:21 on 2026-09-13, one node / four H800 GPUs.
All 32 paired frozen-policy cases executed. Both representative Newton RSL tasks
completed 64-env/5-update smoke, scalar/penalty audit, official export, CPU/BAM
rehearsal/video, native resume and 8192-env/5-update capacity/export. All four
actual periodic callback ONNX files match their independent official-route
exports on 65 inputs with maximum absolute error **0**. The prior metadata defect
is now GPU-validated. Unlearned smoke CPU behavior failures remain expected;
these gates do not demonstrate learned task performance.

Matched standard-profile Walking command response (same policy/backend/seed;
only push amplitude changes):

| Policy / evaluation backend | Forward with / without pushes | Yaw with / without pushes |
|---|---:|---:|
| Repaired MuJoCo SB3 / MuJoCo | 28% / 1% | 97% / 4% |
| Repaired MuJoCo SB3 / Newton | 30% / 1% | 93% / 6% |
| Repaired Newton SB3 / MuJoCo | 24% / approximately 0% | 23% / 6% |
| Repaired Newton SB3 / Newton | 25% / 1% | 79% / 7% |
| Newton RSL 22000 / MuJoCo | 54% / 53% | 98% / 104% |
| Newton RSL 22000 / Newton | 34% / 51% | 108% / 103% |

SB3's apparent motion is strongly confounded by perturbations in these seeded
batteries. Removing them leaves little self-driven command response. This is a
causal replay result about the fixed policies, not proof of a learning root cause.
Newton RSL shows self-driven forward/turn response in **both** backends under
standard/no-push play conditions. Its MuJoCo forward failure in the final-training
profile therefore cannot be labeled a universal simulator-transfer failure:
other training/play configuration differences still matter. Existing lateral
instantaneous-RMSE acceptance remains unmet; no threshold has been relaxed.

Ten 1280×720 recordings and their metrics are retained in
`outputs/diagnostics/push-causal-0913-01/index.html`. Extracted frames were inspected
alongside trajectory metrics: failed SB3 StandUp remains visibly tilted; Walking
must be judged using motion over time, not a still frame alone. Final Newton
StandUp's earlier CPU result remains 14/17 full four-pose batteries, with prone
robustness unresolved. StandUp is not retrained in the pacing experiment below.

## One-factor full training experiment

The next learning intervention tests whether strengthening action smoothness
before skill discovery helps trap Walking in the stationary solution. This is
motivated by the observed stationary policies and the pinned official guidance
on delaying smoothness until discovery, **not an established diagnosis**. It is
kept separate from the official default recipe and from the deferred StandUp
learning-rate experiment.

`configs/experiments/walking-smoothing-delay.json` contains eight independent runs:
MuJoCo/Newton × native RSL-RL/SB3 × original/delayed action-rate curriculum.
Every pair uses seed 42, **8192 environments and the full official 50000-update
Walking budget** (24 steps/environment/update). Fresh paired controls are causal
comparisons, not blind continuations of the failed historical checkpoints.
Only `action_rate_weight` stage times change:

| Weight | Official update threshold | Experimental update threshold |
|---|---:|---:|
| -0.1 | 0 | 0 |
| -0.2 | 500 | 5000 |
| -0.4 | 750 | 5250 |
| -0.6 | 1000 | 5500 |
| -0.8 | 1250 | 5750 |
| -1.0 | 1500 | 6000 |

Thresholds preserve the official strict `common_step_counter > step` rule.
Initial and final weights, other rewards/curricula, commands, DR, BAM, physics,
61 actor observations, 14 servos, HOME and 50 Hz are unchanged. RSL keeps its
native PPO; SB3 keeps the previously repaired native configuration (official
critic, randomized initial episode phase, initial LR 1e-4 with its explicit
previous-rollout KL feedback). The SB3 control is the **official task recipe
with the documented SB3 adapter**, not an official upstream SB3 implementation.

The intervention is recorded in run metadata, SB3 periodic bundles and RSL
runner configuration. Resume cannot silently change it. The owned official
factories remain untouched; `--action-rate-delay-iterations 0` is an exact recipe
no-op. Nonzero delay is explicitly bound only to Flat Walking.

## One submission, complete workflow

One node / eight GPUs, one independent learner per GPU, W&B **offline**, headless:

1. Required 64-env/5-update smoke, sign/finite/export audit, CPU/BAM video,
   native resume and 8192-env capacity gates.
2. Automatically proceed to each 50000-update full learner once its gates pass.
3. Save native checkpoint bundles every 1000 updates. A lightweight background
   follower on that learner's allocated GPU saves checkpoint videos; no repeated
   assistant monitoring or intermediate multi-seed diagnostic batteries.
4. Export and local schema-2 package, standard two-backend replay, CPU/BAM video,
   and explicit **no-push evaluations in both backends**, all with final videos.
   Standard acceptance is retained; no-push replay additionally prevents apparent
   motion from external perturbations passing this experiment's completion gate.

No `--prepare-only`, new environment sweep, public upload, local-host GPU fallback
or arbitrary wall-time cap. Failed startup gates stop their affected learner;
other independent runs can continue. The new training intervention and resume
metadata change training-critical code, so prior preparation cannot be silently
reused for this new source/configuration. Gates run inside the full job.

```bash
python omd.py submit --name omd-walk-pacing-0913-01 --gpus 8 -- \
  .envs/mujoco-sb3/bin/python omd.py campaign \
  --config configs/experiments/walking-smoothing-delay.json \
  --output outputs/experiments/walking-delay-0913-01
```

A positive result requires learned response/trajectory/video improvement over its
paired control and transfer evidence, not a larger total reward alone. Compare
reward values within equivalent curriculum stages; after update 6000 both groups
have the same objective. One training seed per pair does not establish robustness.
StandUp, complete two-task acceptance and native Newton DDP remain open.
