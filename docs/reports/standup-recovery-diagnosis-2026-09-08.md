# StandUp recovery diagnosis — 2026-09-08

The owned MuJoCo/RSL StandUp run is paused after regression in a matched reset
battery. This is a diagnosis pause, not a claim that the official recipe cannot
converge. Original MuJoCo Walking/StandUp and owned MuJoCo Walking continue on
GPU 7. The three previously suspended Newton learners remain separate.

## Paired checkpoint observations

The unchanged CPU/BAM battery was repeated with base reset seeds 100–115 at four
saved checkpoints. Each base seed receives the existing scenario-index offset.
These are **16 reset samples from one training seed**, not 16 independent training
runs. The corrected DOF-load controller, 50 Hz, nominal mechanics, commands,
height/tilt thresholds and one-second final hold were unchanged.

| Checkpoint | Standing | Sitting | Face-down | Face-up |
|---|---:|---:|---:|---:|
| 2000 | 16/16 | 13/16 | 0/16 | 0/16 |
| 2250 | 16/16 | 13/16 | 0/16 | 0/16 |
| 2500 | 16/16 | 10/16 | 0/16 | 0/16 |
| 2750 | 16/16 | 8/16 | 0/16 | 0/16 |

All 256 eight-second trajectories completed with finite state. Sitting outcomes
form two distinct final-state groups (upright around 0.116 m, or down around
0.06 m); the decline is not a tiny threshold crossing. Prone starts remain down.
The earlier single-seed sitting passes therefore cannot establish reliable rise.
Evidence: `outputs/baselines/standup-seed-comparison-0908-01/result.json` and its
per-checkpoint traces; checkpoint 2750 is in `standup-seed-battery-0908-01`.
These diagnostics omit video and do not replace eventual native/CPU video and
multiple-training-seed acceptance.

Following the user's instruction to pause unpromising training for diagnosis,
process group 1051252 was SIGSTOP-suspended at logged iteration 2985. Its latest
complete checkpoint is 2750, SHA `effa3481ac8aa81543638af0a54d21fe3b7fc0e7b6ed0df4dd053a3ec0375e87`.
All three verified group members report stopped state. The saved process identity,
commands and resumption instructions are in the run's `operator-pause.json`.
Do not start a replacement while this group remains live.

## Resume and curriculum checks

Native mjlab RSL checkpoints retain `infos.env_state.common_step_counter`; the
current Walking 2500 / StandUp 2750 saves contain 60048 / 66048 environment steps.
Logged curriculum values continue through the correct stages: StandUp standing
spawn probability reaches 0.15 and body-pose tracking weight reaches 1.5. There
is no evidence of a reset-to-zero curriculum in these runs.

Native resume repeats the saved iteration label when learning continues. After
this one resume, counters are `(iteration + 2) × 24`, versus `(iteration + 1) × 24`
without resume. Equal checkpoint labels consequently differ by one rollout from
the uninterrupted original control. Native resume also does not restore the full
physics/RNG trajectory. Preserve those limits when interpreting later matched
milestones; the initial checkpoint-250 comparison predates the owned resume.
Evidence: `standup-reward-audit-0908-01/resume-curriculum-audit.json`.

## Reward measurement and bounded hypothesis

The recipe enables body-pose tracking at iteration 2500, alongside a harder spawn
mix, then increases its weight and adds further penalties at 3000. On the recorded
last-second states at checkpoint 2750, the actual owned/official-derived
`body_pose_tracking_locomotion` function produces these unweighted means under the
recorded zero body command:

| State origin | Reward (maximum 1) | Reason |
|---|---:|---|
| Standing | 0.989 | Height and both angles track |
| Face-down | 0.636 | Height and roll still track; pitch fails |
| Face-up | 0.331 | Roll still tracks; height and pitch fail |

Independent per-axis reconstruction agrees numerically with the actual function.
The reward is a mean of tracked axes, so a failed pose can retain substantial
reward. This contradicts the recipe comment that all tracked axes score near zero
while prone. It does **not** establish that this term caused regression: recovery
was already absent at 2000, before this term's weight becomes nonzero, and the
measurements are counterfactual training rewards on CPU deployment states.
Evidence: `outputs/baselines/standup-reward-audit-0908-01/result.json` and arrays.
No task reward implementation was changed.

## Official-guided diagnostic fork

The pinned official StandUp recipe explicitly advises first halving the
`body_ang_vel` penalty from −0.05 to −0.025 if recovery freezes, then considering a
softer action-rate ramp. The separate experiment
`outputs/baselines/standup-angular-penalty-0908-01` follows **only the first change**.
It uses the owned official-derived task, native PPO, seed 42 and the preserved
checkpoint 2000. Default original controls remain unchanged.

Before comparison training, it runs 64 environments / five iterations, official
normalized export and finite/penalty audits, corrected CPU/BAM video, then a
4096-environment / five-iteration capacity/export gate. It then performs 501 native
updates so its final saved label is 2500, matching the existing resumed default
branch's label and environment-step count. All recipe curricula, other rewards,
actuator parameters and task thresholds remain intact. Each stage discards its
smoke/capacity weights and starts from the same preserved checkpoint.

The smoke, normalized export, corrected CPU video and 4096-environment capacity/
export gates completed. Comparison training is now running; no improvement is claimed. Its exact script
hash, source snapshot `0792f99`, checkpoint hash, commands and stage outcomes are
recorded in its manifest. A matched 16-reset battery and videos are required after
training before choosing whether this tuning is useful. CPU examples alone do not
prove native-backend or Newton/SB3 success.


The saved default and diagnostic environment YAMLs were compared without
instantiating YAML objects. Across rewards, curricula, commands, observations and
events, the only difference is `rewards.body_ang_vel.weight: -0.05 → -0.025`.
The live comparison worker's environment confirms GPU 7 and W&B offline. Evidence:
`standup-angular-penalty-0908-01/recipe-difference.json`.
