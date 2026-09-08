# RL triage, GPU consolidation and checkpoint previews — 2026-09-08

The user requested stopping unpromising runs, preserving improving runs, checking
actual policy videos and reserving the other GPUs for separate projects.

## Operations and recovery

- Stopped MuJoCo SB3 Walking and StandUp after sustained return regression and
  elevated native PPO KL. Their last retained checkpoints are at rollout 1750
  (`step_000172032000`) and 2250 (`step_000221184000`). Stop reasons and signals
  are recorded in each original run's `operator-stop.json`.
- Briefly stopped the six retained runs and resumed them together on **GPU 7**.
  Four RSL-RL runs and two Newton SB3 runs retain the same native algorithms,
  task recipes, curriculum and training budgets. Other GPU ordinals are not used
  by this campaign, its diagnostics or previews.
- Recovery source: `b0fa5fe`; immutable snapshot under `.job-sources/`.
  Original source/outputs remain preserved. The new manifest is
  `outputs/experiments/shared-gpu7-0908-01/campaign.json`; launch and recovery plan
  are in `outputs/experiments/consolidation-0908-01/`.
- Native checkpoints, SB3 normalizers, identity, hashes, completed progress and
  original smoke/export/rehearsal/resume/capacity gates are checked before
  continuation. Only the remaining budget is requested. At most one 250-iteration
  save interval of unsaved work is discarded; fresh simulator episodes are used.
- Resume progress: MuJoCo RSL Walking 1500, StandUp 2000; Newton RSL Walking and
  StandUp 1000 each; Newton SB3 Walking 500, StandUp 1250. All six produced new
  PPO updates after loading these checkpoints. SB3's printed `iterations`
  restarts for the new segment; native total timesteps retain cumulative progress.
- GPU 7 initially used about 38 GB with six learners and reached 100% compute
  utilization. Sharing fits in memory but reduces per-run throughput. This is a
  user-requested resource tradeoff, not a training-efficiency improvement claim.

The original campaign aggregate reports execution failure because its processes
received operator SIGTERM; accompanying operator-stop records explain this
intentional interruption. It is not a newly observed simulator crash.

## Native SB3 update diagnosis

The official RSL recipe starts at `1e-3` and uses its native adaptive-KL schedule.
Our SB3 mapping retained the initial rate as a constant and uses native target-KL
early stopping. These are materially different update controls. The official
curriculum also increases action-rate penalties, command/CoM ranges and difficult
StandUp spawn proportions; total reward changes across stages are not a controlled
behavior comparison. Task rewards and curricula were not changed in this work.

Controlled test: reload the same stopped MuJoCo SB3 StandUp checkpoint, same seed,
4096 environments and restored curriculum; run five native PPO rollouts, changing
only the learning rate. Probe the same stored rollout observations/actions before
and immediately after native `train()`, sampling 8192 transitions per rollout.
All pre-update KL means were below `5.1e-8`, providing no evidence of a log-probability
mismatch at this boundary. Post-update results:

| Learning rate | Mean post-update KL | Range across five rollouts | Actual optimizer steps |
|---|---:|---|---:|
| 0.001 | 1.67674 | 1.06031–3.02757 | 5 |
| 0.0003 | 0.14564 | 0.05608–0.25439 | 5 |
| 0.0001 | 0.03645 | 0.01648–0.05576 | 15 |

The configured target KL is 0.01. Reducing the rate materially reduces update
overshoot, supporting excessive fixed-rate updates as a contributing cause.
Even `1e-4` still exceeds the target on some rollouts; five updates do not establish
stable convergence or prove the only cause. The stopped full runs remain stopped.
The six retained runs keep their existing rates. An explicit `--learning-rate`
option now permits controlled native SB3 fresh/resume experiments and records the
override; no new adaptive learner or silent change of defaults was introduced.

Evidence: `outputs/diagnostics/sb3-kl-0908-02/` contains scripts, fixed source,
checkpoint hashes, logs and measurements. The earlier `sb3-kl-0908-01` probe is
preserved with `measurement-note.json`: its first four post-update samples were
invalid because SB3 had already reset the buffer. Those measurements are excluded;
the repeated probe samples immediately after native optimization.

## Video previews

`omd preview` renders saved checkpoints through the official normalized export
path and the existing task evaluation, serially on GPU 7. The active watcher uses
source `25fd4df`. It polls for new completed checkpoint saves every 60 seconds;
it does not render training cameras live. Source checkpoint hashes, render logs,
metrics and videos remain distinct for every attempt.

- Continuing runs: `outputs/previews/shared-gpu7-0908-01/index.html`.
- Stopped-policy diagnosis: `outputs/previews/stopped-sb3-0908-01/index.html`.
- Native Newton rendering and explicit MuJoCo OSMesa produce 1280×720 H.264
  videos at 25 FPS. Preview behavior failure is distinct from rendering failure.

Initial RSL previews:

- MuJoCo Walking ~1500: fixed command sequence ended after 10.82 s with instability;
  twist RMSE `[0.1364, 0.1394, 0.7155]`, failing behavior thresholds.
- Newton Walking ~1000: completed all 14 s, with max tilt 0.164 rad and min height
  0.115 m; lateral RMSE 0.145 m/s exceeds 0.1, so behavior still fails.
- Both RSL StandUp previews pass the standing and sitting starts and fail prone
  and supine recovery. These are single-seed checkpoint observations, not success
  rates or final skill acceptance.

Retained training continues while the gallery receives later checkpoints.

## Verification

- All 34 lightweight tests passed, including eleven RL entry/isolation/recovery/preview tests. Recovery
  rejects inconsistent progress, changed bundles and missing prior capacity gates;
  previews ignore partial/incomplete SB3 checkpoint directories.
- Ten SB3 boundary/export/checkpoint tests passed. The native periodic-checkpoint
  test additionally exercises a learning-rate override on a real resumed PPO update.
- New explicit `1e-4` CLI path passed 64-environment / five-iteration StandUp smoke
  with finite/penalty audits, 7,680 timesteps and native reload parity error
  `1.1920928955078125e-06`. Evidence: `outputs/diagnostics/sb3-cli-lr-0908-01/`.
- All six continuing worker environments were inspected: CUDA visibility is `7`
  and W&B mode is `offline`. No stopped full run was automatically restarted.

The initial Newton SB3 Walking preview also completed 14 s without falling but
failed tracking (RMSE `[0.0807, 0.1147, 0.6715]`). The stopped MuJoCo SB3 Walking
checkpoint failed after 9.12 s, with min height 0.0377 m and max tilt 1.253 rad.
StandUp SB3 previews were still rendering their per-spawn videos at this report
update; inspect `previews.json` for later results. Existing completed videos remain
available while those renders continue.
