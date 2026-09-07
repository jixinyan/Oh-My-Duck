# Full representative RL training — 2026-09-07

Submission accepted: `omd-rl-full-0907-01`, queue ID `local-662cf40c7dfb`.
Latest scheduler state at submission: **Pending**; no long-training completion
or learned behavior is claimed. One node, eight GPUs, one independent native
learner per GPU. Source snapshot: `1f45996b2e9ec22c97f492cf1b5a727081cc303f`.
The configuration is `configs/experiments/representative-full.json`.
After submission, the source branch was merged into main at `5ef433b`; the running
campaign continues to use its original immutable snapshot.

| Allocation slot | Simulation | PPO | Task | Environments | Full iterations |
|---|---|---|---|---|---|
| 0 | MuJoCo | RSL-RL | Walking | 4096 | 50,000 |
| 1 | MuJoCo | RSL-RL | StandUp | 4096 | 15,000 |
| 2 | MuJoCo | SB3 | Walking | 4096 | 50,000 |
| 3 | MuJoCo | SB3 | StandUp | 4096 | 15,000 |
| 4 | Isaac/Newton | RSL-RL | Walking | 4096 | 50,000 |
| 5 | Isaac/Newton | RSL-RL | StandUp | 4096 | 15,000 |
| 6 | Isaac/Newton | SB3 | Walking | 4096 | 50,000 |
| 7 | Isaac/Newton | SB3 | StandUp | 4096 | 15,000 |

Slots follow the scheduler's GPU allocation order, not host-global device IDs.
W&B is offline. Checkpoints are saved every 250 PPO updates. There is no extra
wall-clock limit or shared gradient learner across tasks. Each worker runs a
64-environment smoke, export/rehearsal/resume gates and a 4096-environment capacity
check before starting its full budget. Failed workers preserve artifacts while
other independent runs continue. Full checkpoints are exported, locally packaged
and evaluated in both simulators plus CPU/BAM with headless 720p video.

## Evidence and monitoring

- Submission: `outputs/jobs/omd-rl-full-0907-01/submission.json` and
  `submit-response.txt`; scheduler log is in the same directory.
- Runtime manifest (created after allocation):
  `outputs/experiments/full-0907-01/campaign.json`.
- Each worker has a separate `result.json`, stage logs and policy artifacts.
  Native RSL checkpoint paths are recorded in those results.
- Use `squeue --me` for live scheduler state. Do not infer successful training
  from this submission record or from the existence of a checkpoint.

The task refactor preserved all 33 semantic catalog entries, now organized into
14 families with separate environment and PPO configuration. Pre-submission
checks passed 30 lightweight tests, 74 task/SB3 tests and two Newton binding tests.
Two reduced campaign workers (RSL-RL and SB3 Walking) completed every pipeline stage
at 64 environments; both returned `behavior_failed`, with no execution error.
They verified new orchestration and periodic checkpoint recovery; 4096-environment
capacity and long-training behavior are validated by the submitted job.

See [campaign operations](../rl-campaigns.md) and
[single-GPU acceptance](rl-pipeline-acceptance.md).
