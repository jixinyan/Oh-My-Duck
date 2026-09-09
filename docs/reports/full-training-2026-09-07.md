# Full representative RL training — 2026-09-07

Submission accepted: `omd-rl-full-0907-01`, queue ID `local-662cf40c7dfb`.
Initial scheduler state at submission was **Pending**. The 2026-09-08 check
confirms **Failed**; see the follow-up below. No long-training completion
or learned behavior is claimed. One node, eight GPUs, one independent native
learner per GPU. Source snapshot: `1f45996b2e9ec22c97f492cf1b5a727081cc303f`.
The configuration is `configs/experiments/representative-full.json`.
After submission, the source branch was merged into main at `5ef433b`; the submitted
campaign retained its original immutable snapshot.

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
capacity and long-training behavior remain unvalidated until runtime gates finish.

See [campaign operations](../rl-campaigns.md) and
[single-GPU acceptance](rl-pipeline-acceptance.md).

## Scheduler failure and local attempt — 2026-09-08

The platform task ID is `b2bab260-030d-4521-8598-77adb7863b4c`.
`submit --status` reports Failed, start `2026-09-08 05:52:42` and last update
`06:12:08` (scheduler display time, UTC+8 on this host). The local scheduler log
is zero bytes; `submit --logs` returns `no log pod names found`, and status lists
no pods. There is no campaign directory or manifest for `full-0907-01`.
This is insufficient evidence to attribute failure to PPO, CUDA, or task code;
the platform-level root cause is unknown. Preserve `status-20260908.txt` and
`log-query-20260908.txt` beside the original submission records.

The user authorized using the idle local development GPUs. A separate attempt
started at `2026-09-08T01:53:03Z` on eight H200 GPUs, indices 0–7, with the same
source snapshot, configuration, native PPO budgets, headless execution and offline
W&B. The launch is detached from the interactive terminal, with no duration cap.

- Launch metadata and parent log: `outputs/jobs/omd-rl-local-0908-01/`.
- Parent PID at launch: `596483`; worker PIDs are recorded in the manifest.
- Live manifest: `outputs/experiments/full-local-0908-01/campaign.json`.
- All eight 64-environment / five-iteration smoke stages completed. Export,
  rehearsal, resume and capacity gates precede each full training run.
- This is a new attempt, not a successful retry of the original scheduler job.
  Check live stage logs before interpreting this startup record as convergence.

The older Newton DDP task `a52ff51b` is also no longer present in the platform API
according to a separate status query; it supplies no new DDP acceptance evidence.

At the next startup check, all four Walking combinations had completed smoke,
export, CPU/BAM rehearsal, resume, 4096-environment capacity and capacity-export
gates and entered the full stage. All four StandUp combinations had completed
smoke and export and were still running the CPU/BAM rehearsal. No worker had
reported an execution failure. This is a timestamped startup observation; the
manifest and stage logs remain the authority for subsequent progress.
