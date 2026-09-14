# Walking training status — 2026-09-14

Checked at 2026-09-14T01:33:18.959563+00:00. Scheduled job `omd-walk-pacing-0913-01` remains Running on one node / eight H800 GPUs. All eight full learners are active, with no reported execution failures. Every learner retains the declared 50000-update budget and 8192 environments.

| Combination | Official task schedule | Delayed smoothing |
|---|---:|---:|
| mujoco / rsl-rl | 29427 / 50000 | 29647 / 50000 |
| mujoco / sb3 | 19383 / 50000 | 19333 / 50000 |
| isaac-newton / rsl-rl | 21973 / 50000 | 22356 / 50000 |
| isaac-newton / sb3 | 14565 / 50000 | 14299 / 50000 |

Counts are logged rollout updates (native RSL labels are zero-based), not SB3 optimizer epochs. Logs continue to advance after this snapshot. Progress is approximately 29–59%.

167 checkpoint previews have completed across the eight runs. Every latest preview remains below existing Walking acceptance. All eight latest lateral velocity RMSE values exceed 0.1 m/s (approximately 0.115–0.150 m/s); some policies also have insufficient forward response or excessive yaw error. These remain perturbation-enabled play evaluations, not proof of unforced locomotion.

At the 29000-update MuJoCo RSL previews, delayed/original forward response is approximately 98%/47%, but delayed yaw RMSE is 0.791 rad/s and lateral RMSE is 0.150 m/s, so this is not acceptance. At the 19000-update MuJoCo SB3 previews the original schedule has stronger forward response (78%) than delayed (43%). Both groups have reached the same final smoothing weight by these checkpoints; the single-seed observations do not show a consistent benefit from delaying smoothing.

No full run has finished and final no-push/two-backend/CPU-BAM evaluations have not started. Training continues under the existing complete-workflow instruction. This user-requested status check read saved evidence only; it did not launch an intermediate evaluation battery, change rewards, stop/restart learners or submit another job.

Evidence: `outputs/diagnostics/walking-status-0914-01/result.json`. Existing per-run video galleries: `outputs/experiments/walking-delay-0913-01/<run-id>/previews/index.html`. The [experiment protocol](rl-walking-pacing-2026-09-13.md) and [previous snapshot](rl-walking-pacing-status-2026-09-13.md) remain preserved.
