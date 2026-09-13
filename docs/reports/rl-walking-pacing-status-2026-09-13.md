# Walking training status — 2026-09-13

Checked at 2026-09-13T14:31:58.578216+00:00. Job `omd-walk-pacing-0913-01` is **Running**, one node / eight H800 GPUs. All eight runs passed startup gates and entered their full 50000-update training stage. No run reports an execution failure.

| Combination | Official task schedule | Delayed smoothing |
|---|---:|---:|
| mujoco / rsl-rl | 9019 / 50000 | 9085 / 50000 |
| mujoco / sb3 | 6008 / 50000 | 5899 / 50000 |
| isaac-newton / rsl-rl | 6734 / 50000 | 6837 / 50000 |
| isaac-newton / sb3 | 4443 / 50000 | 4370 / 50000 |

Counts are logged rollout updates; native RSL iteration labels are zero-based, and SB3 optimizer epochs are not counted as rollout updates. Logs continue to advance after this snapshot.

Every latest completed saved preview remains below the existing Walking acceptance threshold. At the saved 5000-update MuJoCo SB3 previews, delayed/original forward response is approximately 63%/34%; at the 6000-update Newton RSL previews it is approximately 56%/22%. These are early observations in perturbation-enabled play. In particular, the delayed schedule is still in a different curriculum stage at update 5000. They do not establish causal learning improvement, autonomous motion without pushes, transfer success or convergence. No acceptance threshold has changed.

Checkpoint/video generation continues every 1000 updates. The previous run's causal replay already demonstrated that pushes can inflate apparent response. Final no-push/two-backend/CPU evaluation remains in the scheduled workflow. No intermediate evaluation battery, new training, restart or configuration change was launched for this user-requested status check.

Evidence: `outputs/diagnostics/walking-status-0913-01/result.json`; existing per-run galleries: `outputs/experiments/walking-delay-0913-01/<run-id>/previews/index.html`. Historical checkpoints and controls remain preserved. Training continues toward its declared budget; unified diagnosis follows completion.
