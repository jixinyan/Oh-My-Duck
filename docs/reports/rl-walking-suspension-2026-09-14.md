# Walking job suspended — 2026-09-14

Checked at 2026-09-14T11:53:34.409451+00:00. Scheduler reports **Suspended** for `omd-walk-pacing-0913-01` (`0367fb5b-34e5-40eb-9072-f5a081306253`), with no worker pods listed. Its last update is displayed as 2026-09-14 15:38:44; all eight training logs stopped around **07:38 UTC**. At this check they have been stale for over four hours.

The local campaign/worker JSON files still say `running`; those are stale process-written records, not evidence of continuing training. Preserve them as original artifacts and use the observed scheduler state here and in CLI maturity. Neither status output nor the mirrored scheduler log explains the suspension; do not attribute it to model failure, resource preemption, quota or a user action without evidence.

| Combination | Official logged / saved updates | Delayed logged / saved updates |
|---|---:|---:|
| mujoco / rsl-rl | 40703 / 40000 | 41043 / 41000 |
| mujoco / sb3 | 26783 / 26000 | 26791 / 26000 |
| isaac-newton / rsl-rl | 30378 / 30000 | 30925 / 30000 |
| isaac-newton / sb3 | 20162 / 20000 | 19803 / 19000 |

All budgets are 50000 rollout updates. RSL iteration labels are zero-based. Logged progress after the latest durable save is not guaranteed recoverable. Checkpoint files are present and nonempty; SB3 bundles include model, native normalizer and metadata. This inventory does not claim that a restored learner has been executed successfully.

None of the full training runs reached final export/standard transfer/no-push/CPU-BAM evaluation. All eight latest completed periodic previews remain below the existing acceptance threshold. One delayed MuJoCo RSL preview was interrupted while rendering a newer checkpoint; preserve its partial evidence.

Checkpoint inventory and observed status: `outputs/diagnostics/walking-suspended-0914-01/result.json`. Previous running snapshots and all original logs/checkpoints/videos are retained. This status check did not resume, stop or resubmit any job; the suspension cause remains unresolved.
