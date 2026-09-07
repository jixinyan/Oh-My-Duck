# Representative single-GPU RL pipeline acceptance — 2026-09-06

The domain refactor and representative single-GPU pipelines are implemented and
have completed execution validation. **This is not learned-task acceptance.**
All evaluated five-iteration policies fail the behavior battery. No hardware
validation or public policy upload has occurred.

| Training backend | PPO framework | Task | Train / resume / export / package | Both-backend replay + CPU/BAM video | Learned behavior |
|---|---|---|---|---|---|
| MuJoCo | RSL-RL | Walking | Passed | Completed | Failed |
| MuJoCo | RSL-RL | StandUp | Passed | Completed | Failed |
| MuJoCo | SB3 | Walking | Passed | Completed | Failed |
| MuJoCo | SB3 | StandUp | Passed | Completed | Failed |
| Isaac/Newton | RSL-RL | Walking | Passed | Completed | Failed |
| Isaac/Newton | RSL-RL | StandUp | Passed | Completed | Failed |
| Isaac/Newton | SB3 | Walking | Passed | Completed | Failed |
| Isaac/Newton | SB3 | StandUp | Passed | Completed | Failed |

## Reproducible evidence

- Training: immutable `5971dca`, 64 environments and five native PPO iterations.
  `outputs/acceptance-20260906-{rsl-rl,sb3}-01/result.json`.
- RSL export/packaging: `6303ca4`, four actual checkpoints, numerical parity,
  finite scalar and weighted penalty audits, normalized official runner export,
  schema-2 local packages. `outputs/postmerge-rsl-export-0906-01/result.json`.
- Complete replay: `a0b1f0f`, all eight policies, common seed/commands/manual-reset
  protocols in both simulation backends and official CPU/BAM deployment rehearsal.
  `outputs/replay-acceptance-0906-{rsl-rl,sb3}-01/result.json`.
- Video/trace audit: `outputs/replay-video-validation-0906.json` verifies **60
  videos**, 1280×720 first frames, 25 FPS, exact frame counts against trace ticks,
  and finite 61-observation / 14-action trajectories. Selected first/final frames
  were inspected visually; they show the short policies falling or remaining down.
- SB3 curriculum restoration: `b65e8f9`, both tasks on both backends resume from
  step 120 to 240 and pass fresh export/packaging. A second resume restores the
  saved state and reaches 360. Reports:
  `outputs/sb3-curriculum-resume-0906-01/{mujoco,isaac-newton}-result.json`.
  See [legacy counter limitations](resume-validation.md).

Replay exit code 2 means `behavior_failed`, not an execution error. Completed
means the executable reported an outcome; fallen Walking trajectories stop early.
All 24 replay contexts ran without runtime errors; no automatic reset masked a failed
trajectory. Walking tests hold/forward/stop/turn. StandUp tests standing, sitting,
face-down and face-up starts. The physical audits remain separate evidence for
solver identity, collisions, IMU/state conventions, BAM cadence and DR.

MuJoCo video explicitly uses OSMesa software rasterization on this host because
EGL readback is unreliable. Isaac video remains native Newton. Neither rendering
choice changes training physics or PPO. See [rendering validation](rendering-validation.md).
The plain Newton background provides less ground contrast than the CPU scene;
contact conclusions rely on physical audits and metrics alongside video.

## Source and remaining work

All first-party source is under `src/oh_my_duck` in nine project domains. Old
`training/` packages and generated build directories have been removed. Runtime
locks, source ancestry, licenses and failed experiment artifacts are preserved.
A wheel build checked package resources, included upstream notices/licenses and
excluded Python bytecode. Local Markdown links and all three environment locks
were checked. Tests: 27 lightweight + 72 task/SB3; earlier installed-runtime
checks passed seven Isaac and two Newton binding tests.

Architecture is merged into local `main` (`6303ca4`); follow-up fixes are on
`feat/rl-pipeline-validation`.

Long training and learned gait/recovery acceptance remain. Native RSL MuJoCo DDP
has earlier passing evidence; the two-GPU Newton StandUp smoke job
`omd-newton-stand-rsl-0906-01` started running on 2026-09-07 (job `a52ff51b`); both native worker ranks started.
Completion and DDP acceptance remain pending. Single-GPU work runs directly on the host; multi-GPU runs use the
scheduler. Host GPUs are shared and busy, so no isolated throughput or optimal
GPU/environment-count claim is made. W&B remains offline.
