# Walking policy command dead-zone diagnosis — 2026-09-30

This report evaluates the existing final MuJoCo RSL-RL Walking export without changing the learner, task configuration, or acceptance protocol. It isolates the forward command magnitude by replacing the protocol's nonzero forward command while keeping the stop and turn stages unchanged. The run used the final checkpoint from `jd-walking-rsl-control-20260926`, seed 42, no external pushes, and the CPU MuJoCo backend with the OSMesa renderer.

Policy export:

`outputs/experiments/jd-walking-rsl-control-20260926/mujoco-rsl-rl-walking-official-seed42/export/policy.onnx`

The diagnostic output directories are preserved under `/tmp/omd-walk-diag-20260930-0p1` through `0p4`. They are diagnostic evidence only; because the command was overridden, each result correctly records `acceptance_eligible: false`.

| Forward command (m/s) | Measured forward velocity (m/s) | Response fraction | Whole-trace RMSE (vx, vy, yaw) | Result |
| ---: | ---: | ---: | --- | --- |
| 0.1 | 0.00026 | 0.003 | (0.0548, 0.0952, 0.0956) | forward stage fails |
| 0.2 | 0.15612 | 0.781 | (0.0369, 0.1376, 0.1350) | forward response, strict stages fail |
| 0.3 | 0.22732 | 0.758 | (0.0536, 0.1346, 0.1452) | forward response, strict stages fail |
| 0.4 | 0.28662 | 0.717 | (0.0779, 0.1363, 0.1927) | forward response, strict stages fail |

Yaw response stayed present across the sweep: 0.462–0.475 m/s-equivalent for the 0.5 rad/s command, or 92–95% of the request. The zero-command stages remained quiet. At 0.2–0.4 m/s, the forward stage's mean lateral velocity stayed close to zero, but the turn stage produced enough instantaneous lateral fluctuation to exceed the current global RMSE and per-window quiet-axis limits. This is the existing protocol's known distinction between instantaneous fluctuation and window-mean drift; it is not evidence that the policy has passed Walking acceptance.

The result establishes a command-conditioned failure rather than a uniformly inactive actor. The policy can produce forward motion at moderate commands but collapses to a standing solution at the acceptance protocol's 0.1 m/s command. The task currently uses the official Gaussian velocity-tracking form with weight 2 and `std² = 0.1`. At 0.1 m/s, a stationary body receives `2·exp(-0.1) ≈ 1.81` for this term versus 2.0 at the target, leaving only about 0.19 reward units to pay for stepping. At 0.2 m/s the stationary score is about 1.34, leaving about 0.66 units. This makes the observed low-speed dead zone a credible training objective issue, while preserving the official formula means it is not yet a justified source change.

## Decision

Do not publish this export as a locomotion tool. Keep the controlled diagnostic in the policy-diagnosis branch and use it as the pre/post check for the next Walking training hypothesis. The next candidate intervention should increase the learnable advantage of low nonzero commands while retaining the official command semantics and the 0.1 m/s acceptance scenario. It requires a fresh full workflow with smoke, export, rehearsal, unforced evaluation, and preserved provenance; a short checkpoint is insufficient to claim a learned gait.

