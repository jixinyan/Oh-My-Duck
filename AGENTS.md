# Oh My Duck development constraints

The product scope is `docs/Agentic Microduck - Project Design v0.1.md`. Keep that document, the execution plan, `docs/implementation-status.md`, and the CLI maturity matrix synchronized with verified results. README is the whole-project overview; detailed experiments belong in reports.

This project extends the pinned official `pollen-robotics/microduck_rl` and `pollen-robotics/microduck` implementations. Read `.cache/upstream/microduck_rl/AGENTS.md` before changing RL tasks, actuators, export or publishing. Pins are in `configs/upstream.json`; cached upstream checkouts remain pristine. Reference implementations do not replace the official baseline.

- Preserve 61 actor observations (48 proprioception + twist 3 / head 4 / body 6), 14 named servos, canonical HOME and 50 Hz. HOME is the observation/action reference; do not assume zero-action HOME is a stable held pose. Passive joint names and mappings must follow the official conventions.
- Use official BAM XL330 M6, physics-step delays, per-world friction, non-accumulating DR, matched sensor-view rewards, full noise/NaN protections, and the official task factories wherever possible. Record necessary framework or simulator differences explicitly. Do not simplify task semantics just to pass a smoke test.
- Isaac simulation must use Newton. PD diagnostic success is not BAM/task migration success. Validate actual solver, geometry/contact masks, motor load/friction and timing. Never silently fall back to PhysX or another simulator/framework.
- New combinations require the official 64-environment / 5-iteration smoke, finite reward and penalty-sign audit, normalizer-aware export, and CPU MuJoCo/BAM rehearsal with metrics and video. A short checkpoint does not demonstrate a learned gait. No hardware is available.
- Export through official `run_export` and the runner export method; framework-specific actors must include their native normalization and pass numerical parity checks. Use official schema-2 validation and publisher format. Local package validation is separate from authorization to upload publicly.
- Training/evaluation run headlessly through server jobs. Resource counts are explicit; do not impose arbitrary duration caps. Video resolution is configurable and defaults to 1280x720 for Isaac evaluation.
- Keep optional simulators/frameworks in isolated locked environments. Preserve official pins; document any unavoidable dependency compatibility overrides and verify affected paths.
- Use modular entry points and small focused commits on feature branches. Preserve failed job artifacts. Do not overwrite prior output directories or present automatic retries as successful initial runs. Record project/upstream provenance.
- The external Harness is not ready; future integration uses a deterministic mock, not a substitute internal agent loop. Keep interface-only capabilities visibly unavailable.

- Current RL acceptance scope is the representative tasks in `configs/training.json` (Flat Walking and Flat StandUp), across both backends and both RL frameworks. The complete official registry is an inventory, not a requirement to reproduce every task. New tasks should enter through task registration and shared adapters.
- Preserve each framework's native PPO semantics. RSL-RL uses its native distributed learner; SB3 uses native PPO/vector environments and independent parallel runs. Do not introduce decoupled asynchronous actors or claim SB3 supports distributed gradient updates. Measure throughput before selecting GPU/environment counts.
- W&B must run OFFLINE. The current saved account belongs to someone else. Do not authenticate, upload or sync to it. Propagate offline mode to every distributed worker; keep local run directories and artifacts.
