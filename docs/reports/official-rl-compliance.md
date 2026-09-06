# Official RL compatibility audit — 2026-09-06

Authority: [official AGENTS.md](https://github.com/pollen-robotics/microduck_rl/blob/29e887ecfbf5d37144759e5a9f8a176dfb83d547/AGENTS.md), read in full. Runtime contract: pinned `microduck/docs/policy-manifest.md`. Both upstream checkouts remain unmodified. This is an extension of the official code, not an independently validated replacement.

## Required gates

| Gate | Implementation / evidence | Remaining work |
|---|---|---|
| 61 actor inputs, 14 named servo outputs, HOME, 50 Hz, unfiltered actions | Shared contract and official inference helpers; MuJoCo uses unchanged registered tasks | Isaac locomotion sensor semantics, noise and delay alignment |
| BAM XL330 M6, voltage sag, friction scale, startup expansion, physics-step delay | Unchanged official MuJoCo task and exact BAM Git pin | Newton actuator bridge and physics parity |
| Full task DR, rewards, curricula, NaN guards | Official MuJoCo task factory retained | Isaac migration must preserve the complete stack |
| 3 s noisy-pose settle with height AND tilt | Required before new task training | Explicit battery pending |
| 64 environments / 5 iterations, finite rewards and penalty signs | Official walking job succeeded; checkpoint exported by official exporter | Detailed scalar audit; each new backend/framework/task combination needs its own gate |
| Normalizer baked into ONNX | Official `scripts/export.py` succeeded for MuJoCo | Isaac PD exporter remains diagnostic; SB3 deployment export pending |
| CPU MuJoCo/BAM rehearsal, metrics AND video | Official alpha policy completed 700 ticks / 14 s without falling | Command tracking failed: forward vx RMSE 0.0998 m/s at 0.1 m/s; yaw RMSE 0.485 rad/s at 0.5 rad/s. Command slots verified in saved observations. Do not call this walking success |
| Official schema-2 publisher shape/smoke checks | Publisher read; local dry-run next | No Hub upload performed |
| Official regression tests | 199 passed, 1 skipped (CPU, pinned upstream, ephemeral pytest via uv) | Skip reason to retain with test report |

## Reproducible evidence

- Training: `omd-mj-walk-smoke-20260906-01`, HTrain `dfdff62b-3148-4cc1-b5a4-0aba6c8c53fc`, 64 environments, 5 iterations. Checkpoint `logs/rsl_rl/velocity/2026-09-06_15-27-10_velocity/model_4.pt`.
- Official export: `omd-mj-export-20260906-01`, HTrain `cf565485-a018-4a2b-be57-cb45e433512a`, `outputs/mj-export-20260906-01/policy.onnx`.
- Official-policy CPU replay: `omd-mj-replay-20260906-01`, HTrain `6a8bf0d3-5da6-463c-8908-16948fb659c0`, `outputs/mj-replay-20260906-01/{result.json,trajectory.npz,rollout.mp4}`.
- Tests: `UV_PROJECT_ENVIRONMENT="$PWD/.envs/mujoco" UV_CACHE_DIR="$PWD/.cache/uv" UV_PYTHON_INSTALL_DIR="$PWD/.cache/python" CUDA_VISIBLE_DEVICES='' uv run --project .cache/upstream/microduck_rl --locked --with pytest pytest .cache/upstream/microduck_rl/tests/ -q`.

All three jobs used project commit `e5f604d`. A short training checkpoint is an integration artifact, not a learned gait or hardware-validated policy. The first migrated task remains official flat velocity walking; additional task families and Backlash variants require separate physics/task evidence.
