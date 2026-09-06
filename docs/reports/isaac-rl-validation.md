# Isaac/Newton and RL framework validation — 2026-09-06

## Scope

These results validate asset conversion, finite Newton PD simulation, headless video and short PPO training through two RL frameworks. They do not validate BAM, locomotion quality, sim2sim, framework performance comparisons or hardware deployment. Each GPU experiment ran through an Alaya HTrain job; local checks use CPU only.

## Verified worker results

| Job | Task ID | Evidence |
|---|---|---|
| `omd-isaac-assets-20260906-02` | `e9ec42f9-1680-4831-86bf-48719990a2ce` | Converted official MJCF; generated-file manifest verified |
| `omd-isaac-probe-20260906-02` | `f525ac94-c18b-4fb7-aa85-425e448f46ed` | 2 environments, 100 ticks, actual Newton `SolverMuJoCo` on `cuda:0`; raw asset and action mapping checks passed |
| `omd-isaac-video-20260906-02` | `c683b1dd-173d-4e32-8496-bc661a3d7d71` | Local ground asset, 100 ticks, corrected camera pose assertions and readable video |
| `omd-isaac-rsl-20260906-01` | `7726f6cf-6ec7-45f0-89b7-efea41cfc745` | RSL-RL PPO, 16 environments, 5 iterations; `model_0.pt` through `model_4.pt` saved |
| `omd-isaac-sb3-20260906-01` | `476f7850-fc98-4037-aada-2f147498ab36` | SB3 PPO, 16 environments, 5 rollouts / 1,920 transitions; model and VecNormalize saved after a transient remote-ground failure and worker retry |
| `omd-isaac-sb3-20260906-02` | `9d8ca014-da1f-460f-9618-bea8b0930267` | Corrected reset adapter and local ground; 1,920 transitions with a 0.2 s test horizon; 192 pre-reset snapshots and 192 time-limit truncations |

The physics probe checks joint names/order, joint limits/armature, body masses, principal inertias and body COM against the raw official MJCF. It reports pending collision-training overrides, contact parameters, joint axes and BAM. The generated USD contains PhysX compatibility properties; runtime verification checks Newton's actual solver, with no PhysX simulation fallback.

## Local evidence locations

Generated files are intentionally ignored by Git; these paths refer to the shared workspace.

- `outputs/isaac-video-20260906-02/{result.json,trajectory.npz,frame.png,rollout.mp4}`: accepted finite rollout and video. The first frame has been visually inspected and shows the complete robot.
- `logs/rsl_rl/omd_isaac_pd_diagnostic/2026-09-06_15-12-08/`: native RSL-RL checkpoints and saved environment/agent configurations.
- `logs/sb3/Omd-Microduck-PD-Diagnostic-v0/2026-09-06_15-17-20/`: corrected SB3 run, `model.zip`, `model_vecnormalize.pkl`, `adapter.json`, configs and TensorBoard logs.
- `outputs/rl-checkpoints-20260906-01/result.json`: CPU inspection of first-run native checkpoints. All 60 RSL-RL checkpoint tensors were finite; SB3 model plus restored normalizer produced finite 14D actions for 16 synthetic 61D inputs. This is not resume or physics replay verification.
- `outputs/jobs/<job-name>/`: submitted argv, source commit/dirty state, scheduler response and logs. A scheduler retry may append a later attempt to the same log; the first failure remains evidence.

## Failures found and corrected

1. Kit prompted for a license on stdin. User explicitly agreed to the NVIDIA Omniverse EULA; the conversion job passes `--accept-eula`, scoped to the child environment. Missing acceptance now fails before importing Kit.
2. The first probe submission misplaced `--`; it failed at argument parsing. Its corrected submission passed.
3. Nonuniform images alone missed stale camera poses. Explicit `update_latest_camera_pose`, sensor recomputation and measured pose assertions fix framing. The first video is retained but not accepted as a useful visual diagnostic.
4. The default grid ground fetched a remote USD and failed intermittently. The task now uses a small version-controlled flat USD with explicit collision/material properties. `omd-isaac-video-20260906-01` retains that failure; its automatic retries rejected the already-created output directory rather than overwriting evidence.
5. The pinned SB3 wrapper used post-reset observations as terminal states. The task now captures the pre-reset policy observation and a local wrapper supplies it for SB3 timeout bootstrapping. Upstream source files remain unchanged. Three CPU regressions and the 192-timeout GPU run validate this boundary.
6. The first RSL-RL export constructed a fresh configuration containing a compatibility-only `stochastic` field and failed. Export now reads the checkpoint's actual `params/agent.yaml` (or explicit `--agent-config`) and records its hash. Retry `omd-isaac-export-20260906-02` succeeded (task `740825e0-a468-4c4e-afd9-a2a2e7ce074f`): normalized ONNX matched Torch over 16 inputs with maximum absolute error `3.5762786865234375e-07`. Artifacts: `outputs/isaac-export-20260906-02/{policy.onnx,export.json}`. This validates numerical export, not policy behavior.

## Local checks

- 17 lightweight application/contract/framework tests pass without simulator imports in the application core.
- 7 installed-runtime CPU tests pass with the optional SB3 extra, including exact pre-reset snapshot and timeout semantics.
- Both actual RSL-RL and SB3 task-aware CLI help paths work.
- `uv pip check`: Isaac/Newton with SB3 has 161 compatible installed packages; asset conversion has its independent 166-package environment.
- Syntax, JSON/TOML and Git whitespace checks pass.

SB3 resume and ONNX export remain explicitly unsupported. MuJoCo + SB3 remains planned. The [framework guide](../rl-frameworks.md) records extension boundaries; adding another framework does not require changing the shared simulation task.
