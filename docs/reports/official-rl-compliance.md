# Official RL compatibility audit — 2026-09-06

Authority: [official AGENTS.md](https://github.com/pollen-robotics/microduck_rl/blob/29e887ecfbf5d37144759e5a9f8a176dfb83d547/AGENTS.md), read in full. Runtime contract: pinned `microduck/docs/policy-manifest.md`. Both upstream checkouts remain unmodified. This is an extension of the official code, not an independently validated replacement.

## Required gates

| Gate | Implementation / evidence | Remaining work |
|---|---|---|
| 61 actor inputs, 14 named servo outputs, HOME, 50 Hz, unfiltered actions | Shared contract and official inference helpers; MuJoCo uses unchanged registered tasks | Isaac locomotion sensor semantics, noise and delay alignment |
| BAM XL330 M6, voltage sag, friction scale, startup expansion, physics-step delay | Unchanged official MuJoCo task and exact BAM Git pin | Newton actuator bridge and physics parity |
| Full task DR, rewards, curricula, NaN guards | Official MuJoCo task factory retained | Isaac migration must preserve the complete stack |
| 3 s noisy-pose settle with height AND tilt | Required before new task training | Explicit battery pending |
| 64 environments / 5 iterations, finite rewards and penalty signs | Official walking job succeeded; checkpoint exported by official exporter | 46 scalar tags finite; all 9 penalty terms nonpositive; each new combination needs its own gate |
| Normalizer baked into ONNX | Official `scripts/export.py` succeeded for MuJoCo | Isaac PD exporter remains diagnostic; SB3 deployment export pending |
| CPU MuJoCo/BAM rehearsal, metrics AND video | Official alpha policy completed 700 ticks / 14 s without falling | Command tracking failed: forward vx RMSE 0.0998 m/s at 0.1 m/s; yaw RMSE 0.485 rad/s at 0.5 rad/s. Command slots verified in saved observations. Do not call this walking success |
| Official schema-2 publisher shape/smoke checks | Official dry-run passed: `outputs/mj-package-20260906-01/` | No Hub upload performed |
| Official regression tests | 199 passed, 1 skipped (CPU, pinned upstream, ephemeral pytest via uv) | Skip reason to retain with test report |

## Reproducible evidence

- Training: `omd-mj-walk-smoke-20260906-01`, HTrain `dfdff62b-3148-4cc1-b5a4-0aba6c8c53fc`, 64 environments, 5 iterations. Checkpoint `logs/rsl_rl/velocity/2026-09-06_15-27-10_velocity/model_4.pt`.
- Official export: `omd-mj-export-20260906-01`, HTrain `cf565485-a018-4a2b-be57-cb45e433512a`, `outputs/mj-export-20260906-01/policy.onnx`.
- Official-policy CPU replay: `omd-mj-replay-20260906-01`, HTrain `6a8bf0d3-5da6-463c-8908-16948fb659c0`, `outputs/mj-replay-20260906-01/{result.json,trajectory.npz,rollout.mp4}`.
- Tests: `UV_PROJECT_ENVIRONMENT="$PWD/.envs/mujoco" UV_CACHE_DIR="$PWD/.cache/uv" UV_PYTHON_INSTALL_DIR="$PWD/.cache/python" CUDA_VISIBLE_DEVICES='' uv run --project .cache/upstream/microduck_rl --locked --with pytest pytest .cache/upstream/microduck_rl/tests/ -q`.

All three jobs used project commit `e5f604d`. A short training checkpoint is an integration artifact, not a learned gait or hardware-validated policy. The first migrated task remains official flat velocity walking; additional task families and Backlash variants require separate physics/task evidence.

## SB3 official-task extension

`omd-mj-sb3-smoke-20260906-02` succeeded (HTrain `5634647c-eee1-44eb-833c-5c23e761a3e8`): 64 environments, 5 rollouts, 7,680 transitions, 190 final-state snapshots. Paired native model/normalizer reload agrees within 2.15e-6 in FP32. The first attempt completed training but failed a CPU/GPU comparison because the official Torch setup enabled TF32; it is preserved as failed, and the comparison was corrected to FP32 before the successful rerun.

`omd-mj-sb3-resume-20260906-01` succeeded (HTrain `f892a583-b771-4261-b237-e5e6092b8983`): native checkpoint continuation with an explicit 0.2-second horizon override to exercise timeout bootstrapping. Reports, checkpoint hashes, reward term ranges and recorded algorithm differences are under the corresponding `outputs/mj-sb3-*/run.json` paths. This override is a boundary test, not the walking recipe.

The MuJoCo SB3 lock retains every official dependency version. Two CPU boundary tests exercise real mjlab observation caching/delay behavior and termination-versus-timeout semantics. SB3 ONNX deployment export remains unavailable.

## Newton BAM bridge under validation

The bridge directly constructs the pinned `FrictionDRBamActuator`, delegates `apply_delay` and `compute`, and checks independently writable per-world friction fields. Newton explicit efforts enter `qfrc_applied`; the BAM data view presents these plus native actuator effort as the previous motor-side load. The official fitted armature is read from BAM (0.0018077432831600838), not rounded to the raw asset value. The bridge is not yet enabled as a walking training task.

Newton now includes mjlab 1.3.0 and the exact official BAM Git revision to reuse these implementations. Physics pins (Newton, MuJoCo, MJWarp, Warp, Torch) remain unchanged. Two explicit nonphysics dependency overrides are necessary: typing-extensions 4.15.0 (Isaac declares 4.12.2, mjlab's Tyro requires >=4.13) and retaining Isaac's Pillow 12.2.0 (MoviePy 2.2.1 declares <12). MoviePy is pinned to 2.2.1 to avoid resolution falling back to a 2017 release. Seven existing Isaac CPU tests, BAM construction, and an actual MoviePy MP4 write passed. These exceptions are recorded rather than claiming all upstream package metadata constraints are simultaneously satisfied. Newton runtime regression still needs the worker check.

## Subsequent export, physics and video evidence

- MuJoCo SB3 official-runner export passed: `omd-mj-sb3-export-20260906-01`, HTrain `4e919698-66b1-49e5-9468-454f77f6e07c`. Native actor/normalizer parity on 32 inputs, including clipped outliers: max error 9.54e-7. Artifact `outputs/mj-sb3-export-20260906-01/`.
- CPU BAM HOME settle failed for all 64 noisy initial poses: `omd-mj-settle-20260906-01`, HTrain `f982d371-9ead-4669-bfb9-70b39d89189b`. HOME is a reference frame, not a proven equilibrium target. Do not change the policy HOME contract to hide this result.
- Newton BAM first run failed on flattened USD joint names; fix committed. Second run stepped but detected a fall at tick 43. Third run preserved the full fallen-state trajectory: `omd-newton-bam-20260906-03`, HTrain `a790ef51-c643-410e-9e36-79234128b136`, 64 environments / 150 control ticks / 600 actuator substeps, 0 stable HOME samples. Actual applied motor loads match the bridge's output. This is a failed equilibrium gate with useful physics evidence, not a successful stance test.
- Official alpha-policy Newton replay completed 700 ticks / 14 s with BAM: `omd-newton-policy-20260906-01`, artifact `outputs/newton-policy-20260906-01/`. Whole-sequence twist RMSE `[0.05306, 0.000987, 0.25742]`; this also largely stands still. Contact/hull audit subsequently found deviations, so this run is not final sim2sim acceptance.
- Isaac 720p video passed: `omd-newton-video720-20260906-01`, HTrain `fa314edd-f165-4632-b2f0-5f61a7428386`. Actual MP4 1280x720 / 25 fps / 3 s, frame visually inspected. Default resolution now 720p, configurable with `--video-width` / `--video-height`.
- Compiled model audit found Newton allowed group-2 leg/battery meshes to contact ground/feet, while official MJCF isolates them for self-collision. USD spawn overrides now retain full hull detail and foot priority, and a MODEL_INIT callback adds the official group filters. First validation failed on the callback's event argument; corrected rerun `omd-newton-contact-20260906-02` is pending, including actual compiled-hull and pair-mask checks.

Full Isaac walking task DR/observation/reward/curriculum migration is still outstanding. No Isaac BAM walking training or hardware deployability is claimed.
