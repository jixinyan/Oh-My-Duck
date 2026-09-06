# Isaac Lab / Newton integration

This backend is in development. The code currently implements **asset conversion and a PD diagnostic task**, not the official BAM walking recipe. No Isaac GPU run has passed yet. Keep the full product scope in the [Project Design](Agentic%20Microduck%20-%20Project%20Design%20v0.1.md).

## Environments and source versions

```bash
python omd.py setup --backend isaac-newton
```

This verifies the fixed Isaac Lab source commit, prepares the official Microduck sources, and installs two independent, locked Python 3.12 environments:

| Directory | Purpose |
|---|---|
| `.envs/isaac-newton` | Isaac Lab, Newton MJWarp, RSL-RL and Newton Warp renderer |
| `.envs/isaac-assets` | Isaac Sim MJCF-to-USD converter only |
| `.envs/mujoco` | Existing official baseline; also produces raw MJCF reference properties on CPU |

The conversion environment resolves Isaac Sim's dependencies independently because its published Torch/torchvision requirements conflict with the selected Isaac Lab training versions. No physics dependency overrides or alternate-engine fallback are used. The [resolution report](reports/isaac-environment-resolution.md) records the failures and final separation.

## Server job sequence

Run these stages in order after setup finishes. Each job/output name must be unique; inspect its result before starting the next stage. These are commands to execute, not evidence of completed runs.

```bash
python omd.py submit --name omd-isaac-assets-001 --gpus 1 --   python omd.py assets --backend isaac-newton

python omd.py submit --name omd-isaac-probe-001 --gpus 1 --   python omd.py probe --backend isaac-newton --output outputs/isaac-probe-001

python omd.py submit --name omd-isaac-pd-train-001 --gpus 1 --   python omd.py train --backend isaac-newton --   --task Omd-Microduck-PD-Diagnostic-v0 --num_envs 16 --max_iterations 5

python omd.py submit --name omd-isaac-pd-export-001 --gpus 1 --   python omd.py export --backend isaac-newton   --checkpoint /absolute/path/to/model_4.pt --output outputs/isaac-export-001

python omd.py submit --name omd-isaac-pd-video-001 --gpus 1 --   python omd.py eval --backend isaac-newton --output outputs/isaac-video-001   --num-envs 1 --steps 100 --video
```

Train delegates to the pinned Isaac Lab RSL-RL script with our registration hook and headless flag. Checkpoints are under `logs/rsl_rl/omd_isaac_pd_diagnostic/`. Training without an explicit diagnostic task fails because BAM locomotion is not implemented. The diagnostic task has a small posture reward, zero commands, a compact network and explicit PD actuation. Its checkpoints are not walking-policy candidates and are labeled accordingly when exported.

Probe checks the actual Newton `SolverMuJoCo`, canonical joint/action mapping, raw asset mass/COM/principal inertia/limits/armature, finite stepping and a non-uniform offscreen image. Eval supports bounded zero-action PD holding and optional 61D ONNX replay, but this is still a PD diagnostic, not a BAM sim2sim comparison. It stops on termination rather than treating automatic resets as successful continuation. Export includes the actor's normalizer and compares Torch/ONNX outputs on 16 identical synthetic inputs; these are numerical checks, not behavior certification.

## Modules and artifacts

| Module | Responsibility |
|---|---|
| `src/oh_my_duck/training/isaac_newton.py` | Lightweight process dispatch and explicit availability boundaries |
| `training/isaac_newton/bootstrap_env.py` | Pinned source acquisition and isolated installation |
| `omd_isaac/paths.py`, `contracts.py` | Asset provenance and canonical joint mapping without simulator imports |
| `omd_isaac/assets.py`, `convert_asset.py`, `asset_reference.py` | Isolated conversion, source reference dump and generated-file integrity |
| `omd_isaac/config.py`, `mdp.py`, `environment.py`, `agent.py` | Explicit diagnostic scene, observations, runtime guards and PPO config |
| `omd_isaac/physics.py` | Actual solver inspection and raw-asset comparison; private access stays here |
| `omd_isaac/rollout.py`, `probe.py`, `eval.py`, `export.py` | Finite evidence-producing workers and numerical export validation |

Converted USD and its payloads live under `artifacts/isaac-newton/walk/<fingerprint>/`. The fingerprint includes official source content, upstream pins, converter implementation and its dependency lock. `build.json` hashes generated files; it is written only after conversion succeeds. Parent-process artifact verification prevents a Kit teardown exit code from falsely reporting success. The nested-body transform correction is attributed in [third-party notices](../THIRD_PARTY_NOTICES.md).

Worker outputs include `result.json`, `trajectory.npz`, `frame.png`, and optionally `rollout.mp4`; exports include `policy.onnx` and `export.json`. Reports distinguish raw-asset checks from the still-pending training collision overrides, contacts, joint axes and BAM dynamics.

## Remaining migration work

1. Complete installation/import checks and run conversion/probe when scheduler quota is available.
2. Validate final solver contact/axis parameters and preserve official training overrides.
3. Implement BAM m6 with verified world/DOF mapping, per-world friction/damping storage, voltage and action delays. Current solver field inspection performs **no BAM writes**.
4. Add official delayed observations, encoder bias, commands, rewards and domain randomization; verify the shared contract on actual states.
5. Replay the same official policy in both validated backends; then train, resume, export and measure locomotion. Keep diagnostic results separate from this milestone.
