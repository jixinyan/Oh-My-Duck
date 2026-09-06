# Isaac environment dependency resolution

2026-09-06. These are installation/resolution results, not GPU execution evidence.

- Official Isaac Lab `v3.0.0-beta2.patch1`: `ffff603eafc6b74264a5261cc0183d6a65390d78`; source checkout remains clean.
- `.envs/isaac-newton`: independent Python 3.12 environment. `training/isaac_newton/uv.lock` now resolves 156 packages, including source-installed Isaac Lab, Newton 1.2.1, MuJoCo-Warp 3.8.0.3, Warp 1.13.0, Torch 2.10.0 and torchvision 0.25.0. PyPI's Torch 2.10.0 Linux wheel declares CUDA 12.8 dependencies; the separate download.pytorch.org endpoint failed TLS connection, without disabling certificate checks.
- `.envs/isaac-assets`: separate asset conversion environment. `training/isaac_newton/asset_converter/uv.lock` resolves 167 packages from Isaac Sim 6.0.1.0's actual dependency declarations. Its Torch/CUDA stack is not the training stack.
- Installation was started after successful resolution. Completion, actual imports, asset conversion and worker execution must be checked separately.

A combined environment did not resolve: the pinned Isaac Lab declares coverage 7.6.1 while Isaac Sim declares 7.4.4; more importantly, Isaac Sim core requires torchvision 0.26.0 whereas the selected Isaac Lab/Torch recipe uses 0.25.0. An exploratory coverage override revealed the second conflict and was removed. The committed solution isolates conversion and training and contains no dependency override. Both environments have independent locks. Isaac Sim is used to generate USD; Isaac Lab uses Newton for physics and Newton Warp for diagnostic offscreen rendering.

The package bootstrap script is named `bootstrap_env.py`: naming it `setup.py` caused setuptools to execute its CLI while building the package. Renaming resolved that build error. No upstream source edits were needed.

User steering: keep the MuJoCo probe queued while developing Isaac integration. This permits implementation before MuJoCo runtime validation; it does not mark the baseline complete or remove either backend from scope.

## Installed-environment checks

The Newton training environment installed successfully, including Torch 2.10.0, Isaac Lab 6.1.14, isaaclab-newton 0.13.6, Newton 1.2.1, MuJoCo-Warp 3.8.0.3 and Warp 1.13.0. `uv pip check --python .envs/isaac-newton/bin/python` reports all 156 installed packages compatible.

Actual configuration imports exposed two required upstream source shims: `isaaclab-physx` supplies the material configuration forwarded by the core package, and `isaaclab-contrib` is imported unconditionally by `InteractiveScene`. Both are now in the lock. Installing these source packages does not select their physics: the task configuration and runtime guards explicitly require Newton/MJWarp.

The import check also caught an incorrect local `configclass` import; it now imports the callable from its defining module. Early config rejection exposed a base-class destructor accessing `_is_closed` before initialization; a regression test failed before the fix and passed after initialization was corrected.

The official trainer's help parser inspects a task before invoking `external_callback`. The wrapper now registers the task first and uses `runpy` to execute the unchanged official script in the same interpreter, so both help and normal entry share registration. The actual isolated `omd train --backend isaac-newton -- --task Omd-Microduck-PD-Diagnostic-v0 --help` exits 0. Its output is saved locally at `outputs/isaac-config-20260906-01/train-help.txt`.

Validation: 11 standard-library framework tests plus 4 tests against the installed Isaac packages passed. The latter construct and validate task/camera configs and reject incompatible physics, timing and order without starting simulation. They run with `CUDA_VISIBLE_DEVICES=''`; Warp emits a no-CUDA diagnostic during import, which is expected in this deliberately CPU-only check and is not GPU validation.

CPU-only raw MJCF extraction also ran successfully: 15 bodies, 14 joints, total mass 0.73724318 kg. Evidence: `outputs/isaac-asset-reference-20260906-01/reference.json`. No USD conversion or physics rollout is implied by this extraction. The separate asset environment is still installing at this checkpoint, and worker validation is still waiting for scheduler quota.

## Installation complete and conversion queued

Both locked setup paths completed successfully through `python omd.py setup --backend isaac-newton`. Training has 156 installed packages and conversion has 166 (its 167-entry lock also includes the virtual project). Both pass `uv pip check`. Freeze files and setup provenance are under `artifacts/environments/isaac-newton/` and `artifacts/environments/isaac-assets/`; these are local ignored evidence.

The asset CLI help also exits 0 through the actual conversion interpreter. Asset conversion was submitted as `omd-isaac-assets-20260906-01`, queue ID `local-d0faa6d27c42`, one GPU. At submission it was queued, with no worker/asset result. MuJoCo probe `omd-probe-20260906-01` remains queued for project quota. GPU asset, physics, rendering, training and sim2sim verification remain pending.

## Worker result update

Both jobs left the queue. MuJoCo probe succeeded (device availability, CPU stepping, EGL image only). Isaac conversion failed during Kit first-run EULA input with `EOF when reading a line`, before conversion or Newton simulation. No noninteractive license acceptance is configured. Task IDs and evidence are recorded in [implementation status](../implementation-status.md).

## Successful worker follow-up

The user subsequently approved the EULA; conversion succeeded. Newton finite simulation/video and RSL-RL/SB3 short PPO runs are now verified, with RSL-RL numerical ONNX export. See [the worker validation report](isaac-rl-validation.md) for exact scope, later runtime fixes, task IDs and evidence.
