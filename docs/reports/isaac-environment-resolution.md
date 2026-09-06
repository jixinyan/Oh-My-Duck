# Isaac environment dependency resolution

2026-09-06. These are installation/resolution results, not GPU execution evidence.

- Official Isaac Lab `v3.0.0-beta2.patch1`: `ffff603eafc6b74264a5261cc0183d6a65390d78`; source checkout remains clean.
- `.envs/isaac-newton`: independent Python 3.12 environment. `training/isaac_newton/uv.lock` resolves 154 packages, including source-installed Isaac Lab, Newton 1.2.1, MuJoCo-Warp 3.8.0.3, Warp 1.13.0, Torch 2.10.0 and torchvision 0.25.0. PyPI's Torch 2.10.0 Linux wheel declares CUDA 12.8 dependencies; the separate download.pytorch.org endpoint failed TLS connection, without disabling certificate checks.
- `.envs/isaac-assets`: separate asset conversion environment. `training/isaac_newton/asset_converter/uv.lock` resolves 167 packages from Isaac Sim 6.0.1.0's actual dependency declarations. Its Torch/CUDA stack is not the training stack.
- Installation was started after successful resolution. Completion, actual imports, asset conversion and worker execution must be checked separately.

A combined environment did not resolve: the pinned Isaac Lab declares coverage 7.6.1 while Isaac Sim declares 7.4.4; more importantly, Isaac Sim core requires torchvision 0.26.0 whereas the selected Isaac Lab/Torch recipe uses 0.25.0. An exploratory coverage override revealed the second conflict and was removed. The committed solution isolates conversion and training and contains no dependency override. Both environments have independent locks. Isaac Sim is used to generate USD; Isaac Lab uses Newton for physics and Newton Warp for diagnostic offscreen rendering.

The package bootstrap script is named `bootstrap_env.py`: naming it `setup.py` caused setuptools to execute its CLI while building the package. Renaming resolved that build error. No upstream source edits were needed.

User steering: keep the MuJoCo probe queued while developing Isaac integration. This permits implementation before MuJoCo runtime validation; it does not mark the baseline complete or remove either backend from scope.
