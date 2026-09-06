# Third-party sources

Downloaded upstreams and model weights are stored outside the Git index. The source-of-truth revisions are in `configs/upstream.json`; installed transitive dependencies are fixed by the upstream lockfile and recorded by each worker.

| Source | Upstream declaration / location |
|---|---|
| Pollen Robotics microduck_rl | Code: Apache-2.0 in `LICENSE`. README separately declares 3D model files Creative Commons BY-SA-NC; preserve the original asset notices and inspect exact asset terms before redistribution. |
| Pollen Robotics microduck | Workspace code: Apache-2.0. Vendored ToF drivers and bundled assets retain their own notices. |
| Rhoban BAM | Retain its repository license and model/data notices; installed from the upstream lockfile's Git revision. |
| pollen-robotics/microduck-policies | Public model card declares Apache-2.0; save the README and immutable model revision with downloads. |
| kabilankb/isaaclab-microduck | Static design reference, pinned separately in `configs/upstream.json`; repository declares Apache-2.0. Bundled Microduck assets retain their upstream provenance and require separate notice review before redistribution. The nested rigid-body transform correction is adapted in `training/isaac_newton/oh_my_duck.rl.backends.isaac_newton/convert_asset.py`; retain [its Apache-2.0 license](docs/third_party/isaaclab-microduck-LICENSE.txt). Robot assets are generated from the separately pinned official source and are not committed. |
| Isaac Lab / Newton | Candidate integration only; preserve their own licenses and transitive dependency/asset notices when installed. |

This file is a provenance inventory, not a claim that one code license covers every model, mesh, driver or dataset. No license for this project's original code is selected by this inventory.
