# External scene CPU validation

Immutable source `cbf0c19a53da48c77dd7ca2b7f4baaf8b6433aa6` passed actual
Office and Hospital import and contact-configuration checks in
`.envs/isaac-newton-preflight-20261008-01`. Both runs explicitly disabled CUDA,
verified the unique OpenUSD provider and all downloaded source files, and composed
the actual environment and prepared official robot through USD references.

The actual Newton imports used the pinned Isaac Lab manager's Newton and PhysX
schema resolvers. Environment geometry was global and the robot occupied one
Newton world. `align_collision_model` applied the source contact filters and
scene transformations; each actual model finalized on CPU.

| Measurement | Office / standard feet | Hospital / rollers |
| --- | --- | --- |
| Downloaded files verified | 2291 | 1639 |
| Geometric colliders imported and checked | 3646 | 2036 |
| Imported environment meshes / source points checked | 3645 / 652517 | 2035 / 1448652 |
| Signed-scale transformations checked | 23 | 141 |
| Official robot colliders checked | 70 | 13 |
| Maximum world-coordinate bounding-box error | 0.000025749 m | 0.000002304 m |
| Native bodies / shapes / joints | 15 / 3793 / 15 | 19 / 2132 / 19 |

Every source geometric collider was present in the actual native import.
Each mirrored mesh preserved its world positions, winding and normals. Imported
mesh bounds were independently computed from original USD points and world
transforms, with a `0.0002 m` absolute tolerance. Environment world identity and
collision group zero, source-required robot filters, empty flat-ground pair
tables, finalized filter sets and collision-group arrays passed exact checks.
All source and generated files retained their SHA256.

Hospital also contains 125 Xforms with CollisionAPI annotations. Each Xform had
enabled geometric collider descendants, and every such descendant was checked
against the actual native import. The geometric collider count is 2036;
`collider-annotations.json` retains the complete hierarchy observations.

| Artifact | Location | SHA256 |
| --- | --- | --- |
| Office audit | `outputs/acceptance/external-scene-office-20261008-03/independent-audit.json` | `a73732f7a7e1ab0d0f07896307b6713a78484183e6010492acb28b9a5d1d7601` |
| Hospital audit | `outputs/acceptance/external-scene-hospital-20261008-02/independent-audit.json` | `e7c855b9db575fb275129b5231ff646741768bdff6774a4bfb60798944ee12a6` |
| Audit driver | `.cache/audit_external_scene.py` | `300660878ca8408bf1e477384465c4d9be3316ab23360c2e96932470ec0a2834` |

Raw native diagnostics and all attempted output directories are retained.
Four actual USD imports and two finalized CPU models passed. Warp exposed only
CPU and Torch CUDA remained uninitialized. Both audit processes exited.
Status was independently compared with the actual artifacts in
`outputs/acceptance/external-scene-records-20261008-01.json`; process absence was
recorded in `outputs/acceptance/external-scene-process-audit-20261008-01.log`.
This verifies imported scene geometry and contact configuration. Native Isaac
callback execution, solver stepping, BAM loads, rendered sensors and navigation
behavior require their own execution evidence. GPU acceptance and RL remain stopped.
