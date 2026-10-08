# CPU Newton collision model validation

`task_binding/collision_model.py` configures the actual Newton builder using
official MJCF contact masks, explicit ground pairs and scene geometry corrections.
Its module imports only the standard library. `task_binding/collisions.py`
registers the native Isaac model-initialization callback and passes the active
builder to `align_collision_model`. The calculation body matches the preceding
implementation's AST.

Immutable source `8858d9da5486d34f7d77f3793d7dbcb4adcfa50f` passed actual CPU
validation in `.envs/isaac-newton-preflight-20261008-01`. Each official robot asset
was prepared in an anonymous USD session layer and imported into two actual
Newton worlds. The real SolverMuJoCo custom attributes were registered, the
collision configuration was applied, and both models finalized on CPU.

| Model | Colliders per world | Explicit ground pairs | Source-required filter pairs | Native bodies / shapes / joints |
| --- | --- | --- | --- | --- |
| `allcollisions` | 70 / 70 | 138 | 140 | 30 / 295 / 30 |
| `groundcontact_rollers` | 13 / 13 | 24 | 26 | 38 / 193 / 38 |

Every source-enabled collider was verified in each world. The resulting filter
set exactly equals the native importer filters plus the source-required filters;
native totals were 3454 and 404. All ground-pair world/shape indices, `condim`,
five-component friction, `solref`, `solimp`, margin and gap matched the official
source exactly in the builder and within `1e-7` in finalized CPU arrays. The
ground collision group was checked, and every generated asset file retained its
original SHA256.

Four actual imports and two finalized models passed with Warp CPU only and Torch
CUDA uninitialized. The calculation AST was independently compared with source
`dd5a1fef3b3ead311c31e72eeec840f7f51d37dc`. All 27 configuration/import tests and
source compilation passed. Independent installation outside the checkout checked
422 files, three licenses and 20 actual CLI calls, including 1208 saved ONNX
actions with zero error and the continuous five-motion physical record.
The same fixed source passed all six release metadata stages, four scene
configurations, ten official policies and 29 pinned Harness files without CUDA.

| Artifact | Location | SHA256 |
| --- | --- | --- |
| Actual collision model audit | `outputs/acceptance/collision-model-20261008-01/independent-audit.json` | `53c262a05da8e5bc6f88e3555d6bd4af03430443a1cdce9c7506138b38814264` |
| Independent installed audit | `outputs/acceptance/collision-model-package-20261008-01/result.json` | `5d6795e1d3532031c4740a34572af688a37a8b4314896dc26ac2fbfb17e8c6a8` |
| Installed import audit | `outputs/acceptance/collision-model-package-20261008-01/boundary-audit.json` | `9095e0658cd68db5c44ebfa196e518ebaa4ea3471fad0a769c27f0b98c429506` |
| Complete release preflight | `outputs/acceptance/collision-model-release-preflight-20261008-01/preflight/result.json` | `011e5daf31fee7b1e6b18a52e17466ec667cfd49a81285315516809b1580c83b` |
| Wheel | `outputs/packages/collision-model-20261008-01/oh_my_duck-0.1.0-py3-none-any.whl` | `8317de51e5362c622bfe3ebbe27bda4edc2e92e9047da780c657459f66c04f6f` |
| Source distribution | `outputs/packages/collision-model-20261008-01/oh_my_duck-0.1.0.tar.gz` | `d511377b5f5de11629550598cf4f769129980a3c1c235d9a5dc0a993d4c92b49` |

Raw native diagnostics, build/install logs and saved-evidence checks are retained
beside these results. This validates actual builder configuration and finalized
model data. Solver stepping, native Isaac callback execution, external scene
geometry, BAM loads, rendering and learned-policy behavior retain their own
acceptance requirements. GPU acceptance and RL remain stopped.
All audit processes exited. The remote process observation is retained in
`outputs/acceptance/collision-model-process-audit-20261008-01.log`.
