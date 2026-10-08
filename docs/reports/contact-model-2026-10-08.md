# Official contact model CPU validation

`task_binding/contact_model.py` configures the actual solver model from the
official MJCF geometry. It applies source contact masks, material parameters and
body masks. Its module imports only the standard library. The native manager
calls `configure_contact_model` before compiling MuJoCo Warp contact tables and
capturing CUDA graphs. The calculation body matches the preceding manager's AST.

Immutable source `425bb89696768f6c560970298a9dd9c40c8ac3a8` passed four actual
Newton SolverMuJoCo constructions and four native MuJoCo Warp model compilations
on CPU, with the locked environment `.envs/isaac-newton-preflight-20261008-01`.
The constructor used the library's explicit `use_mujoco_cpu=True` setting.
Robot USD preparation, Newton imports and finalized models used the actual
converted official assets; scene checks used the downloaded Office and Hospital.

| Model | Robot geoms | Environment / ground geoms | Bodies / joints |
| --- | --- | --- | --- |
| Standard feet, flat ground | 70 | 1 | 16 / 15 |
| Rollers, flat ground | 13 | 1 | 20 / 19 |
| Standard feet, Office | 70 | 3646 | 16 / 15 |
| Rollers, Hospital | 13 | 2036 | 20 / 19 |

Every source-enabled robot geom was checked. All ten geometry contact fields
matched the official source exactly. Body masks were independently reconstructed
from their actual geometry membership. Scene masks were one; scene material
fields retained their native values. All 21 protected model fields, including
inertia, joint indexing, geometry transforms and explicit pair data, remained
unchanged. The twelve native MuJoCo Warp geometry/body arrays matched the actual
configured model within `1e-7`. Source and generated asset hashes were unchanged.
The flat-ground models preserved 69 and 12 explicit ground pairs, respectively.

All 28 configuration/import tests and source compilation passed. Fresh wheel
and source distribution installation outside the checkout checked 423 files and
20 CLI calls, including actual saved ONNX and physical-record verification.
The installed contact module left execution dependencies unloaded. The same
fixed source passed six release-preflight stages, four scene configurations,
ten official policies and 29 pinned Harness files with CUDA uninitialized.

| Artifact | Location | SHA256 |
| --- | --- | --- |
| Flat-ground solver models | `outputs/acceptance/contact-model-20261008-01/independent-audit.json` | `019505377b799f1cd02ad32744d4bd3acccf23b6994f6e3fcaec12125759ad5b` |
| Office solver model | `outputs/acceptance/contact-model-office-20261008-01/independent-audit.json` | `500d8733b477e64406cc433bf0c05b8e6785f918033a0fa8cb84cdb21a83d872` |
| Hospital solver model | `outputs/acceptance/contact-model-hospital-20261008-01/independent-audit.json` | `585d0ac942161b0d5db0d13518f216cd82c74f55661f0037f93fc695b4fd7cd8` |
| Independent installed audit | `outputs/acceptance/contact-model-package-20261008-01/result.json` | `83b29c67c2210a9aa09fe74e61795d611964e34bdec364ec0dba29552e59233a` |
| Installed import audit | `outputs/acceptance/contact-model-package-20261008-01/boundary-audit.json` | `a1c7484a90923eff0bde3500271a41978a5ea02b0f7a5ff97aadc137042a425d` |
| Complete release preflight | `outputs/acceptance/contact-model-release-preflight-20261008-01/preflight/result.json` | `50e0edd4cdf872708a4a42040b60721aa23b84d45018fbc00e11968c0e6b4bf4` |
| Wheel | `outputs/packages/contact-model-20261008-01/oh_my_duck-0.1.0-py3-none-any.whl` | `6efde6fc910c1833c1d2904366ecb47d8e0a6d132a55cbec66f9ee2f8693ce3c` |
| Source distribution | `outputs/packages/contact-model-20261008-01/oh_my_duck-0.1.0.tar.gz` | `23c70d0d5e1725f0865390f64c055d42b8d4a5d5edbd214609fbcb20e4ad0430` |

All audits exposed Warp CPU only and left Torch CUDA uninitialized. Raw native
diagnostics and installation logs are retained. Processes exited; absence is
recorded in `outputs/acceptance/contact-model-process-audit-20261008-01.log`.
Project status was independently checked against all native, installed-package
and release-preflight results in `outputs/acceptance/contact-model-records-20261008-01.json`.
This validates actual model configuration and native compiled data. Native Isaac
callback execution, CUDA graphs, solver stepping, BAM loads, rendering and
task behavior require their own execution evidence. GPU acceptance and RL remain stopped.
