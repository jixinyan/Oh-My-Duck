# Native acceptance modules and CPU verification

Application entry points are `omd validate`, `omd replay` and `python -m oh_my_duck`.
The [source map](../../src/oh_my_duck/validation/README.md) associates each command
with its implementation. Metric, native Harness and release checks use separate
package directories. Thirteen existing executable scripts dispatch to these
modules; configuration schemas import plan definitions directly. The `validation`
extra declares JSON Schema, image, NumPy, ONNX and ONNX Runtime dependencies.

## Source and execution

| Evidence | Immutable source | Saved result |
| --- | --- | --- |
| Actual CPU continuous motion and independent audit | `a3cbcc160ca9bb25a12e350d0125e877e53b0934` | `outputs/acceptance/refactor-cpu-sequence-20261008-01/independent-audit.json` |
| Control-channel closure during actual motion | `a3cbcc160ca9bb25a12e350d0125e877e53b0934` | `outputs/acceptance/refactor-cancel-eof-20261008-01/acceptance.json` |
| SIGTERM during actual motion | `a3cbcc160ca9bb25a12e350d0125e877e53b0934` | `outputs/acceptance/refactor-cancel-sigterm-20261008-01/acceptance.json` |
| Remote dependency, model and scene metadata | `a3cbcc160ca9bb25a12e350d0125e877e53b0934` | `outputs/acceptance/refactor-preflight-20261008-02/preflight/result.json` |
| Built distributions and independent installed commands | `19e594b2dbd5170ed3dd0f1929354232aab5dbaf` | `outputs/acceptance/refactor-installed-package-20261008-02.json` |

The CPU session executed 1208 controls and 4832 physical substeps through the
native ActionGate, official `velstand` and `alpha_walking`, and actual MuJoCo/BAM.
Each endpoint has measured upright stopping, zero external obstacle contacts and
the original 0.05 m / 5° tolerance. Native termination and resource release passed.

| Requested motion | Independently recomputed final error |
| --- | --- |
| Walk +1.0 m | 0.021102 m |
| Rotate −78° | 4.496450° |
| Walk −0.5 m | 0.035872 m |
| Rotate +78° | 4.712026° |
| Walk −0.5 m | 0.040380 m |

All 1208 admitted 61→14 actions were independently recomputed from actual recorded
inputs. Maximum action error was zero; the 1220 serialized native updates preserved
matching inference identity. Schema 2 provenance verifies `metric/case.py` and its
compatibility entry point against the recorded Git revision. The two cancellation
cases executed 84 and 86 controls, returned supervisor status 130, preserved the
aborted case and released the native session within their graceful cleanup interval.

## Installed and saved-evidence checks

The independent installation verifies 412 tracked source/resource files and three
licenses against the wheel, source distribution and installed package. Nineteen
CLI calls passed outside the checkout with `PYTHONPATH` unset, including actual
release-plan checking, ONNX action recomputation and physical campaign verification.
Additional reports are `refactor-installed-package-20261008-02-policy.json` and
`refactor-installed-package-20261008-02-metric.json` in `outputs/acceptance/`.

The complete recorded CPU three-case matrix passed at
`outputs/acceptance/refactor-historical-matrix-20261008-01.json`. The recorded Hospital
navigation run `cf059bda-3f7e-4294-93e7-8b5fe9d70df1` passed the new audit: 380 original
observer frames, 5.438808 m summed walking-segment displacement, 0.095637 m final
goal error, zero external obstacle contacts, matching native stopping and independent
Verifier identity. Its report is `outputs/acceptance/refactor-historical-navigation-20261008-01.json`.
This checks saved physical evidence from the existing run.

Twenty-one configuration and import checks passed. Configuration, control transport,
replay export and dispatcher imports leave execution libraries unloaded. Eight
environment lock checks passed locally; package-extra metadata is synchronized.
Python compilation and the native server syntax check passed.

Source `19e594b2dbd5170ed3dd0f1929354232aab5dbaf` also passed compilation of all Python
modules and scripts in the remote Linux environment, and actual native Harness
and perception-client imports with CUDA visibility empty. The clean-source output
is `outputs/acceptance/refactor-linux-imports-20261008-01.txt`.

The remote metadata check passed six declared stages, four scene configurations,
nine input-file hashes, 29 pinned native Harness source files, all ten official
policy models and the complete Office/Hospital resource inventories. Its report
records `cuda_runtime_checked: false`; the campaign records `physical_gpu: null`
and `gpu_acceptance_performed: false`.

## Artifact identity

| Artifact | SHA256 |
| --- | --- |
| CPU independent audit | `24d3573cda23c5711ef3ad9a1b5f0ac65c7dd268b5097ca5bd3d272d93fe3313` |
| Installed package report | `d9762360893b911fbd6d13283b9040285805e5cd605478768b394985468a704a` |
| Remote metadata result | `1a15179aec60b1b5bfe880f7bd66209cc658058ce1b3a33ba8229c6f304adc16` |
| Recorded Hospital navigation audit | `d5c66b5a22b21c55ee7307829b6ca2d8bf6ac7cb058fde1e08ac349675c9f08b` |

GPU acceptance and RL remain stopped. Current Newton motion, fresh kernel
initialization, complete Office navigation, image-only VLN, perception accuracy,
object interaction, trained-policy behavior and hardware retain their separate
acceptance requirements in [release readiness](../release-readiness.md).
