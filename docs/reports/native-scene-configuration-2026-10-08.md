# Native scene configuration validation

`omd harness --scene-config` accepts the official CPU apartment and
Isaac/Newton scenes through one packaged JSON Schema. Python uses `jsonschema`;
the pinned native Node deployment uses its actual Ajv 8.17.1 dependency.
Validation precedes native execution. The retained verification source is
`daf5594d9d7bd5294ee9937fa776ced4942cd2e2`.

| Check | Verified result |
| --- | --- |
| Declared scene configurations | Seven actual files accepted by both implementations |
| Invalid scene configurations | 23 variants rejected by both implementations |
| Source tests | 64 passed from a fresh immutable source checkout |
| Public startup | Actual Python CLI and pinned Node server; CPU profile, native tools, registered-package option and `gpt-6.1-sol` / `high` metadata |
| Owned service cleanup | SIGINT, exit code 0, retained lifecycle record, no cleanup timeout |
| Distribution and installation | 431 source/resource files, three licenses and 24 actual CLI calls outside the checkout |
| Installed schema | Seven scenes validated through the independently installed resource |
| Physical source preservation | 345 robot, RL and native execution files unchanged from `b3d3a13e541c0d82edb50ffb89952047e7d0db27` |
| Remote release preparation | Six stages, four scene configurations, all ten official graphs and 29 pinned Harness files; CUDA runtime checks disabled |

CPU scenes select native room, dock or object goals and a finite XY/yaw spawn.
Newton scenes select point goals with optional ordered waypoints, USD provenance
and an XYZ/yaw spawn. Robot variants, device, sensor renderer, bounded hold count,
finite values and positive budgets are validated before startup. CPU CUDA-device
assignment is rejected before loading the native SDK or model provider.

Schema SHA256 is
`01b5f7c01e88df9da9b4de2f6b132f380964ba1919673d1652f454d8ab43f2bd`.
The source audit compares complete Git tree bytes for robot/RL and the native
environment, device, session and worker modules. Their SHA256 is
`019b3623f3077380130618d050af705a493461d128fb170827a1a114f5342461`.
Python source compilation and both Node entry modules pass syntax checks.

| Evidence | Location | SHA256 |
| --- | --- | --- |
| Python and Node decisions | `outputs/acceptance/native-scene-schema-20261008-02/result.json` | `33bcd8981cd2eb4b4a252a576deeddcdfce5a01edfa518f98318840ec8e4e55b` |
| Actual public startup | `outputs/acceptance/native-scene-startup-20261008-01/result.json` | `4d6819265587e7ef8f9d71202fefc5fe4e153fbf90c15efe6609cb14d75a73c3` |
| Independent installation | `outputs/acceptance/native-scene-install-20261008-01/result.json` | `1cf9fa288002afe390d6a6ee7a6790c9a61694da4ab8e639a1c31ed44748b314` |
| Preserved physical source | `outputs/acceptance/native-scene-source-20261008-01.json` | `34e8f46fa86b9090bfc69812b104b164decceb994bb02bc5b214eee11c7c78ed` |
| Remote preparation | `outputs/acceptance/native-scene-preflight-20261008-01/preflight/result.json` | `8ccc2ffbbbe9f049e7704f2024c8132b3da470cdb7f3c98496cc24698c7bf210` |
| Wheel | `outputs/packages/native-scene-20261008-01/oh_my_duck-0.1.0-py3-none-any.whl` | `5758570c8d2a5e7393543d6ec7519e7ed7e1fc332db20ba9be0359b46c3d01cc` |
| Source distribution | `outputs/packages/native-scene-20261008-01/oh_my_duck-0.1.0.tar.gz` | `a5099577353e437aab429b86a24d37faf43802a6c723453ba6b1840a25171d19` |

The startup check reads the actual native `/api/config` and preserves its original
metadata and process record. It performs no model inference or physical task.
Installed audits also recompute retained official ONNX and continuous physical
records. Current policy execution and package integration have their separately
recorded CPU evidence. Newton runtime, new model-task results and hardware
require their own acceptance. GPU acceptance and RL remain stopped.

Usage and field responsibilities are documented in
[native deployment](../harness-native-integration.md#场景配置).
