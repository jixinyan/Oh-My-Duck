# Native application interfaces — 2026-10-08

Verified source: `d54040d7d1da0e41ca4e877891dbc57c9c73bd23`.
Local source: `.cache/source-native-application-20261008-01`.
Remote source: `/home/jixin/workspace/code/Oh-My-Duck/.job-sources/native-application-20261008-01`.
Native EDH: `8a5e685b22d032207f53db20454f0992a4ad60fd`.

## Application and tools

`ApplicationServices.harness` accepts the `HarnessBridge` protocol implemented
by `integrations/native_client.py`. Operations are `open`, `submit`, `status`,
`wait`, `stop` and `close`. Voice and application composition use this same
client. The native server owns sessions, planning, tool registration, execution
and experience records. Run records retain their native state and identity;
stopping requires terminal execution and device confirmation.

`ToolCatalog` uses JSON Schema Draft 2020-12 through `jsonschema`, available in
the `validation` extra and locked execution environments. Registration checks
the schema; invocation checks finite JSON arguments before the actual handler.
The seven simulation tools declare required arguments, additional-property
rules and movement ranges. References, conditional requirements, positional
arrays and dependent requirements use the standard JSON Schema implementation.
The application and tool source map is [agentic/README.md](../../src/oh_my_duck/agentic/README.md).

## Executed verification

| Verification | Actual result |
| --- | --- |
| Targeted source tests | 77 passed; 33 subtests passed |
| Native HTTP session | Actual public CLI, pinned Node server and remote CPU MuJoCo/BAM environment |
| Application interface | Actual `NativeTaskClient` attached to `ApplicationServices`; native task catalogue read; undeclared task rejected; idle stop returned |
| Session close | `state=closed`, `resources=released`, task history count 0 |
| Resource release | Server PID 68088 exited with code 0; cleanup 0.039760 s; no remaining native worker |
| Independent installation | Wheel and source distribution match 430 tracked source/resource files and three licenses; 24 public CLI calls pass outside the checkout |
| Installed interfaces | Actual native client, capability handler, seven schemas and invalid movement rejection; simulator/model libraries uninitialized |
| Physical source | 345 robot, RL, environment, device, session and worker files unchanged from `b3d3a13`; native task and voice client implementations also unchanged |
| Remote preparation | Six stages, four scene configurations, ten official policies, 29 pinned Harness files and actual provider metadata pass; `cuda_runtime_checked=false` |

The CPU worker uses the actual isolated Python environment and OSMesa with
`CUDA_VISIBLE_DEVICES=''`, `LIBGL_ALWAYS_SOFTWARE=1` and `GALLIUM_DRIVER=llvmpipe`.
Native initialization loaded the original 61-input/14-output official graph and
14 BAM M6 actuators. This session verification covers service lifecycle and task
catalogue admission. Model-task and physical-task acceptance remain independent.
GPU acceptance and RL stay stopped.

The installed schema library is `jsonschema==4.26.0`; the locked remote runtime
uses `4.25.1`. Both execute the same Draft 2020-12 validation. Seven installed
tool definitions have SHA256
`1041f6d31efbaa5b7d4ab626ad7b453e13d5606fea283dc47bb4d1fc6fc27ad8`.

## Artifacts

| Artifact under the repository root | SHA256 |
| --- | --- |
| `outputs/acceptance/native-application-tests-20261008-01.log` | `49b7765b2fdbf42df6348d4fee1c33e1509a68dab042ad56af7a3826afa8a272` |
| `outputs/acceptance/native-application-session-20261008-01/result.json` | `9c5e58cad1a39a39fc777fdefff33ca58064aaff9389d3d32ab489b16c651aa8` |
| `outputs/acceptance/native-application-session-20261008-01/independent-audit.json` | `8554c925566ac26a562297d7a37dbc4db164e74b70e3224d85d12f1bbfabb2ba` |
| `outputs/acceptance/native-application-session-20261008-01/process.json` | `9395c7c856690521f5918015b8c5c800ea9923a230f52ead63a44428c0b8bad1` |
| `outputs/acceptance/native-application-install-20261008-01/result.json` | `c90e45ecd9a69b0930eb46cf3c5370db2a1bc709f5387dcc159b82514c2be585` |
| `outputs/acceptance/native-application-install-20261008-01/interface.json` | `ad16febe1c5637a9cd2838e70d48c6ac2aaaf2434acd53fc81c6b924e7c7df94` |
| `outputs/acceptance/native-application-source-20261008-01.json` | `c3dfa6ae38e4655d3bfffa7d24def8f298fb2f3da562f21f17d56dcab0f499ed` |
| `outputs/acceptance/native-application-preflight-20261008-01/preflight/result.json` | `2f1e7aa6f7752c6f475564d8e03ceb2f439928d4dd5a89dacacd162154f564ca` |

Wheel SHA256:
`f8c47e30272935b6bed18c24c22adf3d47277f4c5f116ee540cf266295b8aa19`.
Source distribution SHA256:
`0907fe2f6b5a6cda274ab90d4f3ce0202398cc98aa19217bce448951248bb31a`.
Native physical tree SHA256:
`019b3623f3077380130618d050af705a493461d128fb170827a1a114f5342461`.

Reproduce the source checks in the locked CPU environment:

```sh
CUDA_VISIBLE_DEVICES='' TMPDIR="$PWD/.cache/tmp" PYTHONPATH=src python -m pytest \
  tests/test_tool_schemas.py tests/test_harness_interface.py tests/test_framework.py \
  tests/test_native_scene_configuration.py tests/test_policy_registry_cli.py \
  tests/test_release_plan.py tests/test_validation_imports.py -q
```

Actual native deployment follows the [deployment guide](../harness-native-integration.md).
Independent installed verification uses `omd validate package-audit` with a
fresh installation and preserved actual policy/metric records.
The [CPU readiness record](cpu-development-readiness-2026-10-08.md) and
[runtime campaign](../runtime-release-acceptance.md) define subsequent acceptance.
