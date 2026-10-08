# CPU Newton asset readiness

The locked Linux x86_64 kitless environment uses `usd-exchange==3.0.0` as the unique
provider of `pxr` and OpenUSD 26.08. This version includes the native multithreaded
physics parser correction documented by [Isaac Lab](https://isaac-sim.github.io/IsaacLab/develop/source/refs/issues.html).
The explicit Linux provider override preserves the pinned Isaac Lab checkout and
all other resolved dependency versions. Asset conversion remains in its separate
locked environment.

`infrastructure/usd_runtime.py` verifies namespace ownership and every hashed
installation file through `importlib.metadata`. Doctor checks and online backend
initialization perform this verification before CUDA. Missing files, altered bytes,
multiple providers or an unexpected version terminate execution.

`omd setup --backend isaac-newton` includes native Harness dependencies.
`--rl-framework sb3` also selects locked SB3 dependencies. Setup reinstalls the
selected OpenUSD provider, verifies its files and records the result in
`artifacts/environments/isaac-newton/setup.json`. Installer temporary files use
`.cache/tmp`.

`task_binding/collision_assets.py` owns source MJCF compilation, USD instance
expansion, source geometry mapping and physics material preparation. It imports
only the standard library until called. `task_binding/collisions.py` owns simulator
spawning and the Newton callback for geometry orientation, contact filters, ground
pairs and solver parameters. The callback and preparation semantics match the
baseline AST; geometry mapping also matches execution of the actual baseline code.

Actual validation uses immutable implementation
`d39d5df1ed04b474d9866e90e6f421e422ab8901`. The dedicated remote environment is
`/home/jixin/workspace/code/Oh-My-Duck/.envs/isaac-newton-usd-exchange-20261008-01`.
The shared canonical environment retained the hashes of its 117 recorded namespace
files. Unrelated processes were preserved. CUDA visibility was empty throughout.

| Actual check | Verified result |
| --- | --- |
| Installation and native USD | One provider, 244 verified installation files, actual OpenUSD 26.08 |
| Actual setup function | RSL-RL → SB3 → RSL-RL passed with native Harness dependencies |
| Initialization checks | Doctor and online backend reject incompatible ownership before CUDA and Isaac Lab imports |
| Robot composition | Allcollisions: 742 prims / 794304 mesh points; rollers: 495 prims / 490966 mesh points; nine resolved layers each |
| Collider authoring | Allcollisions: 70 enabled source geoms / 280 expanded instances; rollers: 13 geoms / 178 instances |
| Materials and contact attributes | Source friction, physics binding, restitution, `condim`, priority and hull attributes verified for every enabled collider |
| Geometry preservation | All 229 meshes, 1285270 points, world transforms and generated asset hashes unchanged |
| Actual Newton CPU imports | Allcollisions: 15 bodies / 147 shapes / 15 joints; rollers: 19 bodies / 96 shapes / 19 joints; all 14 servos retained |
| Repeated native imports | Three independent processes / six imports passed with 256 OpenUSD worker threads; all per-process audit hashes match |
| Execution boundary | Warp devices: CPU only; Torch CUDA initialization false |
| Actual imports | Newton, online backend, native worker, perception client/validation and task simulation binding |
| Release preparation | Six stages, four scene configurations, ten policies, 29 pinned Harness files, complete Office/Hospital inventories |
| Independent installed package | 420 source/resource files, three licenses and 19 CLI calls outside the checkout |
| Saved policy and physical evidence | 1208 ONNX actions / 1220 serialized updates, zero action error and complete five-motion audit |
| Configuration and lazy imports | 23 tests passed; optional execution libraries remain unloaded |

Asset edits are authored in anonymous USD session layers. Each native import uses
actual `ModelBuilder.add_usd`; the original generated files are unchanged. Native
diagnostics are retained alongside the reports. Fresh GPU solver execution,
behavior, rendering and hardware require their separate acceptance. GPU acceptance
and RL remain stopped.

| Artifact | Location |
| --- | --- |
| Actual environment update | `outputs/acceptance/usd-environment-update-20261008-05/independent-audit.json` |
| Runtime and composition audit | `outputs/acceptance/usd-provider-runtime-20261008-06/independent-audit.json` |
| Native collision preparation and stability | `outputs/acceptance/collision-assets-stability-20261008-03/independent-audit.json` |
| Release metadata preflight | `outputs/acceptance/collision-assets-preflight-20261008-02/preflight/result.json` |
| Independent installed package | `outputs/acceptance/collision-assets-package-20261008-02/result.json` |

| Artifact | SHA256 |
| --- | --- |
| Environment update | `a219a3b99a8359b0a6bb2494a07d15ea09179ba3be36aac27fd78f0a4df488e1` |
| Runtime/composition audit | `fcae7b7ab30b44f335535831d94937f859d9284446b7fd8a07cb190bae382fc6` |
| Complete native stability audit | `851d5f08f5a20fb617fba63a57762c9863d77dbe047d7ee75b5d6c60929f84ee` |
| Per-process collision audit | `3fb1c1a3465c427f1388b83ec6c09a039424a33b1aef2b6191a8363f6d763c71` |
| Metadata preflight | `2575339eba881f8c50c4fe60aac02cfa90d7ebcba702917bf19cdb9cbca00107` |
| Installed audit | `e643016c4a7d19e3cb705578c884c7222254228029bd56197af6c1c431fdc5fe` |
| Wheel | `231a180fc3f1c1933ac5b4973f655e90b2819562c075861c494d9ec96cd67ce0` |
| Source distribution | `6d7cc45b52594ef24128c52bbf2269ddd5e74e45a4d5512abcadb051c29be5d3` |

Build, installation, native import, subprocess and saved-evidence logs are retained
beside these results. Output paths are unique.
The remote process audit confirmed the CPU audit processes had exited. No GPU
execution or RL process was started by this validation.
