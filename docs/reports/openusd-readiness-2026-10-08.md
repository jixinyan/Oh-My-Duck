# OpenUSD readiness before Newton execution

The locked Linux x86_64 Newton environment uses `usd-core==25.11` as the unique
provider of `pxr`. The explicit architecture override selects `usd-exchange` on
ARM; the project's supported Newton environment remains Linux x86_64. All other
resolved versions remain unchanged. The pinned Isaac Lab checkout is pristine.
NVIDIA documents the shared native-file requirements in its
[USD deployment guide](https://docs.omniverse.nvidia.com/usd/code-docs/usd-exchange-sdk/latest/docs/deployments.html).

`infrastructure/usd_runtime.py` checks namespace ownership and every hashed
installation file through `importlib.metadata`. `doctor --runtime isaac-newton`
performs this check before CUDA initialization, including metadata-only mode.
The online Newton backend checks it before importing Isaac Lab or initializing
CUDA. Missing files, changed bytes, multiple providers or an unexpected version
terminate execution.

`omd setup --backend isaac-newton` includes the native Harness dependencies.
`--rl-framework sb3` additionally selects the locked SB3 dependencies. Setup
reinstalls the selected OpenUSD provider, verifies its files, and records the
verified provider in `artifacts/environments/isaac-newton/setup.json`.
Temporary installer files use `.cache/tmp`.

Actual validation used immutable source
`4bb90aa79c9fdb00045e652862e8acb06f7d23ef`. The independent remote environment is
`/home/jixin/workspace/code/Oh-My-Duck/.envs/isaac-newton-usd-core-20261008-01`.
The shared canonical environment and unrelated processes were preserved.
CUDA visibility was empty throughout.

| Actual check | Verified result |
| --- | --- |
| Installation records and native USD | One provider, 123 verified installation files, actual USD version 25.11 |
| Actual setup function | RSL-RL → SB3 → RSL-RL installation selections passed; native Harness dependencies verified |
| Existing coinstalled environment | Doctor and online backend rejected it before CUDA or Isaac Lab import; original 117 recorded namespace files retained their hashes |
| Robot composition | Allcollisions: 742 prims and 794304 mesh points; rollers: 495 prims and 490966 mesh points; nine resolved layers per model |
| Actual Newton USD import on CPU | Allcollisions: 15 bodies, 147 shapes, 15 joints; rollers: 19 bodies, 96 shapes, 19 joints; all 14 canonical servos present |
| Execution boundary | Available Warp devices: CPU only; Torch CUDA initialization remained false |
| Actual imports | Newton, online backend, native worker, perception client/validation and task simulation binding |
| Complete release preparation | Six stages, four scene configurations, ten official policies, 29 pinned Harness files, complete Office/Hospital inventories |
| Independent installed package | 419 source/resource files, three licenses and 19 installed CLI calls outside the checkout |
| Actual saved policy and physics evidence | 1208 ONNX actions and 1220 serialized updates, zero action error, complete five-motion physical audit |

The Newton import check exercises actual `ModelBuilder.add_usd` on the CPU. Its
native import diagnostics are retained. Online collision materials, filters and
contact parameters are prepared through the owned `spawn_official` and
`align_collisions` implementation. Fresh solver execution, behavior, rendering and
hardware require their separate acceptance. GPU acceptance and RL remain stopped.

| Artifact | Location |
| --- | --- |
| Actual environment update | `outputs/acceptance/usd-environment-update-20261008-03/independent-audit.json` |
| Runtime and composition audit | `outputs/acceptance/usd-provider-runtime-20261008-04/independent-audit.json` |
| CPU Newton importer | `outputs/acceptance/newton-usd-import-20261008-04.json` |
| Complete metadata preflight | `outputs/acceptance/usd-provider-preflight-20261008-03/preflight/result.json` |
| Independent installed package | `outputs/acceptance/usd-provider-package-20261008-03/result.json` |

Build, installation, native import, subprocess and saved-evidence logs are retained
beside these results. Output paths are unique.

| Artifact | SHA256 |
| --- | --- |
| Environment update | `7bf4e662010e3a7b4371c7a9d380f0247fd603cae41a39a371b3d04e7666a982` |
| Runtime/composition audit | `8c7149effc11a18746a1188215889dd02c6c8d3ca95f30a0821fbb2e9d30ff24` |
| CPU Newton importer | `bc3a5e197bc527596a11f4f6de1c45294b27e6e69c3752631d1e3be97fbee137` |
| Metadata preflight | `eedd698006bef5798070468de4d943636642cade4784570cce8fc89e2b617147` |
| Installed audit | `a2cde3ea333a1779ee7e71895eb3edf5d6b5e93c41c357cbc266f61b952a70d7` |
| Wheel | `1350719525ee8835e000fec8b2ff57807433669930d08bc2eeecef751b059f98` |
| Source distribution | `a5bed2d5d26f3ad1f170907f8782d7b7403f35b52c7a113542be9f5a38af7643` |
