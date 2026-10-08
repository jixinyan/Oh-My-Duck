# Public CPU model asset validation

`omd validate model-assets` checks actual robot USD preparation and native Newton
imports from the locked Newton environment. Both standard-foot and roller variants
are checked by default. `--models` selects explicit variants and `--repeat` repeats
their checks in the same process. CUDA visibility must be explicitly empty, and
the output directory must be new.

```bash
CUDA_VISIBLE_DEVICES='' omd validate model-assets --repeat 2 \
  --output outputs/acceptance/model-assets-unique-run
```

The implementation is `validation/release/assets.py`. Its module and help entry
point load without execution dependencies. Actual invocation verifies the unique
OpenUSD provider, generated asset files, source MJCF colliders and joints, USD
friction/contact attributes, physics material bindings and Newton imports.
Anonymous session layers contain the authored changes. All source mesh points,
world transforms and generated files retain their original contents.

Immutable source `1abea13d4bcd084b08c0065d1673f9630d2a9a41` passed actual validation:

| Check | Result |
| --- | --- |
| Native public CLI | Two repetitions / four actual Newton imports passed |
| Standard-foot asset | 70 source colliders, 140 meshes, 14 nonfree source joints |
| Roller asset | 13 source colliders, 89 meshes, 18 nonfree source joints including four passive wheel joints |
| Runtime | OpenUSD 26.08, 244 verified provider files, 256 worker threads, Warp CPU only, Torch CUDA false |
| Progress records | Eight ordered start/verification records independently checked |
| Output and argument protection | Actual invalid repeat, repeated model, absent CPU mask and existing-output invocations rejected; prior output hashes preserved |
| Actual installed dependency failure | Missing provider terminates before model progress, preserves the failed result and leaves execution dependencies unloaded |
| Import/configuration tests | 24 tests passed |
| Independent package | 421 source/resource files, three licenses and 20 installed CLI calls passed outside the checkout |
| Saved policy/physics audit | 1208 ONNX actions / 1220 serialized updates, zero action error, five-motion physical audit passed |

The result records source/configuration hashes, actual dependency versions,
generated-file hashes and every verified model. `progress.jsonl` retains completed
checks if execution terminates; Python failures also retain `result.json` and
propagate to the caller. Actual solver execution, policy behavior, rendering and
hardware remain separate acceptance. GPU acceptance and RL are stopped.

| Artifact | Location | SHA256 |
| --- | --- | --- |
| Actual CLI result | `outputs/acceptance/model-assets-cli-20261008-01/result.json` | `0302ca839f6bdf8d8f6e14cba69e51bdc3a46ccb91070088602f85206cf19287` |
| Independent boundary audit | `outputs/acceptance/model-assets-cli-boundaries-20261008-01/independent-audit.json` | `e8b65a85201e0161604d4555cbd8e7fe034bffa7c3fff6413a359a4ea9dbf216` |
| Installed failure audit | `outputs/acceptance/model-assets-cli-installed-boundaries-20261008-01/independent-audit.json` | `c27ad60436468f406d46aed643c91983c605e05e682f03230b7c5ea331c2161f` |
| Installed package audit | `outputs/acceptance/model-assets-cli-package-20261008-01/result.json` | `a877f6dac5c8f476b21a5cb9f1d3d3fcd930292d2d46256e6528240fd38d455b` |
| Wheel | `outputs/packages/model-assets-cli-20261008-01/oh_my_duck-0.1.0-py3-none-any.whl` | `4c511942e28f527d25f162358e11fbdb90b76e5be6fd91b3ff3ab9cd455979bb` |
| Source distribution | `outputs/packages/model-assets-cli-20261008-01/oh_my_duck-0.1.0.tar.gz` | `4af884627e78af9333896b4b851f4b9ae482c68c1ab5bec86b70688cbc294175` |

Complete build, native diagnostics, boundary checks and installed-command logs
are retained beside the results. The actual CLI process exited normally.
