# Native campaign input admission

Verified source: `075569be5e8f8fb7024b02408bfe8ef59a12b3fe`.
Remote source: `.job-sources/campaign-input-admission-20261008-01`.

`rl/experiments/campaign.py` checks declared training counts before creating
output directories or launching workers. Schema version, smoke iterations,
checkpoint interval, environment count, training iterations, seed, search updates
and allocation counts require integer JSON values. Booleans, decimal values,
strings and missing required values terminate at admission. Seed preserves the
existing integer range; counts preserve their documented minimum values.

Declared environment search requires an object. Memory and throughput fractions
must be finite numeric values in their documented ranges. Warmup requires at
least one update and measured execution requires additional updates. Every
committed campaign retains its complete loaded configuration. The eight-run
8192-environment plan retains its declared tasks, native frameworks and budgets.

## Executed verification

The actual `load_plan` API, `assigned_devices` API and campaign CLI were exercised
without replacing dependencies or process execution. Invalid CLI counts terminate
before output creation; the committed eight-run CLI dry-run returns the original
runs and allocation declaration without launching training.

Local admission, preparation and framework checks passed 108 tests and three
subtests. Two checks skipped because their local optional environment paths were
absent. The same files passed 110 tests and three subtests on Linux with the
actual configured interpreter paths. The remote test process exited with code 0.

An immutable-source wheel and source distribution passed independent installation
outside the checkout: 430 source/resource files, three licenses and 24 actual
public CLI calls, including retained policy and motion-record audits. No CUDA
runtime was initialized. Task recipes, BAM, actuator timing, native PPO,
normalization, export and physical execution remain unchanged.

| Artifact | SHA256 |
| --- | --- |
| `outputs/acceptance/campaign-input-install-20261008-01/result.json` | `e8733abdbc7577fad4dcb60c1dc3e1e81597887702a3ab3a7238cff5aefadf2d` |

GPU acceptance and RL remain stopped. These results establish input validation
and installed-package integrity; learned behavior and Newton execution retain
their separate acceptance requirements.

## Reproduction

```sh
mkdir -p .cache/tmp
CUDA_VISIBLE_DEVICES='' TMPDIR="$PWD/.cache/tmp" PYTHONPATH=src \
  python -m pytest tests/test_campaign_input_admission.py \
  tests/test_rl_preparation_inputs.py tests/test_rl_frameworks.py \
  -q --basetemp .cache/tmp/campaign-input-tests

CUDA_VISIBLE_DEVICES='' TMPDIR="$PWD/.cache/tmp" omd campaign \
  --config configs/experiments/representative-8192.json \
  --output outputs/experiments/campaign-input-inspection --dry-run
```
