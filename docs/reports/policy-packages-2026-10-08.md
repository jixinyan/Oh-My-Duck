# Native training-package integration

Registered schema-2 API-1 training packages execute through native Microduck
tools, ActionGate and CPU MuJoCo/BAM. The actual campaign completed 725 controls,
2900 physics substeps, 145 original observer frames and 87 consecutive stopped
samples. Independent recorded-action and installed-package checks passed.
GPU acceptance and RL remain stopped.

## Source and package identity

The physical campaign uses immutable source
`837066bd37da309575deef379a492c4e15108c54`. Final installation, input checks and
Newton admission use `b3d3a13e541c0d82edb50ffb89952047e7d0db27`. Independent
review compares ten CPU execution modules between those revisions; their bytes
are unchanged. The Newton constructor validates the catalogue before local
Isaac imports and CUDA allocation.

| Alias | Retained training package | ONNX SHA256 |
| --- | --- | --- |
| `trained_walking` | MuJoCo RSL-RL Walking, `model_49999.pt`, training source `4253ee790` | `83205f46ab1ef0e492ee6aa4f45a6d698bb8b68695919ec325d3f88e630be11c` |
| `trained_standup` | Newton RSL-RL StandUp seed 42, `model_14999.pt`, training source `eb8eb0ec5` | `5a8231ece7ee1a266d69893ea2afb01ee1260ca4630ed7183e20d03f60486bd5` |

Manifest SHA256 values are
`aa92f0553a28a0a73f7daacb2b54666e80610e477444d75d4e80cff748bf10ce`
and `c9f3b7e2eaab7ad91de87e9c1a4b4958410ff98198902dc223be623861b04db9`.
Original packages remain unchanged. The native registry has SHA256
`2cd9cd48271ca8d5363b7adcaf38ac9207e7ce66d307c88b07cc99c688382ef5`.
It preserves ten official policies and adds two aliases. The local audit registry
resolves copied package locations and verifies that every remaining registration
field, manifest hash and model hash matches.

## Physical execution

| Policy | Controls | Verified operation |
| --- | ---: | --- |
| Official `velstand` | 75 | Native warmup and stopped selection boundary |
| `trained_walking` | 150 | Metric `walk` preparation selects the registered alias; 50 bounded twist controls and 100 zero-command controls |
| `trained_standup` | 400 | Complete eight-second episode; nonzero velocity rejected; native episode termination |
| Official `alpha_stand` | 100 | Transition preserves pose, velocity, episode and simulation sequence; measured standing and stopping |

There are 725 distinct controls and 742 serialized control publications.
Independent review checks repeated physical evidence, four complete substeps per
control, fourteen action values, the policy inventory and zero external-contact
controls. All 145 PNG bytes and simulation timestamps match native events; each
frame decodes at 640×480 with nonuniform RGB content.

Final height is `0.11641552464337189` m and tilt is `0.004955518530077827` rad.
Stopping has 87 consecutive samples. The final execution ends with `policy_stop`;
the action device and both native servers are closed. The renderer is
`llvmpipe (LLVM 15.0.7, 256 bits)` and Torch CUDA remains uninitialized.

Linux recomputation of all 725 actions has maximum error zero. Independently
installed macOS/ARM `policy-audit` runs outside the checkout and recomputes the
same actions with maximum error `5.364418029785156e-7`, within the unchanged
`1e-6` tolerance.

The campaign exercises integration from a standing apartment spawn. Historical
learned Walking and multi-spawn StandUp behavior remains unaccepted. Metric
preparation establishes policy selection here; distance completion, model task
verdicts, Newton runtime and hardware require their own acceptance.

## Input and release checks

- Twelve invalid registries and eighteen invalid packages are rejected. Package
  cases retain actual trained graph initializers and exercise invalid metadata.
- Four actual Newton constructor calls reject invalid registrations before CUDA
  allocation; installed OpenUSD verification passes and CUDA stays uninitialized.
- Official descriptions, 44 command blocks and 5750 inference comparisons across
  ten policies and 575 recorded observations exactly match source `9b9c22e`.
- Thirty-seven configuration, transport-argument and import tests pass.
- Fresh distributions and installation verify 429 source/resource files, three
  licenses and 24 CLI calls. Installed registry admission, prior 1208-action
  parity and continuous metric records also pass.
- Six-stage remote preparation passes four scenes, ten official models and 29
  pinned Harness files. The final process audit finds no native policy worker.

## Preserved artifacts

| Evidence | Location | SHA256 |
| --- | --- | --- |
| Native execution | `outputs/acceptance/policy-packages-runtime-20261008-02/result.json` | `8025a613bd44365f058ed40320b46ee59641e4a79111e05050c962259941efeb` |
| Newton admission | `outputs/acceptance/newton-policy-admission-20261008-03/result.json` | `84f697ce58d719d19551aec28741cf7f4b8169d2a63cd1836b9126379bb19ab1` |
| Installed package | `outputs/acceptance/policy-packages-install-20261008-03/result.json` | `af12ab16be257887da35d8ba6a5b48f8521262988051463003a770f655c765dd` |
| Remote preparation | `outputs/acceptance/policy-packages-preflight-20261008-03/preflight/result.json` | `742d1a7d76d48713a9ea7326f389c8fc9c2f6b72070b609c0cf3ce7c1e8c7ee8` |
| Wheel | `outputs/packages/policy-packages-20261008-03/oh_my_duck-0.1.0-py3-none-any.whl` | `bf122612fd4c1f9cdd8c3fc70dd9a62a97a9d82e38413b631e9d78d920d12023` |
| Source distribution | `outputs/packages/policy-packages-20261008-03/oh_my_duck-0.1.0.tar.gz` | `ff8de6ede5d452242a7fcdd8f8c71d97a8fbfe0449c3786ada538f771317c61f` |

Independent physical/image review is
`outputs/acceptance/policy-packages-independent-20261008-03.json`.
Installed recomputation is
`outputs/acceptance/policy-packages-installed-parity-20261008-03.json`.
Input and official-control results are
`policy-registry-inputs-20261008-01/result.json`,
`policy-package-inputs-20261008-03/result.json` and
`official-policy-preservation-20261008-03.json` under `outputs/acceptance/`.
Public commands are documented in the [registry guide](../policy-registry.md).
