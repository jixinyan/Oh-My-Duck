# Newton execution preflight

`infrastructure/usd_runtime.py` verifies the unique OpenUSD provider and installed
file contents before Newton execution. Registered RSL-RL and SB3 entry points,
native RSL workers, evaluation and direct environment creation use
`verify_execution_runtime`. Diagnostic launchers verify before loading Isaac.
MuJoCo execution retains its independent dependency environment.

Immutable source `82451e7c41c711854aacd62c52e670776a4f451e` passed actual CPU
validation with real installed dependencies. The dedicated environment was
`.envs/isaac-newton-preflight-20261008-01`, including native Harness and SB3 extras.
The shared canonical environment was preserved. CUDA visibility was explicitly
empty throughout validation.

| Actual check | Result |
| --- | --- |
| Registered RSL-RL and SB3 training | Both reject coinstalled OpenUSD before GPU selection/initialization |
| Registered task evaluation | Rejects before EGL setup, policy loading and output creation |
| Diagnostic RSL-RL, SB3, probe and evaluation | All four reject before the native Isaac launcher |
| Direct environment and native RSL worker | Both reject before simulation/device initialization |
| All nine failed invocations | Expected provider error and exit code 1; CUDA uninitialized, Isaac and pxr unloaded; no W&B session; launch environment unchanged |
| Output protection | No training/evaluation directories created |
| Actual locked setup | Removes the duplicate provider and reinstalls `usd-exchange==3.0.0`; all 244 files verified |
| MuJoCo selection | Succeeds without an OpenUSD provider; metadata/helper imports keep execution dependencies unloaded |
| Configuration and import tests | 26 passed from the fixed source |
| Source compilation | Passed |
| Complete release metadata | Six stages, four scene configurations, ten official policies and 29 pinned Harness files passed; CUDA uninitialized |
| Independent installation | 421 source/resource files, three licenses and 20 actual CLI calls outside the checkout |
| Installed saved-evidence audit | 1208 ONNX actions / 1220 serialized updates; zero action error; five-motion physical audit passed |

The negative invocations used actual `usd-core==25.11` and `usd-exchange==3.0.0`
installed together in the dedicated environment. Registered calls used the actual
Flat Walking task and downloaded `alpha_walking.onnx`; direct calls built the
official configuration. Native learner, task recipe, BAM, timing, reward,
normalization and export semantics remain unchanged.

| Artifact | Location | SHA256 |
| --- | --- | --- |
| Actual entry-point audit | `outputs/acceptance/newton-execution-preflight-20261008-02/independent-audit.json` | `92e6a8cf362f3bf9a61c63d44d9e801d12bde7afd00e9e55d6ba069cf908a6b5` |
| Installed package audit | `outputs/acceptance/newton-execution-preflight-package-20261008-01/result.json` | `5c0678f486d284d35d182eda3bd3602888778425da8ef58630e5354ad550eb48` |
| Installed dependency boundaries | `outputs/acceptance/newton-execution-preflight-package-20261008-01/boundary-audit.json` | `64c7622b3b1c0e8b7a19886641b9c9642ca8f666350648f6ba47b0c763f886ba` |
| Complete release preflight | `outputs/acceptance/newton-execution-release-preflight-20261008-01/preflight/result.json` | `fa021f57baabb028e6680c8656aea64af0451d9de4d6ac13d07dbad213acda96` |
| Wheel | `outputs/packages/newton-execution-preflight-20261008-01/oh_my_duck-0.1.0-py3-none-any.whl` | `37967b31d19938580fb441d0dc0ea44415cb7c7d0a4b21e78a596e88746a6b64` |
| Source distribution | `outputs/packages/newton-execution-preflight-20261008-01/oh_my_duck-0.1.0.tar.gz` | `d92a9447e11b6b4b165d097e180453fef8c35e001084f0575d2149eba70c4aa0` |

Each actual invocation preserves its log and observed process state. Build,
installation, command records and saved-evidence reports accompany the results.
All validation processes exited. Remote process and GPU ownership observations
are retained in `outputs/acceptance/newton-execution-preflight-process-audit-20261008-01.log`
and the adjacent GPU audit/owner logs. Existing GPU workloads were preserved.
GPU physics, learner execution, rendering and learned-policy behavior require
separate acceptance. GPU acceptance and RL remain stopped.
