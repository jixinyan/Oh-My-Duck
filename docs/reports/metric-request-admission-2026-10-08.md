# Native metric request admission

Prepared `walk` and `rotate` requests retain their original identity until execution completes, fails, encounters a measured hazard or receives an explicit interruption. A subsequent metric request reports the pending request ID before changing the command or physical state. `set_command` explicitly selects a replacement command, which executes through native `execution.resume`.

Native acceptance waits shield the owned policy pump. A caller deadline ends that wait while the original execution continues under its declared native budget. Physical stopping uses the native stop interface and device confirmation.

## Actual CPU execution

The public command `omd validate metric-admission` ran from immutable source `d113f3b9eb2d4ff6ff43662a040b9ca978327535` on `jd_B300`, using CPU MuJoCo/BAM and `llvmpipe (LLVM 15.0.7, 256 bits)`. CUDA visibility was empty and CUDA remained uninitialized. The fixed apartment spawn was `(-2.5, -1.9, 0)` with sensor seed `20261003`.

| Check | Recorded result |
| --- | --- |
| Pending walk/rotate request combinations | Eight rejected calls; original request, command, native boundary, complete qpos/qvel/ctrl and physical time remained unchanged |
| Forward distance | Requested 0.5 m; stopped target error 0.0222649826 m, within the original 0.05 m tolerance |
| Rotation | Requested 45°; measured 49.0870304733°, stopped error 4.0870304733°, within the original 5° tolerance |
| Physical stopping | Both metric actions reached five consecutive upright stopped samples |
| Rotation translation | 0.1951671116 m; subsequent navigation uses the resulting measured pose |
| Caller deadlines | Four actual 0.001-second waits expired; each original execution remained running and its policy pump continued |
| Explicit command replacement | Two pending metric requests replaced by zero twist; each replacement executed 50 real controls and confirmed upright stopping |
| Policy and physics | 641 admitted policy controls, 2564 physical substeps; all sequences were present and external-obstacle contact counts remained zero |
| Termination and cleanup | Confirmed native `policy_stop`, closed device and both control servers, owned process exited with status 0 |

The original 61 actor observations, 14 servo actions, HOME, 50 Hz cadence and four 0.005-second substeps remain unchanged. All files under `robotics/` and `rl/` match the preceding `bb2457d` source. The native Harness remains pinned to `8a5e685b22d032207f53db20454f0992a4ad60fd`; the official policy catalogue remains `1b56c396825c052a4e26e95cf2b8d8298af9e9b4`.

## Independent verification

An independent installed package on the operator computer reconstructed the distance target and accumulated yaw from the saved continuous physical samples. It recomputed all 641 actions through the actual official ONNX files. Maximum action difference was `5.364418029785156e-7`, within the unchanged `1e-6` numerical threshold. Serialized native updates, complete sequences, four-substep counters, contact evidence, source hashes and resource records passed.

Seventeen subprocess import tests passed with simulator and execution dependencies unloaded. Wheel, source distribution and a separate installation outside the checkout passed 432 source/resource file checks, three license checks and 25 actual CLI invocations, including `metric-admission --help`.

| Evidence | SHA256 |
| --- | --- |
| `outputs/acceptance/metric-request-admission-20261008-07/result.json` | `7b6ba31de1f736388eda3c4acc9564fb35e41fccbc71d79390e13f8fdaf7e45a` |
| `outputs/acceptance/metric-request-admission-20261008-07/independent-audit.json` | `f53062234ac2600b44fc7752eb59365edd5f5fca1df87837035aa854396705c4` |
| `outputs/acceptance/metric-request-admission-20261008-07-install/result.json` | `d307457d3fda71b81ba5f099f5022989a9c7e45b3237f99cbdf4324250debc06` |

Raw `events.json`, `samples.json`, `tools.json` and `remote-process-audit.json` remain with the immutable run. The acceptance implementation is `validation/harness/metric_admission.py`; the compatibility script delegates to that same module.

## Reproduction and scope

Use the locked Linux CPU runtime with OSMesa and Mesa software rendering, the pinned native SDK on `PYTHONPATH`, empty CUDA visibility and a new output directory:

```bash
python -m oh_my_duck validate metric-admission \
  --scene-config configs/simulation-demo/apartment-metric.json \
  --catalog POLICY_DIRECTORY \
  --edh-source PINNED_HARNESS_DIRECTORY \
  --output outputs/acceptance/metric-admission-new
```

This execution verifies native request admission, measured policy motion, caller deadlines, explicit replacement and lifecycle cleanup. Luna Office navigation, current Newton execution, varied motion parameters, perception accuracy and hardware retain their own acceptance requirements. GPU work and RL remain stopped.
