# Native execution retries and goal evidence

Every admitted execution receives a bounded command belonging to its own
execution ID and action generation. A retry prepares 75 zero-twist control steps
when the preceding command belongs to a confirmed terminal execution. An
explicitly prepared command keeps its request ID, parameters and step limit.
The retained scene preserves physical position, velocity, actuator state,
simulation time, stop counters and the policy's preceding action.

`task_progress`, `observe` and `wait_for_motion` expose `goal_check` from the
native backend at the same episode, sequence and simulation time. Its evidence
contains the actual target, upright requirements and held/required ticks.
Repeated reads preserve the physical state and hold counters. The independent
Verifier produces the formal task verdict.

Source `11fdb1100609aa22ab1d39f9b071441c30f55d41` passed actual CPU MuJoCo/BAM
execution with Mesa llvmpipe, the pinned Harness
`8a5e685b22d032207f53db20454f0992a4ad60fd` and official policy revision
`1b56c396825c052a4e26e95cf2b8d8298af9e9b4`. The furnished apartment, seed
`20260929`, original room goal and budget were retained.

| Execution | Controls | Physical sequence | Stopped samples | Terminal reason |
| --- | --- | --- | --- | --- |
| Initial command | 75 | 75 | 37 | policy_stop |
| Automatic retry command | 75 | 150 | 112 | policy_stop |
| Explicitly prepared retry command | 50 | 200 | 162 | policy_stop |

All three executions preserve the complete physical state at startup and use
unique execution and command IDs. Six rejected start requests preserve that
state and command identity. Three repeated observation/progress checks preserve
physics and counters. Every terminal boundary is device-confirmed.

The 200 admitted controls executed 800 physical substeps. An independent
installed validator recomputed every ONNX action; maximum error was
`7.450580596923828e-7`. Native serialization retained 206 matching updates.
All resources closed, the recorded worker PID exited and CUDA remained
uninitialized. This validates execution retries and native goal reporting.
Complete model navigation, Newton GPU execution and hardware require their
corresponding acceptance results. GPU acceptance and RL remain stopped.

Eighteen actual import tests and Node syntax checks passed. Independent wheel,
source distribution and installation checks verified 433 source/resource files,
three licenses and 26 CLI calls, including retained policy and physical-record
audits outside the checkout.

Evidence:

- `outputs/acceptance/native-execution-retry-20261008-02/result.json`
- `outputs/acceptance/native-execution-retry-20261008-02/independent-audit.json`
- `outputs/acceptance/native-execution-retry-20261008-02/remote-process-audit.json`
- `outputs/acceptance/native-execution-retry-20261008-02-install/recheck/result.json`

Run `omd validate execution-retry --help` for the CPU validation inputs. The
implementation is `validation/harness/execution_retry.py`; execution admission
is handled by `integrations/edh/device.py` and `integrations/edh/session.py`.
