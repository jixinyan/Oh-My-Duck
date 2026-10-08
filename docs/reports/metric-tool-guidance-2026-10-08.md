# Metric policy tool parameters

The native tool schemas advertise units, motion limits, default speeds and measured
completion requirements. `walk` defaults to 0.4 m/s and `rotate` to 45 degrees/s.
Standard feet command 0.2 m/s forward and 0.25 m/s lateral during rotation.
The Planner reserves travel space and observes the actual position after turning.
The accepted speed range requires scene-specific policy response measurements.

Production source `4200252329d1b8ab4f573163db3e31c18aaa578a` passed actual
OpenAI Responses `gpt-6-luna` high startup and native tool execution in the CPU
MuJoCo/BAM apartment. The original seed `20260929`, spawn `(0,0,0)`, office goal,
five hold ticks and 4000-control/1800-second budget were preserved. The pinned
Harness is `8a5e685b22d032207f53db20454f0992a4ad60fd`, and the official policy
revision is `1b56c396825c052a4e26e95cf2b8d8298af9e9b4`.

| Request | Measured result | Final error | Stop samples |
| --- | --- | --- | --- |
| Walk +0.2 m | +0.191707 m | 0.009957 m | 5 |
| Rotate +30 degrees | +26.165450 degrees | 3.834550 degrees | 5 |
| Walk +0.3 m | +0.302388 m | 0.020484 m | 5 |
| Rotate -37 degrees | -34.194138 degrees | 2.805862 degrees | 5 |

These four measured requests preserve the original 0.05 m / 5-degree thresholds.
The two turns translated 0.080609 m and 0.172163 m. The original task export is
`outputs/acceptance/cpu-luna-navigation-20261008-04/replay/`, run
`2bc7a2da-8aa3-44d7-8a4a-d4c5b6a28306`. Session
`5c932f10-a15b-474a-b2a8-3b9ccd262a89` closed with released resources, and the
owned server exited with code 0. This record verifies tool execution and parameter
guidance. Complete office navigation remains pending.

Node syntax checks and 17 actual import tests passed. A new wheel and source
distribution passed independent installation outside the checkout, including 432
source/resource files, three licenses, 25 CLI calls and retained ONNX/physical-record
verification. The installation report is
`outputs/acceptance/metric-tool-guidance-20261008-01-install/result.json`.
Robot, RL, controller math, task predicates, actuators and solver inputs preserve
the preceding verified source. GPU acceptance and RL remain stopped.
