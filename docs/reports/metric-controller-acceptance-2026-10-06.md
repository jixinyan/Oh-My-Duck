# Metric controller acceptance — 2026-10-06

## Physical results

The declared native policy-tool matrix passed three independent CPU apartment sessions and three independent Newton Office sessions. Each suite covers +0.5 m, −0.5 m, +1.0 m and ±45°. The unchanged requirements are ≤0.05 m position error or ≤5° angle error, upright stopping, five consecutive measured stopped samples, zero external-obstacle contacts and confirmed resource release.

| Scene | Session | Position error (m) | Angle error (degrees) | Control / physics steps |
|---|---|---|---|---|
| CPU apartment | `forward-turn` | 0.019558 | 0.656126 | 501 / 2004 |
| CPU apartment | `reverse-turn` | 0.035112 | 3.777766 | 437 / 1748 |
| CPU apartment | `long-forward` | 0.021102 | — | 410 / 1640 |
| Newton Office | `forward-turn` | 0.014251 | 1.182402 | 369 / 1476 |
| Newton Office | `reverse-turn` | 0.001029 | 4.519054 | 417 / 1668 |
| Newton Office | `long-forward` | 0.022884 | — | 406 / 1624 |

All ten actions passed. Each native policy control produced 14 servo actions and four 0.005-second BAM physics steps. The endpoint remains an actual stationary measurement. Independent artifact verification reconstructs the movement from continuous physical samples, verifies native image bytes and timestamps, checks execution identities and source hashes, and requires the complete declared case plan. Every child process also exited successfully.

## Controller behavior

Standard-foot turns retain the requested angular command through measured approach, with coupled forward and lateral commands required by the official walking policy. Braking uses the current angle and at most three corrections derived from the measured braking rotation. Turning also produces translation: Office measured 0.167710 m and 0.222898 m during its two turns. Navigation must observe the resulting position before choosing its next movement.

The controller checks target progress over 50 physical controls. Walking requires 0.01 m progress and rotation requires 0.05 rad progress over this interval. Insufficient target progress reports `metric_progress_stalled` and uses the existing zero-command stopping path. A corrective turn measures progress toward its active braking-compensation target; final acceptance always uses the original requested angle. Failure artifacts retain the final execution status and metric motion evidence.

Policy inference sets `ORT_DISABLE_TELEMETRY=1` before ONNX Runtime initialization. The locked macOS runtime completed all three actual simulation child processes with exit code zero. The setting follows ONNX Runtime's documented telemetry control and changes no policy weights, observation values or servo actions.

## Provenance

| Evidence | CPU apartment | Newton Office |
|---|---|---|
| Source revision | `6e2757e8d02e3bd52353f445b45bc9890e84e69b` | `dca35a13a71f3f070c5a4a0037ba86dd78610d84` |
| Campaign | `outputs/acceptance/metric-apartment-rotation-20261006-04` | `outputs/acceptance/metric-office-feet-20261006-04` |
| Campaign SHA256 | `4a79d7985e787ad3f24ce3583547c0545ae9e807d4731b826332893d2d08d0bd` | `290f3138ca9aa8a14bb37add4ec0ce7b88cdc7a7c880767e4c015b6740be6173` |
| Independent `verification.json` SHA256 | `bca79e5b6dd7a5ea8f819c5026975f59a23b332fc5f0f07d7050be8b8766913f` | `098cad572a470e51f66bb8cc401bd332a6b5b89eb9a0df11fa574a1c6d858671` |
| Physical GPU | None | 4 |

Both immutable sources contain the identical `metric_motion.py` SHA256 `614fec90681ac17d2c6e213022fb14f2618d13ab86e9fbe5459d17bdef007c20`. The later CPU source additionally contains the ONNX telemetry initialization setting and records that setting in provenance. The pinned Harness revision is `8a5e685b22d032207f53db20454f0992a4ad60fd`; the official policy catalogue revision is `1b56c396825c052a4e26e95cf2b8d8298af9e9b4`. Office uses the actual Newton SolverMuJoCo/MuJoCo-Warp path, native ActionGate, complete scene collisions and the official `alpha_walking` policy.

## Acceptance scope

These measurements establish bounded motion tools at the declared apartment and Office spawn poses. They do not establish recognition accuracy, image-only VLN, object carrying, arbitrary scene generalization, learned-policy behavior or hardware acceptance. Model task completion continues to require the native independent Verifier. Hospital roller matrix acceptance remains pending. Authorized GPU devices are 2–4 with at most one physical device used concurrently; RL remains stopped.
