# Metric controller acceptance — 2026-10-06

## Physical results

The declared native policy-tool matrix passed three independent CPU apartment sessions, three Newton Office sessions and three Newton Hospital roller sessions. Each suite covers +0.5 m, −0.5 m, +1.0 m and ±45°. The unchanged requirements are ≤0.05 m position error or ≤5° angle error, upright stopping, five consecutive measured stopped samples, zero external-obstacle contacts and confirmed resource release.

| Scene | Session | Position error (m) | Angle error (degrees) | Control / physics steps |
|---|---|---|---|---|
| CPU apartment | `forward-turn` | 0.019558 | 0.656126 | 501 / 2004 |
| CPU apartment | `reverse-turn` | 0.035112 | 3.777766 | 437 / 1748 |
| CPU apartment | `long-forward` | 0.021102 | — | 410 / 1640 |
| Newton Office | `forward-turn` | 0.014251 | 1.182402 | 369 / 1476 |
| Newton Office | `reverse-turn` | 0.001029 | 4.519054 | 417 / 1668 |
| Newton Office | `long-forward` | 0.022884 | — | 406 / 1624 |
| Newton Hospital rollers | `forward-turn` | 0.019067 | 1.624581 | 332 / 1328 |
| Newton Hospital rollers | `reverse-turn` | 0.043211 | 4.162124 | 480 / 1920 |
| Newton Hospital rollers | `long-forward` | 0.041882 | — | 379 / 1516 |

All fifteen actions passed. Each native policy control produced 14 servo actions and four 0.005-second BAM physics steps. The endpoint remains an actual stationary measurement. Independent artifact verification reconstructs the movement from continuous physical samples, verifies native image bytes and timestamps, checks execution identities and source hashes, and requires the complete declared case plan. Every child process also exited successfully.

## Controller behavior

Standard-foot turns retain the requested angular command through measured approach, with coupled forward and lateral commands required by the official walking policy. Braking uses the current angle and at most three corrections derived from the measured braking rotation. Turning also produces translation: Office measured 0.167710 m and 0.222898 m during its two turns. Navigation must observe the resulting position before choosing its next movement.

The controller checks target progress over 50 physical controls. Walking requires 0.01 m progress and rotation requires 0.05 rad progress over this interval. Insufficient target progress reports `metric_progress_stalled` and uses the existing zero-command stopping path. A corrective turn measures progress toward its active braking-compensation target; final acceptance always uses the original requested angle. Failure artifacts retain the final execution status and metric motion evidence.

Roller turns retain the requested angular command and use measured braking compensation with at most three corrections. Their translation was 0.013598 m and 0.033180 m in the two Hospital cases. The four passive wheel joints remain unactuated. Motion commands are policy inputs; actual speed, position, yaw and stopping are measured from the simulator.

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

Hospital uses source `b0b4c2afe264c1b92222d2fc088d8fb9ccbc5c11` from `.job-sources/roller-turn-20261006-01` on `jd_B300`, physical GPU4. Its campaign is `outputs/acceptance/metric-hospital-rollers-20261006-05`; campaign SHA256 is `3b2fcbdd5b63726b2e9c5e55561dbf55d84914d6306b4b07f905924a51736209`, and independent `verification.json` SHA256 is `8ae3e1d002155e43ba522bc99cec0f39b9b238cddfd85c396216bae5647d996e`. Its controller SHA256 is `57c8f8243b1ad1276d2e9cf13f7aab321942c6b39d3663f70db633aab553a9e0`. This source extends turning compensation to rollers and preserves standard-foot motion behavior. Hospital and Office matrices executed sequentially on GPU4. Earlier unsuccessful runs remain preserved in their original directories.

## Acceptance scope

These measurements establish bounded motion tools at the declared apartment, Office and Hospital spawn poses. Recognition accuracy, image-only VLN, object carrying, arbitrary scene generalization, learned-policy behavior and hardware require their own acceptance. Model task completion continues to require the native independent Verifier. Authorized GPU devices are 2–4 with at most one physical device used concurrently; RL remains stopped.
