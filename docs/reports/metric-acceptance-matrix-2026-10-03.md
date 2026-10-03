# Measured policy-tool acceptance — 2026-10-03

## Executed cases

The official `alpha_walking` policy executed through the pinned native physical Harness, ActionGate and CPU MuJoCo/BAM in the furnished apartment. The spawn was `x=-2.5 m, y=-1.9 m, yaw=0`. Three independent sessions ran the declared `apartment-feet` suite. Each motion required final position error ≤ 0.05 m or angle error ≤ 5°, upright posture and five measured stopped samples.

| Session | Sensor seed | Commands | Final errors | Control / physics steps |
|---|---|---|---|---|
| `forward-turn` | 20261003 | +0.5 m, +45° | 0.019558 m, 0.737881° | 516 / 2064 |
| `reverse-turn` | 20261004 | −0.5 m, −45° | 0.035112 m, 0.605836° | 476 / 1904 |
| `long-forward` | 20261005 | +1.0 m | 0.021102 m | 410 / 1640 |

All five motions passed. External-obstacle contact samples remained zero throughout all three sessions. Native executions ended with device confirmation, and the device owner and both control servers closed. Saved state values were finite, ToF readings remained within 0–4000 mm, and the final head RGB frames had nonzero contrast. Seed changes affect the simulated sensor stream; the spawn and policy remain fixed. These measurements cover tool behavior, with independent model-task verdicts recorded in the separate end-to-end report.

## Controller and acceptance workflow

Foot walking uses the full measured position target, lateral correction and heading correction. Its braking condition checks cross-track distance. Foot turning limits angular commands using current angle error and angular velocity. Bounded corrections use the angular change measured between the braking boundary and the stopped pose. Each request allows at most three corrections and retains its final measured error, motion phase and stopping evidence.

`scripts/accept_metric_policy_tools.py` accepts scene configuration, seed, distance, angle and command speeds. It records source revision, tracked-change hash, runner hash, scene hash, catalogue hash, installed runtime versions, physical device visibility, raw events, samples and PNG frames. A case becomes passed after measured motion, terminal device confirmation and resource release. Sensor image payloads remain in artifact files.

`scripts/accept_metric_campaign.py` validates `configs/experiments/metric-policy-acceptance.json`, executes cases sequentially in unique directories and records progress in `campaign.json`. A failed case terminates the campaign and preserves its artifacts. Existing output directories are rejected. Newton suites require an explicit physical GPU allocation; CPU suites clear GPU visibility. The plan contains Apartment feet, Office feet and Hospital rollers suites.

The CPU ToF adapter reports noisy valid measurements within the configured sensor range. Regression checks use actual MuJoCo ray intersections at 2 m and 4 m and an actual surface beyond 4 m. Both tests and the two range subcases passed.

## Provenance

- Source: `3265a3d38c405a7ae0e3354ef175b472955b7f54`.
- Immutable source: `.job-sources/metric-cpu-20261003-03`.
- Harness: `8a5e685b22d032207f53db20454f0992a4ad60fd`.
- Official catalogue: `1b56c396825c052a4e26e95cf2b8d8298af9e9b4`.
- Campaign: `outputs/acceptance/metric-campaign-apartment-20261003-05/campaign.json`.
- Independent artifact checks: `outputs/acceptance/metric-campaign-apartment-20261003-05/verification.json`.
- Campaign SHA256: `2639a6edc6431cad7e81be147f37a3b09e4a35e6610926dcdd72a321f7553505`.

| Result | SHA256 |
|---|---|
| `forward-turn/result.json` | `b920384d81c31ea8bd3d929d768c48b44998e490f2c16fc92ecfa5acd2535014` |
| `reverse-turn/result.json` | `db70b4285349f9d16e8cbe1514fe5c6753893c5a81e88e8c534ebb29deca178f` |
| `long-forward/result.json` | `cef1d5caa6ec47bdaa6edd64aac3d82aaeaebc6d79561ca8ca339666a3d7abf5` |

## Execution scope

The current foot-controller version requires Newton Office revalidation. Hospital roller behavior under the matrix also remains pending. GPU work is restricted to physical GPUs 2, 3 and 4; all three were occupied during this iteration, so no GPU workloads were launched. Existing processes continued, and RL remained stopped.

Varied spawn poses, surfaces, longer routes, other command speeds, perception accuracy, object effects and hardware require their own measurements. The current three cases establish bounded metric execution at the declared fixed apartment pose.
