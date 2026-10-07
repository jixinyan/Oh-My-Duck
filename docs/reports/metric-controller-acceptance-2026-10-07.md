# Metric controller acceptance — 2026-10-07

## Measured requirements

The native policy-tool campaign uses immutable source `1640fda2ecf5c081beaf09a4796635a707f06c8d`. Each scene declares three independent sessions covering +0.5 m, −0.5 m, +1.0 m and ±45°. Acceptance requires ≤0.05 m position error or ≤5° angle error, upright stopping, at least five consecutive physical stop samples, zero external-obstacle contacts, confirmed native termination and released session resources. Every child process must exit successfully.

| Scene | Session | Position error (m) | Angle error (degrees) | Control / physics steps |
|---|---|---|---|---|
| CPU apartment | `forward-turn` | 0.019558 | 0.656126 | 501 / 2004 |
| CPU apartment | `reverse-turn` | 0.035112 | 3.777766 | 437 / 1748 |
| CPU apartment | `long-forward` | 0.021102 | — | 410 / 1640 |
| Newton Office | `forward-turn` | 0.015363 | 0.556092 | 374 / 1496 |
| Newton Office | `reverse-turn` | 0.004519 | 4.952898 | 562 / 2248 |
| Newton Office | `long-forward` | 0.021602 | — | 410 / 1640 |
| Newton Hospital rollers | `forward-turn` | 0.015056 | 1.206051 | 335 / 1340 |
| Newton Hospital rollers | `reverse-turn` | 0.049210 | 3.746165 | 395 / 1580 |
| Newton Hospital rollers | `long-forward` | 0.043232 | — | 430 / 1720 |

All nine sessions and fifteen actions passed. The three whole-campaign artifact checks passed independently on the operator computer. Newton Office and Hospital ran sequentially on physical GPU4, and their child processes exited successfully. Maximum distance/angle errors are 0.035112 m / 3.777766° for CPU apartment, 0.021602 m / 4.952898° for Newton Office and 0.049210 m / 3.746165° for Newton Hospital.

## Physical control

The controller measures the original requested distance or accumulated angle on every control sample. When motion reaches its approach target, or is physically stationary within the requested tolerance, it enters zero-command braking. Completion requires a subsequent measured upright endpoint and five consecutive stop samples under zero-command control. Braking compensation uses the measured settling angle and permits at most three corrections. Final error remains relative to the original request.

Target progress is checked over 50 controls, using 0.01 m for translation and 0.05 rad for yaw. Correction progress uses the active compensation target. Insufficient progress reports `metric_progress_stalled` through the confirmed stopping path. The independent physical motion, ToF and obstacle guards remain active. Turning can translate the body; callers must read the resulting position before selecting their next action.

Standard-foot deployment uses the official sitstand solver settings: 30 iterations, 50 line-search iterations and direct Newton execution. Roller deployment uses 10/20 and CUDA graphs. Both use Newton SolverMuJoCo/MuJoCo-Warp and official BAM XL330 M6, preserving 61 observations, 14 named servo actions and four 0.005-second physics steps per 50 Hz control. Training configurations remain unchanged.

## Reproduction and evidence

Use a new output directory with the locked CPU environment for `apartment-feet` or the locked Isaac/Newton environment for `office-feet` and `hospital-rollers`:

```bash
python scripts/accept_metric_campaign.py \
  --suite office-feet --catalog POLICY_DIRECTORY --gpu 4 \
  --output outputs/acceptance/office-matrix-new
python scripts/verify_metric_campaign.py \
  --campaign outputs/acceptance/office-matrix-new \
  --output outputs/acceptance/office-matrix-new/verification.json
```

The verifier independently reconstructs distances and accumulated yaw from continuous physical samples. It checks the complete declared plan, unchanged thresholds, native action/boundary identities, 14-servo and four-substep counters, source revisions, result hashes, decoded observer image bytes and timestamps, obstacle contacts and session cleanup. It requires one unchanged source revision throughout each campaign. Per-case checks run before campaign acceptance; whole-campaign checks also run on the copied artifacts on the operator computer.

| Evidence | CPU apartment | Newton Office | Newton Hospital |
|---|---|---|---|
| Campaign directory | `outputs/acceptance/metric-apartment-feet-20261007-07` | `outputs/acceptance/metric-office-feet-20261007-07` | `outputs/acceptance/metric-hospital-rollers-20261007-07` |
| Campaign SHA256 | `590ed9a0e63fdd3d54f3d5ee21a469e0655248c60f4c5ef6396dac3e48d75740` | `e526d967449e350e98628c6a86c21f764db506c271b0a5f70dd41858b4b762da` | `1eb209fc82de0375e70f2b14932cfb25cc551d4d8690696308c81e9fcef516c5` |
| Whole-campaign `verification.json` SHA256 | `b53d5e5c8706ae0cfedc0025a51e869759c3be5def91b497874d0a42bfb89f17` | `688c9123745c72cc1d65260412d23215ceaab906c2a68a364a365bd402fdd028` | `bb81b5aaeba29cd6767fc236419d27a1c13bb3fdc1d40adb8c34e265ab272b72` |
| Physical GPU | None | 4 | 4 |

The shared `metric_motion.py` SHA256 is `8eb6153fcd15d9287fd1d6f35d4cc96341d5f9e1518188a1e8a74f0503cc37db`; `isaac_official.py` SHA256 is `5449afbebf98da3a655f53c665b28a476e12f04a8b8b1b82a1b26d665f9fd883`. The pinned native Harness is `8a5e685b22d032207f53db20454f0992a4ad60fd`, and the official policy catalogue is `1b56c396825c052a4e26e95cf2b8d8298af9e9b4`. CPU uses the official `alpha_walking` policy with MuJoCo/BAM; Office uses that policy with Newton/BAM and complete scene collisions. Hospital uses the official `roller` policy and four unactuated passive wheel joints. GPU suites run sequentially on `jd_B300` from `.job-sources/metric-stationary-20261007-01`. ONNX Runtime telemetry is disabled before initialization. Post-campaign process checks confirm that the campaign and GPU policy workers have exited.

## Acceptance scope

The motion matrix covers the declared spawn poses, seeds, distances and speeds. The [four-policy native Office task](office-skills-acceptance-2026-10-07.md) has separate source-pinned model, recovery, navigation, formal Verifier and MP4 evidence. Image-only VLN, recognition accuracy, object carrying, arbitrary scene generalization, trained-policy behavior, fresh-install kernel compilation and hardware require independent acceptance. Physical GPU authorization is limited to devices 2–4 with at most one device used concurrently. RL remains stopped.
