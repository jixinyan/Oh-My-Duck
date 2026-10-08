# Runtime observation acceptance — 2026-10-07

Implementation commit: `32e35dd09ce7c278c6a264cf1fc9ab9926996efd`. The actual CPU campaign ran in a clean detached checkout of that commit, using pinned official ONNX policies, MuJoCo/BAM, native EDH ActionGate and actual camera/sensor samples. RL training remains stopped.

| Continuous operation | Requested magnitude | Final error |
|---|---|---|
| `walk` | +1.0 m | 0.021102 m |
| `rotate` | −78° | 4.496450° |
| `walk` | −0.5 m | 0.035872 m |
| `rotate` | +78° | 4.712026° |
| `walk` | −0.5 m | 0.040380 m |

All five operations satisfied the original 0.05 m / 5° tolerances, measured upright stopping and zero external obstacle contact. The episode executed 1208 control steps and 4832 physical substeps. Repeated `observe` results preserved the paused episode, sequence, measured joints, odometry and stop samples; remaining wall time decreased. The original head PNG decoded successfully. Repeated motion-boundary waits preserved the same confirmed boundary. The session closed and resources were released.

Campaign: `outputs/acceptance/runtime-sequence-clean-20261007-01/campaign.json`. Independent audit: `outputs/acceptance/runtime-sequence-clean-20261007-01/independent-audit.json`. The independent verifier reconstructed each motion from actual physical samples and checked input parameters, source provenance, stopping, sensor/image evidence and execution counts.

Existing artifact regression checks passed for the three-case CPU feet matrix, the formally completed Hospital navigation run `cf059bda-3f7e-4294-93e7-8b5fe9d70df1`, and the native voice run `5dfb18d9-b31b-4226-bc94-9391ddd847ac`. These checks review existing physical evidence. Current-source GPU execution is recorded separately by the unified campaign.

Wheel and sdist built successfully from the fixed checkout. The wheel installed in an independent environment and its `omd` command entry worked outside the source directory. Python compilation, JavaScript syntax and source whitespace checks passed.

The updated renderer generated `outputs/demos/runtime-render-regression-20261007.mp4` from the existing native voice run. The video is 1920×1080 at 10 fps, 1473 frames / 147.3 seconds, with all recorded 96 observer frames and the original head camera. Every encoded frame passed text-boundary checks, the recorded motion-time schedule passed, and full FFmpeg decoding succeeded. SHA256: `c876824af8da107ee8ac014138c077960273455c08179ca55f5b72243590f60f`. This video validates rendering of the existing run.

The GPU campaign uses matching fixed local and remote source and runs Office/Hospital matrices, continuous sequences and model-driven ordered navigation sequentially on one allocated GPU from devices 2–4. GPU acceptance, initialization reliability, long-motion precision and Office completion remain pending until terminal records and independent audits are available. Hardware, image-only VLN, perception accuracy, object interaction and trained-policy behavior require their own evidence. See [campaign operation](../runtime-release-acceptance.md).

Campaign `runtime-release-20261007-01` stopped during GPU admission. The preserved final occupation sample showed GPU 2 at 91% utilization, GPU 4 at 99%, and GPU 3 occupied by multiple existing workloads. No stage or physical worker started. The original samples and process inventory remain in the fixed checkout's `outputs/acceptance/runtime-release-20261007-01`. The release controller requires ten seconds of consecutive idle observations and records admission failures as an aborted campaign. It preserves unrelated workloads and keeps allocation within devices 2–4.

Campaign `runtime-release-20261007-02` at commit `960a9db2226f46a05846d10391830081a646af0f` is stopped at the user's request. The local controller PID `55721` and remote campaign/worker PIDs `179574` / `179584` exited; the NVIDIA process inventory confirmed the worker PID was absent. The native campaign preserves `aborted` and `KeyboardInterrupt` at `2026-10-08T03:13:16Z`. Artifacts and startup timing records are retained at `outputs/acceptance/runtime-release-20261007-02`. Automatic resumption is disabled.

The Office matrix passed three independent sessions and five actual motions. Errors were 0.017634 m / 0.717625° for the forward/turn case, 0.001558 m / 3.239445° for reverse/clockwise, and 0.022404 m for +1.0 m. Actual stopping, zero external obstacle contact, observation/image evidence, source checks and resource release passed. Independent local verification of all three copied cases also passed. Office startup timings included 137.3 and 147.4 seconds with the existing kernel cache.

Hospital forward/turn passed at 0.018369 m / 0.821840°. The reverse case returned `blocked`, with measured −0.447341 m for a −0.5 m request, final position error 0.052917 m, `motion_stalled`, zero external obstacle contact and resources released. The Office continuous sequence was interrupted. Hospital long sequences and both model-driven navigation stages were not started. Overall GPU acceptance remains incomplete.

Future admission requires zero compute processes on the selected GPU in addition to sustained zero utilization. This requirement excludes sharing with haomin or any other workload, including currently idle CUDA processes. The admission check uses actual NVIDIA process inventory and rejects occupied devices without starting physics or stopping other users' processes.
