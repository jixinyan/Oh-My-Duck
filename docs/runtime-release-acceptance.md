# Native runtime acceptance campaign

The release campaign validates the combined observation and native motion-wait interfaces with actual official policies, native ActionGate execution and measured physical state. The external Harness owns planning and formal task verification. Current GPU acceptance of this iteration is pending.

`microduck.observe` captures sensors at a confirmed paused or ended execution boundary. It returns head RGB, the 8×8 ToF array, IMU, 14 servo measurements, odometry, motion progress and remaining native wall time. Optional `prompt` and `source` select current-frame ground truth or the configured model perception service. Episode, sequence and execution boundary checks reject inconsistent observations. `microduck.wait_for_motion` waits up to 90 seconds for the native confirmed boundary and reports actual progress. Repeated reads preserve physical sequence. Neither tool admits physical actions.

## Local validation

The clean fixed-source campaign at commit `32e35dd09ce7c278c6a264cf1fc9ab9926996efd` completed +1.0 m, −78°, −0.5 m, +78°, −0.5 m. Distance errors were 0.021102, 0.035872 and 0.040380 m; angle errors were 4.496450° and 4.712026°. Actual stopping, repeated observations, source provenance, physical counts and resource release passed. The separate artifact verifier also passed. Artifacts are at `outputs/acceptance/runtime-sequence-clean-20261007-01`; see the [acceptance evidence](reports/runtime-observation-acceptance-2026-10-07.md).

The sequence plan is `configs/experiments/navigation-motion-sequences.json`. Each `motions` entry preserves its own operation and parameters, allowing repeated `walk` and `rotate` operations. The independent verifier reconstructs each displacement and continuous yaw from native physical samples, validates original request tolerances, decoded camera bytes, sensor dimensions, stopping, control/physics counts, source hashes and resource release.

## Unified GPU validation

`--preflight-only` checks matching clean source, both pinned Harness revisions, the actual remote environment versions, every declared stage and motion parameter, all downloaded scene file sizes and SHA256 values, public map references, converted robot assets, the ten official policy models and the selected provider metadata. CUDA visibility is cleared and no simulation worker starts. The output records `preflight_passed`, `gpu_acceptance_performed: false` and the actual per-scene checks. Full GPU execution performs the same preflight before admission.

```bash
python scripts/run_release_campaign.py --preflight-only \
  --worker-host jd_B300 --worker-root "$OMD_WORKER_ROOT" \
  --worker-python "$OMD_WORKER_PYTHON" --worker-edh-source "$OMD_WORKER_EDH" \
  --worker-policy-dir "$OMD_WORKER_POLICIES" --edh-source "$OMD_LOCAL_EDH" \
  --remote-provider-config "$OMD_PROVIDER_CONFIG" \
  --output outputs/acceptance/runtime-preflight-new
```

`configs/experiments/runtime-release-acceptance.json` declares six sequential stages:

1. Office feet metric matrix.
2. Hospital roller metric matrix.
3. Office walking followed by clockwise rotation.
4. Hospital long roller motions.
5. Actual model-driven Office ordered navigation.
6. Actual model-driven Hospital ordered navigation.

Run the controller from a clean fixed checkout and use a remote clean checkout at the same commit. Set `OMD_WORKER_ROOT`, `OMD_WORKER_PYTHON`, `OMD_WORKER_EDH`, `OMD_WORKER_POLICIES`, `OMD_LOCAL_EDH`, and `OMD_PROVIDER_CONFIG` to the actual paths. The provider file stays on the remote host. Select an idle device from GPU 2–4 after checking current ownership.

```bash
python scripts/run_release_campaign.py \
  --worker-host jd_B300 --worker-root "$OMD_WORKER_ROOT" \
  --worker-python "$OMD_WORKER_PYTHON" --worker-edh-source "$OMD_WORKER_EDH" \
  --worker-policy-dir "$OMD_WORKER_POLICIES" --edh-source "$OMD_LOCAL_EDH" \
  --remote-provider-config "$OMD_PROVIDER_CONFIG" --gpu 2 \
  --output outputs/acceptance/runtime-release-20261007-01
```

The controller checks clean source before each stage and holds an exclusive local campaign lease. The selected GPU must have zero compute processes and remain at zero utilization for ten seconds with at least 16 GiB available. Any existing compute PID rejects admission, including an idle process. Allocation remains within devices 2–4 and never shares a GPU with haomin or another user's workload. Occupancy and process samples are preserved. Admission stops after thirty seconds without sufficient idle observations. Metric artifacts are copied from the remote worker. Navigation exports actual events, camera bytes, formal verdicts and independent audit results. Each completed stage records its exit code and elapsed time; a failed stage prevents overall acceptance. Configuration and transport errors propagate and record the aborted campaign and any active stage. RL training remains stopped.

The controller owns each launched process and saves its PID, process group, cancellation signals, exit status and cleanup timing. Remote metric stages use `run_owned_acceptance.py`: closing the SSH input channel requests cancellation, including after a transport disconnect. The remote supervisor interrupts only its own campaign, and that campaign closes its own physical worker. Interrupted metric stages copy available artifacts and lifecycle records back to the controller. Navigation cancellation gives the runner time to close its native session before the controller closes its server. Graceful cleanup has a 120-second deadline; exceeded deadlines produce termination evidence and an explicit failure.

Actual CPU cancellation checks cover both control-channel closure and SIGTERM during official policy motion. Each executed 85 control steps, preserved the cancelled result, closed the native session and released resources without a cleanup timeout. Reproduce with the locked CPU environment and pinned policies:

```bash
PYTHONPATH=src python scripts/accept_campaign_cancellation.py \
  --catalog POLICY_DIRECTORY --mode control-eof \
  --output outputs/acceptance/cancel-eof-new
PYTHONPATH=src python scripts/accept_campaign_cancellation.py \
  --catalog POLICY_DIRECTORY --mode sigterm \
  --output outputs/acceptance/cancel-signal-new
```

## Initialization and media evidence

Newton initialization writes stage timings under `outputs/runtime-startup/<identity>/startup.json`, including backend import, simulation launch, environment construction, BAM binding and sensor binding. Ready sessions expose the record path and SHA256. A partial record identifies the last completed initialization stage; native startup limits remain unchanged.

Replay export, independent navigation review and MP4 overlays accept actual `observe` and `wait_for_motion` events. Frames come from the recorded simulator cameras and the trace comes from actual public Planner events. Formal navigation acceptance requires the original physical target, ordered waypoint visits, measured final stopping, independent Verifier and native task completion. Hardware, image-only VLN, model-perception accuracy, policy-specific object interaction and trained-policy behavior retain their separate acceptance requirements.
