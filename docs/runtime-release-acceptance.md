# Native runtime acceptance campaign

The release campaign validates the combined observation and native motion-wait interfaces with actual official policies, native ActionGate execution and measured physical state. The external Harness owns planning and formal task verification. Current GPU acceptance of this iteration is pending.

`microduck.observe` captures sensors at a confirmed paused or ended execution boundary. It returns head RGB, the 8×8 ToF array, IMU, 14 servo measurements, odometry, motion progress and remaining native wall time. Optional `prompt` and `source` select current-frame ground truth or the configured model perception service. Episode, sequence and execution boundary checks reject inconsistent observations. `microduck.wait_for_motion` waits up to 90 seconds for the native confirmed boundary and reports actual progress. Repeated reads preserve physical sequence. Neither tool admits physical actions.

## Local validation

The actual CPU apartment functional run at `outputs/acceptance/runtime-observation-cpu-20261007-01` completed a 0.5 m walk and −45° rotation. The five-motion run at `outputs/acceptance/cpu-navigation-sequence-20261007-02` completed +1.0 m, −78°, −0.5 m, +78°, −0.5 m. Distance errors were 0.021102, 0.035872 and 0.040380 m; angle errors were 4.496450° and 4.712026°. Actual stopping, repeated observations and resource release passed. These development runs used a working checkout; formal acceptance additionally requires clean fixed source provenance.

The sequence plan is `configs/experiments/navigation-motion-sequences.json`. Each `motions` entry preserves its own operation and parameters, allowing repeated `walk` and `rotate` operations. The independent verifier reconstructs each displacement and continuous yaw from native physical samples, validates original request tolerances, decoded camera bytes, sensor dimensions, stopping, control/physics counts, source hashes and resource release.

## Unified GPU validation

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

The controller checks clean source before each stage, holds an exclusive local campaign lease, requires an idle selected GPU with at least 16 GiB available, records occupancy samples and preserves foreign process records. Metric artifacts are copied from the remote worker. Navigation exports actual events, camera bytes, formal verdicts and independent audit results. Each completed stage records its exit code and elapsed time; a failed stage prevents overall acceptance. Configuration and transport errors propagate and record an aborted stage. The controller closes its own navigation server and the navigation runner closes its own physics session. RL training remains stopped.

## Initialization and media evidence

Newton initialization writes stage timings under `outputs/runtime-startup/<identity>/startup.json`, including backend import, simulation launch, environment construction, BAM binding and sensor binding. Ready sessions expose the record path and SHA256. A partial record identifies the last completed initialization stage; native startup limits remain unchanged.

Replay export, independent navigation review and MP4 overlays accept actual `observe` and `wait_for_motion` events. Frames come from the recorded simulator cameras and the trace comes from actual public Planner events. Formal navigation acceptance requires the original physical target, ordered waypoint visits, measured final stopping, independent Verifier and native task completion. Hardware, image-only VLN, model-perception accuracy, policy-specific object interaction and trained-policy behavior retain their separate acceptance requirements.
