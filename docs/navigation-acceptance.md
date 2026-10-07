# Native navigation acceptance

Navigation runs through the pinned native Harness and actual configured model. Model calls select bounded metric policy actions through ActionGate; the independent native Verifier checks the declared physical destination and ordered checkpoints. The exported artifact audit additionally checks policy identity, physical timing, observations, images, measured segment displacement and final stopping.

## Declared routes

| Scene configuration | Robot / policy | Ordered route in world meters |
|---|---|---|
| `configs/simulation-demo/office-vln.json` | Standard feet / `alpha_walking` | Lobby center (−17.5, 32.2), right detour (−16.35, 32.4), final desk approach (−16.8, 34.4) |
| `configs/simulation-demo/hospital-navigation.json` | Four passive wheels / `roller` | East checkpoint (9.7, 7.2), north checkpoint (9.7, 9.4), west destination (8.1, 9.4) |

Both routes use Newton SolverMuJoCo/MuJoCo-Warp, official BAM, 61 observations, 14 named servo actions and 50 Hz control with four physics substeps. Scene files contain authored USD assets, explicit spawn poses, checkpoint radii, final radius and budgets. The model receives current head RGB, ToF, odometry and public authored geometry. These benchmarks explicitly permit simulator-ground-truth segmentation, bearing and distance. They do not establish image-only VLN or recognition accuracy.

Instructions are in the matching `office-vln-instruction.md` and `hospital-navigation-instruction.md` files. They specify route constraints and measured observations. The Planner chooses each movement using the current physical state; commands remain subject to native pause, lease, interruption, obstacle and motion-progress checks. Turns report their actual body translation. Failed or blocked metric motions remain recorded even when subsequent navigation succeeds.

## Run and export

Start `omd.py harness` with the pinned EDH source, actual model provider, locked Isaac/Newton worker environment, scene configuration and a new data directory. On a remote NVIDIA host, provide `--worker-host`, `--worker-root`, `--worker-python`, `--worker-edh-source`, `--worker-policy-dir` and one explicit `--worker-cuda-device`. Check actual process ownership and activity before allocation. Authorized devices are 2–4, with at most one physical GPU concurrently. Run the scene sessions sequentially and confirm resource release between them. RL remains stopped.

The scene ID is the native launch profile. Its scenario ID is `navigate-` followed by the scene ID:

```bash
python scripts/record_harness_demo.py open \
  --base-url http://127.0.0.1:4360 --profile nvidia-hospital-navigation \
  --session .cache/navigation-new-session.json
python scripts/record_harness_demo.py task \
  --base-url http://127.0.0.1:4360 --session .cache/navigation-new-session.json \
  --scenario navigate-nvidia-hospital-navigation \
  --instruction configs/simulation-demo/hospital-navigation-instruction.md \
  --output .cache/navigation-new-task.json
python scripts/record_harness_demo.py export \
  --base-url http://127.0.0.1:4360 --run-id RUN_ID \
  --data-directory .cache/harness-navigation-new \
  --output outputs/demos/navigation-new
python scripts/accept_navigation_replay.py outputs/demos/navigation-new \
  --scene-config configs/simulation-demo/hospital-navigation.json \
  --minimum-distance-m 3 --output outputs/acceptance/navigation-new.json
python scripts/record_harness_demo.py close \
  --base-url http://127.0.0.1:4360 --session .cache/navigation-new-session.json
```

Export requires a terminal native run. Preserve unsuccessful runs in separate directories. Close sessions and verify that their GPU workers exit regardless of task outcome. The [release readiness guide](release-readiness.md#record-the-agentic-video) describes rendering the original scene cameras and public agentic trace with source hashes and full MP4 decoding.

## Artifact requirements

`accept_navigation_replay.py` fails at the unmet requirement and creates its report exclusively in a new file. It requires actual model walk/rotate calls, one physical episode, all ordered checkpoints within their configured radii, a passed formal goal check, confirmed `policy_stop`, upright final stopping and zero recorded external-obstacle contacts. It verifies every original exported image hash and byte count, image dimensions and nonuniform pixels, physical observer timing, original event sequences and policy/action cadence.

Walking target error is recomputed from admitted signed distance, recorded starting yaw and measured endpoint. Rotation error is checked against the recorded accumulated physical yaw. Completed metric actions retain 0.05 m / 5° thresholds and five stopped samples. The minimum-distance measure sums the measured displacement of distinct terminal walking segments; it does not measure continuous trajectory length. Native checkpoint evidence records the actual first qualifying pose and sequence at each waypoint. The report retains completed, failed and blocked metric results separately from the final task verdict.

Acceptance applies to the declared scene, spawn, seed, policy and observation access. Results across these routes establish measured tasks in multiple fixed environments. Broader scene generalization, independently measured recognition accuracy, object interactions, trained policies and hardware require their own evidence.
