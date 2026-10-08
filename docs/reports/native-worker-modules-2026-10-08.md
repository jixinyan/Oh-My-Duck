# Native worker modules and actual CPU verification

The native worker implementation is organized in
[`integrations/edh/`](../../src/oh_my_duck/integrations/edh/README.md): environment
and sensors, action device, session and control tools, and process transport.
`oh_my_duck.integrations.edh_native` remains the public executable and class import
entry point. CPU MuJoCo/BAM and Isaac Newton/BAM share these modules.

## Source and evidence

All runtime checks below use immutable source
`edffd76594a2347996af82e59f3389ad0f480eb5`. Python AST comparison verified that the
three migrated classes, wire timestamp function and transport function preserve
their complete implementation. The 61 actor observations, 14 named servo actions,
50 Hz control timing, four physics substeps, ActionGate admission and physical
identity checks remain in the native execution path.

| Actual check | Saved result |
| --- | --- |
| Continuous five-motion CPU session and physical audit | `outputs/acceptance/native-worker-cpu-sequence-20261008-01/independent-audit.json` |
| 1208 recorded ONNX actions and 1220 serialized updates | `outputs/acceptance/native-worker-cpu-sequence-20261008-01/walk-clockwise-return/policy-verification.json` |
| Queued selection, queued command and old-task command rejection | `outputs/acceptance/native-worker-lease-20261008-01.log` |
| Native SDK → independent worker process → actual policy and camera data | `outputs/acceptance/native-worker-transport-20261008-02/result.json` |
| Control-channel closure during actual movement | `outputs/acceptance/native-worker-cancel-eof-20261008-01/acceptance.json` |
| SIGTERM during actual movement | `outputs/acceptance/native-worker-cancel-sigterm-20261008-01/acceptance.json` |
| Actual `ground_pick` → `alpha_stand` transition and stopping | `outputs/acceptance/native-worker-policy-transition-20261008-01.json` |
| Independent wheel installation and actual installed commands | `outputs/acceptance/native-worker-installed-package-20261008-01.json` |
| Remote Python compilation and native/Newton/perception imports | `outputs/acceptance/native-worker-linux-imports-20261008-01.txt` |

The CPU sequence executed 1208 controls and 4832 physics substeps. Requests were
+1.0 m, −78°, −0.5 m, +78° and −0.5 m; final errors were 0.021102 m, 4.496450°,
0.035872 m, 4.712026° and 0.040380 m. Each endpoint had measured upright stopping,
zero external obstacle contacts and the original 0.05 m / 5° tolerance. All
recorded ONNX outputs recomputed exactly, with maximum action error zero.
Native termination, physical provenance and session resource release passed.

The native SDK test used the pinned `createNativeWorkerEnvironment`, real
`LocalImageStore`, a separate Python worker, official `velstand`, actual sensors
and ActionGate. It executed 75 controls / 300 physical substeps, saved 80 updates
and 15 camera frames, and confirmed device termination and process exit. The
standing task returned the actual native `goal_reached=false` observation.

Both cancellation checks interrupted actual movement after 85 controls. The
supervisor returned status 130, saved the aborted case and confirmed resource
release within its graceful cleanup interval. The policy transition executed
140 `ground_pick` controls followed by 100 `alpha_stand` controls and measured
95 stopped samples before native policy termination.

## Installed package and runtime preparation

A fresh independent environment verified 417 tracked Python/resource files,
three licenses and 19 installed CLI calls outside the checkout. This includes
the actual 1208-action ONNX audit and independent physical-record verification.
Configuration and dependency-boundary checks passed 21 tests. Source compilation
and JavaScript syntax checks passed.

The remote Newton environment compiled the source and imported the actual native
worker, Newton backend and perception client with `CUDA_VISIBLE_DEVICES` empty.
Current-source release metadata requires robot conversion assets matching the
current conversion-source fingerprint. The
[source-verified asset preparation](source-verified-assets-2026-10-08.md) supplies
both required robot variants and records the complete metadata preflight at
`bf8abf881f57880a352fcb765f1182f80b710bb7`.
GPU simulation, fresh Newton kernel execution and hardware acceptance remain
pending. GPU acceptance and RL stay stopped.

| Artifact | SHA256 |
| --- | --- |
| Independent CPU physical audit | `652489c74f95261ad4f0b57dbdbd40c9b8cde216dfddc7e54ee342010af87751` |
| Native SDK/process transport result | `176299cf023f514f29bc023b53e67167da8839e7ad5135e98df47ade70bf9c60` |
| Installed package result | `ea138c4b8a1caa7c3ae5a9481ab54683f6fc4a273542acb82c322478a765a295` |
| Wheel | `4906515d5c3ecedfb7c426fb2d5fb2e52987962e33607fcfb8cffadcaf01fd6b` |
| Source distribution | `921f3a2d2f6d8b863137cd84e4e6d15560b6d3f837d9e04cae18298b969556ea` |
