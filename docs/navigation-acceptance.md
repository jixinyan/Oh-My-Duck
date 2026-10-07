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

`run_navigation_acceptance.py` 使用已启动的原生服务，创建会话并提交指令，保存任务身份和状态，等待终态，确认资源释放，导出原始记录并运行独立复核。复核通过且会话已经释放资源后才写入 `result.json`。错误终止执行，已有输出目录禁止重复使用。关闭会话之后保留服务进程，原始事件和图像通过原生持久记录导出。

```bash
python scripts/run_navigation_acceptance.py \
  --base-url http://127.0.0.1:4361 \
  --scene-config configs/simulation-demo/office-vln.json \
  --instruction configs/simulation-demo/office-vln-instruction.md \
  --data-directory .cache/harness-office-new \
  --output outputs/acceptance/office-navigation-new \
  --minimum-distance-m 3
```

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
python scripts/record_harness_demo.py close \
  --base-url http://127.0.0.1:4360 --session .cache/navigation-new-session.json
python scripts/record_harness_demo.py export \
  --base-url http://127.0.0.1:4360 --run-id RUN_ID \
  --data-directory .cache/harness-navigation-new \
  --output outputs/demos/navigation-new
python scripts/accept_navigation_replay.py outputs/demos/navigation-new \
  --scene-config configs/simulation-demo/hospital-navigation.json \
  --minimum-distance-m 3 --output outputs/acceptance/navigation-new.json
```

Export requires a terminal native run. In the manual workflow, wait for that terminal state before closing, then confirm `closed` / `released` before export. Preserve unsuccessful runs in separate directories. Verify that GPU workers exit regardless of task outcome. The [release readiness guide](release-readiness.md#record-the-agentic-video) describes rendering the original scene cameras and public agentic trace with source hashes and full MP4 decoding.

## Artifact requirements

`accept_navigation_replay.py` fails at the unmet requirement and creates its report exclusively in a new file. It requires actual model walk/rotate calls, one physical episode, all ordered checkpoints within their configured radii, a passed formal goal check, confirmed `policy_stop`, upright final stopping and zero recorded external-obstacle contacts. It verifies every original exported image hash and byte count, image dimensions and nonuniform pixels, physical observer timing, original event sequences and policy/action cadence.

Walking target error is recomputed from admitted signed distance, recorded starting yaw and measured endpoint. Rotation error is checked against the recorded accumulated physical yaw. Completed metric actions retain 0.05 m / 5° thresholds and five stopped samples. The minimum-distance measure sums the measured displacement of distinct terminal walking segments; it does not measure continuous trajectory length. Native checkpoint evidence records the actual first qualifying pose and sequence at each waypoint. The report retains completed, failed and blocked metric results separately from the final task verdict.

Acceptance applies to the declared scene, spawn, seed, policy and observation access. Results across these routes establish measured tasks in multiple fixed environments. Broader scene generalization, independently measured recognition accuracy, object interactions, trained policies and hardware require their own evidence.
