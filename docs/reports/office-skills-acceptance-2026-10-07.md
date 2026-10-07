# Office skill and navigation acceptance — 2026-10-07

## Native task

Run `b2e0fe5d-beaf-4cc5-bbc9-4278610abbe7` completed the furnished NVIDIA Office skill demonstration and navigation task through the pinned native Harness, an actual `gpt-6-astra` Responses/high provider, official ONNX policies, Newton SolverMuJoCo/MuJoCo-Warp, BAM and the independent Verifier. The scene and physical episode remained continuous across policy transitions and the two native executions.

| Measurement | Actual result |
|---|---|
| `sitstand`, seated endpoint | Height 0.060976 m after 100 controls |
| `sitstand`, standing endpoint | Height 0.115825 m after 100 controls; increase 0.054849 m |
| `alpha_stand`, head/body command | Head `[0,0.2,0.4,0]`, body `[0,0,0.01,0,0,0]`; measured head-yaw change 0.333259 rad relative to the subsequent neutral endpoint |
| `ground_pick`, reaching endpoint | Height 0.087211 m after 100 controls |
| `ground_pick`, complete episode | 140 controls / 2.8 s, native `episode_terminated` |
| `alpha_stand`, standing recovery | Height 0.116068 m; 97 consecutive stopped samples |
| `alpha_walking`, 0.6 m tool | Measured translation 0.576529 m; endpoint error 0.025469 m; five stopped samples |
| Final navigation | Target error 0.045512 m against 0.2 m; upright hold 147 / 5 ticks |
| Final physical stopping | 75 additional zero-command controls; 80 consecutive stopped samples |
| External-obstacle contact samples | Zero |
| Total execution | 999 policy calls / controls, 3996 physics substeps |

The first execution ends when the reaching policy completes. Its formal destination check records the robot outside the goal. Native `tasks.retry` creates attempt 2; `transition_policy` preserves physical state, and the second execution performs standing recovery and measured navigation. Final verdict `66d5c675-8279-4fa2-8958-e5a1e1e19b50` passes at boundary `59ed115b-cd7f-4fd6-a03e-4e41b25c3cbc`. The Planner completes the native plan and calls `tasks.finish`; run state is `succeeded`.

## Runtime and perception

The interactive `allcollisions` robot uses the official sitstand solver iteration settings, 30 iterations and 50 line-search iterations, with direct Newton physics execution. The scene solver retains capacity for 5000 contacts and 5000 constraint rows. The roller deployment model retains 10/20 iterations and CUDA graphs. Training configurations are unchanged. Policy observations remain 61D, actions retain 14 named servos, and each 50 Hz action executes four 0.005 s BAM physics steps. Public scene information reports the solver settings and CUDA graph mode.

Six current-frame model observations execute SAM3.1 and YOLO26. The returned floor queries contain no valid targets; the original empty outputs remain in the replay. ToF at these open-area poses reports an invalid/out-of-range measurement. Navigation uses explicitly available simulator odometry and public scene geometry. These observations establish actual inference and refreshed frames; recognition accuracy and image-only VLN remain open.

The SAM checkpoint SHA256 is `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`; YOLO26 SHA256 is `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`. The independent checker validates physical skill endpoints, current joint measurements, episodic duration, recovery, refreshed model observations, original PNG bytes, execution counters and the formal navigation boundary. All checks pass without changing distance, angle, upright or stopping thresholds.

## Reproduction and provenance

Use `configs/simulation-demo/office-skills.json` and its committed instruction `configs/simulation-demo/office-skills-instruction.md` with the native server. Export a terminal run and check the saved evidence:

```bash
environments/cpu-apartment/.venv/bin/python scripts/accept_policy_showcase_replay.py \
  outputs/demos/office-skills-new --office-skills \
  --policies sitstand alpha_stand ground_pick alpha_walking \
  --robot-model robot_allcollisions --stop-reason policy_stop \
  --output outputs/acceptance/office-skills-new.json
```

| Source or artifact | Identity |
|---|---|
| Project physics source | `341e9035045a61faf67529835f47ffbb866ee4de` |
| Deployment backend SHA256 | `5449afbebf98da3a655f53c665b28a476e12f04a8b8b1b82a1b26d665f9fd883` |
| Native Harness | `8a5e685b22d032207f53db20454f0992a4ad60fd` |
| Official policy catalogue | `1b56c396825c052a4e26e95cf2b8d8298af9e9b4` |
| Host / physical GPU | `jd_B300` / 4 |
| Export | `outputs/demos/office-native-skills-20261007-03` |
| Task snapshot | 2472 events, 212 images, 199 observer frames |
| Export manifest SHA256 | `78c1afb0f7a7be8bc51fc6dec9d833fcbeb7ac3324c3c486b5ebe7fbe7b7e3e2` |
| Run JSON SHA256 | `4bc15196766ed0cdbaaef307adf4c75be8f66f857d46bc3b70866a809737c4db` |
| Events JSON SHA256 | `2e11b444d0f7d9bf7ff63c8249009d5337f0b441080bb738f3b5150c0881e677` |
| Independent acceptance | `outputs/acceptance/office-native-skills-20261007-03.json` |
| Acceptance SHA256 | `a99ce1a443b0edb4dfce9059f9e9c666769aa141baca5e468dc8dade05cc641e` |

## Video

`outputs/demos/oh-my-duck-office-skills-20261007-demo.mp4` is a 169.1-second, 1920 × 1080, 10 fps recording with 1691 frames. It combines the actual Newton/Warp scene camera, head RGB, public Planner text, native plan, tool feedback and formal verdict. Model waiting time is compressed at 16×; all 199 recorded observer frames retain at least their original physical time interval. Every frame's text-boundary checks, image/source hashes and complete FFmpeg decoding pass. MP4 SHA256 is `d2c4ef0514d77b105a0b868a3d2f28787cceff1f85c1c34749991bc0be406608`; the accompanying JSON records the schedule and provenance.

The 4× waiting-time version is `outputs/demos/oh-my-duck-office-skills-20261007-verified.mp4`, 437.3 seconds, SHA256 `6556ed2dc6053b1266b38f6ed0b212bc1c4479f24ef29c51858b12fe8f11f841`. Its full decoding also passes. The rendered public trace contains no private reasoning content. Recognition accuracy, object carrying and RTX appearance retain their separate acceptance requirements.

Session `3d3e16d6-eefd-477d-adba-568c1d432e7b` is closed with resources released. Its GPU worker and dedicated perception service have exited. At most one physical GPU was used; other projects' processes were preserved. RL remains stopped.

This acceptance covers the declared Office commands and fixed spawn. Ground-pick reaching has measured posture effects; grasping and object carrying require their own physical evidence. Arbitrary fallen-state recovery, broad scene generalization, hardware behavior, trained-policy acceptance, independently annotated recognition accuracy and first-install kernel compilation remain open.
