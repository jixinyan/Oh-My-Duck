# Native multi-policy demonstrations

The fixed NVIDIA Hospital scenario passed a real native-Harness task with the official roller robot, `roller` and `crouch` policies, actual SAM3.1 + YOLO26 inference, Newton/BAM actions and independent destination verification. This validates that specific task and its measured policy effects. Cross-scene generalization, precise roller endpoints, object carrying and recognition accuracy remain open.

## Hospital video and native evidence

- MP4: `outputs/demos/oh-my-duck-hospital-rollers-20261001-verified.mp4`, 1920×1080, 10 fps, 634 frames, 63.4 seconds.
- SHA256: `b866f28c4b1c67126bea2c5055c025f109cc1d3fbdf4c9abd95d3a1554fa1c34`.
- Native run: `48317c87-66ce-4c91-a2f9-1157511b3df9`, `succeeded`.
- Independent verdict: `60590ac9-8755-4b51-a13e-706fbb40fe8b`, `passed`.
- Confirmed boundary: `0cefbdb9-6843-4f39-9697-cf8ac9f50e51`; stop reason `episode_terminated`.
- Actual execution: 677 policy calls/control steps, 2708 physics substeps, 50 Hz with four 0.005-second substeps per action.
- Destination error: 0.188404 m against 0.3 m; upright hold: 42 ticks against five; final stopped samples: 26.
- Final body height: 0.111623 m; tilt: 0.060464 rad; cumulative non-ground external contact samples: zero.
- Actual observer frames: 135; current-image joint-model observations: three, six targets with valid simulator ray depths.
- Source export: `.cache/perception/replay-hospital-rollers-08`; physical audit: [recorded checks](evidence/hospital-showcase-20261001.json).

The video uses recorded observer/head images, public Planner text, native plans, tool parameters, sensor feedback and the final verdict. Internal model reasoning is excluded. Image hashes, chronological event completeness, physical timing, wheel measurements, episodic duration, final stopping, text bounds and complete video decoding passed.

All four passive wheel joints have actual sensor measurements, including simultaneous wheel rotation during movement. At control sequence 502, the measured body height was 0.114175 m. After 100 crouch controls it was 0.067142 m; after the full 175-control episodic duration it recovered to 0.111623 m. Navigation verification used native pose and upright hold at that confirmed boundary.

The single `walk(distance_m=0.5, speed_m_s=0.4)` command measured 1.130831 m along its starting heading and 1.147115 m total translation. Its endpoint error was 0.659577 m against the unchanged 0.05 m tolerance, so the metric command remains failed. The Planner used the measured endpoint and completed the crouch demonstration within the final destination region. Final destination success does not establish metric endpoint precision.

## Office measurements and remaining task

Independent Newton/BAM physics measurements completed 790 controls/3160 substeps:

| Demonstrated policy | Measured effect |
|---|---|
| `sitstand` | Seated height 0.061134 m, standing height 0.115785 m |
| `alpha_stand` | Head/body command execution, neutral recovery and measured stopping |
| `ground_pick` | Minimum body height 0.087013 m, followed by upright `alpha_stand` recovery at 0.116069 m with 147 stopped samples |

Physical evidence is under `jd_B300:/home/jixin/workspace/code/Oh-My-Duck/.job-sources/multiskill-20260930-05/outputs/demos/office-skills-physics-05/`.

The latest native Office multi-skill run `462259a8-50b1-448d-804a-a262a829e621` executed the skill phases but ended with `Responses model returned a failed event.` Its first episodic boundary was outside the navigation destination and received a failed verdict. The complete source export is `.cache/perception/replay-office-skills-08`; full task success remains unverified. The earlier recorded Office skill video at `outputs/demos/oh-my-duck-office-skills-observed-20261001-06.mp4` also preserves its terminal failure. Ground-pick reaching does not establish gripping or carrying an object. Roulade recovery and object effects are outside the accepted scope.

## Runtime and perception

Both scenes use imported, provenance-checked NVIDIA USD assets, Newton `SolverMuJoCo`/MJWarp and XL330 M6 BAM. The owned official roller factory and conversion path preserve the canonical 61 observations and 14 servos; four interleaved passive wheel joints remain unactuated. The all-collisions regression matched all 70 collision geometries and retained body/armature parity.

The executed conversion used the existing local Isaac-assets lock with SHA256 `ea626b92d8a428575142e877e94a1412b1e3666874fc45a65cbac3791289ef65`. That preexisting workspace change remains unstaged. Reproduction must regenerate converted assets for the lock actually in use; cached conversion fingerprints from a different lock are not interchangeable. This milestone adds pretrained-policy scene execution; it does not establish a roller RL training task.

Full signed-transform audits are stored as hash-addressed JSON files. `scene_info` returns their references and nearby authored-map geometry, including stored coverage and query regions. Querying after movement refreshes that local map view. Native action admission and the external Harness agent loop retain their pinned implementations.

| Source | Pin / identity |
|---|---|
| EDH | `8a5e685b22d032207f53db20454f0992a4ad60fd` |
| Official policy catalogue | `1b56c396825c052a4e26e95cf2b8d8298af9e9b4` |
| Executed Hospital worker | `.job-sources/multiskill-20260930-07` on `jd_B300`, GPU1 |
| Brain | Actual `gpt-6-astra`, Responses, high |
| SAM3.1 source | `2345a4ad109ac29c569da749c91d84f10dc08c40` |
| Existing SAM checkpoint | `/home/jixin/workspace/checkpoints/sam3.1/sam3.1_multiplex.pt` |
| SAM checkpoint SHA256 | `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6` |
| YOLO26 checkpoint SHA256 | `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b` |

The service executes actual text-prompt segmentation and YOLO association. Empty results and incorrect associated categories remain in the original outputs. Distances are explicitly `simulator_ground_truth`; depth-estimation accuracy and recognition accuracy have no independent acceptance yet.

The recorded renderer is `newton_warp` with shadows. NVIDIA MDL appearance is not fully reproduced. An isolated graphics-library probe could not create a Vulkan instance through the host NVIDIA ICD; RTX device/render acceptance remains unavailable. No photorealistic rendering claim is made.

Both current native sessions report `closed` with resources `released`. Their GPU workers and the dedicated SAM service have exited; other projects' GPU processes remain untouched. RL training remains stopped; no new learner was launched. The next unresolved work is roller braking/endpoint precision, completion of the Office native task, independently annotated perception evaluation and a working RTX graphics environment.
