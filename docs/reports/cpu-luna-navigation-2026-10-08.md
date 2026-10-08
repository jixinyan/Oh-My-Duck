# Luna native CPU office navigation

OpenAI Responses `gpt-6-luna` with high reasoning completed the original office
navigation task through the pinned native Harness and official ONNX policies.
Run `8b12dd3c-7204-4e23-b6cc-30185fdc74fb` has terminal state `succeeded`.
Independent Verifier verdict `5b70056f-723c-45f7-9250-e6a262d28465` is `passed`.

The run used source `11fdb1100609aa22ab1d39f9b071441c30f55d41`, Harness
`8a5e685b22d032207f53db20454f0992a4ad60fd` and official policy revision
`1b56c396825c052a4e26e95cf2b8d8298af9e9b4`. CPU MuJoCo/BAM retained seed
`20260929`, spawn `(0,0,0)`, the native office room goal, five required hold ticks
and the 4000-control/1800-second budget. The instruction was submitted as text.
Actual head RGB and distance tools were used; scene inspection explicitly used
`simulator_ground_truth`.

| Request | Measurement | Error | Stopped samples |
| --- | --- | --- | --- |
| Rotate +30 degrees | +26.657412 degrees | 3.342588 degrees | 5 |
| Walk +0.4 m | +0.386406 m | 0.025793 m | 5 |
| Walk +0.55 m | +0.530458 m | 0.034599 m | 5 |

The task executed 569 controls and 2276 physical substeps. Its final zero-twist
hold executed 75 controls and reported 80 consecutive stopped samples. The
confirmed terminal reason is `policy_stop`. Native goal evidence reports
position approximately `(0.82186, 0.56203, 0.11677)` m, office bounds
`x=[0.7,3.8], y=[-0.8,0.8]`, height 0.11677 m, tilt 0.01134 rad and 153 held ticks.

The independent installed audit at source
`f0d844e7be364672302c59334b7ca23d6c305b29` checked all 1040 events, 113 observer
frames and 12 stop-progress records. It verifies the current confirmed boundary,
preceding pause, unchanged native action counters, command identity, episode,
physical sequence, action generation, metric accuracy and formal goal evidence.
The retained failed navigation export is rejected. The owned server exited with
code 0, and session `0a8db2b5-6021-4e6a-931a-f51a94d88b25` closed with released
resources. GPU acceptance and RL remain stopped.

The MP4 contains recorded simulation frames, head RGB, tool calls, planner text
when present and formal verification. It is H264, 1920×1080, 10 fps, 1534 frames
and 153.4 seconds. Full decoding and every-frame text-boundary checks passed.
Renderer source `cf3864d`, source/event hashes and video SHA256 are recorded in
the adjacent video report. This demonstrates the single CPU apartment task.
Ordered multi-scene navigation, image-only VLN, current Newton GPU execution and
hardware retain their separate acceptance requirements.

Evidence:

- `outputs/acceptance/cpu-luna-navigation-20261008-05/acceptance-recheck/result.json`
- `outputs/acceptance/cpu-luna-navigation-20261008-05/replay/`
- `outputs/acceptance/cpu-luna-task-acceptance-20261008-01-install/result.json`
- `outputs/demos/cpu-luna-office-navigation-20261008-05-v2.mp4`
- `outputs/demos/cpu-luna-office-navigation-20261008-05-v2.json`

The installed package checked 433 source/resource files, three licenses and 26
CLI calls outside the checkout, including retained ONNX and physical-record
audits. Original runtime records remain immutable; acceptance uses the separate
installed audit report above.
