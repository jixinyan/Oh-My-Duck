# Native target tracking

`microduck.inspect_scene` and the perception part of `microduck.observe` preserve
each detection's original `target_id` and attach a stable `track_id`. The native
worker owns tracking state for the current task. `microduck.read_track` queries
that state against the current physical episode and control sequence.

| Query parameter | Meaning | Accepted values |
| --- | --- | --- |
| `track_id` | Identity from a current-task perception result | Required string, 1–512 characters |
| `max_age_ms` | Maximum wall-clock age of the original measurement | Integer, 100–60000; default 60000 |

```json
{"track_id": "episode:track:1", "max_age_ms": 60000}
```

| Query status | Meaning | `current_target` |
| --- | --- | --- |
| `current` | Confirmed target measured at the current physical sequence within the requested age | Original measured target |
| `tentative` | Model association needs another distinct observation | `null` |
| `missing` | A later image in the same stream has no accepted measurement for this identity | `null` |
| `ambiguous` | Nearby observations allow competing identity assignments | `null` |
| `stale` | The measurement belongs to an earlier physical sequence or exceeds its wall-clock age | `null` |
| `expired` | The track exceeds its retention interval at the queried physical sequence | `null` |

`last_observation` always identifies the original sequence, timestamp and target.
The query reports its source, prompt and model hashes. Historical geometry stays
under `last_observation`. The tool reads current physical state and does not
capture a new perception image, advance physics or prepare a motion.

## Association and identity

Streams are separated by source, exact prompt and model hashes. Reobserve with
the same prompt and source to update an existing track. New task creation clears
the streams; identities are never reused during one worker's lifetime. The
tracker also clears its streams when the physical episode changes.

`simulator_ground_truth` associates authoritative native instance IDs. `models`
associates measured world surface points with same-label tracks through SciPy
`linear_sum_assignment`. Motion prediction is used for association; the returned
target retains its original RGBD measurement. Model confirmation requires two
consecutive observations at distinct physical sequences. After an observed gap
or ambiguous assignment, confirmation requires two fresh observations.

The shared parameters are declared in `TrackingParameters`:

| Parameter | Default |
| --- | --- |
| Physical control interval | 0.02 seconds |
| Maximum predicted-point association distance | 0.75 meters |
| Competing-assignment distance margin | 0.10 meters |
| Maximum velocity used for prediction | 2.0 meters/second |
| Retention after the last accepted measurement | 150 physical control steps |
| Stream capacity | 16 |
| Track capacity per stream | 64 |

Conflicting assignments return `track_id=null` and their candidate identities.
Fewer than eight valid depth pixels return an unmeasured target. Repeated reads
of the same physical sequence preserve confirmation counts and the original
measurement time. Different geometry at the same sequence, earlier sequences,
invalid timestamps and capacity overflow raise errors.

An expired identity can be removed during a subsequent update; a query for a
removed or unknown identity raises an error. RGBD association has a defined
distance gate and requires measured world coordinates. Recognition accuracy,
object crossings and long occlusions require their own evidence.

## Execution and recording

`read_track` supplies observation state to the upper Harness. The current
`approach_object(target_id, standoff_m)` tool prepares a fixed observed surface
point. Autonomous following needs continuing measurements, bounded movement,
target-loss stopping and its own physical task verification.

The optional service flag `--record-inspections` retains every actual model
request under the service's `--output`: `frame.json`, `response.json` and
`inspection.json`. The manifest identifies source time, episode, sequence, prompt,
device, model hashes and the SHA256 of both data files. Recording is opt-in.

Public CPU validation commands are:

The native runtime is the locked `environments/cpu-apartment` environment.
The actual perception service runs in its isolated model environment with
`--device cpu --record-inspections`. Keep `CUDA_VISIBLE_DEVICES=''` for both.
Prepare OSMesa through the native setup entry point and select it explicitly
before importing MuJoCo. Independent record audits require the declared
`validation` package extra, including SciPy.

```bash
python -m oh_my_duck.infrastructure.headless --mujoco-renderer osmesa -- \
  omd validate tracking --catalog POLICY_DIRECTORY --edh-source HARNESS_DIRECTORY \
  --endpoint http://127.0.0.1:PORT --inspections SERVICE_RECORD_DIRECTORY \
  --output NEW_CAPTURE_DIRECTORY

omd validate tracking-audit --capture CAPTURE_DIRECTORY \
  --catalog POLICY_DIRECTORY --output NEW_AUDIT_JSON

omd validate replay EXPORT_DIRECTORY --expected-verdict passed \
  --expected-stop-reason policy_stop --require-stop-progress \
  --require-target-approach --require-target-tracking
```

The native campaign runs official CPU MuJoCo/BAM policies, records an actual
moving ball and real YOLO observations, queries native tools, checks task cleanup,
and saves original captures and control evidence. The independent audit verifies
measurements against source RGBD, confirmation counts, native instance pixels,
observer PNGs and optional ONNX action parity. The replay check requires the same
identity queried at two physical sequences in a formal native task.

The model task uses
[`apartment-target-tracking.json`](../configs/simulation-demo/apartment-target-tracking.json).
Source locations are listed in the
[perception source map](../src/oh_my_duck/perception/README.md).
Actual CPU execution, model task, independent installation and video evidence
are recorded in the [acceptance report](reports/target-tracking-2026-10-12.md).

The [`apartment-model-target.json`](../configs/simulation-demo/apartment-model-target.json)
scene uses actual YOLO26 observations for both target selection and approach.
Add `--require-model-perception` to the replay command to require the original
model detection and SHA256 on every approach request. The actual CPU task queried
the same model track at sequences 85 and 95 before approaching and finishing with
an independent passed verdict. See the [model task and MP4](reports/model-target-task-2026-10-13.md).
