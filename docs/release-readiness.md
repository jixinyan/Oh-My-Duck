# Release readiness

The release workflow uses recorded speech, the pinned native Harness, official policies, actual simulation sensors, independent task verification, and the confirmed voice profile. Each acceptance run has a unique directory and preserves source revisions, model identity, task events and measured physics.

## Runtime preparation

Use Python 3.12, Node.js and the locked environments in `environments/`. Install the voice client on the operator computer and the ASR, TTS and Isaac/Newton environments on the NVIDIA host. Keep the model services bound to localhost and use SSH forwarding between hosts. Voice setup requires an explicitly confirmed profile before task feedback can be synthesized.

```bash
mkdir -p .cache/tmp
TMPDIR="$PWD/.cache/tmp" uv sync --project environments/voice-client --locked --no-editable
environments/voice-client/.venv/bin/python omd.py doctor --runtime voice-client \
  --asr-url http://127.0.0.1:18761 --tts-url http://127.0.0.1:18762 \
  --harness-url http://127.0.0.1:4318 --output outputs/acceptance/doctor-client-new.json
```

The doctor checks installed dependency versions and the actual service identities. On the simulation host, `--runtime isaac-newton --scene-config configs/simulation-demo/office.json --catalog POLICY_DIRECTORY` also checks CUDA, the pinned Isaac Lab revision, the converted robot asset and the complete official ONNX catalogue. Readiness checks and behavioral acceptance have separate evidence.

## End-to-end acceptance

```bash
environments/voice-client/.venv/bin/python omd.py voice-task \
  --audio /absolute/path/command.wav --persona duck \
  --harness-url http://127.0.0.1:4318 \
  --profile nvidia-office-6.0 --scenario navigate-nvidia-office-6.0 \
  --output outputs/acceptance/voice-task-new
```

`voice-task` preserves the recorded audio SHA256 and ASR model revision, submits the recognized instruction unchanged with the native catalogue digest, waits for the actual task result, and synthesizes a status response using the active voice profile. It returns exit code 0 for a successful native task and 2 for a completed task that failed acceptance. Service and transport errors terminate execution. The command produces audio files; interactive device playback is provided by `voice-session`.

Acceptance requires a real native model task, actual pretrained policy actions, current sensor observations, a passed independent Verifier result, confirmed execution termination and session cleanup. Metric motion checks also require measured distance/angle error and five upright stopped samples. Operator interruption must confirm the native execution boundary and demonstrate that control and physics counters remain unchanged afterwards. Physical braking and execution-clock interruption are recorded separately.

## Metric policy matrix

Run the declared cases in the locked CPU apartment environment:

```bash
TMPDIR="$PWD/.cache/tmp" uv sync --project environments/cpu-apartment --locked
environments/cpu-apartment/.venv/bin/python scripts/accept_metric_campaign.py \
  --suite apartment-feet --catalog POLICY_DIRECTORY \
  --output outputs/acceptance/apartment-matrix-new
```

The pinned native Harness source must be available in `.cache/edh/8a5e685b22d032207f53db20454f0992a4ad60fd`. The campaign checks all declared cases, stops at the first failure and preserves per-case events, state samples, images, provenance and resource-release evidence. Output paths must be new. `office-feet` and `hospital-rollers` use the locked Isaac/Newton environment and require `--gpu` with an allocated physical device. Current authorized host devices are 2–4; inspect their ownership and activity before running a GPU suite. Current-controller Newton behavior is pending. See the [CPU measurements](reports/metric-acceptance-matrix-2026-10-03.md).

Every new case also runs an independent artifact check before campaign acceptance. Recompute a saved campaign with:

```bash
environments/cpu-apartment/.venv/bin/python scripts/verify_metric_campaign.py \
  --campaign outputs/acceptance/apartment-matrix-new \
  --output outputs/acceptance/apartment-matrix-new/verification.json
```

The verifier uses native tool admissions and continuous physical samples to reconstruct position and accumulated yaw. It checks unchanged 0.05 m / 5° thresholds, five consecutive measured stop samples, upright endpoints, zero obstacle contacts, 14-servo actions, four physics substeps per action and confirmed termination. It compares observer PNG bytes and timestamps with native events, decodes images, verifies result hashes and checks runner/scene hashes against the recorded Git revision. Saved cases must cover the complete declared plan and share one source revision. Existing verification output files are rejected. These checks measure policy-tool execution; independent model task verdicts have their own acceptance.

Current resource authorization permits at most one physical GPU from devices 2–4. Run Newton suites sequentially on that device and confirm resource release before the next workload. RL remains stopped.

## Record the agentic video

Keep the native server running while exporting the completed task. Use the run ID from the voice result, the matching server data directory, and new output paths:

```bash
TMPDIR="$PWD/.cache/tmp" uv sync --project environments/demo --locked
environments/demo/.venv/bin/python scripts/record_harness_demo.py export --base-url http://127.0.0.1:4318 \
  --run-id RUN_ID --data-directory .cache/harness-data --output outputs/demos/run-new
environments/demo/.venv/bin/python scripts/render_microduck_run.py --export outputs/demos/run-new \
  --output outputs/demos/run-new-silent.mp4 --review-dir outputs/demos/run-new-review
environments/demo/.venv/bin/python scripts/add_voice_to_demo.py --video outputs/demos/run-new-silent.mp4 \
  --voice-result outputs/acceptance/voice-task-new/result.json \
  --instruction-audio /absolute/path/command.wav --output outputs/demos/run-new-voice.mp4
```

The renderer uses actual public events and source camera images. The audio step checks both WAV hashes against the task result, requires a successful native run with a passed Verifier result, and fully decodes the finished MP4. The generated manifest preserves source and output hashes, model identity and voice profile revision. The isolated `environments/demo` dependencies are locked; the renderer also requires `ffmpeg` and `ffprobe` on PATH. Supply `--font /absolute/path/unicode.ttf` on Linux for a font that covers the trace text; macOS uses Arial Unicode by default.

## Evidence available on 2026-10-03

| Capability | Evidence | Scope |
|---|---|---|
| Recorded speech → native agent → Newton → Verifier → fixed voice | Run `5dfb18d9-b31b-4226-bc94-9391ddd847ac`, succeeded; target error 0.090916 m; 480 control steps | NVIDIA Office, one seed, simulator ground truth |
| Standard-foot metric tools | 0.5 m request: 0.024196 m final error; 45° request: 0.522914° error; five stopped samples | Official `alpha_walking`, Newton/BAM and native ActionGate |
| Current foot controller, CPU matrix | Three independent sessions; +0.5/−0.5/+1.0 m and ±45°; maximum errors 0.035112 m and 0.737881° | Fixed apartment pose, actual native tools, zero obstacle contacts and closed sessions; Newton revalidation pending |
| Roller metric tools | 0.5 m request: 0.018337 m final error; 45° request: 3.893902° error; five stopped samples | Official `roller`, Hospital, one seed; zero external-contact samples |
| Runtime readiness | Local voice client and remote Newton doctor passed | Installed dependencies, actual model services, scene assets and ten-policy catalogue |
| Agentic video with speech | 91.4 s, 1920 × 1080 MP4, full decoding passed | Actual public events and camera frames; recorded input and confirmed-voice feedback |
| Voice task interruption | CPU run `4a4b3357-257c-40d0-a4e5-00bb2d1e2830` and Newton run `21ff8bf8-cd08-414f-ab88-14dc6d9a9659`; device-confirmed termination; no actions for two seconds afterwards | Actual ASR, model and policy actions; both sessions closed with resources released |
| Newton process initialization | Office worker initialized within the pinned native 180-second budget | Fresh process on the shared host with existing compiled-kernel cache |

The [acceptance report](reports/end-to-end-2026-10-03.md) records paths, revisions and hashes. RL remains stopped. Hardware behavior, recognition accuracy, object carrying, image-only VLN, RTX rendering and broad scene generalization require their own acceptance. Release scope and public capability statements must use these evidence boundaries.

The deployment backend directly constructs the same simulation configuration as the maintained Walking recipe. The interactive worker preserves Newton CUDA graphs, SolverMuJoCo, BAM and the 50 Hz control cadence. Office process initialization passed the pinned native 180-second budget with the existing compiled-kernel cache; installation-time compilation and startup under other host loads need separate measurements.
